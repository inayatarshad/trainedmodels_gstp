"""Synthetic blur for the deblurring stage (sharp image -> blurred, noisy image).

The GSTP camera's real blur is not characterised yet, so training and test
pairs are made by blurring sharp satellite images with the blur types an
orbiting push-frame camera can suffer:

  gaussian : optics / detector MTF (always present to some degree)
  defocus  : focus shift (thermal, launch loads)  -> disk PSF
  motion   : platform motion during the exposure  -> straight line PSF
  jitter   : reaction-wheel / attitude jitter     -> random wandering-path PSF
  none     : no extra blur (checks that a method does not damage sharp input)

Every blurred image also gets the base optics blur and sensor noise
(read noise + photon noise). Units are reflectance in [0, 1]; the blur size is
in image pixels. When the camera PSF / MTF and exposure time are known, the
parameter ranges below should be replaced with the measured values.

All PSFs are odd-sized, non-negative and sum to 1, so blurring keeps the mean
brightness.
"""

from __future__ import annotations

import numpy as np

BLUR_TYPES = ("none", "gaussian", "defocus", "motion", "jitter")

# Parameter ranges in pixels (train draws from these; test uses fixed seeded draws)
RANGES = {
    "optics_sigma": (0.3, 0.7),   # base MTF blur added to every image
    "gaussian_sigma": (0.8, 2.5),
    "defocus_radius": (1.0, 4.0),
    "motion_length": (3.0, 15.0),
    "jitter_length": (4.0, 15.0),
    "read_noise": (0.001, 0.008),  # std in reflectance units
    "shot_gain": (0.0, 1.5e-4),    # variance per unit signal (photon noise)
}


# ───────────────────────────────────────────────────────────── PSF shapes
def _odd(n: float) -> int:
    n = int(np.ceil(n))
    return n + 1 if n % 2 == 0 else n


def _normalise(k: np.ndarray) -> np.ndarray:
    k = np.clip(k, 0, None).astype(np.float64)
    s = k.sum()
    if s <= 0:
        raise ValueError("empty PSF")
    return (k / s).astype(np.float32)


def delta_psf() -> np.ndarray:
    return np.ones((1, 1), np.float32)


def gaussian_psf(sigma: float, size: int | None = None) -> np.ndarray:
    if sigma <= 0:
        return delta_psf()
    size = size or _odd(6 * sigma + 1)
    r = np.arange(size) - size // 2
    g = np.exp(-(r**2) / (2 * sigma**2))
    return _normalise(np.outer(g, g))


def _supersample(fn, size: int, ss: int = 8) -> np.ndarray:
    """Evaluate a 0/1 shape on an ss-times finer grid and average -> anti-aliased PSF."""
    c = (np.arange(size * ss) + 0.5) / ss - size / 2
    yy, xx = np.meshgrid(c, c, indexing="ij")
    fine = fn(yy, xx).astype(np.float64)
    return fine.reshape(size, ss, size, ss).mean(axis=(1, 3))


def defocus_psf(radius: float) -> np.ndarray:
    """Uniform disk (geometric defocus) of the given radius in pixels."""
    size = _odd(2 * radius + 3)
    return _normalise(_supersample(lambda y, x: (x**2 + y**2) <= radius**2, size))


def _splat_path(points: np.ndarray, size: int) -> np.ndarray:
    """Accumulate a densely sampled path (N, 2) of (y, x) offsets onto a grid with bilinear weights."""
    k = np.zeros((size, size), np.float64)
    p = points + size // 2
    y0, x0 = np.floor(p[:, 0]).astype(int), np.floor(p[:, 1]).astype(int)
    fy, fx = p[:, 0] - y0, p[:, 1] - x0
    for dy, dx, w in ((0, 0, (1 - fy) * (1 - fx)), (1, 0, fy * (1 - fx)), (0, 1, (1 - fy) * fx), (1, 1, fy * fx)):
        yy, xx = y0 + dy, x0 + dx
        ok = (yy >= 0) & (yy < size) & (xx >= 0) & (xx < size)
        np.add.at(k, (yy[ok], xx[ok]), w[ok])
    return k


def motion_psf(length: float, angle_deg: float) -> np.ndarray:
    """Straight-line motion of `length` pixels at `angle_deg` (0 = along x)."""
    if length < 1:
        return delta_psf()
    t = np.linspace(-0.5, 0.5, int(length * 16) + 2) * length
    a = np.deg2rad(angle_deg)
    pts = np.stack([t * np.sin(a), t * np.cos(a)], 1)
    return _normalise(_splat_path(pts, _odd(length + 3)))


def jitter_psf(length: float, rng: np.random.Generator, inertia: float = 0.7) -> np.ndarray:
    """Random camera-shake path (Boracchi & Foi, 2012, simplified).

    The line of sight moves with random, partly persistent velocity changes;
    the path is rescaled to `length` pixels of total travel.
    """
    n = 64
    v = rng.normal(size=2)
    vel = []
    for _ in range(n):
        v = inertia * v + (1 - inertia) * rng.normal(size=2) * 2.0
        vel.append(v.copy())
    path = np.cumsum(np.array(vel), 0)
    path -= path.mean(0)
    travel = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
    path *= length / max(travel, 1e-6)
    # resample densely so the splatted path has no gaps
    t = np.linspace(0, n - 1, n * 32)
    dense = np.stack([np.interp(t, np.arange(n), path[:, i]) for i in range(2)], 1)
    size = _odd(2 * np.abs(dense).max() + 3)
    return _normalise(_splat_path(dense, size))


def convolve_psfs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    from scipy.signal import fftconvolve

    return _normalise(fftconvolve(a, b, mode="full"))


# ───────────────────────────────────────────────────── degradation recipe
def sample_recipe(rng: np.random.Generator, blur_type: str | None = None) -> dict:
    """Draw one degradation: blur type + parameters + noise levels (all JSON-serialisable)."""
    bt = blur_type or str(rng.choice(BLUR_TYPES))
    u = lambda key: float(rng.uniform(*RANGES[key]))  # noqa: E731
    r = {"blur_type": bt, "optics_sigma": u("optics_sigma"), "read_noise": u("read_noise"),
         "shot_gain": u("shot_gain"), "psf_seed": int(rng.integers(2**31))}
    if bt == "gaussian":
        r["sigma"] = u("gaussian_sigma")
    elif bt == "defocus":
        r["radius"] = u("defocus_radius")
    elif bt == "motion":
        r["length"], r["angle"] = u("motion_length"), float(rng.uniform(0, 180))
    elif bt == "jitter":
        r["length"] = u("jitter_length")
    elif bt != "none":
        raise ValueError(f"unknown blur type {bt}")
    return r


def psf_from_recipe(r: dict) -> np.ndarray:
    base = gaussian_psf(r["optics_sigma"])
    bt = r["blur_type"]
    if bt == "none":
        return base
    if bt == "gaussian":
        extra = gaussian_psf(r["sigma"])
    elif bt == "defocus":
        extra = defocus_psf(r["radius"])
    elif bt == "motion":
        extra = motion_psf(r["length"], r["angle"])
    else:  # jitter
        extra = jitter_psf(r["length"], np.random.default_rng(r["psf_seed"]))
    return convolve_psfs(base, extra)


def psf_radius(psf: np.ndarray) -> int:
    return max(psf.shape) // 2


# ─────────────────────────────────────────────────── convolution (torch FFT)
def _to_tensor(x):
    import torch

    return x if isinstance(x, torch.Tensor) else torch.as_tensor(np.ascontiguousarray(x))


def pad_reflect(x, p: int):
    """Reflect-pad the last two dims of a (..., H, W) tensor by p (handles p larger than the image)."""
    import torch.nn.functional as F

    if p <= 0:
        return x
    shape = x.shape
    x4 = x.reshape(-1, 1, *shape[-2:])
    while p > 0:  # torch reflect padding needs p < size; pad in steps if necessary
        step = min(p, x4.shape[-1] - 1, x4.shape[-2] - 1)
        x4 = F.pad(x4, (step, step, step, step), mode="reflect")
        p -= step
    return x4.reshape(*shape[:-2], *x4.shape[-2:])


def psf_to_otf(psf, shape, device=None, dtype=None):
    """Zero-pad the PSF to `shape` and centre it at (0, 0) so that FFT multiplication = convolution."""
    import torch

    psf = _to_tensor(psf).to(device=device, dtype=dtype or torch.float32)
    k = torch.zeros(shape, device=psf.device, dtype=psf.dtype)
    kh, kw = psf.shape
    k[:kh, :kw] = psf
    k = torch.roll(k, shifts=(-(kh // 2), -(kw // 2)), dims=(0, 1))
    return torch.fft.rfft2(k)


def fft_conv(x, otf, shape):
    import torch

    return torch.fft.irfft2(torch.fft.rfft2(x) * otf, s=shape)


def blur(x, psf):
    """Blur a (..., H, W) image with a PSF, reflect-padded at the borders (no wrap-around)."""
    import torch

    x = _to_tensor(x).float()
    p = psf_radius(np.asarray(psf)) + 1
    xp = pad_reflect(x, p)
    shape = xp.shape[-2:]
    y = fft_conv(xp, psf_to_otf(psf, shape, x.device), shape)
    return y[..., p:-p, p:-p].contiguous() if p > 0 else y


def add_noise(y, read_noise: float, shot_gain: float, generator=None):
    """Sensor noise: Gaussian read noise + signal-dependent photon noise (Gaussian approximation)."""
    import torch

    std = torch.sqrt(read_noise**2 + shot_gain * torch.clamp(y, min=0))
    return y + std * torch.randn(y.shape, generator=generator, device=y.device, dtype=y.dtype)


def degrade(x, recipe: dict, seed: int | None = None):
    """Apply a recipe to a sharp (..., H, W) reflectance image. Returns (blurred_noisy, psf).

    The same PSF is applied to every band: platform motion, jitter and defocus
    affect all bands of one exposure the same way.
    """
    import torch

    psf = psf_from_recipe(recipe)
    y = blur(x, psf)
    g = torch.Generator(device=y.device)
    g.manual_seed(int(recipe["psf_seed"] if seed is None else seed))
    y = add_noise(y, recipe["read_noise"], recipe["shot_gain"], g)
    return y, psf
