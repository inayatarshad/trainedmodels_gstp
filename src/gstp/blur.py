"""Synthetic blur for the deblurring stage (sharp image -> blurred image).

There are no blurry GSTP images with a sharp "right answer" yet, so sharp
satellite images are blurred on purpose and the deblurring methods are scored
against the sharp original. Blur types (as agreed with the manager):

  motion   : platform motion / attitude jitter during the exposure -> streaks
             (straight line, or a random wandering path for jitter)
  defocus  : lens out of focus -> disk PSF
  gaussian : general optical softness -> Gaussian PSF
  none     : check group only - no extra blur, to see whether a method damages
             an image that is already sharp (double-sharpening risk)

Every image also gets a weak base optics blur and a SMALL leftover noise.
Noise removal is done on the FPGA (dark-reference subtraction, FIR filtering),
so the deblurring model must not be a denoiser; the leftover noise is only
there so that a method that amplifies noise is penalised. Set NOISE_STD to the
value measured after the FPGA once it is known (0 if almost none is left).

Units are reflectance in [0, 1]; blur sizes are in image pixels and are
placeholders until the camera PSF / MTF and exposure time are known.
All PSFs are odd-sized, non-negative and sum to 1 (brightness is kept).
"""

from __future__ import annotations

import numpy as np

BLUR_TYPES = ("motion", "defocus", "gaussian")   # the three blur types we train and report on
CHECK_TYPE = "none"                              # extra test group: no extra blur
ALL_TYPES = BLUR_TYPES + (CHECK_TYPE,)
TRAIN_MIX = {"motion": 0.3, "defocus": 0.3, "gaussian": 0.3, "none": 0.1}

# Parameter ranges in pixels (train draws from these; val/test use fixed seeded draws)
RANGES = {
    "optics_sigma": (0.3, 0.6),    # weak base optics blur added to every image
    "gaussian_sigma": (0.8, 2.5),
    "defocus_radius": (1.0, 4.0),
    "motion_length": (3.0, 15.0),  # straight-line motion and jitter path length
    "noise_std": (0.0, 0.002),     # leftover noise after the FPGA, reflectance units (placeholder)
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
    """Draw one degradation: blur type + parameters + noise level (all JSON-serialisable)."""
    if blur_type is None:
        blur_type = str(rng.choice(list(TRAIN_MIX), p=list(TRAIN_MIX.values())))
    bt = blur_type
    u = lambda key: float(rng.uniform(*RANGES[key]))  # noqa: E731
    r = {"blur_type": bt, "optics_sigma": u("optics_sigma"), "noise_std": u("noise_std"),
         "psf_seed": int(rng.integers(2**31))}
    if bt == "gaussian":
        r["sigma"] = u("gaussian_sigma")
    elif bt == "defocus":
        r["radius"] = u("defocus_radius")
    elif bt == "motion":
        r["kind"] = "line" if rng.random() < 0.5 else "jitter"
        r["length"] = u("motion_length")
        if r["kind"] == "line":
            r["angle"] = float(rng.uniform(0, 180))
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
    elif bt == "motion" and r.get("kind", "line") == "line":
        extra = motion_psf(r["length"], r["angle"])
    elif bt == "motion":
        extra = jitter_psf(r["length"], np.random.default_rng(r["psf_seed"]))
    else:
        raise ValueError(f"unknown blur type {bt}")
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


def add_noise(y, noise_std: float, generator=None):
    """Small leftover Gaussian noise (what is left after the FPGA noise removal)."""
    import torch

    if noise_std <= 0:
        return y
    return y + noise_std * torch.randn(y.shape, generator=generator, device=y.device, dtype=y.dtype)


def degrade(x, recipe: dict, seed: int | None = None):
    """Apply a recipe to a sharp (..., H, W) reflectance image. Returns (blurred, psf).

    The same PSF is applied to every band: platform motion, jitter and defocus
    affect all bands of one exposure the same way.
    """
    import torch

    psf = psf_from_recipe(recipe)
    y = blur(x, psf)
    g = torch.Generator(device=y.device)
    g.manual_seed(int(recipe["psf_seed"] if seed is None else seed))
    y = add_noise(y, recipe["noise_std"], g)
    return y, psf
