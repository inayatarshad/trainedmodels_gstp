"""Classic deblurring baselines and image-quality metrics.

  Wiener filter            : one-step inverse filter with a noise-to-signal constant K
  Richardson-Lucy (RL)     : iterative maximum-likelihood deconvolution; iterations act as regularisation
  Unsharp mask             : plain sharpening, no PSF needed (what an FPGA FIR filter can do)

Wiener and RL need the PSF. They are scored two ways:
  * true PSF    - the exact PSF used to blur each test image (best case for the classic methods;
                  on orbit the PSF is never known this exactly)
  * nominal PSF - one fixed PSF for every image (what you have if only a lab-measured PSF is known)
The learned model never gets a PSF.

All free parameters (K, number of RL iterations, unsharp amount) are tuned on the
VALIDATION images; the test images are scored once with the tuned values.

Everything runs on torch tensors, so it uses the GPU when one is available.
"""

from __future__ import annotations

import numpy as np

from .blur import gaussian_psf, convolve_psfs, pad_reflect, psf_radius, psf_to_otf, fft_conv, blur

# A "nominal" PSF: what a lab measurement of a slightly soft camera might give.
NOMINAL_PSF = convolve_psfs(gaussian_psf(0.5), gaussian_psf(1.2))


def _margin(psf) -> int:
    return max(16, 3 * psf_radius(np.asarray(psf)))


def wiener(y, psf, k: float):
    """Wiener deconvolution of a (..., H, W) tensor with constant noise-to-signal ratio k."""
    import torch

    m = _margin(psf)
    yp = pad_reflect(y, m)
    shape = yp.shape[-2:]
    H = psf_to_otf(psf, shape, y.device)
    X = torch.conj(H) * torch.fft.rfft2(yp) / (H.abs() ** 2 + k)
    x = torch.fft.irfft2(X, s=shape)
    return x[..., m:-m, m:-m]


def richardson_lucy(y, psf, iterations, eps: float = 1e-4):
    """Richardson-Lucy deconvolution.

    `iterations` may be an int or a list of ints; with a list, the estimate at
    each listed iteration is returned as a dict {iteration: image}, which lets
    the iteration count be tuned in a single run.
    """
    import torch

    its = [iterations] if isinstance(iterations, int) else sorted(iterations)
    m = _margin(psf)
    yp = torch.clamp(pad_reflect(y, m), min=eps)
    shape = yp.shape[-2:]
    H = psf_to_otf(psf, shape, y.device)
    Hc = torch.conj(H)  # correlation with the PSF = convolution with the flipped PSF
    x = yp.clone()
    out = {}
    for i in range(1, its[-1] + 1):
        est = torch.clamp(fft_conv(x, H, shape), min=eps)
        x = x * fft_conv(yp / est, Hc, shape)
        if i in its:
            out[i] = x[..., m:-m, m:-m].clone()
    return out[its[0]] if isinstance(iterations, int) else out


def unsharp(y, amount: float, sigma: float = 1.5):
    return y + amount * (y - blur(y, gaussian_psf(sigma)))


# ───────────────────────────────────────────────────────────────── metrics
def eval_mask(valid: np.ndarray, border: int = 8) -> np.ndarray:
    """Pixels used for scoring: valid data, away from the image border."""
    m = valid.copy().astype(bool)
    if border:
        m[:border] = m[-border:] = False
        m[:, :border] = m[:, -border:] = False
    return m


def psnr(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray, data_range: float = 1.0) -> float:
    """PSNR (dB) over masked pixels of all bands. gt/pred: (C, H, W)."""
    err = (gt - pred)[:, mask]
    mse = float(np.mean(err.astype(np.float64) ** 2))
    return float(10 * np.log10(data_range**2 / max(mse, 1e-12)))


def ssim(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray, data_range: float = 1.0) -> float:
    """Mean SSIM over masked pixels, averaged over bands."""
    from skimage.metrics import structural_similarity

    vals = []
    for c in range(gt.shape[0]):
        _, smap = structural_similarity(gt[c], pred[c], data_range=data_range, full=True)
        vals.append(float(smap[mask].mean()))
    return float(np.mean(vals))


def score(gt, pred, valid, border=8) -> dict:
    m = eval_mask(valid, border)
    pred = np.clip(pred, 0, 1)
    return {"psnr": psnr(gt, pred, m), "ssim": ssim(gt, pred, m)}


# ─────────────────────────────────────────────────────── tune & evaluate
def _np(t):
    return t.detach().float().cpu().numpy()


def _psf_for(item, psf_mode):
    return item["psf"] if psf_mode == "true" else NOMINAL_PSF


def tune(val_items, device="cpu", k_grid=None, rl_iters=(5, 10, 15, 20, 30, 50, 80), amounts=None, log=print) -> dict:
    """Choose K (Wiener), RL iterations and unsharp amount on validation items (PSNR)."""
    import torch

    k_grid = k_grid if k_grid is not None else np.logspace(-4, -0.5, 15)
    amounts = amounts if amounts is not None else [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    acc = {("wiener", mode, float(k)): [] for mode in ("true", "nominal") for k in k_grid}
    acc.update({("rl", mode, it): [] for mode in ("true", "nominal") for it in rl_iters})
    acc.update({("unsharp", "none", float(a)): [] for a in amounts})
    for it_idx, item in enumerate(val_items):
        y = torch.as_tensor(item["blurred"], device=device)
        gt, m = item["sharp"], eval_mask(item["valid"])
        p = lambda t: psnr(gt, np.clip(_np(t), 0, 1), m)  # noqa: E731  (PSNR only: SSIM is too slow for a grid)
        for mode in ("true", "nominal"):
            psf = _psf_for(item, mode)
            for k in k_grid:
                acc[("wiener", mode, float(k))].append(p(wiener(y, psf, k)))
            for it, x in richardson_lucy(y, psf, list(rl_iters)).items():
                acc[("rl", mode, it)].append(p(x))
        for a in amounts:
            acc[("unsharp", "none", float(a))].append(p(unsharp(y, a)))
        if (it_idx + 1) % 10 == 0:
            log(f"  tuned on {it_idx + 1}/{len(val_items)} validation images")
    mean = {key: float(np.mean(v)) for key, v in acc.items()}
    best = {}
    for method, mode in (("wiener", "true"), ("wiener", "nominal"), ("rl", "true"), ("rl", "nominal"), ("unsharp", "none")):
        cands = {p: s for (m, md, p), s in mean.items() if m == method and md == mode}
        p = max(cands, key=cands.get)
        best[f"{method}_{mode}"] = {"param": p, "val_psnr": cands[p], "curve": cands}
    return best


def method_table(best: dict) -> dict:
    """Map display names to functions (y_tensor, item) -> restored tensor, using tuned parameters."""
    k_t, k_n = best["wiener_true"]["param"], best["wiener_nominal"]["param"]
    i_t, i_n = int(best["rl_true"]["param"]), int(best["rl_nominal"]["param"])
    a = best["unsharp_none"]["param"]
    return {
        "Blurred input (no fix)": lambda y, it: y,
        f"Unsharp mask (amount {a:g})": lambda y, it: unsharp(y, a),
        f"Wiener, nominal PSF (K={k_n:.2g})": lambda y, it: wiener(y, NOMINAL_PSF, k_n),
        f"Richardson-Lucy, nominal PSF ({i_n} it)": lambda y, it: richardson_lucy(y, NOMINAL_PSF, i_n),
        f"Wiener, true PSF (K={k_t:.2g})": lambda y, it: wiener(y, it["psf"], k_t),
        f"Richardson-Lucy, true PSF ({i_t} it)": lambda y, it: richardson_lucy(y, it["psf"], i_t),
    }


def evaluate(items, methods: dict, device="cpu", log=print, keep=None) -> list[dict]:
    """Score every method on every item. Returns one row per (item, method).

    keep: optional set of item ids whose restored images are returned (for figures)."""
    import torch

    rows, kept = [], {}
    for n, item in enumerate(items):
        y = torch.as_tensor(item["blurred"], device=device)
        for name, fn in methods.items():
            out = _np(fn(y, item))
            rows.append({"id": item["id"], "dataset": item.get("dataset", ""), "blur_type": item["blur_type"],
                         "method": name, **score(item["sharp"], out, item["valid"])})
            if keep and item["id"] in keep:
                kept.setdefault(item["id"], {})[name] = np.clip(out, 0, 1)
        if (n + 1) % 20 == 0:
            log(f"  scored {n + 1}/{len(items)} test images")
    return (rows, kept) if keep else rows
