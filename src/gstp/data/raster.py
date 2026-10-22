"""Raster reading and normalisation shared by all image datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

IGNORE_INDEX = 255


def read_raster(path: str | Path, bands: Sequence[int] | None = None) -> np.ndarray:
    """Read a GeoTIFF as float32 (C, H, W). `bands` are 0-based indices."""
    import rasterio

    with rasterio.open(path) as src:
        idx = None if bands is None else [b + 1 for b in bands]
        arr = src.read(idx).astype(np.float32)
    if arr.ndim == 2:
        arr = arr[None]
    return arr


def read_mask(path: str | Path) -> np.ndarray:
    """Read a label mask and return uint8 (H, W) with 0, 1 or IGNORE_INDEX.

    Handles the three conventions used by our datasets:
      * GeoTIFF with -1 (no data), 0, 1    -> Sen1Floods11, HLS Burn Scars
      * PNG with 0 / 255 (or 0 / 1)        -> Amazon / Atlantic Forest masks
    """
    path = Path(path)
    if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        from PIL import Image

        m = np.array(Image.open(path))
        if m.ndim == 3:
            m = m[..., 0]
        return (m > (127 if m.max() > 1 else 0)).astype(np.uint8)
    m = read_raster(path)[0]
    out = np.full(m.shape, IGNORE_INDEX, dtype=np.uint8)
    valid = np.isfinite(m) & (m >= 0)
    out[valid & (m > 0.5)] = 1
    out[valid & (m <= 0.5)] = 0
    return out


def clean(arr: np.ndarray, nodata: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Replace NaN / inf / nodata with 0 and return (array, valid_pixel_mask)."""
    valid = np.all(np.isfinite(arr), axis=0)
    if nodata is not None:
        valid &= np.all(arr != nodata, axis=0)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    if nodata is not None:
        arr = np.where(arr == nodata, 0.0, arr)
    return arr.astype(np.float32), valid


def compute_band_stats(paths, reader, n_max: int = 200, nodata=None, seed: int = 0) -> dict:
    """Per-band mean / std estimated from up to n_max training images.

    Statistics are always computed on the *training* split only so that no
    information from validation or test data leaks into preprocessing.
    """
    rng = np.random.default_rng(seed)
    paths = list(paths)
    if len(paths) > n_max:
        paths = [paths[i] for i in rng.choice(len(paths), n_max, replace=False)]
    sums = sq = None
    count = None
    for p in paths:
        arr, valid = clean(reader(p), nodata)
        flat = arr[:, valid].astype(np.float64)
        if sums is None:
            sums = np.zeros(arr.shape[0])
            sq = np.zeros(arr.shape[0])
            count = 0
        sums += flat.sum(1)
        sq += (flat**2).sum(1)
        count += flat.shape[1]
    mean = sums / max(count, 1)
    std = np.sqrt(np.maximum(sq / max(count, 1) - mean**2, 1e-12))
    return {"mean": mean.astype(np.float32).tolist(), "std": std.astype(np.float32).tolist()}


def normalise(arr: np.ndarray, method: str, stats: dict | None = None) -> np.ndarray:
    """Normalise a (C, H, W) array.

    zscore      -> (x - mean) / std using training-set statistics
    percentile  -> per-image, per-band 2-98 % stretch to [0, 1]; makes the model
                   insensitive to the unknown byte-scaling used by a dataset and
                   to radiometric differences between processing levels.
    """
    if method == "zscore":
        mean = np.asarray(stats["mean"], dtype=np.float32)[:, None, None]
        std = np.asarray(stats["std"], dtype=np.float32)[:, None, None]
        return (arr - mean) / std
    if method == "percentile":
        out = np.empty_like(arr, dtype=np.float32)
        for c in range(arr.shape[0]):
            lo, hi = np.percentile(arr[c], (2, 98))
            out[c] = np.clip((arr[c] - lo) / max(hi - lo, 1e-6), 0, 1)
        return out
    if method == "none":
        return arr.astype(np.float32)
    raise ValueError(f"Unknown normalisation: {method}")
