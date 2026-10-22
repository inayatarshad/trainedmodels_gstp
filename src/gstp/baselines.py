"""Spectral-index threshold baselines — the approach CRISP uses today.

CRISP classifies hazards with fixed thresholds on indices (NDVI, NBR, SAR
backscatter). To measure what the learned models add, each one is compared
with the best single-index threshold, where the threshold is tuned on the
validation split (a stronger baseline than a fixed textbook value) and then
scored on the same test chips as the model.
"""

from __future__ import annotations

import numpy as np

from .data.raster import IGNORE_INDEX
from .data.segmentation import SegDataset
from .metrics import BinarySegMeter


def normalized_difference(a, b):
    return (a - b) / np.maximum(a + b, 1e-6)


def _index_maps(samples, index_fn, nodata=None):
    ds = SegDataset(samples, norm="none", nodata=nodata)
    for i in range(len(ds)):
        img, mask = ds.load(i)
        yield index_fn(img), mask


def tune_and_test(val_samples, test_sets: dict, index_fn, direction: str, nodata=None, n_thresholds=99):
    """direction '>' : positive class where index > t ; '<' : where index < t.

    Returns dict with chosen threshold, validation scores and per-test-set scores.
    """
    sign = 1.0 if direction == ">" else -1.0
    vals = []
    for idx, mask in _index_maps(val_samples, index_fn, nodata):
        v = mask != IGNORE_INDEX
        vals.append(idx[v][:: max(1, v.sum() // 20000)])
    grid = np.unique(np.quantile(np.concatenate(vals) * sign, np.linspace(0.01, 0.99, n_thresholds)))

    def score(samples):
        meter = BinarySegMeter(grid)
        for idx, mask in _index_maps(samples, index_fn, nodata):
            meter.update(idx * sign, mask == 1, mask != IGNORE_INDEX)
        return meter

    val_meter = score(val_samples)
    best = val_meter.best("iou")
    t_idx = int(np.argmin(np.abs(grid - best["threshold"])))
    out = {"direction": direction, "threshold": float(best["threshold"] * sign), "val": best}
    for name, samples in test_sets.items():
        out[name] = score(samples).compute(t_idx)
        out[name]["threshold"] = float(best["threshold"] * sign)
    return out
