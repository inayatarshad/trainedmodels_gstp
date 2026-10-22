"""Plotting helpers used by the notebooks. All figures are saved to results/<module>/figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .data.raster import IGNORE_INDEX


def _save(fig, path):
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def training_curves(history, metric="val_iou", title="", path=None):
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(ep, [h["train_loss"] for h in history])
    ax[0].set(title="Training loss", xlabel="epoch")
    ax[1].plot(ep, [h[metric] for h in history], color="tab:green")
    ax[1].set(title=metric.replace("_", " "), xlabel="epoch")
    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, path)


def rgb(img: np.ndarray, idx=(0, 1, 2)) -> np.ndarray:
    """Percentile-stretched RGB from a [C,H,W] array."""
    x = np.stack([img[i] for i in idx], -1).astype(np.float32)
    lo, hi = np.nanpercentile(x, 2), np.nanpercentile(x, 98)
    return np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)


def overlay(base_rgb, mask, color=(1.0, 0.1, 0.1), alpha=0.45):
    out = base_rgb.copy()
    m = mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * np.asarray(color)
    return out


def prediction_grid(examples, title="", path=None, color=(0.1, 0.4, 1.0)):
    """examples: list of dict(rgb=[H,W,3], mask=[H,W] (0/1/255), prob=[H,W], thr=float, name=str)."""
    n = len(examples)
    fig, axes = plt.subplots(n, 4, figsize=(13, 3.2 * n))
    axes = np.atleast_2d(axes)
    for r, ex in enumerate(examples):
        m = ex["mask"]
        pred = ex["prob"] >= ex["thr"]
        valid = m != IGNORE_INDEX
        err = np.zeros(m.shape + (3,), np.float32)
        err[valid & pred & (m == 1)] = (0.2, 0.8, 0.2)   # true positive  - green
        err[valid & pred & (m == 0)] = (0.9, 0.2, 0.2)   # false positive - red
        err[valid & ~pred & (m == 1)] = (1.0, 0.8, 0.0)  # false negative - yellow
        panels = [ex["rgb"], overlay(ex["rgb"], m == 1, color), overlay(ex["rgb"], pred, color), err]
        names = ["Input", "Ground truth", "Prediction", "TP green / FP red / FN yellow"]
        for c, (p, nm) in enumerate(zip(panels, names)):
            axes[r, c].imshow(p)
            axes[r, c].set_xticks([]), axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(nm, fontsize=10)
        axes[r, 0].set_ylabel(ex.get("name", ""), fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, path)


def confusion(cm, class_names, title="", path=None, normalise=True):
    cm = np.asarray(cm, dtype=np.float64)
    if normalise:
        cm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalise else None)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cm[i, j]:.2f}" if normalise else int(cm[i, j]), ha="center", va="center",
                    fontsize=7, color="white" if cm[i, j] > 0.5 * cm.max() else "black")
    ax.set(xlabel="Predicted", ylabel="True", title=title)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    return _save(fig, path)


def threshold_curve(meter, chosen, title="", path=None):
    s = [meter.compute(i) for i in range(len(meter.thresholds))]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for k in ("iou", "f1", "precision", "recall"):
        ax.plot(meter.thresholds, [x[k] for x in s], label=k)
    ax.axvline(chosen, color="k", ls="--", lw=1, label=f"chosen {chosen:.2f}")
    ax.set(xlabel="threshold", title=title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, path)
