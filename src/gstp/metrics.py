"""Evaluation metrics.

Segmentation metrics are accumulated as pixel counts over the whole split
(micro-averaged), which is the convention used by the Sen1Floods11 and
HLS Burn Scars benchmarks. Pixels marked as no-data are excluded.
"""

from __future__ import annotations

import numpy as np


class BinarySegMeter:
    """Accumulates TP/FP/FN/TN for a binary mask at one or more thresholds."""

    def __init__(self, thresholds=(0.5,)):
        self.thresholds = np.asarray(thresholds, dtype=np.float32)
        self.reset()

    def reset(self):
        n = len(self.thresholds)
        self.tp = np.zeros(n, dtype=np.int64)
        self.fp = np.zeros(n, dtype=np.int64)
        self.fn = np.zeros(n, dtype=np.int64)
        self.tn = np.zeros(n, dtype=np.int64)

    def update(self, prob: np.ndarray, target: np.ndarray, valid: np.ndarray | None = None):
        """prob: probabilities in [0,1]; target: {0,1}; valid: bool mask of pixels to score."""
        prob = np.asarray(prob, dtype=np.float32).ravel()
        target = np.asarray(target).ravel().astype(bool)
        if valid is not None:
            v = np.asarray(valid).ravel().astype(bool)
            prob, target = prob[v], target[v]
        for i, t in enumerate(self.thresholds):
            pred = prob >= t
            self.tp[i] += np.count_nonzero(pred & target)
            self.fp[i] += np.count_nonzero(pred & ~target)
            self.fn[i] += np.count_nonzero(~pred & target)
            self.tn[i] += np.count_nonzero(~pred & ~target)

    def compute(self, idx: int = 0) -> dict:
        return binary_scores(self.tp[idx], self.fp[idx], self.fn[idx], self.tn[idx], self.thresholds[idx])

    def best(self, key: str = "iou") -> dict:
        scores = [self.compute(i) for i in range(len(self.thresholds))]
        return max(scores, key=lambda s: s[key])


def _safe_div(a, b):
    return float(a) / float(b) if b > 0 else 0.0


def binary_scores(tp, fp, fn, tn, threshold=0.5) -> dict:
    iou_pos = _safe_div(tp, tp + fp + fn)
    iou_neg = _safe_div(tn, tn + fp + fn)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "threshold": float(threshold),
        "iou": iou_pos,
        "iou_background": iou_neg,
        "miou": (iou_pos + iou_neg) / 2.0,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": _safe_div(tp + tn, tp + fp + fn + tn),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def classification_report(y_true, y_pred, class_names) -> dict:
    """Accuracy, macro/weighted F1, per-class precision/recall/F1 and confusion matrix."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    k = len(class_names)
    cm = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    per_class = {}
    f1s, supports = [], []
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        p = _safe_div(tp, tp + fp)
        r = _safe_div(tp, tp + fn)
        f = _safe_div(2 * p * r, p + r) if (p + r) > 0 else 0.0
        per_class[name] = {"precision": p, "recall": r, "f1": f, "support": int(cm[i, :].sum())}
        f1s.append(f)
        supports.append(cm[i, :].sum())
    supports = np.asarray(supports, dtype=np.float64)
    return {
        "accuracy": _safe_div(np.trace(cm), cm.sum()),
        "macro_f1": float(np.mean(f1s)),
        "weighted_f1": float(np.sum(np.asarray(f1s) * supports) / max(supports.sum(), 1)),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "class_names": list(class_names),
    }


def regression_scores(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    ss_res = np.sum(err**2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "n": int(len(y_true)),
    }
