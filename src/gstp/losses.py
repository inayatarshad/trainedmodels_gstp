"""Loss for binary segmentation with ignored pixels.

BCE + Dice: BCE gives stable per-pixel gradients; Dice directly optimises
overlap and counters class imbalance (water / burn scars are a minority of
pixels in most chips).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .data.raster import IGNORE_INDEX


class BCEDiceLoss(torch.nn.Module):
    def __init__(self, bce_weight: float = 0.5, pos_weight: float | None = None, smooth: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.pos_weight = pos_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = logits.squeeze(1)
        valid = target != IGNORE_INDEX
        if valid.sum() == 0:
            return logits.sum() * 0.0
        y = target.clone().float()
        y[~valid] = 0
        pw = None if self.pos_weight is None else torch.tensor(self.pos_weight, device=logits.device)
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none", pos_weight=pw)
        bce = (bce * valid).sum() / valid.sum()
        p = torch.sigmoid(logits) * valid
        y = y * valid
        inter = (p * y).sum()
        dice = 1 - (2 * inter + self.smooth) / (p.sum() + y.sum() + self.smooth)
        return self.bce_weight * bce + (1 - self.bce_weight) * dice
