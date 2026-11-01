"""Downstream check: do the hazard models work better on deblurred images?

For each test chip the trained hazard model (flood S2 U-Net or burn-scar
U-Net) is run on the sharp original, on the blurred image and on every
deblurred version. Pixel counts are accumulated per method (and per blur
type), giving IoU against the hand-made labels. The decision threshold is the
one each hazard model chose on its own validation set - it is not re-tuned.

All versions go through the same preprocessing as the hazard model's training
(reflectance x scale, z-score with the model's training statistics, no-data
set to 0), so the only difference between rows is the image itself.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .metrics import BinarySegMeter, binary_scores

SHARP = "Sharp original (upper bound)"


class HazardScorer:
    def __init__(self, ckpt_path, scale: float, device="cpu", name=""):
        import torch

        from .models import build_unet

        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ck["config"]
        self.model = build_unet(cfg["in_channels"], pretrained=False)
        self.model.load_state_dict(ck["state_dict"])
        self.model = self.model.eval().to(device)
        st = cfg["extra"]["stats"]
        self.mean = np.asarray(st["mean"], np.float32)[:, None, None]
        self.std = np.asarray(st["std"], np.float32)[:, None, None]
        self.threshold = float(ck.get("threshold", 0.5))
        self.scale, self.device, self.name = scale, device, name
        self.meters = defaultdict(lambda: BinarySegMeter([self.threshold]))
        self._sharp_done = set()

    def _prob(self, refl: np.ndarray, valid_data: np.ndarray) -> np.ndarray:
        import torch

        x = (np.clip(refl, 0, 1) * self.scale - self.mean) / self.std
        x[:, ~valid_data] = 0.0
        with torch.no_grad():
            p = torch.sigmoid(self.model(torch.from_numpy(x.astype(np.float32))[None].to(self.device)))
        return p.squeeze().cpu().numpy()

    def _update(self, key, item, refl):
        lab = item["label"]
        ok = (lab != 255) & item["valid_data"]
        self.meters[key].update(self._prob(refl, item["valid_data"]), lab == 1, ok)

    def __call__(self, item, method, refl):
        """Callback for deblur_classic.evaluate(on_output=...)."""
        if item.get("label") is None:
            return
        if item["id"] not in self._sharp_done:
            self._sharp_done.add(item["id"])
            for key in ((SHARP, "all"), (SHARP, item["blur_type"])):
                self._update(key, item, item["sharp"])
        for key in ((method, "all"), (method, item["blur_type"])):
            self._update(key, item, refl)

    def table(self) -> list[dict]:
        rows = []
        for (method, group), m in self.meters.items():
            s = binary_scores(m.tp[0], m.fp[0], m.fn[0], m.tn[0], self.threshold)
            rows.append({"hazard_model": self.name, "method": method, "blur_type": group, "iou": s["iou"],
                         "f1": s["f1"], "precision": s["precision"], "recall": s["recall"]})
        return rows
