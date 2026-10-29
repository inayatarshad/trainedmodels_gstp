"""Blurred / sharp image pairs for the deblurring stage.

Sharp images are the optical chips already used for the hazard models
(HLS Burn Scars, 6 bands at 30 m; Sen1Floods11 Sentinel-2, 12 bands at 10 m),
converted to reflectance in [0, 1]. Blur and noise are added synthetically
(see gstp.blur). Using the same chips and the same train / val / test splits as
the hazard models means the test images can later be passed through the
flood and burn-scar models to measure whether deblurring restores their IoU.

Train  : random chip, random band, random crop, new random blur every time.
Val/test: every chip gets ONE fixed, seeded blur (blur types assigned in turn,
          so each type is equally represented), applied to all its bands.
          The recipes are saved so the exact test set can be rebuilt.
"""

from __future__ import annotations

import numpy as np

from .. import blur as B
from .raster import clean
from .segmentation import Sample, read_sample_image


def load_reflectance(sample: Sample, scale: float, nodata: float | None = None):
    """Chip as reflectance (C, H, W) clipped to [0, 1], plus its valid-pixel mask."""
    img, valid = clean(read_sample_image(sample), nodata)
    img = np.clip(img / scale, 0.0, 1.0).astype(np.float32)
    img[:, ~valid] = 0.0
    return img, valid


def guess_scale(samples, nodata=None, n=10) -> float:
    """10000 if the chips store reflectance x 10000, else 1."""
    med = np.median([np.median(clean(read_sample_image(s), nodata)[0]) for s in samples[:n]])
    return 10000.0 if med > 10 else 1.0


class DeblurEvalSet:
    """Deterministic blurred/sharp items for validation or test (one chip per item, all bands)."""

    def __init__(self, samples, scale, nodata=None, seed=0, name="", recipes=None):
        self.samples, self.scale, self.nodata, self.name = samples, scale, nodata, name
        if recipes is None:
            recipes = []
            for i in range(len(samples)):
                rng = np.random.default_rng([seed, i])
                recipes.append(B.sample_recipe(rng, B.BLUR_TYPES[i % len(B.BLUR_TYPES)]))
        self.recipes = recipes

    def __len__(self):
        return len(self.samples)

    def recipe_table(self) -> list[dict]:
        return [{"id": s.id, **r} for s, r in zip(self.samples, self.recipes)]

    def __getitem__(self, i):
        from scipy.ndimage import binary_erosion

        s, r = self.samples[i], self.recipes[i]
        sharp, valid = load_reflectance(s, self.scale, self.nodata)
        y, psf = B.degrade(sharp, r)
        rad = B.psf_radius(psf)
        if rad and not valid.all():  # pixels whose blur footprint touches no-data are not scored
            valid = binary_erosion(valid, iterations=rad + 1)
        return {"id": s.id, "dataset": self.name, "blur_type": r["blur_type"], "recipe": r,
                "sharp": sharp, "blurred": y.numpy().astype(np.float32), "psf": psf, "valid": valid}

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class DeblurTrainSet:
    """Random single-band crops with a fresh random blur each time: (blurred[1,h,w], sharp[1,h,w]).

    Single-band pairs make the deblurring model independent of the band set,
    so the same network can run on any band of the GSTP camera.
    """

    def __init__(self, samples, scale, nodata=None, crop=128, length=None, min_valid=0.98, seed=0):
        self.samples, self.scale, self.nodata, self.crop = samples, scale, nodata, crop
        self.length = length or 20 * len(samples)
        self.min_valid = min_valid
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.length

    def _crop(self, rng):
        for _ in range(20):
            s = self.samples[int(rng.integers(len(self.samples)))]
            img, valid = load_reflectance(s, self.scale, self.nodata)
            c = int(rng.integers(img.shape[0]))
            h, w = valid.shape
            yy, xx = int(rng.integers(0, h - self.crop + 1)), int(rng.integers(0, w - self.crop + 1))
            if valid[yy: yy + self.crop, xx: xx + self.crop].mean() >= self.min_valid:
                break
        patch = img[c, yy: yy + self.crop, xx: xx + self.crop]
        patch = np.rot90(patch, int(rng.integers(4)))
        if rng.random() < 0.5:
            patch = patch[:, ::-1]
        return np.ascontiguousarray(patch)[None]

    def __getitem__(self, i):
        import torch

        rng = np.random.default_rng([self.seed, i, int(self.rng.integers(2**31))])
        sharp = self._crop(rng)
        y, _ = B.degrade(sharp, B.sample_recipe(rng), seed=int(rng.integers(2**31)))
        return y.float(), torch.from_numpy(sharp)
