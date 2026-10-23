"""Generic binary-segmentation dataset used by the flood, burn-scar and forest models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .raster import IGNORE_INDEX, clean, normalise, read_mask, read_raster


@dataclass
class Sample:
    """One training chip. `sources` lets a chip stack bands from several files
    (e.g. Sentinel-1 + Sentinel-2 for flood fusion)."""

    id: str
    sources: list[tuple[str, Sequence[int] | None]]
    mask: str | None = None
    meta: dict = field(default_factory=dict)


def read_sample_image(sample: Sample) -> np.ndarray:
    parts = [read_raster(path, bands) for path, bands in sample.sources]
    return np.concatenate(parts, axis=0)


def augment(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator):
    """Random dihedral transform (flips + 90° rotations).

    Satellite imagery has no canonical 'up', so all 8 orientations are valid.
    Radiometric augmentations are deliberately avoided: band values carry
    physical meaning (reflectance, backscatter).
    """
    k = int(rng.integers(4))
    img = np.rot90(img, k, axes=(1, 2))
    mask = np.rot90(mask, k, axes=(0, 1))
    if rng.random() < 0.5:
        img = img[:, :, ::-1]
        mask = mask[:, ::-1]
    return np.ascontiguousarray(img), np.ascontiguousarray(mask)


def random_crop(img, mask, size, rng, min_valid_frac=0.25, tries=10):
    h, w = mask.shape
    if size is None or (h <= size and w <= size):
        return img, mask
    for _ in range(tries):
        y = int(rng.integers(0, h - size + 1))
        x = int(rng.integers(0, w - size + 1))
        m = mask[y : y + size, x : x + size]
        if np.mean(m != IGNORE_INDEX) >= min_valid_frac:
            break
    return img[:, y : y + size, x : x + size], m


class SegDataset:
    """Torch-compatible dataset returning (image float32 [C,H,W], mask int64 [H,W]).

    Mask values: 0 = negative, 1 = positive, 255 = ignore (no data / invalid input).
    """

    def __init__(
        self,
        samples: list[Sample],
        norm: str = "zscore",
        stats: dict | None = None,
        nodata: float | None = None,
        train: bool = False,
        crop_size: int | None = None,
        seed: int = 0,
        pre=None,
    ):
        """pre: optional function applied to the raw image before normalisation
        (e.g. Landsat-8 digital numbers -> top-of-atmosphere reflectance)."""
        self.samples = samples
        self.pre = pre
        self.norm = norm
        self.stats = stats
        self.nodata = nodata
        self.train = train
        self.crop_size = crop_size
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.samples)

    def load(self, i: int):
        s = self.samples[i]
        img, valid = clean(read_sample_image(s), self.nodata)
        mask = read_mask(s.mask) if s.mask else np.zeros(img.shape[1:], np.uint8)
        if mask.shape != img.shape[1:]:
            raise ValueError(f"{s.id}: mask {mask.shape} != image {img.shape[1:]}")
        mask = mask.copy()
        mask[~valid] = IGNORE_INDEX
        if self.pre is not None:
            img = self.pre(img)
        img = normalise(img, self.norm, self.stats)
        img[:, ~valid] = 0.0
        return img, mask

    def __getitem__(self, i: int):
        import torch

        img, mask = self.load(i)
        if self.train:
            img, mask = random_crop(img, mask, self.crop_size, self.rng)
            img, mask = augment(img, mask, self.rng)
        return torch.from_numpy(img.astype(np.float32)), torch.from_numpy(mask.astype(np.int64))


def split_samples(samples: list[Sample], val_frac: float, seed: int = 42, group_key=None):
    """Random train/val split. If group_key is given, whole groups (e.g. flood
    events or source scenes) go to one side, preventing leakage between
    near-duplicate chips."""
    rng = np.random.default_rng(seed)
    if group_key is None:
        idx = rng.permutation(len(samples))
        n_val = max(1, int(round(val_frac * len(samples))))
        val_idx = set(idx[:n_val].tolist())
        return (
            [s for i, s in enumerate(samples) if i not in val_idx],
            [s for i, s in enumerate(samples) if i in val_idx],
        )
    groups = sorted({group_key(s) for s in samples})
    rng.shuffle(groups)
    target = val_frac * len(samples)
    val_groups, n = set(), 0
    for g in groups:
        if n >= target:
            break
        val_groups.add(g)
        n += sum(1 for s in samples if group_key(s) == g)
    return (
        [s for s in samples if group_key(s) not in val_groups],
        [s for s in samples if group_key(s) in val_groups],
    )
