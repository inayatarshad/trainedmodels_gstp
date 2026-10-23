"""Amazon and Atlantic Forest Sentinel-2 forest masks (Bragagnolo et al., 2021).

Zenodo record 4498086 (CC BY 4.0). Per biome: 499/485 training images,
100 validation images and 20 unlabelled test images; 512x512 px, Sentinel-2
L2A bands 4-3-2-8 (Red, Green, Blue, NIR) rescaled to bytes. Masks are PNGs,
forest = white, non-forest = black.

The official test images have no masks, so we use the official validation
images as our TEST set and carve a validation set out of training.

Because the byte rescaling applied by the dataset authors is not documented,
the model uses per-image percentile normalisation (see raster.normalise),
which makes it insensitive to that choice and lets us apply it to raw
Sentinel-2 L2A reflectance at the CRISP sites.
"""

from __future__ import annotations

import re
from pathlib import Path

from .segmentation import Sample

BANDS = ["B04", "B03", "B02", "B08"]
IMG_EXT = {".tif", ".tiff"}
MASK_EXT = {".png", ".tif", ".tiff"}


def _split_of(path: Path) -> str | None:
    """Split from folder names *relative to the dataset root* (never the absolute
    path, which may itself contain words like 'train')."""
    s = "/" + path.as_posix().lower()
    if "train" in s:
        return "training"
    if "valid" in s or "/val" in s:
        return "validation"
    if "test" in s:
        return "test"
    return None


def _biome_of(path: Path) -> str:
    return "atlantic" if "atlantic" in path.as_posix().lower() else "amazon"


def _is_mask(path: Path) -> bool:
    s = "/" + path.as_posix().lower()
    return path.suffix.lower() == ".png" or any(k in s for k in ("mask", "label", "/gt", "reference"))


def _key(path: Path) -> str:
    """Pairing key: the full file stem. Images and masks share the same Sentinel-2
    tile name (e.g. S2A_MSIL2A_..._01_07); a partial key such as the last number
    collides across tiles and silently drops most of the dataset."""
    return path.stem.lower()


def build_samples(root: str | Path, split: str) -> tuple[list[Sample], dict]:
    """Pair images with masks inside one split ('training' or 'validation').

    Returns (samples, report). The report counts unmatched files so the
    notebook can show the pairing is complete before training.
    """
    root = Path(root)
    images, masks = {}, {}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if _split_of(rel) != split:
            continue
        key = (_biome_of(rel), _key(p))
        if _is_mask(rel) and p.suffix.lower() in MASK_EXT:
            masks[key] = p
        elif p.suffix.lower() in IMG_EXT:
            images[key] = p
    samples = [
        Sample(id=f"{b}_{k}", sources=[(str(images[(b, k)]), None)], mask=str(masks[(b, k)]), meta={"biome": b})
        for (b, k) in sorted(images)
        if (b, k) in masks
    ]
    report = {
        "split": split,
        "images": len(images),
        "masks": len(masks),
        "paired": len(samples),
        "per_biome": {b: sum(1 for s in samples if s.meta["biome"] == b) for b in ("amazon", "atlantic")},
    }
    return samples, report
