"""HLS Burn Scars (NASA / IBM, 2023) — burn-scar masks on Harmonized Landsat-Sentinel-2.

804 chips of 512x512 px at 30 m from fires across the contiguous USA (2018-2021).
Image bands (HLS S30 naming, = Sentinel-2 bands):
  Blue (B02), Green (B03), Red (B04), NIR-narrow (B8A), SWIR-1 (B11), SWIR-2 (B12)
Mask: 1 burn scar, 0 not burned, -1 no data.

The public release has two folders, 'training' and 'validation'. To keep the
final test untouched, we treat the official 'validation' folder as the TEST set
and carve our own validation set (for early stopping / threshold choice) out of
'training'.

Download (Colab): huggingface_hub.snapshot_download("ibm-nasa-geospatial/hls_burn_scars", repo_type="dataset")
"""

from __future__ import annotations

import re
from pathlib import Path

from .segmentation import Sample

BANDS = ["B02", "B03", "B04", "B8A", "B11", "B12"]
NODATA = -9999.0


def _scene_key(name: str) -> str:
    # subsetted_512x512_HLS.S30.T10SDH.2020248.v1.4_merged.tif -> T10SDH.2020248
    m = re.search(r"(T\d{2}[A-Z]{3})\.(\d{7})", name)
    return f"{m.group(1)}.{m.group(2)}" if m else name


def build_samples(root: str | Path, folder: str) -> list[Sample]:
    """folder: 'training' or 'validation' (official release folders)."""
    root = Path(root)
    dirs = [p for p in root.rglob(folder) if p.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"'{folder}' folder not found under {root} (did you extract the archive?)")
    samples = []
    for img in sorted(dirs[0].rglob("*merged*.tif")):
        mask = img.with_name(img.name.replace("merged", "mask"))
        if not mask.exists():
            continue
        samples.append(
            Sample(
                id=img.stem.replace("_merged", ""),
                sources=[(str(img), None)],
                mask=str(mask),
                meta={"tile": _scene_key(img.name).split(".")[0]},
            )
        )
    return samples
