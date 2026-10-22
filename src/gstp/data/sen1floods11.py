"""Sen1Floods11 (Bonafilia et al., CVPRW 2020) — hand-labelled flood chips.

446 chips of 512x512 px at 10 m from 11 flood events on 6 continents.
  S1Hand   : Sentinel-1 GRD, 2 bands (VV, VH) in dB
  S2Hand   : Sentinel-2 L1C, 13 bands, TOA reflectance x 10000
  LabelHand: -1 no data, 0 not water, 1 water

Official splits: train / valid / test, plus a 'bolivia' split — a whole
flood event never seen in training, used to measure geographic generalisation.

Download (Colab):  gsutil -m cp -r gs://sen1floods11/v1.1/data/flood_events/HandLabeled <root>
                   gsutil -m cp -r gs://sen1floods11/v1.1/splits <root>
"""

from __future__ import annotations

import csv
from pathlib import Path

from .segmentation import Sample

S1_BANDS = ["VV", "VH"]
S2_BANDS_ALL = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
# B10 (cirrus) is not distributed in Sentinel-2 L2A products, which is what
# CRISP and the Planetary Computer serve. Dropping it keeps the trained model
# applicable to L2A imagery at inference time.
S2_BAND_IDX = [i for i, b in enumerate(S2_BANDS_ALL) if b != "B10"]
S2_BANDS = [S2_BANDS_ALL[i] for i in S2_BAND_IDX]

MODALITIES = {
    "s1": {"bands": S1_BANDS},
    "s2": {"bands": S2_BANDS},
    "s1s2": {"bands": S1_BANDS + S2_BANDS},
}

SPLIT_FILES = {
    "train": "flood_train_data.csv",
    "val": "flood_valid_data.csv",
    "test": "flood_test_data.csv",
    "bolivia": "flood_bolivia_data.csv",
}


def _find(root: Path, name: str) -> Path | None:
    hits = list(root.rglob(name))
    return hits[0] if hits else None


def _find_dir(root: Path, name: str) -> Path:
    hits = [p for p in root.rglob(name) if p.is_dir()]
    if not hits:
        raise FileNotFoundError(f"Folder '{name}' not found under {root}")
    return hits[0]


def chip_ids_for_split(root: Path, split: str) -> list[str]:
    csv_path = _find(root, SPLIT_FILES[split])
    if csv_path is None:
        raise FileNotFoundError(f"{SPLIT_FILES[split]} not found under {root}")
    ids = []
    with open(csv_path, newline="") as f:
        for row in csv.reader(f):
            if not row or not row[0].strip():
                continue
            name = Path(row[0].strip()).name  # e.g. Ghana_103272_S1Hand.tif
            ids.append(name.replace("_S1Hand.tif", "").replace("_S2Hand.tif", ""))
    return ids


def build_samples(root: str | Path, split: str, modality: str = "s1s2") -> list[Sample]:
    root = Path(root)
    s1_dir, s2_dir, lab_dir = (_find_dir(root, d) for d in ("S1Hand", "S2Hand", "LabelHand"))
    samples = []
    for cid in chip_ids_for_split(root, split):
        sources = []
        if modality in ("s1", "s1s2"):
            sources.append((str(s1_dir / f"{cid}_S1Hand.tif"), [0, 1]))
        if modality in ("s2", "s1s2"):
            sources.append((str(s2_dir / f"{cid}_S2Hand.tif"), S2_BAND_IDX))
        mask = lab_dir / f"{cid}_LabelHand.tif"
        if not mask.exists():
            continue
        samples.append(Sample(id=cid, sources=sources, mask=str(mask), meta={"event": cid.split("_")[0]}))
    return samples
