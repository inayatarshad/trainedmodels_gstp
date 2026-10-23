"""Landsat-8 Active Fire dataset (Pereira et al., ISPRS J. Photogramm. 2021) — flame / active-fire pixels.

>150k patches of 256x256 px at 30 m from Landsat-8 scenes worldwide (Aug-Sep 2020), released per
region as zip files (CC BY 4.0, github.com/pereira-gha/activefire). Labels:
  * training: masks from three classic active-fire algorithms (Schroeder 2016, Murphy 2016,
    Kumar-Roy 2018) and their 'voting' combination (>= 2 of 3 agree). We train on the voting masks.
  * test: a separate set of patches annotated BY HAND. These human labels are our ground truth,
    and the three algorithms' masks on the same patches are the classic baselines.

Input bands: Landsat-8 7 (SWIR-2), 6 (SWIR-1), 2 (blue) -> Sentinel-2 B12, B11, B02 equivalents,
so the trained model can be applied to Sentinel-2. Landsat-8 L1 digital numbers are converted to
top-of-atmosphere reflectance (DN * 2e-5 - 0.1, the standard L8 rescaling) so both sensors are
fed reflectance.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .segmentation import Sample

L8_BANDS = [7, 6, 2]                 # 1-based Landsat-8 band numbers = channel numbers in the patches
S2_EQUIVALENT = ["B12", "B11", "B02"]
ALGORITHMS = ["Schroeder", "Murphy", "Kumar-Roy", "Intersection", "Voting"]

_PATCH_RE = re.compile(r"^(?P<scene>.+?)_p(?P<idx>\d+)\.tif$", re.IGNORECASE)


def l8_dn_to_toa(dn: np.ndarray) -> np.ndarray:
    """Landsat-8 Collection-1 reflectance rescaling (same coefficients for bands 1-9)."""
    return np.clip(dn.astype(np.float32) * 2.0e-5 - 0.1, 0, None)


def strip_tokens(name: str, tokens=ALGORITHMS + ["v1"]) -> str:
    """Remove algorithm / version tokens so a mask name matches its image name."""
    for t in tokens:
        name = re.sub(rf"_{re.escape(t)}(?=_)", "", name, flags=re.IGNORECASE)
    return name


def scene_of(name: str) -> str:
    m = _PATCH_RE.match(strip_tokens(Path(name).name))
    return m.group("scene") if m else Path(name).stem


def build_training_index(root: str | Path, max_fire: int | None = None, nonfire_ratio: float = 0.5,
                         exclude_scenes: set | None = None, seed: int = 42):
    """Pair training images with their 'voting' masks; add a sample of fire-free patches.

    A patch with no voting mask had no fire found by the algorithms -> all-zero mask.
    Scenes in `exclude_scenes` (those of the hand-labelled test set) are dropped so the test
    scenes are never seen in training.
    """
    root = Path(root)
    images = {p.name.lower(): p for p in (root / "images" / "patches").glob("*.tif")}
    vote_dir = root / "masks" / "voting"
    masks = {strip_tokens(p.name).lower(): p for p in vote_dir.glob("*.tif")}
    exclude = {s.lower() for s in (exclude_scenes or set())}
    rng = np.random.default_rng(seed)

    fire, nonfire = [], []
    for key, img in images.items():
        if scene_of(key).lower() in exclude:
            continue
        s = Sample(id=img.stem, sources=[(str(img), [b - 1 for b in L8_BANDS])],
                   mask=str(masks[key]) if key in masks else None, meta={"scene": scene_of(img.name)})
        (fire if key in masks else nonfire).append(s)
    if max_fire and len(fire) > max_fire:
        fire = [fire[i] for i in rng.choice(len(fire), max_fire, replace=False)]
    n_non = min(len(nonfire), int(round(len(fire) * nonfire_ratio)))
    nonfire = [nonfire[i] for i in rng.choice(len(nonfire), n_non, replace=False)] if n_non else []
    report = {"images": len(images), "voting_masks": len(masks), "fire_patches": len(fire),
              "nonfire_patches": len(nonfire), "excluded_scenes": len(exclude)}
    return fire + nonfire, report


def build_manual_index(root: str | Path):
    """Hand-labelled test patches with the three algorithms' masks for the same patches.

    Returns (samples, algo_masks, report): samples carry the human mask; algo_masks maps
    algorithm -> {patch name: mask path} (missing = that algorithm found no fire there).
    """
    root = Path(root)
    tifs = list(root.rglob("*.tif"))
    manual, algo, images = {}, {a: {} for a in ALGORITHMS}, {}
    for p in tifs:
        low = p.as_posix().lower()
        if "manual_annotation" in low:
            if "_v2" in p.name.lower():   # the dataset ships a second annotation pass; v1 is the reference
                continue
            manual[strip_tokens(p.name).lower()] = p
        elif "masks_patches" in low or "/masks" in low:
            for a in ALGORITHMS:
                if f"_{a.lower()}_" in p.name.lower():
                    algo[a][strip_tokens(p.name).lower()] = p
        else:
            images[p.name.lower()] = p
    samples = [Sample(id=images[k].stem, sources=[(str(images[k]), [b - 1 for b in L8_BANDS])], mask=str(m),
                      meta={"scene": scene_of(k), "key": k}) for k, m in sorted(manual.items()) if k in images]
    report = {"manual_masks": len(manual), "images": len(images), "paired": len(samples),
              "scenes": len({s.meta["scene"] for s in samples}),
              **{f"{a}_masks": len(v) for a, v in algo.items()}}
    return samples, algo, report
