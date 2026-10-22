"""Small shared helpers: seeding, device selection, paths, JSON IO."""

from __future__ import annotations

import json
import os
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def seed_everything(seed: int = 42) -> None:
    """Make runs repeatable (up to non-deterministic CUDA kernels)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serialisable: {type(o)}")


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_sites(focus: str | None = None) -> list[dict]:
    """Return the 60 CRISP sites, optionally only those with a given hazard focus
    ('agriculture', 'floods', 'wildfires', 'forestry', 'air_quality', or 'all' for
    the Spain/UK sites that every CRISP module monitors)."""
    sites = load_json(CONFIG_DIR / "crisp_sites.json")["sites"]
    if focus is None:
        return sites
    return [s for s in sites if s["focus"] == focus]


def bbox_center(bbox: list[float]) -> tuple[float, float]:
    """(lat, lon) of a [min_lon, min_lat, max_lon, max_lat] box."""
    return (bbox[1] + bbox[3]) / 2.0, (bbox[0] + bbox[2]) / 2.0
