"""Land-use composition of one CRISP site, run as its own process.

Notebook 06 calls this once per site. Each site gets a fresh Python process with a memory cap, so one heavy
Sentinel-2 load can only fail that site instead of crashing the whole Colab session (12.7 GB RAM on the free tier).

    python tools/apply_agriculture.py <checkpoint.pt> <site_id> <out.json> [--mem-gb 9]
"""
import argparse
import json
import resource
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("site_id")
    ap.add_argument("out")
    ap.add_argument("--mem-gb", type=float, default=9.0)
    ap.add_argument("--max-items", type=int, default=3)
    ap.add_argument("--tiles", type=int, default=2, help="split the site into tiles x tiles pieces")
    a = ap.parse_args()

    cap = int(a.mem_gb * 1024**3)
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))   # over the cap -> MemoryError here, not a dead session

    import numpy as np
    import torch
    from gstp import inference as INF
    from gstp.models import build_s2_classifier
    from gstp.utils import CONFIG_DIR, load_json, load_sites

    site = {s["id"]: s for s in load_sites()}[a.site_id]
    events = load_json(CONFIG_DIR / "crisp_events.json")["agriculture"]
    win = events["site_windows"].get(site["id"], events["default_window"])

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    ag = ck["config"]["extra"]
    classes = ag["classes"]
    model = build_s2_classifier(len(classes), drop_bands=["B10"], pretrained=False, input_size=ag["input_size"])
    model.load_state_dict(ck["state_dict"])

    # Load the site in tiles (a 12-band stack of a 500 km2 site at 10 m is several GB) and add up the patch counts.
    x0, y0, x1, y1 = site["bbox"]
    n = a.tiles
    counts = np.zeros(len(classes))
    scenes, valid_px, total_px = set(), 0, 0
    for i in range(n):
        for j in range(n):
            sub = [x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * j / n, x0 + (x1 - x0) * (i + 1) / n, y0 + (y1 - y0) * (j + 1) / n]
            arr, valid, info = INF.load_s2(sub, win, ag["bands"], max_cloud=20, max_items=a.max_items)
            cls, _ = INF.patch_classify(model, arr / ag["scale"], device="cpu")  # CPU: CUDA reserves more address space than the cap
            counts += np.bincount(cls.ravel(), minlength=len(classes))
            scenes.update(info["scenes"]); valid_px += int(valid.sum()); total_px += valid.size
            del arr, valid, cls
    counts /= counts.sum()
    row = {"site": site["label"], "site_id": site["id"], "country": site["country"], "window": win,
           "scenes": len(scenes), "valid_fraction": round(valid_px / total_px, 3), "tiles": n * n,
           **{c: round(float(v), 3) for c, v in zip(classes, counts)}}
    Path(a.out).write_text(json.dumps(row))
    print("ok", site["id"])


if __name__ == "__main__":
    main()
