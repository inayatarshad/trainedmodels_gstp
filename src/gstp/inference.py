"""Apply trained models to the CRISP sites using live Sentinel imagery.

Imagery comes from the Microsoft Planetary Computer STAC catalogue (free,
no account needed): Sentinel-2 L2A and Sentinel-1 RTC.

Known domain differences between training data and this imagery are handled
explicitly and documented in each model card:
  * Sentinel-2 processing baseline >= 04.00 (from 25 Jan 2022) adds a +1000
    offset to L2A digital numbers; it is removed here.
  * Sen1Floods11 was built from L1C (top-of-atmosphere); CRISP-side imagery is
    L2A (surface reflectance). Water is spectrally distinct in both, but small
    differences in scores are expected.
  * Sen1Floods11 SAR is GRD sigma0 in dB; Planetary Computer serves RTC gamma0
    (linear), converted to dB here. Over flat flood plains the two are close.
"""

from __future__ import annotations

import numpy as np

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
SCL_INVALID = {0, 1, 3, 8, 9, 10}  # no data, saturated, cloud shadow, cloud (med/high), cirrus


def _catalog():
    import planetary_computer
    import pystac_client

    return pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)


def search(collection, bbox, date_range, max_cloud=None, limit=50):
    query = {"eo:cloud_cover": {"lt": max_cloud}} if max_cloud is not None else None
    items = _catalog().search(collections=[collection], bbox=bbox, datetime=date_range, query=query, limit=limit)
    return list(items.items())


def load_s2(bbox, date_range, bands, resolution=10, max_cloud=30, max_items=6):
    """Cloud-masked median composite of the clearest Sentinel-2 L2A scenes.

    Returns (array [C,H,W] in L2A DN = reflectance x 10000, valid mask [H,W], info).
    """
    import odc.stac

    items = search("sentinel-2-l2a", bbox, date_range, max_cloud)
    if not items:
        raise RuntimeError(f"No Sentinel-2 scenes with <{max_cloud}% cloud in {date_range}")
    items = sorted(items, key=lambda it: it.properties.get("eo:cloud_cover", 100))[:max_items]
    ds = odc.stac.load(items, bands=list(bands) + ["SCL"], bbox=bbox, resolution=resolution,
                       crs="utm", groupby="solar_day", resampling={"SCL": "nearest", "*": "bilinear"})
    stack = []
    for t in range(ds.sizes["time"]):
        scl = ds["SCL"].isel(time=t).values
        bad = np.isin(scl, list(SCL_INVALID))
        arrs = []
        for b in bands:
            a = ds[b].isel(time=t).values.astype(np.float32)
            a[a == 0] = np.nan
            arrs.append(a)
        a = np.stack(arrs)
        date = str(ds.time.values[t])[:10]
        if date >= "2022-01-25":  # processing baseline 04.00 radiometric offset
            a = a - 1000.0
        a[:, bad] = np.nan
        stack.append(a)
    comp = np.nanmedian(np.stack(stack), axis=0)
    valid = np.all(np.isfinite(comp), axis=0)
    comp = np.nan_to_num(comp, nan=0.0).clip(min=0)
    info = {"scenes": [it.id for it in items], "dates": [str(t)[:10] for t in ds.time.values],
            "crs": str(ds.odc.crs), "resolution_m": resolution, "valid_fraction": float(valid.mean())}
    return comp, valid, info


def load_s1_db(bbox, date_range, resolution=10):
    """Sentinel-1 RTC VV/VH in dB, from the scene closest to the start of date_range."""
    import odc.stac

    items = search("sentinel-1-rtc", bbox, date_range)
    if not items:
        raise RuntimeError(f"No Sentinel-1 RTC scenes in {date_range}")
    items = sorted(items, key=lambda it: it.datetime)
    first_day = items[0].datetime.date()
    items = [it for it in items if it.datetime.date() == first_day]
    ds = odc.stac.load(items, bands=["vv", "vh"], bbox=bbox, resolution=resolution, crs="utm", groupby="solar_day")
    arr = np.stack([ds["vv"].isel(time=0).values, ds["vh"].isel(time=0).values]).astype(np.float32)
    valid = np.all(np.isfinite(arr) & (arr > 0), axis=0)
    arr = 10.0 * np.log10(np.where(valid, arr, 1.0))
    info = {"scenes": [it.id for it in items], "date": str(first_day), "resolution_m": resolution,
            "valid_fraction": float(valid.mean())}
    return arr, valid, info


def sliding_window_probs(model, img: np.ndarray, tile=512, overlap=64, device="cpu", batch=4):
    """Binary-segmentation probabilities for an arbitrarily sized image [C,H,W].

    Tiles overlap and are blended with a cosine window to avoid seams.
    """
    import torch

    c, h, w = img.shape
    step = tile - overlap
    ph, pw = max(tile, int(np.ceil((h - overlap) / step)) * step + overlap), max(tile, int(np.ceil((w - overlap) / step)) * step + overlap)
    pad = np.zeros((c, ph, pw), np.float32)
    pad[:, :h, :w] = img
    acc = np.zeros((ph, pw), np.float32)
    wsum = np.zeros((ph, pw), np.float32)
    win1d = np.hanning(tile) + 1e-3
    win = np.outer(win1d, win1d).astype(np.float32)
    coords = [(y, x) for y in range(0, ph - tile + 1, step) for x in range(0, pw - tile + 1, step)]
    model.eval().to(device)
    with torch.no_grad():
        for i in range(0, len(coords), batch):
            chunk = coords[i : i + batch]
            x = torch.from_numpy(np.stack([pad[:, y : y + tile, xx : xx + tile] for y, xx in chunk])).to(device)
            p = torch.sigmoid(model(x).float()).squeeze(1).cpu().numpy()
            for (y, xx), pi in zip(chunk, p):
                acc[y : y + tile, xx : xx + tile] += pi * win
                wsum[y : y + tile, xx : xx + tile] += win
    return (acc / np.maximum(wsum, 1e-6))[:h, :w]


def patch_classify(model, img: np.ndarray, patch=64, device="cpu", batch=256):
    """Classify non-overlapping patches -> (class map [H//patch, W//patch], prob map)."""
    import torch

    c, h, w = img.shape
    ny, nx = h // patch, w // patch
    tiles = img[:, : ny * patch, : nx * patch].reshape(c, ny, patch, nx, patch).transpose(1, 3, 0, 2, 4)
    tiles = tiles.reshape(ny * nx, c, patch, patch)
    model.eval().to(device)
    out = []
    with torch.no_grad():
        for i in range(0, len(tiles), batch):
            x = torch.from_numpy(np.ascontiguousarray(tiles[i : i + batch])).float().to(device)
            out.append(torch.softmax(model(x).float(), 1).cpu().numpy())
    probs = np.concatenate(out).reshape(ny, nx, -1)
    return probs.argmax(-1), probs


def area_km2(mask: np.ndarray, resolution_m: float) -> float:
    return float(mask.sum()) * (resolution_m**2) / 1e6
