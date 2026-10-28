"""Flood extent and area for any place, following operational practice.

Why this module exists
----------------------
Notebook 06/08 ran the U-Net on Planetary Computer *RTC gamma0*, while it was trained on Sen1Floods11, which is
Google Earth Engine *S1_GRD sigma0* in dB. Terrain-flattened gamma0 is brighter than sigma0 by about
-10*log10(cos(incidence)) = 0.6 to 2 dB over the Sentinel-1 IW swath, so open water looks less dark than in
training and the model under-detects floods. The operational services do not rely on one model with one fixed
threshold either. This module follows them:

  * Copernicus Global Flood Monitoring (GFM): three independent algorithms, majority vote, plus exclusion masks
    (reference water, no-sensitivity areas).
  * UN-SPIDER Recommended Practice (Sentinel-1): speckle smoothing, before/after change ratio > 1.25 on dB values,
    JRC permanent water mask, slope > 5 % mask, remove flood patches of 8 or fewer connected pixels.

Three detectors, each giving water/flood on the 10 m grid of the post-flood image:
  A. U-Net (Sen1Floods11) on sigma0-harmonised input  -> water after, minus water before
  B. Change detection (UN-SPIDER): VH_dB_after / VH_dB_before > 1.25 (both negative dB)
  C. Split-based Otsu on VV dB of the after image (adaptive threshold, Martinis-style), minus water before
Final flood = at least 2 of 3 agree, then masks. Areas are reported per method so disagreement is visible.
"""

from __future__ import annotations

import numpy as np

from . import inference as INF

PX_KM2 = 1e-4  # 10 m pixels


# ── input harmonisation ─────────────────────────────────────────────────────────────────────────────────────
def gamma0_to_sigma0_db(db: np.ndarray, incidence_deg: float | np.ndarray = 37.0) -> np.ndarray:
    """RTC gamma0 (dB) -> sigma0-like (dB): sigma0 ~= gamma0 * cos(theta). Default = IW mid-swath angle."""
    return db + 10.0 * np.log10(np.cos(np.deg2rad(incidence_deg)))


def incidence_across_swath(shape, orbit_state: str | None, near=30.5, far=45.5) -> np.ndarray:
    """Approximate incidence angle per column for a site that lies inside one IW swath. Without the annotation
    grid we use the swath mid value; kept as a function so a per-pixel grid can replace it later."""
    return np.full(shape, (near + far) / 2.0, np.float32)


def speckle_smooth(db: np.ndarray, size: int = 5) -> np.ndarray:
    """Median filter per band (5 x 5 at 10 m = 50 m window, as in the UN-SPIDER practice)."""
    from scipy.ndimage import median_filter

    return np.stack([median_filter(b, size=size) for b in db]).astype(np.float32)


# ── detectors ───────────────────────────────────────────────────────────────────────────────────────────────
def unet_water(model, db_sigma0, stats, threshold, device):
    from .data.raster import normalise

    prob = INF.sliding_window_probs(model, normalise(db_sigma0, "zscore", stats), device=device)
    return prob >= threshold, prob


def change_ratio_flood(vh_after_db, vh_before_db, ratio=1.25):
    """UN-SPIDER: flood where after/before > 1.25 on dB values (both negative, so darker after -> ratio > 1)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = vh_after_db / vh_before_db
    return np.nan_to_num(r, nan=0.0) > ratio


def split_based_otsu_water(vv_db, valid, tile=256, min_water_frac=0.1, max_water_frac=0.9):
    """Otsu threshold estimated on tiles that contain both water and land (bimodal), applied to the whole image.
    Falls back to -18 dB (common VV water threshold) when no bimodal tile is found."""
    from skimage.filters import threshold_otsu

    h, w = vv_db.shape
    ths = []
    for y in range(0, h - tile + 1, tile):
        for x in range(0, w - tile + 1, tile):
            v = vv_db[y:y + tile, x:x + tile][valid[y:y + tile, x:x + tile]]
            if v.size < tile * tile * 0.5:
                continue
            t = threshold_otsu(v)
            frac = (v < t).mean()
            if min_water_frac <= frac <= max_water_frac and np.mean(v[v >= t]) - np.mean(v[v < t]) > 4:
                ths.append(t)
    t = float(np.median(ths)) if ths else -18.0
    return (vv_db < t) & valid, t, len(ths)


# ── masks ───────────────────────────────────────────────────────────────────────────────────────────────────
def steep_mask(bbox, geobox, max_slope_pct=5.0):
    """Slope from the Copernicus 30 m DEM, resampled to the 10 m grid; True where slope > 5 % (no floods)."""
    import odc.stac

    items = INF.search("cop-dem-glo-30", bbox, None)
    if not items:
        return np.zeros(geobox.shape, bool)
    ds = odc.stac.load(items, bands=["data"], geobox=geobox, resampling="bilinear")
    da = ds["data"]
    dem = (da.max("time") if "time" in da.dims else da).values.astype(np.float32)
    gy, gx = np.gradient(dem, 10.0)
    return np.hypot(gx, gy) * 100.0 > max_slope_pct


def remove_small(mask, min_pixels=9):
    """Drop flood patches of 8 or fewer connected pixels (UN-SPIDER connectivity rule)."""
    from scipy import ndimage

    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_pixels
    keep[0] = False
    return keep[lab]


# ── full pipeline for one place ─────────────────────────────────────────────────────────────────────────────
def map_flood(bbox, dry_range, flood_range, model, stats, threshold, device, max_passes=8, max_dry=3,
              harmonise=True, smooth=True):
    """Flood extent for a bounding box. Returns a dict with the final mask, per-method masks, per-pass areas and
    the masks used, all on the grid of the first flood-period radar pass."""
    days = INF._s1_days(bbox, flood_range)
    d0, it0 = days[0]
    db0, v0, gb = INF._s1_db(it0, bbox=bbox)
    posts = [(d0, db0, v0)] + [(d, *INF._s1_db(it, geobox=gb)[:2]) for d, it in days[1:max_passes]]
    dry_days = INF._s1_days(bbox, dry_range)
    dry = [(d, *INF._s1_db(it, geobox=gb)[:2]) for d, it in dry_days[::max(1, len(dry_days) // max_dry)][:max_dry]]

    prep = (lambda db: speckle_smooth(db)) if smooth else (lambda db: db)
    harm = (lambda db: gamma0_to_sigma0_db(db)) if harmonise else (lambda db: db)

    # dry-season reference: water in any dry pass (U-Net) and median VH for the change ratio
    dry_water = np.zeros(gb.shape, bool)
    dry_ok = np.zeros(gb.shape, bool)
    vh_stack = []
    for _, db, v in dry:
        p = prep(db)
        w, _ = unet_water(model, harm(p), stats, threshold, device)
        dry_water |= w & v
        dry_ok |= v
        vh_stack.append(np.where(v, p[1], np.nan))
    vh_dry = np.nanmedian(np.stack(vh_stack), axis=0)

    perm, _ = INF.load_permanent_water(bbox, gb)
    try:
        steep = steep_mask(bbox, gb)
    except Exception:
        steep = np.zeros(gb.shape, bool)

    per_pass, methods_any = [], {"unet": np.zeros(gb.shape, bool), "change": np.zeros(gb.shape, bool),
                                 "otsu": np.zeros(gb.shape, bool), "final": np.zeros(gb.shape, bool)}
    post_ok = np.zeros(gb.shape, bool)
    for d, db, v in posts:
        p = prep(db)
        ok = v & dry_ok
        a, _ = unet_water(model, harm(p), stats, threshold, device)
        a = a & ~dry_water
        b = change_ratio_flood(p[1], vh_dry)
        c, t_otsu, n_tiles = split_based_otsu_water(p[0], v)
        c = c & ~dry_water
        votes = a.astype(np.uint8) + b.astype(np.uint8) + c.astype(np.uint8)
        excl = perm | steep | ~ok
        final = remove_small((votes >= 2) & ~excl)
        res = {"date": d, "otsu_threshold_db": round(t_otsu, 2), "otsu_tiles": n_tiles}
        for k, m in (("unet", a), ("change", b), ("otsu", c)):
            m = remove_small(m & ~excl)
            methods_any[k] |= m
            res[f"{k}_km2"] = round(float(m.sum()) * PX_KM2, 1)
        res["final_km2"] = round(float(final.sum()) * PX_KM2, 1)
        methods_any["final"] |= final
        post_ok |= v
        per_pass.append(res)
    return {"geobox": gb, "per_pass": per_pass, "max_extent": methods_any, "permanent": perm, "steep": steep,
            "post_db": db0, "dates": [p[0] for p in posts], "dry_dates": [d[0] for d in dry],
            "max_extent_km2": {k: round(float(m.sum()) * PX_KM2, 1) for k, m in methods_any.items()}}


def pass_masks(bbox, date, result):
    """Helper for evaluation: index of the pass on `date` in result['per_pass'] (None if absent)."""
    for i, p in enumerate(result["per_pass"]):
        if p["date"] == date:
            return i
    return None
