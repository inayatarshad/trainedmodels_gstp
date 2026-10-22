"""Air-quality forecasting for the CRISP sites.

Why forecasting and not image segmentation: Sentinel-5P TROPOMI pixels are
~3.5 x 5.5 km, so a CRISP site (~15 km across) is only a handful of pixels —
there is no spatial structure for a CNN to learn. The useful product for a
city or industrial site is a short-range forecast of pollutant levels.

Data: CAMS atmospheric-composition fields (which assimilate satellite
retrievals including Sentinel-5P) served by the Open-Meteo Air Quality API,
plus ERA5 weather from the Open-Meteo Archive API. Same source and domain for
all 60 sites so the training data are consistent.

Task: predict tomorrow's daily-mean concentration (NO2, PM2.5) at each site.
Model: one LightGBM model shared by all sites, compared with persistence and
weekly-persistence baselines on a chronologically held-out year.

Licensing note: Open-Meteo is free for non-commercial use; commercial use needs
an API key (https://open-meteo.com/en/pricing). CAMS and Open-Meteo must be
credited. For production, the same CAMS data can be obtained directly from the
Copernicus Atmosphere Data Store under the Copernicus licence.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WX_URL = "https://archive-api.open-meteo.com/v1/archive"
POLLUTANTS = ["nitrogen_dioxide", "pm2_5", "pm10", "ozone", "carbon_monoxide", "sulphur_dioxide"]
WEATHER = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m",
           "precipitation", "surface_pressure", "shortwave_radiation"]
TARGETS = {"no2": "nitrogen_dioxide", "pm25": "pm2_5"}


def _get(url, params, retries=4):
    import requests

    for k in range(retries):
        r = requests.get(url, params=params, timeout=120)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(10 * (k + 1))
            continue
        raise RuntimeError(f"{url} -> {r.status_code}: {r.text[:300]}")
    raise RuntimeError(f"{url}: gave up after {retries} tries")


def fetch_site(site: dict, start: str, end: str, domain: str = "cams_global") -> pd.DataFrame:
    """Hourly pollutants + weather for the centre of a CRISP site bbox."""
    lat = (site["bbox"][1] + site["bbox"][3]) / 2
    lon = (site["bbox"][0] + site["bbox"][2]) / 2
    aq = _get(AQ_URL, {"latitude": lat, "longitude": lon, "hourly": ",".join(POLLUTANTS),
                       "start_date": start, "end_date": end, "domains": domain, "timezone": "UTC"})
    wx = _get(WX_URL, {"latitude": lat, "longitude": lon, "hourly": ",".join(WEATHER),
                       "start_date": start, "end_date": end, "timezone": "UTC"})
    a = pd.DataFrame(aq["hourly"]).rename(columns={"time": "ts"})
    w = pd.DataFrame(wx["hourly"]).rename(columns={"time": "ts"})
    df = a.merge(w, on="ts", how="outer")
    df["ts"] = pd.to_datetime(df["ts"])
    df["site_id"], df["lat"], df["lon"] = site["id"], lat, lon
    return df


def fetch_all(sites, start, end, cache_dir, domain="cams_global", log=print) -> pd.DataFrame:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for i, s in enumerate(sites, 1):
        f = cache_dir / f"{s['id']}_{start}_{end}_{domain}.csv"
        if f.exists():
            df = pd.read_csv(f, parse_dates=["ts"])
        else:
            df = fetch_site(s, start, end, domain)
            df.to_csv(f, index=False)
            time.sleep(1.0)  # be polite to the free API
        log(f"[{i:2d}/{len(sites)}] {s['id']:<24} {len(df):6d} hourly rows")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def to_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    h = hourly.copy()
    h["date"] = h["ts"].dt.floor("D")
    rad = np.deg2rad(h["wind_direction_10m"])
    h["wind_u"] = -h["wind_speed_10m"] * np.sin(rad)
    h["wind_v"] = -h["wind_speed_10m"] * np.cos(rad)
    agg = {p: "mean" for p in POLLUTANTS}
    agg.update({"temperature_2m": "mean", "relative_humidity_2m": "mean", "wind_speed_10m": "mean",
                "wind_u": "mean", "wind_v": "mean", "precipitation": "sum", "surface_pressure": "mean",
                "shortwave_radiation": "mean", "lat": "first", "lon": "first"})
    d = h.groupby(["site_id", "date"]).agg(agg).reset_index()
    # Require most hours present for a valid daily mean.
    counts = h.groupby(["site_id", "date"])[POLLUTANTS[0]].count().reset_index(name="n_hours")
    d = d.merge(counts, on=["site_id", "date"])
    d.loc[d["n_hours"] < 18, POLLUTANTS] = np.nan
    return d.drop(columns="n_hours")


WEATHER_DAILY = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_u", "wind_v",
                 "precipitation", "surface_pressure", "shortwave_radiation"]


def make_features(daily: pd.DataFrame, target: str, use_next_day_weather: bool = True):
    """Build the supervised table: features known at day t -> target at day t+1.

    use_next_day_weather: include day-t+1 weather. Operationally this comes from
    a weather forecast; here ERA5 reanalysis stands in for it, which is slightly
    optimistic. Set False for a strict 'no future information' variant.
    """
    col = TARGETS[target]
    rows = []
    for sid, g in daily.sort_values("date").groupby("site_id"):
        g = g.set_index("date").asfreq("D")
        g["site_id"] = sid
        f = pd.DataFrame(index=g.index)
        f["site_id"] = sid
        f["lat"], f["lon"] = g["lat"].ffill().bfill(), g["lon"].ffill().bfill()
        for lag in (0, 1, 2, 3, 6, 13):
            f[f"{col}_lag{lag}"] = g[col].shift(lag)
        f[f"{col}_mean7"] = g[col].rolling(7, min_periods=5).mean()
        f[f"{col}_mean28"] = g[col].rolling(28, min_periods=20).mean()
        f[f"{col}_std7"] = g[col].rolling(7, min_periods=5).std()
        for p in POLLUTANTS:
            if p != col:
                f[f"{p}_lag0"] = g[p]
        for w in WEATHER_DAILY:
            f[f"{w}_d0"] = g[w]
            if use_next_day_weather:
                f[f"{w}_d1"] = g[w].shift(-1)
        tgt_date = g.index + pd.Timedelta(days=1)
        f["dow_target"] = tgt_date.dayofweek
        doy = tgt_date.dayofyear
        f["doy_sin"], f["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
        f["target"] = g[col].shift(-1)
        f["target_date"] = tgt_date
        rows.append(f.reset_index(drop=True))
    df = pd.concat(rows, ignore_index=True)
    df = df.dropna(subset=["target", f"{col}_lag0", f"{col}_lag6"])
    df["site_id"] = df["site_id"].astype("category")
    return df


def chrono_split(df, val_start, test_start):
    tr = df[df["target_date"] < val_start]
    va = df[(df["target_date"] >= val_start) & (df["target_date"] < test_start)]
    te = df[df["target_date"] >= test_start]
    return tr, va, te


def feature_columns(df):
    return [c for c in df.columns if c not in ("target", "target_date")]


def train_lgbm(tr, va, seed=42):
    import lightgbm as lgb

    cols = feature_columns(tr)
    params = dict(objective="regression", learning_rate=0.03, num_leaves=63, min_data_in_leaf=40,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  verbose=-1, seed=seed)
    # Train on log1p: concentrations are right-skewed; this stops a few
    # pollution episodes from dominating the fit.
    dtr = lgb.Dataset(tr[cols], np.log1p(tr["target"]), categorical_feature=["site_id"])
    dva = lgb.Dataset(va[cols], np.log1p(va["target"]), categorical_feature=["site_id"], reference=dtr)
    model = lgb.train(params, dtr, num_boost_round=5000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
    return model, cols


def predict_lgbm(model, df, cols):
    return np.expm1(model.predict(df[cols], num_iteration=model.best_iteration)).clip(min=0)


def baselines(df, col):
    return {"persistence": df[f"{col}_lag0"].values, "weekly_persistence": df[f"{col}_lag6"].values,
            "mean_7day": df[f"{col}_mean7"].values}
