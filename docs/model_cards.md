# Model cards

One card per trained model. Metrics are filled from `results/RESULTS.md` after the training runs.

---

## flood_s1 · flood_s2 · flood_s1s2 — flood-water segmentation

| | |
|---|---|
| **CRISP module** | Floods |
| **Architecture** | U-Net, ResNet-34 encoder (ImageNet init, first conv adapted to the band count), 1 output logit |
| **Inputs** | `s1`: Sentinel-1 VV, VH in dB · `s2`: 12 Sentinel-2 bands (B1–B12 without B10, ×10000) · `s1s2`: both (14) |
| **Output** | per-pixel water probability; threshold chosen on validation and stored in the checkpoint |
| **Training data** | Sen1Floods11 hand-labelled chips (official train split), random 256-px crops, dihedral flips/rotations |
| **Validation** | official `valid` split — early stopping (patience 12) and threshold selection |
| **Test** | official `test` split, and the `bolivia` split (an entire event withheld from training) |
| **Loss / optimiser** | 0.5 BCE + 0.5 Dice, no-data ignored · AdamW, one-cycle LR 3e-4, AMP, up to 60 epochs |
| **Baselines** | MNDWI (green vs SWIR-1) threshold; VV backscatter threshold — both tuned on validation |
| **Intended use** | Mapping flood extent at CRISP flood sites; `s1` for cloudy conditions, `s2` when clear |
| **Limitations** | Permanent water is labelled water too, so event floods are mapped as *new* water (after minus before). Optical L1C→L2A and SAR σ⁰→γ⁰ differences at application time. Urban and vegetated floods are under-represented in the training data. |

## burn_scar — burned-area segmentation

| | |
|---|---|
| **CRISP module** | Wildfires |
| **Architecture** | U-Net, ResNet-34 encoder, 1 output logit |
| **Inputs** | 6 bands: Blue, Green, Red, NIR-narrow (B8A), SWIR-1, SWIR-2 at 30 m (HLS ≡ Sentinel-2 band set) |
| **Training / val / test** | HLS Burn Scars `training` folder split 85 / 15 by Sentinel-2 tile; official `validation` folder = test |
| **Baseline** | NBR (NIR vs SWIR-2) threshold tuned on validation — CRISP's current burn rule |
| **Intended use** | Burned area after a fire at CRISP wildfire sites; applied before and after, counting only new scars |
| **Limitations** | Training fires are all in the contiguous USA (2018–2021); Mediterranean shrubland and Australian eucalypt fires are out of distribution. 30 m resolution. Not an active-fire detector. |

## forest — forest / non-forest segmentation and forest loss

| | |
|---|---|
| **CRISP module** | Forestry |
| **Architecture** | U-Net, ResNet-34 encoder, 1 output logit |
| **Inputs** | Sentinel-2 Red, Green, Blue, NIR (B4, B3, B2, B8) at 10 m, per-image 2–98 % percentile normalisation |
| **Training / val / test** | Amazon + Atlantic Forest training images split 88 / 12; official validation images (both biomes) = test |
| **Baseline** | NDVI threshold tuned on validation |
| **Intended use** | Forest cover at two dates → forest loss (km²) at CRISP forestry sites |
| **Limitations** | Trained on tropical / subtropical Brazilian forest; boreal (Dalarna) and Congo-basin forest are out of distribution. Cloud persistence in the tropics limits clear composites. |

## landuse_eurosat — land-use / crop-land classification

| | |
|---|---|
| **CRISP module** | Agriculture |
| **Architecture** | ResNet-18, pretrained with MoCo on SSL4EO-S12 (1 M Sentinel-2 scenes, 13 bands); B10 filter removed → 12 inputs |
| **Inputs** | 64×64 px Sentinel-2 patch (640 m), 12 bands, reflectance (DN / 10000) |
| **Classes** | AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake |
| **Splits** | Official TorchGeo EuroSAT train / val / test |
| **Intended use** | Land-use map of each CRISP site → cropland mask for crop-health (NDVI) monitoring; farmland share per site |
| **Limitations** | EuroSAT covers European countries; worldwide farm sites (Punjab, Nile Delta, Iowa) differ in field size and crops. Patch-level (640 m) labels, not field boundaries. Does not measure crop stress directly. |

## lgbm_no2_* · lgbm_pm25_* — next-day air-quality forecast

| | |
|---|---|
| **CRISP module** | Air quality / Industrial |
| **Model** | LightGBM gradient-boosted trees, one model shared by all 60 sites (site as categorical), trained on log1p target |
| **Features** | today's and lagged pollutant levels, 7/28-day statistics, other pollutants, weather today (and tomorrow, in the `with_nextday_weather` variant), day-of-week and season |
| **Target** | tomorrow's daily-mean NO₂ / PM2.5 (µg/m³) from CAMS global |
| **Splits** | chronological: train to 31 Mar 2025, validate Apr–Sep 2025, test Oct 2025 – Sep 2026 |
| **Baselines** | persistence (tomorrow = today), weekly persistence, 7-day mean |
| **Limitations** | Target is CAMS model output (~45 km grid), not station measurements; next-day weather comes from reanalysis, standing in for a weather forecast (slightly optimistic — the `no_nextday_weather` variant removes it). |
