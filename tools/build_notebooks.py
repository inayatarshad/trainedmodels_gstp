"""Generate the Colab notebooks in notebooks/ from the cell definitions below.

Keeping notebook source in one Python file means every notebook shares the
same setup cell and can be reviewed as plain code. Run:  python tools/build_notebooks.py
"""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"

SETUP = r'''# ── Setup: run once per Colab session (Runtime ▸ Change runtime type ▸ T4 GPU) ──
import os, sys, subprocess, shutil
from pathlib import Path

REPO_URL = "https://github.com/inayatarshad/trainedmodels_gstp.git"
IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    WORK = Path("/content")
    DRIVE = Path("/content/drive/MyDrive/gstp_runs")       # checkpoints + results persist here
    REPO = WORK / "trainedmodels_gstp"
    if not REPO.exists():
        if subprocess.run(["git", "clone", "-q", REPO_URL, str(REPO)]).returncode != 0:
            # Private repo? Upload the repo zip to MyDrive/trainedmodels_gstp.zip and re-run.
            shutil.unpack_archive("/content/drive/MyDrive/trainedmodels_gstp.zip", WORK)
            REPO = next(WORK.glob("trainedmodels_gstp*"))
    os.chdir(REPO)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt"], check=True)
else:  # local / Kaggle
    REPO = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
    os.chdir(REPO)
    WORK = DRIVE = REPO / "runs"

sys.path.insert(0, str(REPO / "src"))
DRIVE.mkdir(parents=True, exist_ok=True)

import torch
from gstp.utils import seed_everything, get_device
seed_everything(42)
print("Repo:", REPO, "| device:", get_device(), "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
'''

FINISH = r'''# ── Package results: metrics + figures go to Drive as a zip you can download ──
import shutil
archive = shutil.make_archive(str(DRIVE / f"results_{MODULE}"), "zip", RESULTS)
print("Results zip :", archive)
print("Checkpoints :", sorted(str(p) for p in CKPT_DIR.glob("*.pt")))
'''


def nb(title, intro, cells):
    n = nbf.v4.new_notebook()
    n.metadata = {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    }
    out = [nbf.v4.new_markdown_cell(f"# {title}\n\n{intro}"), nbf.v4.new_code_cell(SETUP)]
    for kind, src in cells:
        out.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src.strip("\n")))
    n.cells = out
    return n


def md(s):
    return ("md", s)


def code(s):
    return ("code", s)


# ─────────────────────────────────────────────────────────────── 01 Floods
FLOODS = nb(
    "01 · Floods — water segmentation (Sen1Floods11)",
    """**CRISP module:** Floods · **Task:** per-pixel flood-water segmentation · **Model:** U-Net / ResNet-34

Trains three models on the hand-labelled Sen1Floods11 chips and compares them:

| Model | Input | Why |
|---|---|---|
| `flood_s1` | Sentinel-1 radar VV, VH | Sees through cloud — floods happen under storm cloud |
| `flood_s2` | Sentinel-2, 12 bands | Best accuracy when the sky is clear |
| `flood_s1s2` | both | Fusion |

Each is scored on the official **test** split and on **Bolivia**, a flood event never seen in training
(geographic generalisation), and compared with the index-threshold rules CRISP uses today.
Runtime on a T4: about 25–35 min for all three.""",
    [
        code(r'''
MODULE = "floods"
from gstp.data import sen1floods11 as S
from gstp.data.segmentation import SegDataset
from gstp.data.raster import compute_band_stats, read_raster
from gstp.models import build_unet, count_parameters
from gstp import train_seg, baselines, viz
from gstp.utils import save_json, load_json
import numpy as np, pandas as pd, torch

DATA = WORK / "data" / "sen1floods11"
RESULTS = REPO / "results" / MODULE
CKPT_DIR = DRIVE / MODULE
for p in (DATA, RESULTS / "figures", CKPT_DIR):
    p.mkdir(parents=True, exist_ok=True)
'''),
        md("## 1. Download the hand-labelled chips (~3 GB, public Google Cloud bucket)"),
        code(r'''
if not any(DATA.rglob("*_LabelHand.tif")):
    for sub in ("S1Hand", "S2Hand", "LabelHand"):
        subprocess.run(["gsutil", "-m", "-q", "cp", "-r",
                        f"gs://sen1floods11/v1.1/data/flood_events/HandLabeled/{sub}", str(DATA)], check=True)
    subprocess.run(["gsutil", "-m", "-q", "cp", "-r", "gs://sen1floods11/v1.1/splits", str(DATA)], check=True)
print({d: len(list(DATA.rglob(f"*_{d}.tif"))) for d in ("S1Hand", "S2Hand", "LabelHand")})
print([p.name for p in DATA.rglob("*.csv")])
'''),
        md("## 2. Build the splits and inspect the data"),
        code(r'''
splits = {m: {sp: S.build_samples(DATA, sp, m) for sp in ("train", "val", "test", "bolivia")} for m in S.MODALITIES}
print({sp: len(v) for sp, v in splits["s1s2"].items()})
assert all(len(v) for v in splits["s1s2"].values()), "An empty split — check the downloaded CSVs"
events = pd.Series([s.meta["event"] for s in splits["s1s2"]["train"]]).value_counts()
print("Training chips per flood event:\n", events.to_string())
'''),
        code(r'''
# Training-set statistics (never computed on val/test) for z-score normalisation
from gstp.data.segmentation import read_sample_image
stats = {m: compute_band_stats([s for s in splits[m]["train"]], read_sample_image) for m in S.MODALITIES}
for m in stats:
    print(m, "bands:", S.MODALITIES[m]["bands"])
    print("   mean:", np.round(stats[m]["mean"], 2))
save_json(stats, RESULTS / "band_stats.json")
'''),
        md("## 3. Train the three models"),
        code(r'''
EPOCHS = 60          # early stopping usually ends earlier
summaries, models = {}, {}
for m in ("s2", "s1", "s1s2"):
    seed_everything(42)
    bands = S.MODALITIES[m]["bands"]
    cfg = train_seg.SegTrainConfig(name=f"flood_{m}", in_channels=len(bands), bands=bands, epochs=EPOCHS,
                                   batch_size=16, crop_size=256, lr=3e-4, patience=12,
                                   extra={"norm": "zscore", "stats": stats[m], "modality": m,
                                          "units": "S1: dB; S2: TOA reflectance x 10000", "dataset": "Sen1Floods11 v1.1"})
    tr = SegDataset(splits[m]["train"], "zscore", stats[m], train=True, crop_size=256)
    va = SegDataset(splits[m]["val"], "zscore", stats[m])
    model = build_unet(len(bands))
    print(f"\n=== flood_{m}: {len(bands)} bands, {count_parameters(model)/1e6:.1f} M parameters ===")
    summaries[m] = train_seg.train(model, tr, va, cfg, CKPT_DIR)
    models[m] = model
    viz.training_curves(summaries[m]["history"], title=f"flood_{m}", path=RESULTS / "figures" / f"curves_flood_{m}.png")
'''),
        md("## 4. Test once — official test split and the unseen Bolivia event"),
        code(r'''
reports = {}
for m, model in models.items():
    thr = summaries[m]["chosen_threshold"]
    tests = {sp: SegDataset(splits[m][sp], "zscore", stats[m]) for sp in ("test", "bolivia")}
    reports[m] = train_seg.test_report(model, tests, thr, RESULTS / f"test_flood_{m}.json",
                                       group_key=lambda s: s.meta["event"])
rows = []
for m, r in reports.items():
    for sp in ("test", "bolivia"):
        x = r[sp]["at_chosen_threshold"]
        rows.append({"model": f"U-Net {m}", "split": sp, "IoU (water)": x["iou"], "F1": x["f1"],
                     "precision": x["precision"], "recall": x["recall"], "mIoU": x["miou"]})
'''),
        md("## 5. Baselines: the index thresholds CRISP-style rules use"),
        code(r'''
s2_bands = S.MODALITIES["s2"]["bands"]
G, SW1 = s2_bands.index("B3"), s2_bands.index("B11")
bl = {
    "MNDWI threshold (S2)": baselines.tune_and_test(
        splits["s2"]["val"], {"test": splits["s2"]["test"], "bolivia": splits["s2"]["bolivia"]},
        lambda x: baselines.normalized_difference(x[G], x[SW1]), ">"),
    "VV backscatter threshold (S1)": baselines.tune_and_test(
        splits["s1"]["val"], {"test": splits["s1"]["test"], "bolivia": splits["s1"]["bolivia"]},
        lambda x: x[0], "<"),
}
for name, r in bl.items():
    for sp in ("test", "bolivia"):
        x = r[sp]
        rows.append({"model": name, "split": sp, "IoU (water)": x["iou"], "F1": x["f1"],
                     "precision": x["precision"], "recall": x["recall"], "mIoU": x["miou"]})
save_json(bl, RESULTS / "baselines.json")
table = pd.DataFrame(rows).sort_values(["split", "IoU (water)"], ascending=[True, False])
table.to_csv(RESULTS / "results_table.csv", index=False)
table.style.format(precision=3).hide(axis="index")
'''),
        md("## 6. Figures — threshold choice, per-event scores and example predictions"),
        code(r'''
from gstp.data.raster import IGNORE_INDEX
best_m = max(reports, key=lambda m: reports[m]["test"]["at_chosen_threshold"]["iou"])
print("Best model on test:", best_m)
val_meter, _ = train_seg.evaluate(models[best_m], SegDataset(splits[best_m]["val"], "zscore", stats[best_m]))
viz.threshold_curve(val_meter, summaries[best_m]["chosen_threshold"], f"flood_{best_m}: validation",
                    RESULTS / "figures" / "threshold_curve.png")

per_event = pd.DataFrame({m: {g: v["iou"] for g, v in reports[m]["test"]["per_group"].items()} for m in reports})
per_event.to_csv(RESULTS / "per_event_iou.csv")
ax = per_event.plot.bar(figsize=(9, 3.5), title="Test IoU per flood event"); ax.set_ylabel("IoU")
ax.figure.tight_layout(); ax.figure.savefig(RESULTS / "figures" / "per_event_iou.png", dpi=150)
'''),
        code(r'''
def examples(split, m, n=4):
    ds = SegDataset(splits[m][split], "zscore", stats[m])
    raw = SegDataset(splits["s2"][split], "none")
    model = models[m].eval()
    order = np.argsort([-np.mean(ds.load(i)[1] == 1) for i in range(len(ds))])[: n * 3 : 3]  # flood-rich chips
    out = []
    for i in order:
        x, y = ds.load(i)
        with torch.no_grad():
            p = torch.sigmoid(model(torch.from_numpy(x)[None].to(get_device()))).squeeze().cpu().numpy()
        img, _ = raw.load(i)
        out.append({"rgb": viz.rgb(img, (3, 2, 1)), "mask": y, "prob": p,
                    "thr": summaries[m]["chosen_threshold"], "name": ds.samples[i].id})
    return out
for sp in ("test", "bolivia"):
    viz.prediction_grid(examples(sp, best_m), f"flood_{best_m} on {sp}", RESULTS / "figures" / f"examples_{sp}.png")
'''),
        md("## 7. Export: ONNX for the later Jetson / TensorRT step, plus a results summary"),
        code(r'''
from gstp.models import export_onnx
for m, model in models.items():
    export_onnx(model, len(S.MODALITIES[m]["bands"]), 512, str(CKPT_DIR / f"flood_{m}.onnx"))
save_json({"summaries": {m: {k: v for k, v in s.items() if k != "history"} for m, s in summaries.items()},
           "test": reports}, RESULTS / "summary.json")
'''),
        code(FINISH),
    ],
)

# ─────────────────────────────────────────────────────────────── 02 Wildfires
WILDFIRES = nb(
    "02 · Wildfires — burn-scar segmentation (HLS Burn Scars)",
    """**CRISP module:** Wildfires · **Task:** per-pixel burn-scar segmentation · **Model:** U-Net / ResNet-34

Input bands are the six Sentinel-2-equivalent HLS bands (Blue, Green, Red, NIR-narrow, SWIR-1, SWIR-2), so the
model can be applied directly to Sentinel-2 imagery of the CRISP wildfire sites. The official `validation` folder is
held out as the **test** set; validation for early stopping is carved from `training`, grouped by Sentinel-2 tile so that
neighbouring chips never sit on both sides. Compared against an NBR threshold — CRISP's current burn rule.
Runtime on a T4: about 20 min.""",
    [
        code(r'''
MODULE = "wildfires"
from gstp.data import burn_scars as B
from gstp.data.segmentation import SegDataset, split_samples, read_sample_image
from gstp.data.raster import compute_band_stats
from gstp.models import build_unet, count_parameters, export_onnx
from gstp import train_seg, baselines, viz
from gstp.utils import save_json
import numpy as np, pandas as pd, torch, tarfile

DATA = WORK / "data" / "hls_burn_scars"
RESULTS = REPO / "results" / MODULE
CKPT_DIR = DRIVE / MODULE
for p in (DATA, RESULTS / "figures", CKPT_DIR):
    p.mkdir(parents=True, exist_ok=True)
'''),
        md("## 1. Download from Hugging Face"),
        code(r'''
from huggingface_hub import snapshot_download
if not any(DATA.rglob("*merged*.tif")):
    snapshot_download("ibm-nasa-geospatial/hls_burn_scars", repo_type="dataset", local_dir=str(DATA))
    for t in DATA.rglob("*.tar.gz"):
        with tarfile.open(t) as tf:
            tf.extractall(DATA)
print("chips:", len(list(DATA.rglob("*merged*.tif"))))
'''),
        code(r'''
train_all = B.build_samples(DATA, "training")
test = B.build_samples(DATA, "validation")
assert train_all and test, "Burn-scar chips not found — check the extracted folder layout"
train, val = split_samples(train_all, 0.15, seed=42, group_key=lambda s: s.meta["tile"])
print(f"train {len(train)} | val {len(val)} | test {len(test)}")
stats = compute_band_stats(train, read_sample_image, nodata=B.NODATA)
units_scale = 10000.0 if np.median(stats["mean"]) > 10 else 1.0   # reflectance units of this release
print("band means:", np.round(stats["mean"], 4), "| reflectance scale:", units_scale)
save_json({"stats": stats, "reflectance_scale": units_scale}, RESULTS / "band_stats.json")
'''),
        md("## 2. Train"),
        code(r'''
cfg = train_seg.SegTrainConfig(name="burn_scar", in_channels=6, bands=B.BANDS, epochs=60, batch_size=16,
                               crop_size=256, lr=3e-4, patience=12,
                               extra={"norm": "zscore", "stats": stats, "reflectance_scale": units_scale,
                                      "resolution_m": 30, "dataset": "HLS Burn Scars"})
tr = SegDataset(train, "zscore", stats, nodata=B.NODATA, train=True, crop_size=256)
va = SegDataset(val, "zscore", stats, nodata=B.NODATA)
model = build_unet(6)
print(f"{count_parameters(model)/1e6:.1f} M parameters")
summary = train_seg.train(model, tr, va, cfg, CKPT_DIR)
viz.training_curves(summary["history"], title="burn_scar", path=RESULTS / "figures" / "curves_burn_scar.png");
'''),
        md("## 3. Test once, and compare with the NBR-threshold rule"),
        code(r'''
thr = summary["chosen_threshold"]
report = train_seg.test_report(model, {"test": SegDataset(test, "zscore", stats, nodata=B.NODATA)}, thr,
                               RESULTS / "test_burn_scar.json")
NIR, SW2 = B.BANDS.index("B8A"), B.BANDS.index("B12")
bl = baselines.tune_and_test(val, {"test": test}, lambda x: baselines.normalized_difference(x[NIR], x[SW2]), "<",
                             nodata=B.NODATA)
save_json({"NBR threshold": bl}, RESULTS / "baselines.json")
rows = [{"model": "U-Net (6 bands)", **{k: report["test"]["at_chosen_threshold"][k] for k in ("iou", "f1", "precision", "recall", "miou")}},
        {"model": f"NBR < {bl['threshold']:.3f}", **{k: bl["test"][k] for k in ("iou", "f1", "precision", "recall", "miou")}}]
table = pd.DataFrame(rows).rename(columns={"iou": "IoU (burn)"})
table.to_csv(RESULTS / "results_table.csv", index=False)
table.style.format(precision=3).hide(axis="index")
'''),
        code(r'''
val_meter, _ = train_seg.evaluate(model, va)
viz.threshold_curve(val_meter, thr, "burn_scar: validation", RESULTS / "figures" / "threshold_curve.png")
ds, raw = SegDataset(test, "zscore", stats, nodata=B.NODATA), SegDataset(test, "none", nodata=B.NODATA)
order = np.argsort([-np.mean(ds.load(i)[1] == 1) for i in range(len(ds))])[:20:5]
ex = []
for i in order:
    x, y = ds.load(i)
    with torch.no_grad():
        p = torch.sigmoid(model.eval()(torch.from_numpy(x)[None].to(get_device()))).squeeze().cpu().numpy()
    img, _ = raw.load(i)
    ex.append({"rgb": viz.rgb(img, (5, 3, 2)), "mask": y, "prob": p, "thr": thr, "name": ds.samples[i].id[-30:]})
viz.prediction_grid(ex, "Burn scars — test (false colour SWIR2/NIR/Red)", RESULTS / "figures" / "examples_test.png",
                    color=(1.0, 0.3, 0.0));
'''),
        code(r'''
export_onnx(model, 6, 512, str(CKPT_DIR / "burn_scar.onnx"))
save_json({"summary": {k: v for k, v in summary.items() if k != "history"}, "test": report, "baseline": bl},
          RESULTS / "summary.json")
'''),
        code(FINISH),
    ],
)

# ─────────────────────────────────────────────────────────────── 03 Forestry
FORESTRY = nb(
    "03 · Forestry — forest segmentation & forest-loss mapping (Amazon + Atlantic Forest)",
    """**CRISP module:** Forestry · **Task:** forest / non-forest segmentation; forest *loss* is the difference between
two dates · **Model:** U-Net / ResNet-34 on Sentinel-2 Red, Green, Blue, NIR

Data: Bragagnolo et al. (2021), Zenodo 4498086, CC BY 4.0 — hand-checked forest masks from two Brazilian biomes.
The official test images are unlabelled, so the official **validation** images are our test set.
Per-image percentile normalisation makes the model independent of how the dataset was scaled to bytes, so it can
run on raw Sentinel-2 L2A imagery at the CRISP forestry sites. Runtime on a T4: about 20 min (+ ~5 min download).""",
    [
        code(r'''
MODULE = "forestry"
from gstp.data import forest as F
from gstp.data.segmentation import SegDataset, split_samples
from gstp.models import build_unet, count_parameters, export_onnx
from gstp import train_seg, baselines, viz
from gstp.utils import save_json
import numpy as np, pandas as pd, torch

DATA = WORK / "data" / "forest"
RESULTS = REPO / "results" / MODULE
CKPT_DIR = DRIVE / MODULE
for p in (DATA, RESULTS / "figures", CKPT_DIR):
    p.mkdir(parents=True, exist_ok=True)
'''),
        md("## 1. Download (2 × 1.2 GB) and extract"),
        code(r'''
import urllib.request
subprocess.run("apt-get -qq install -y unrar > /dev/null", shell=True)
for name in ("AMAZON.rar", "ATLANTIC FOREST.rar"):
    dst = DATA / name
    if not dst.exists():
        url = "https://zenodo.org/records/4498086/files/" + name.replace(" ", "%20") + "?download=1"
        print("downloading", name); urllib.request.urlretrieve(url, dst)
    marker = DATA / (name + ".done")
    if not marker.exists():
        subprocess.run(["unrar", "x", "-o+", "-inul", str(dst), str(DATA) + "/"], check=True); marker.touch()
'''),
        code(r'''
train_all, rep_tr = F.build_samples(DATA, "training")
test, rep_te = F.build_samples(DATA, "validation")
print(rep_tr); print(rep_te)
assert rep_tr["paired"] > 0.95 * rep_tr["images"], "Image/mask pairing incomplete — inspect the folder layout"
assert rep_te["paired"] > 0, "No labelled validation images found — inspect the folder layout"
train, val = split_samples(train_all, 0.12, seed=42)
print(f"train {len(train)} | val {len(val)} | test {len(test)}")
# Visual check that masks line up with images
viz.prediction_grid([{"rgb": viz.rgb(SegDataset([s], "none").load(0)[0], (0, 1, 2)),
                      "mask": SegDataset([s], "none").load(0)[1],
                      "prob": SegDataset([s], "none").load(0)[1].astype(float), "thr": 0.5, "name": s.id}
                     for s in train[:2]], "Pairing check (prediction panel = ground truth)", color=(0.1, 0.8, 0.2));
'''),
        md("## 2. Train"),
        code(r'''
cfg = train_seg.SegTrainConfig(name="forest", in_channels=4, bands=F.BANDS, epochs=60, batch_size=16, crop_size=256,
                               lr=3e-4, patience=12,
                               extra={"norm": "percentile", "dataset": "Amazon + Atlantic Forest (Zenodo 4498086)",
                                      "resolution_m": 10})
tr = SegDataset(train, "percentile", train=True, crop_size=256)
va = SegDataset(val, "percentile")
model = build_unet(4)
summary = train_seg.train(model, tr, va, cfg, CKPT_DIR)
viz.training_curves(summary["history"], title="forest", path=RESULTS / "figures" / "curves_forest.png");
'''),
        md("## 3. Test once (per biome), compare with an NDVI threshold"),
        code(r'''
thr = summary["chosen_threshold"]
report = train_seg.test_report(model, {"test": SegDataset(test, "percentile")}, thr, RESULTS / "test_forest.json",
                               group_key=lambda s: s.meta["biome"])
bl = baselines.tune_and_test(val, {"test": test}, lambda x: baselines.normalized_difference(x[3], x[0]), ">")
save_json({"NDVI threshold": bl}, RESULTS / "baselines.json")
rows = [{"model": "U-Net (R,G,B,NIR)", **report["test"]["at_chosen_threshold"]},
        {"model": f"NDVI > {bl['threshold']:.3f}", **bl["test"]}]
for b, v in report["test"]["per_group"].items():
    rows.append({"model": f"U-Net — {b} only", **v})
table = pd.DataFrame(rows)[["model", "iou", "f1", "precision", "recall", "miou", "accuracy"]].rename(columns={"iou": "IoU (forest)"})
table.to_csv(RESULTS / "results_table.csv", index=False)
table.style.format(precision=3).hide(axis="index")
'''),
        code(r'''
ds, raw = SegDataset(test, "percentile"), SegDataset(test, "none")
ex = []
for i in np.linspace(0, len(ds) - 1, 4).astype(int):
    x, y = ds.load(i)
    with torch.no_grad():
        p = torch.sigmoid(model.eval()(torch.from_numpy(x)[None].to(get_device()))).squeeze().cpu().numpy()
    ex.append({"rgb": viz.rgb(raw.load(i)[0], (0, 1, 2)), "mask": y, "prob": p, "thr": thr, "name": ds.samples[i].id})
viz.prediction_grid(ex, "Forest — test", RESULTS / "figures" / "examples_test.png", color=(0.1, 0.8, 0.2))
export_onnx(model, 4, 512, str(CKPT_DIR / "forest.onnx"))
save_json({"summary": {k: v for k, v in summary.items() if k != "history"}, "test": report, "baseline": bl},
          RESULTS / "summary.json")
'''),
        code(FINISH),
    ],
)

# ─────────────────────────────────────────────────────────────── 04 Agriculture
AGRI = nb(
    "04 · Agriculture — land-use / crop-land classification (EuroSAT, Sentinel-2)",
    """**CRISP module:** Agriculture · **Task:** classify 64×64 Sentinel-2 patches (640 m) into 10 land-use classes,
including *Annual crop*, *Permanent crop*, *Pasture* and *Herbaceous vegetation* · **Model:** ResNet-18
pretrained on 1 M Sentinel-2 scenes (SSL4EO-S12, MoCo)

Run patch-by-patch over a site, the classifier produces a land-use map, which tells CRISP where the farmland is —
the mask its NDVI crop-health readings should be computed over. Uses 12 bands (B10 dropped) so the model
accepts Sentinel-2 L2A. Official TorchGeo train / val / test splits. Runtime on a T4: about 25 min.""",
    [
        code(r'''
MODULE = "agriculture"
from torchgeo.datasets import EuroSAT
from gstp.models import build_s2_classifier, classifier_bands, count_parameters
from gstp import train_cls, viz
from gstp.utils import save_json
import numpy as np, pandas as pd, torch

DATA = WORK / "data" / "eurosat"
RESULTS = REPO / "results" / MODULE
CKPT_DIR = DRIVE / MODULE
for p in (DATA, RESULTS / "figures", CKPT_DIR):
    p.mkdir(parents=True, exist_ok=True)
'''),
        code(r'''
# Bands are requested in the channel order of the SSL4EO-S12 pretrained weights
# (B8A after B08). TorchGeo lists B8A last by default, so the order must be explicit.
BANDS = classifier_bands(drop_bands=["B10"])
ds = {sp: EuroSAT(str(DATA), split=sp, bands=BANDS, download=True) for sp in ("train", "val", "test")}
CLASSES = list(ds["train"].classes)
print({k: len(v) for k, v in ds.items()}, "\nclasses:", CLASSES, "\nbands:", BANDS)
'''),
        code(r'''
cfg = train_cls.ClsTrainConfig(name="landuse_eurosat", class_names=CLASSES, bands=BANDS, epochs=30, batch_size=128,
                               lr=1e-3, patience=8)
model = build_s2_classifier(len(CLASSES), drop_bands=["B10"], input_size=128)
print(f"{count_parameters(model)/1e6:.1f} M parameters")
summary = train_cls.train(model, train_cls.ScaledDataset(ds["train"], cfg.scale, True),
                          train_cls.ScaledDataset(ds["val"], cfg.scale, False), cfg, CKPT_DIR)
viz.training_curves(summary["history"], metric="val_macro_f1", title="landuse_eurosat",
                    path=RESULTS / "figures" / "curves.png");
'''),
        md("## Test once"),
        code(r'''
rep = train_cls.test_report(model, train_cls.ScaledDataset(ds["test"], cfg.scale, False), cfg, RESULTS / "test_landuse.json")
print(f"Test accuracy {rep['accuracy']:.4f} | macro-F1 {rep['macro_f1']:.4f}")
viz.confusion(rep["confusion_matrix"], CLASSES, "EuroSAT test — row-normalised", RESULTS / "figures" / "confusion.png")
per = pd.DataFrame(rep["per_class"]).T
AGRI_CLASSES = ["AnnualCrop", "PermanentCrop", "Pasture", "HerbaceousVegetation"]
print("Mean F1 over agricultural classes:", per.loc[[c for c in AGRI_CLASSES if c in per.index], "f1"].mean().round(4))
per.to_csv(RESULTS / "per_class.csv")
per.style.format(precision=3)
'''),
        code(r'''
from gstp.models import export_onnx
ck = torch.load(CKPT_DIR / "landuse_eurosat_best.pt", weights_only=False)
ck["config"]["extra"] = {"scale": cfg.scale, "patch": 64, "bands": BANDS, "classes": CLASSES, "input_size": 128}
torch.save(ck, CKPT_DIR / "landuse_eurosat_best.pt")
export_onnx(model, len(BANDS), 64, str(CKPT_DIR / "landuse_eurosat.onnx"))
save_json({"summary": {k: v for k, v in summary.items() if k != "history"}, "test": rep}, RESULTS / "summary.json")
'''),
        code(FINISH),
    ],
)

# ─────────────────────────────────────────────────────────────── 05 Air quality
AIR = nb(
    "05 · Air quality — next-day NO₂ and PM2.5 forecasts for all 60 CRISP sites",
    """**CRISP module:** Air quality / Industrial · **Task:** forecast tomorrow's daily-mean NO₂ and PM2.5 at every
CRISP site · **Model:** one LightGBM model shared by all 60 sites

Data: CAMS global atmospheric-composition fields (which assimilate satellite retrievals, including Sentinel-5P) and
ERA5 weather, both via the free Open-Meteo APIs, Aug 2022 → Sep 2026. Chronological split — train until Mar 2025,
validate Apr–Sep 2025, **test on the whole following year (Oct 2025 – Sep 2026)**, so every season is tested.
Scored against persistence baselines ("tomorrow = today"), which are hard to beat for day-ahead air quality.
CPU only, about 10 min (most of it downloading).""",
    [
        code(r'''
MODULE = "air_quality"
from gstp import air_quality as AQ
from gstp.metrics import regression_scores
from gstp.utils import load_sites, save_json
import numpy as np, pandas as pd, matplotlib.pyplot as plt

RESULTS = REPO / "results" / MODULE
CKPT_DIR = DRIVE / MODULE
CACHE = DRIVE / "aq_cache"
for p in (RESULTS / "figures", CKPT_DIR, CACHE):
    p.mkdir(parents=True, exist_ok=True)
START, END = "2022-08-01", "2026-09-30"
VAL_START, TEST_START = pd.Timestamp("2025-04-01"), pd.Timestamp("2025-10-01")
sites = load_sites()
print(len(sites), "CRISP sites")
'''),
        code(r'''
hourly = AQ.fetch_all(sites, START, END, CACHE)
daily = AQ.to_daily(hourly)
daily.to_csv(RESULTS / "daily_site_data.csv.gz", index=False)
print(daily.shape, daily["date"].min().date(), "→", daily["date"].max().date())
'''),
        code(r'''
results, models = {}, {}
for target in ("no2", "pm25"):
    for weather in (True, False):
        tag = f"{target}_{'with' if weather else 'no'}_nextday_weather"
        df = AQ.make_features(daily, target, use_next_day_weather=weather)
        tr, va, te = AQ.chrono_split(df, VAL_START, TEST_START)
        model, cols = AQ.train_lgbm(tr, va)
        pred = AQ.predict_lgbm(model, te, cols)
        res = {"LightGBM": regression_scores(te["target"], pred)}
        for name, b in AQ.baselines(te, AQ.TARGETS[target]).items():
            ok = ~np.isnan(b)
            res[name] = regression_scores(te["target"].values[ok], b[ok])
        res["skill_vs_persistence_mae"] = 1 - res["LightGBM"]["mae"] / res["persistence"]["mae"]
        res["n_train"], res["n_val"], res["n_test"] = len(tr), len(va), len(te)
        results[tag], models[tag] = res, (model, cols, te, pred)
        model.save_model(str(CKPT_DIR / f"lgbm_{tag}.txt"))
        print(f"{tag:<34} MAE {res['LightGBM']['mae']:.2f} vs persistence {res['persistence']['mae']:.2f}"
              f"  → skill {res['skill_vs_persistence_mae']:+.1%}   R² {res['LightGBM']['r2']:.3f}")
save_json(results, RESULTS / "test_metrics.json")
'''),
        code(r'''
rows = []
for tag, r in results.items():
    for m in ("LightGBM", "persistence", "weekly_persistence", "mean_7day"):
        rows.append({"setting": tag, "model": m, **{k: r[m][k] for k in ("mae", "rmse", "r2", "bias")}})
table = pd.DataFrame(rows)
table.to_csv(RESULTS / "results_table.csv", index=False)
table.style.format(precision=3).hide(axis="index")
'''),
        md("## Per-site accuracy, feature importance and example forecasts"),
        code(r'''
model, cols, te, pred = models["no2_with_nextday_weather"]
te = te.assign(pred=pred, persist=te["nitrogen_dioxide_lag0"])
per_site = te.groupby("site_id", observed=True).apply(
    lambda g: pd.Series({"MAE_model": np.mean(np.abs(g.pred - g.target)),
                         "MAE_persistence": np.mean(np.abs(g.persist - g.target)),
                         "mean_NO2": g.target.mean()})).sort_values("mean_NO2", ascending=False)
per_site["skill"] = 1 - per_site.MAE_model / per_site.MAE_persistence
per_site.to_csv(RESULTS / "per_site_no2.csv")
print(f"Model beats persistence at {(per_site.skill > 0).sum()} of {len(per_site)} sites")

imp = pd.Series(model.feature_importance("gain"), index=cols).sort_values(ascending=False).head(15)
ax = imp[::-1].plot.barh(figsize=(6, 4.5), title="NO₂ model — top features (gain)")
ax.figure.tight_layout(); ax.figure.savefig(RESULTS / "figures" / "feature_importance_no2.png", dpi=150)

fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
for ax, sid in zip(axes, ["delhi-ncr", "madrid-sur", "manchester-centre", "ruhr-essen"]):
    g = te[te.site_id == sid].sort_values("target_date")
    ax.plot(g.target_date, g.target, label="observed (CAMS)", lw=1)
    ax.plot(g.target_date, g.pred, label="LightGBM forecast", lw=1)
    ax.set_title(sid, fontsize=9); ax.set_ylabel("NO₂ µg/m³")
axes[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(RESULTS / "figures" / "no2_forecasts.png", dpi=150)
'''),
        code(r'''
save_json({"results": results, "period": [START, END], "val_start": str(VAL_START.date()),
           "test_start": str(TEST_START.date()), "data": "CAMS global via Open-Meteo; ERA5 via Open-Meteo archive"},
          RESULTS / "summary.json")
'''),
        code(FINISH),
    ],
)

# ─────────────────────────────────────────────────────────────── 06 CRISP sites
CRISP = nb(
    "06 · Applying the trained models to the CRISP sites",
    """Runs each trained model on live Sentinel imagery of its CRISP sites for real events, and writes maps and a
summary table. Imagery: Microsoft Planetary Computer (free, no account). **Run notebooks 01–04 first**; this one
loads their checkpoints from Drive.

| Module | Sites | Imagery | Output |
|---|---|---|---|
| Floods | 6 flood events (Sukkur 2022, Valencia 2024, …) | Sentinel-1 radar, before & after | new-water area (km²) |
| Wildfires | 5 fires (Pedrógão 2017, Tubbs 2017, …) | Sentinel-2, before & after | burned area (km²) |
| Forestry | 4 forest sites | Sentinel-2, two years apart | forest cover & loss (km²) |
| Agriculture | 6 farm sites + 36 Spain/UK sites | Sentinel-2, growing season | land-use composition |

These are **applications, not accuracy tests** — the test-set scores are in notebooks 01–04. Comparing the
mapped areas with published event reports is a useful sanity check for the write-up. Runtime: 20–40 min (mostly downloads).""",
    [
        code(r'''
MODULE = "crisp_sites"
from gstp import inference as INF, viz
from gstp.models import build_unet, build_s2_classifier
from gstp.data.raster import normalise
from gstp.utils import load_json, load_sites, save_json, CONFIG_DIR
import numpy as np, pandas as pd, torch, matplotlib.pyplot as plt, traceback

RESULTS = DRIVE / "results" / MODULE   # on Drive, so maps survive a disconnect
CKPT_DIR = DRIVE
(RESULTS / "maps").mkdir(parents=True, exist_ok=True)
EVENTS = load_json(CONFIG_DIR / "crisp_events.json")
SITES = {s["id"]: s for s in load_sites()}
DEV = get_device()

def load_seg(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = build_unet(ck["config"]["in_channels"], pretrained=False); m.load_state_dict(ck["state_dict"])
    return m.eval().to(DEV), ck["config"], ck.get("threshold", 0.5)

def save_map(panels, titles, title, path):
    fig, ax = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.2))
    for a, p, t in zip(np.atleast_1d(ax), panels, titles):
        a.imshow(p); a.set_title(t, fontsize=9); a.axis("off")
    fig.suptitle(title, fontsize=10); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
'''),
        md("## Floods — Sentinel-1 radar model, change between before and after"),
        code(r'''
flood_m, flood_cfg, flood_thr = load_seg(CKPT_DIR / "floods" / "flood_s1_best.pt")
st = flood_cfg["extra"]["stats"]
rows = []
for ev in EVENTS["floods"]:
    site = SITES[ev["site"]]
    try:
        out = {}
        for when in ("pre", "post"):
            arr, valid, info = INF.load_s1_db(site["bbox"], ev[when])
            prob = INF.sliding_window_probs(flood_m, normalise(arr, "zscore", st), device=DEV)
            out[when] = (arr, (prob >= flood_thr) & valid, info)
        new_water = out["post"][1] & ~out["pre"][1]
        rows.append({"site": site["label"], "event": ev["event"], "pre_date": out["pre"][2]["date"],
                     "post_date": out["post"][2]["date"], "water_before_km2": INF.area_km2(out["pre"][1], 10),
                     "water_after_km2": INF.area_km2(out["post"][1], 10), "new_water_km2": INF.area_km2(new_water, 10)})
        vv = lambda a: np.clip((a[0] + 25) / 25, 0, 1)
        save_map([vv(out["pre"][0]), vv(out["post"][0]), viz.overlay(np.dstack([vv(out["post"][0])] * 3), new_water, (0.1, 0.4, 1))],
                 [f"VV before {out['pre'][2]['date']}", f"VV after {out['post'][2]['date']}", "new flood water (blue)"],
                 f"{site['label']} — {ev['event']}", RESULTS / "maps" / f"flood_{site['id']}.png")
        print("✓", rows[-1])
    except Exception as e:
        print("✗", site["id"], e)
pd.DataFrame(rows).to_csv(RESULTS / "floods.csv", index=False)
pd.DataFrame(rows)
'''),
        md("## Wildfires — burn-scar model on Sentinel-2 (30 m), burned after but not before"),
        code(r'''
burn_m, burn_cfg, burn_thr = load_seg(CKPT_DIR / "wildfires" / "burn_scar_best.pt")
ex = burn_cfg["extra"]; S2B = ["B02", "B03", "B04", "B8A", "B11", "B12"]
rows = []
for ev in EVENTS["wildfires"]:
    site = SITES[ev["site"]]
    try:
        out = {}
        for when in ("pre", "post"):
            arr, valid, info = INF.load_s2(site["bbox"], ev[when], S2B, resolution=30)
            x = arr / 10000.0 * ex["reflectance_scale"]
            prob = INF.sliding_window_probs(burn_m, normalise(x, "zscore", ex["stats"]), device=DEV)
            out[when] = (arr, (prob >= burn_thr) & valid, info)
        burned = out["post"][1] & ~out["pre"][1]
        rows.append({"site": site["label"], "event": ev["event"], "post_scenes": ", ".join(out["post"][2]["dates"]),
                     "burned_area_km2": INF.area_km2(burned, 30)})
        save_map([viz.rgb(out["pre"][0], (5, 3, 2)), viz.rgb(out["post"][0], (5, 3, 2)),
                  viz.overlay(viz.rgb(out["post"][0], (2, 1, 0)), burned, (1, 0.3, 0))],
                 ["before (SWIR/NIR/Red)", "after (SWIR/NIR/Red)", "new burn scar (orange)"],
                 f"{site['label']} — {ev['event']}", RESULTS / "maps" / f"burn_{site['id']}.png")
        print("✓", rows[-1])
    except Exception as e:
        print("✗", site["id"], e)
pd.DataFrame(rows).to_csv(RESULTS / "wildfires.csv", index=False)
pd.DataFrame(rows)
'''),
        md("## Wildfires (flames) — active-fire model on every Sentinel-2 image taken during each fire"),
        code(r'''
af_m, af_cfg, af_thr = load_seg(CKPT_DIR / "wildfires_active_fire" / "active_fire_best.pt")
afx = af_cfg["extra"]; rows = []
for ev in EVENTS["active_fire"]:
    site = SITES[ev["site"]]
    try:
        scenes = INF.load_s2_scenes(site["bbox"], ev["during"], afx["sentinel2_bands"], resolution=30)
        best = None
        for date, refl, valid, sid in scenes:
            prob = INF.sliding_window_probs(af_m, normalise(refl, "zscore", afx["stats"]), tile=256, overlap=32, device=DEV)
            fire = (prob >= af_thr) & valid
            rows.append({"site": site["label"], "event": ev["event"], "date": date, "scene": sid,
                         "fire_pixels": int(fire.sum()), "fire_area_km2": INF.area_km2(fire, 30), "valid_fraction": float(valid.mean())})
            if best is None or fire.sum() > best[2].sum():
                best = (date, refl, fire)
        if best is not None:
            fc = np.dstack([best[1][0], best[1][1], best[1][2]]); fc = np.clip(fc / np.percentile(fc, 99), 0, 1)
            save_map([fc, viz.overlay(fc, best[2], (1.0, 0.9, 0.0), alpha=0.8)], [f"SWIR2/SWIR1/Blue {best[0]}", "detected flames (yellow)"],
                     f"{site['label']} — {ev['event']}", RESULTS / "maps" / f"activefire_{site['id']}.png")
        print("✓", site["id"], len(scenes), "scenes")
    except Exception as e:
        print("✗", site["id"], e)
pd.DataFrame(rows).to_csv(RESULTS / "active_fire.csv", index=False)
pd.DataFrame(rows).sort_values("fire_pixels", ascending=False).head(15)
'''),
        md("## Forestry — forest cover in two years and the loss between them"),
        code(r'''
forest_m, _, forest_thr = load_seg(CKPT_DIR / "forestry" / "forest_best.pt")
rows = []
for ev in EVENTS["forestry"]:
    site = SITES[ev["site"]]
    try:
        out = {}
        for when in ("before", "after"):
            arr, valid, info = INF.load_s2(site["bbox"], ev[when], ["B04", "B03", "B02", "B08"], max_cloud=50)
            prob = INF.sliding_window_probs(forest_m, normalise(arr, "percentile"), device=DEV)
            out[when] = (arr, (prob >= forest_thr) & valid, valid)
        both = out["before"][2] & out["after"][2]
        loss = out["before"][1] & ~out["after"][1] & both
        rows.append({"site": site["label"], "period": ev["event"],
                     "forest_before_km2": INF.area_km2(out["before"][1] & both, 10),
                     "forest_after_km2": INF.area_km2(out["after"][1] & both, 10),
                     "forest_loss_km2": INF.area_km2(loss, 10), "valid_fraction": float(both.mean())})
        save_map([viz.rgb(out["before"][0]), viz.rgb(out["after"][0]), viz.overlay(viz.rgb(out["after"][0]), loss, (1, 0, 0))],
                 ["before", "after", "forest loss (red)"], f"{site['label']} — {ev['event']}",
                 RESULTS / "maps" / f"forest_{site['id']}.png")
        print("✓", rows[-1])
    except Exception as e:
        print("✗", site["id"], e)
pd.DataFrame(rows).to_csv(RESULTS / "forestry.csv", index=False)
pd.DataFrame(rows)
'''),
        md("## Agriculture — land-use composition of every farm, Spain and UK site"),
        code(r'''
# Each site runs in its own process with a memory cap and is loaded in tiles (tools/apply_agriculture.py): a heavy Sentinel-2 load
# then fails that one site instead of crashing the Colab session, and results are saved after every site.
import json, os
AG_CKPT = CKPT_DIR / "agriculture" / "landuse_eurosat_best.pt"
targets = [s for s in SITES.values() if s["focus"] in ("agriculture", "all")]
part = RESULTS / "agriculture_sites"; part.mkdir(exist_ok=True)
env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
for site in targets:
    out = part / f"{site['id']}.json"
    if out.exists():
        continue
    for attempt, (tiles, items) in enumerate([(2, 3), (3, 2)]):
        r = subprocess.run([sys.executable, "tools/apply_agriculture.py", str(AG_CKPT), site["id"], str(out),
                            "--mem-gb", "6", "--max-items", str(items), "--tiles", str(tiles)], capture_output=True, text=True, env=env)
        if out.exists():
            print("✓", site["id"]); break
        print("✗", site["id"], f"attempt {attempt + 1}:", (r.stderr.strip().splitlines() or ["?"])[-1][:140])
rows = [json.loads(p.read_text()) for p in sorted(part.glob("*.json"))]
agri = pd.DataFrame(rows)
agri["farmland_share"] = agri[["AnnualCrop", "PermanentCrop", "Pasture"]].sum(1)
agri.sort_values("farmland_share", ascending=False).to_csv(RESULTS / "agriculture_landuse.csv", index=False)
agri.sort_values("farmland_share", ascending=False).head(15)
'''),
        code(r'''
import shutil
archive = shutil.make_archive(str(DRIVE / f"results_{MODULE}"), "zip", RESULTS)
print("Results zip:", archive)
'''),
    ],
)

# ─────────────────────────────────────────────────────────────── 07 Active fire (flames)
ACTIVE_FIRE = nb(
    "07 · Wildfires — active-fire (flame) segmentation (Landsat-8 Active Fire)",
    """**CRISP module:** Wildfires · **Task:** find burning pixels — flames — in a single image, so a fire can be flagged
while it is happening · **Model:** U-Net / ResNet-34 on SWIR-2, SWIR-1, Blue (Landsat-8 bands 7-6-2 = Sentinel-2 B12-B11-B02)

Data: Pereira et al. (2021), *Active fire detection in Landsat-8 imagery*, ISPRS J. Photogramm. Remote Sens., CC BY 4.0.
Training labels are the 'voting' masks of three classic active-fire algorithms (Schroeder, Murphy, Kumar-Roy).
The **test set is the hand-labelled subset**: the model and each classic algorithm are scored against human
annotations on the same patches. Scenes in the hand-labelled set are removed from training.
Runtime on a T4: ~15 min download/unzip + ~40 min training.""",
    [
        code(r'''
MODULE = "wildfires_active_fire"
from gstp.data import active_fire as AF
from gstp.data.segmentation import SegDataset, split_samples, read_sample_image
from gstp.data.raster import compute_band_stats, read_mask, IGNORE_INDEX
from gstp.models import build_unet, count_parameters, export_onnx
from gstp.metrics import BinarySegMeter
from gstp import train_seg, viz
from gstp.utils import save_json
import numpy as np, pandas as pd, torch, zipfile, glob

DATA = WORK / "data" / "activefire"
RESULTS = REPO / "results" / MODULE
CKPT_DIR = DRIVE / MODULE
for p in (DATA / "compressed", RESULTS / "figures", CKPT_DIR):
    p.mkdir(parents=True, exist_ok=True)
pre = AF.l8_dn_to_toa
'''),
        md("## 1. Download: hand-labelled test set + three regions for training (~15 GB from Google Drive)"),
        code(r'''
import gdown
MANUAL = {"landsat_patches": "1uZnc65_GRFdAoavGoUKkJfeVQOUI8lVg", "masks_patches": "1RCURItVvqsT_oMxlhB5NYiRp8SJ9_xZ3",
          "manual_annotations_patches": "1LdsX-rH5hy_82jfc1akO8p4n0_N8lRgf"}
REGIONS = {"Oceania": "1xHYTICHKU0u3-kIrq-pM9k0YaeQt60Bt", "North_America1": "1zW_pEIggJ5Li7uQX9XKMfHkcgL3kiUoi",
           "Asia4": "1lyR6y6u8tSozfv3AQJ1PuUcdR0BU9yUk"}
# Asia2 ("1w_wv0_QZhnH9jO1ygJg6ssTrXupIbTHp", 9.7 GB) is left out: the three regions above already give >6,800 fire
# patches, and adding it overflows the free Colab disk.
MAN_DIR = DATA / "manual"

_drive = None
def fetch(fid, out):
    """Download a public Google Drive file. The anonymous route (gdown) is often blocked by Drive's
    'too many users downloaded this file' quota, so fall back to the signed-in Drive API, and if the
    shared file's quota is still exhausted, make a temporary copy in your own Drive, download that,
    and remove the copy."""
    global _drive
    out = str(out)
    try:
        gdown.download(id=fid, output=out, quiet=True)
        if os.path.exists(out) and zipfile.is_zipfile(out):
            return
    except Exception as e:
        print("  gdown blocked:", str(e).splitlines()[0][:90])
    from google.colab import auth
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaIoBaseDownload
    if _drive is None:
        auth.authenticate_user(); _drive = build("drive", "v3")

    def pull(file_id):
        req = _drive.files().get_media(fileId=file_id, supportsAllDrives=True)
        with open(out, "wb") as f:
            dl, done = MediaIoBaseDownload(f, req, chunksize=256 * 1024 * 1024), False
            while not done:
                _, done = dl.next_chunk()

    try:
        pull(fid); print("  downloaded via Drive API"); return
    except HttpError as e:
        print("  Drive API direct download refused (", e.status_code, ") - using a temporary copy")
    copy = _drive.files().copy(fileId=fid, body={"name": f"tmp_gstp_{fid}.zip"}, supportsAllDrives=True).execute()
    try:
        pull(copy["id"]); print("  downloaded via temporary copy")
    finally:
        _drive.files().delete(fileId=copy["id"]).execute()

for name, fid in MANUAL.items():
    z = DATA / "compressed" / f"{name}.zip"
    if not (MAN_DIR / name).exists():
        if not z.exists():
            print("downloading", name); fetch(fid, z)
        zipfile.ZipFile(z).extractall(MAN_DIR / name)
        z.unlink()                                          # free disk
print("manual set ready:", sum(1 for _ in MAN_DIR.rglob("*.tif")), "tifs")
'''),
        code(r'''
import shutil
img_dir, mask_dir = DATA / "images" / "patches", DATA / "masks" / "patches"
vote_dir, inter_dir = DATA / "masks" / "voting", DATA / "masks" / "intersection"
for d in (img_dir, mask_dir, vote_dir, inter_dir):
    d.mkdir(parents=True, exist_ok=True)
for region, fid in REGIONS.items():
    done = DATA / f"{region}.done"
    if done.exists():
        continue
    z = DATA / "compressed" / f"{region}.zip"
    if not z.exists():
        print("downloading", region); fetch(fid, z)
    tmp = DATA / "tmp"; shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir()
    zipfile.ZipFile(z).extractall(tmp)
    for inner in glob.glob(str(tmp / "**" / "*.zip"), recursive=True):
        if inner.endswith("masks_derivates.zip"):
            with zipfile.ZipFile(inner) as zf:
                for n in zf.namelist():
                    if n.lower().endswith(".tif") and "_voting_" in n.lower():
                        zf.extract(n, vote_dir)
        elif inner.endswith("masks.zip"):
            zipfile.ZipFile(inner).extractall(mask_dir)
        else:
            zipfile.ZipFile(inner).extractall(img_dir)
    shutil.rmtree(tmp); z.unlink(); done.touch()        # free disk as we go
for d in (img_dir, vote_dir):                            # flatten any nested folders
    for f in list(d.rglob("*.tif")):
        if f.parent != d:
            f.rename(d / f.name)
print({"images": len(list(img_dir.glob("*.tif"))), "voting masks": len(list(vote_dir.glob("*.tif")))})
'''),
        md("## 2. Build the splits (test scenes removed from training)"),
        code(r'''
test, ALGO, rep_man = AF.build_manual_index(MAN_DIR)
print(rep_man)
assert rep_man["paired"] > 0, "Hand-labelled patches not paired — inspect MAN_DIR layout"
test_scenes = {s.meta["scene"] for s in test}
samples, rep_tr = AF.build_training_index(DATA, max_fire=5000, nonfire_ratio=0.5, exclude_scenes=test_scenes)
print(rep_tr)
assert rep_tr["fire_patches"] > 500, "Too few fire patches — check the voting-mask names"
train, val = split_samples(samples, 0.12, seed=42, group_key=lambda s: s.meta["scene"])
print(f"train {len(train)} | val {len(val)} | test (hand-labelled) {len(test)}")
stats = compute_band_stats(train, lambda s: pre(read_sample_image(s)), nodata=0)
print("band means (TOA reflectance, SWIR2/SWIR1/Blue):", np.round(stats["mean"], 4))
save_json({"stats": stats, "manual": rep_man, "train_index": rep_tr}, RESULTS / "data_report.json")
'''),
        md("## 3. Train"),
        code(r'''
cfg = train_seg.SegTrainConfig(name="active_fire", in_channels=3, bands=AF.S2_EQUIVALENT, epochs=25, batch_size=16,
                               crop_size=None, lr=3e-4, patience=8,
                               extra={"norm": "zscore", "stats": stats, "input": "TOA reflectance",
                                      "landsat8_bands": AF.L8_BANDS, "sentinel2_bands": AF.S2_EQUIVALENT,
                                      "resolution_m": 30, "dataset": "Landsat-8 Active Fire (Pereira et al. 2021)"})
tr = SegDataset(train, "zscore", stats, nodata=0, train=True, pre=pre)
va = SegDataset(val, "zscore", stats, nodata=0, pre=pre)
model = build_unet(3)
print(f"{count_parameters(model)/1e6:.1f} M parameters")
summary = train_seg.train(model, tr, va, cfg, CKPT_DIR)
viz.training_curves(summary["history"], title="active_fire", path=RESULTS / "figures" / "curves.png");
'''),
        md("## 4. Test once against the hand-labelled patches — model vs the three classic algorithms"),
        code(r'''
thr = summary["chosen_threshold"]
te = SegDataset(test, "zscore", stats, nodata=0, pre=pre)
report = train_seg.test_report(model, {"test": te}, thr, RESULTS / "test_active_fire.json")
rows = [{"method": "U-Net (SWIR2, SWIR1, Blue)", **report["test"]["at_chosen_threshold"]}]
algo_scores = {}
for a in ("Schroeder", "Murphy", "Kumar-Roy", "Voting", "Intersection"):
    m = BinarySegMeter([0.5])
    for s in test:
        gt = read_mask(s.mask); valid = gt != IGNORE_INDEX
        p = AF.strip_tokens(Path(s.sources[0][0]).name).lower()
        pred = read_mask(ALGO[a][p]) == 1 if p in ALGO[a] else np.zeros_like(gt, bool)
        m.update(pred.astype(np.float32), gt == 1, valid)
    algo_scores[a] = m.compute(); rows.append({"method": f"{a} algorithm", **algo_scores[a]})
save_json(algo_scores, RESULTS / "baselines.json")
table = pd.DataFrame(rows)[["method", "iou", "f1", "precision", "recall"]].rename(columns={"iou": "IoU (fire)"})
table.to_csv(RESULTS / "results_table.csv", index=False)
table.style.format(precision=3).hide(axis="index")
'''),
        code(r'''
raw = SegDataset(test, "none", nodata=0, pre=pre)
fire_rich = np.argsort([-(te.load(i)[1] == 1).sum() for i in range(len(te))])[:4]
ex = []
for i in fire_rich:
    x, y = te.load(i)
    with torch.no_grad():
        p = torch.sigmoid(model.eval()(torch.from_numpy(x)[None].to(get_device()))).squeeze().cpu().numpy()
    ex.append({"rgb": viz.rgb(raw.load(i)[0], (0, 1, 2)), "mask": y, "prob": p, "thr": thr, "name": te.samples[i].id[-24:]})
viz.prediction_grid(ex, "Active fire — hand-labelled test patches (false colour SWIR2/SWIR1/Blue)",
                    RESULTS / "figures" / "examples_test.png", color=(1.0, 0.85, 0.0))
export_onnx(model, 3, 256, str(CKPT_DIR / "active_fire.onnx"))
import json
print("GSTPJSON" + json.dumps({"test": report["test"]["at_chosen_threshold"], "n_test": len(test), "thr": thr,
      "algos": algo_scores, "split": [len(train), len(val), len(test)], "data": rep_tr, "manual": rep_man,
      "ep": [summary["best_epoch"], summary["epochs_run"], summary["train_time_s"]],
      "hist": [[h["epoch"], round(h["train_loss"], 4), round(h["val_iou"], 4)] for h in summary["history"]]}, default=float) + "GSTPEND")
'''),
        code(FINISH),
    ],
)

NOTEBOOKS = {
    "01_floods_sen1floods11.ipynb": FLOODS,
    "02_wildfires_burn_scars.ipynb": WILDFIRES,
    "03_forestry_forest_segmentation.ipynb": FORESTRY,
    "04_agriculture_landuse_eurosat.ipynb": AGRI,
    "05_air_quality_forecasting.ipynb": AIR,
    "06_apply_to_crisp_sites.ipynb": CRISP,
    "07_wildfires_active_fire.ipynb": ACTIVE_FIRE,
}

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, n in NOTEBOOKS.items():
        nbf.write(n, OUT / name)
        print("wrote", OUT / name)
