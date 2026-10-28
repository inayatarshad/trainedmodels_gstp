"""Smoke tests: run every training path end-to-end on tiny synthetic data.

They check the plumbing (readers, normalisation, loss, metrics, training loop,
threshold selection, test report, baselines, sliding-window inference, air-quality
features) without needing the real datasets or a GPU:

    pytest -q tests
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

rasterio = pytest.importorskip("rasterio")
torch = pytest.importorskip("torch")

from gstp import baselines, train_seg  # noqa: E402
from gstp.data import burn_scars, forest, sen1floods11  # noqa: E402
from gstp.data.raster import IGNORE_INDEX, compute_band_stats, read_mask  # noqa: E402
from gstp.data.segmentation import SegDataset, read_sample_image, split_samples  # noqa: E402
from gstp.metrics import BinarySegMeter, classification_report, regression_scores  # noqa: E402


def _write_tif(path, arr, dtype="float32"):
    from rasterio.transform import from_origin

    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[None]
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2], count=arr.shape[0],
                       dtype=dtype, transform=from_origin(0, 0, 10, 10)) as dst:
        dst.write(arr.astype(dtype))


def _blob(rng, size):
    yy, xx = np.mgrid[:size, :size]
    cy, cx, r = rng.integers(10, size - 10, 2).tolist() + [int(rng.integers(6, size // 3))]
    return ((yy - cy) ** 2 + (xx - cx) ** 2 < r**2).astype(np.float32)


@pytest.fixture(scope="module")
def fake_sen1floods11(tmp_path_factory):
    root = tmp_path_factory.mktemp("s1f11")
    rng = np.random.default_rng(0)
    splits = {"train": [], "valid": [], "test": [], "bolivia": []}
    size = 64
    for i in range(24):
        event = ["Ghana", "India", "Spain", "Bolivia"][i % 4]
        cid = f"{event}_{1000 + i}"
        water = _blob(rng, size)
        s1 = np.stack([np.where(water > 0, -22, -8), np.where(water > 0, -28, -15)]) + rng.normal(0, 1.5, (2, size, size))
        s2 = rng.uniform(500, 3000, (13, size, size))
        s2[2] += water * 1500  # water: green up ...
        s2[11] = np.where(water > 0, rng.uniform(50, 300, (size, size)), s2[11])  # ... SWIR-1 down
        lab = water.copy()
        lab[:3, :3] = -1
        _write_tif(root / "HandLabeled/S1Hand" / f"{cid}_S1Hand.tif", s1)
        _write_tif(root / "HandLabeled/S2Hand" / f"{cid}_S2Hand.tif", s2, "uint16")
        _write_tif(root / "HandLabeled/LabelHand" / f"{cid}_LabelHand.tif", lab, "int16")
        sp = "bolivia" if event == "Bolivia" else ["train", "train", "valid", "test"][i % 4 if i % 4 < 3 else 3]
        splits[sp].append(cid)
    # ensure each split has members
    splits["test"] += splits["train"][:2]
    d = root / "splits/flood_handlabeled"
    d.mkdir(parents=True)
    for sp, name in [("train", "train"), ("valid", "valid"), ("test", "test"), ("bolivia", "bolivia")]:
        (d / f"flood_{name}_data.csv").write_text("".join(f"{c}_S1Hand.tif,{c}_LabelHand.tif\n" for c in splits[sp]))
    return root


def test_mask_reader_handles_nodata(tmp_path):
    p = tmp_path / "m.tif"
    _write_tif(p, np.array([[-1, 0], [1, 1]], np.float32))
    m = read_mask(p)
    assert m.tolist() == [[IGNORE_INDEX, 0], [1, 1]]


def test_metrics_known_values():
    meter = BinarySegMeter([0.5])
    meter.update(np.array([0.9, 0.9, 0.1, 0.1]), np.array([1, 0, 1, 0]))
    s = meter.compute()
    assert s["tp"] == 1 and s["fp"] == 1 and s["fn"] == 1 and s["tn"] == 1
    assert abs(s["iou"] - 1 / 3) < 1e-9 and abs(s["f1"] - 0.5) < 1e-9
    rep = classification_report([0, 1, 1, 2], [0, 1, 2, 2], ["a", "b", "c"])
    assert rep["accuracy"] == 0.75
    assert regression_scores([1, 2, 3], [1, 2, 3])["mae"] == 0


@pytest.mark.parametrize("modality", ["s1", "s2", "s1s2"])
def test_flood_training_end_to_end(fake_sen1floods11, tmp_path, modality):
    from gstp.models import build_unet

    sp = {s: sen1floods11.build_samples(fake_sen1floods11, s, modality) for s in ("train", "val", "test", "bolivia")}
    assert all(len(v) > 0 for v in sp.values())
    bands = sen1floods11.MODALITIES[modality]["bands"]
    stats = compute_band_stats(sp["train"], read_sample_image)
    assert len(stats["mean"]) == len(bands)
    cfg = train_seg.SegTrainConfig(name=f"t_{modality}", in_channels=len(bands), bands=bands, epochs=2,
                                   batch_size=4, crop_size=32, num_workers=0, pretrained=False, amp=False,
                                   extra={"stats": stats})
    model = build_unet(len(bands), pretrained=False)
    summ = train_seg.train(model, SegDataset(sp["train"], "zscore", stats, train=True, crop_size=32),
                           SegDataset(sp["val"], "zscore", stats), cfg, tmp_path, log=lambda *_: None)
    assert 0.05 <= summ["chosen_threshold"] <= 0.95
    rep = train_seg.test_report(model, {"test": SegDataset(sp["test"], "zscore", stats)}, summ["chosen_threshold"],
                                tmp_path / "r.json", group_key=lambda s: s.meta["event"], num_workers=0)
    assert 0 <= rep["test"]["at_chosen_threshold"]["iou"] <= 1
    ck = torch.load(tmp_path / f"t_{modality}_best.pt", weights_only=False)
    assert "threshold" in ck and ck["config"]["extra"]["stats"] == stats


def test_index_baseline_separates_water(fake_sen1floods11):
    sp = {s: sen1floods11.build_samples(fake_sen1floods11, s, "s2") for s in ("val", "test")}
    b = sen1floods11.S2_BANDS
    r = baselines.tune_and_test(sp["val"], {"test": sp["test"]},
                                lambda x: baselines.normalized_difference(x[b.index("B3")], x[b.index("B11")]), ">")
    assert r["test"]["iou"] > 0.6  # synthetic water is spectrally obvious


def test_burn_scar_and_forest_indexers(tmp_path):
    rng = np.random.default_rng(1)
    for split in ("training", "validation"):
        for i in range(3):
            stem = f"subsetted_512x512_HLS.S30.T10SDH.20202{i}8.v1.4"
            _write_tif(tmp_path / "burn" / split / f"{stem}_merged.tif", rng.uniform(0, 0.4, (6, 32, 32)))
            _write_tif(tmp_path / "burn" / split / f"{stem}.mask.tif", (rng.random((32, 32)) > 0.8).astype(np.int16), "int16")
    s = burn_scars.build_samples(tmp_path / "burn", "training")
    assert len(s) == 3 and s[0].meta["tile"] == "T10SDH"
    tr, va = split_samples(s, 0.34, group_key=lambda x: x.meta["tile"])
    assert len(tr) + len(va) == 3

    from PIL import Image

    for split in ("Training", "Validation"):
        for i in range(4):
            _write_tif(tmp_path / "forest/AMAZON" / split / "image" / f"img_{i}.tif",
                       rng.integers(0, 255, (4, 32, 32)), "uint8")
            label_dir = tmp_path / "forest/AMAZON" / split / "label"
            label_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray((rng.random((32, 32)) > 0.5).astype(np.uint8) * 255).save(label_dir / f"img_{i}.png")
    samples, rep = forest.build_samples(tmp_path / "forest", "training")
    assert rep["paired"] == 4 and rep["images"] == 4
    # real naming: same tile-position suffix in two different scenes must not collide
    for scene in ("S2A_SCENEA", "S2B_SCENEB"):
        _write_tif(tmp_path / "forest2/AMAZON/Training/image" / f"{scene}_01_07.tif", rng.integers(0, 255, (4, 16, 16)), "uint8")
        _write_tif(tmp_path / "forest2/AMAZON/Training/label" / f"{scene}_01_07.tif", (rng.random((16, 16)) > 0.5).astype(np.uint8), "uint8")
    s2, rep2 = forest.build_samples(tmp_path / "forest2", "training")
    assert rep2["paired"] == 2
    img, mask = SegDataset(samples, "percentile").load(0)
    assert img.shape == (4, 32, 32) and 0 <= img.min() and img.max() <= 1 and set(np.unique(mask)) <= {0, 1}


def test_sliding_window_matches_shape():
    from gstp.inference import sliding_window_probs
    from gstp.models import build_unet

    model = build_unet(2, pretrained=False)
    p = sliding_window_probs(model, np.random.rand(2, 300, 200).astype(np.float32), tile=128, overlap=32)
    assert p.shape == (300, 200) and np.all((p >= 0) & (p <= 1))


def test_air_quality_features_no_leakage():
    from gstp import air_quality as AQ

    days = pd.date_range("2023-01-01", periods=400, freq="D")
    rows = []
    for sid in ("a", "b"):
        for d in days:
            row = {"site_id": sid, "date": d, "lat": 1.0, "lon": 2.0}
            row.update({p: float(d.dayofyear % 30) for p in AQ.POLLUTANTS})
            row.update({w: 1.0 for w in AQ.WEATHER_DAILY})
            rows.append(row)
    daily = pd.DataFrame(rows)
    f = AQ.make_features(daily, "no2")
    # target is tomorrow's value; lag0 is today's
    r = f.iloc[100]
    assert r["target_date"] - pd.Timedelta(days=1) >= days[0]
    today = daily[(daily.site_id == r["site_id"]) & (daily.date == r["target_date"] - pd.Timedelta(days=1))]
    assert r["nitrogen_dioxide_lag0"] == today["nitrogen_dioxide"].iloc[0]
    tr, va, te = AQ.chrono_split(f, pd.Timestamp("2023-10-01"), pd.Timestamp("2023-12-01"))
    assert tr["target_date"].max() < va["target_date"].min() <= va["target_date"].max() < te["target_date"].min()


def test_classifier_training_end_to_end(tmp_path):
    from gstp import train_cls
    from gstp.models import build_s2_classifier, classifier_bands

    class FakeEuroSAT:
        def __init__(self, n, seed):
            g = torch.Generator().manual_seed(seed)
            self.y = torch.randint(0, 3, (n,), generator=g)
            self.x = torch.rand(n, 12, 64, 64, generator=g) * 3000 + self.y[:, None, None, None] * 2000

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            return {"image": self.x[i], "label": self.y[i]}

    names = ["a", "b", "c"]
    cfg = train_cls.ClsTrainConfig(name="cls", class_names=names, bands=classifier_bands(), epochs=2,
                                   batch_size=8, num_workers=0, amp=False)
    model = build_s2_classifier(3, drop_bands=["B10"], pretrained=False, input_size=96)
    summ = train_cls.train(model, train_cls.ScaledDataset(FakeEuroSAT(48, 0), cfg.scale, True),
                           train_cls.ScaledDataset(FakeEuroSAT(24, 1), cfg.scale, False), cfg, tmp_path,
                           log=lambda *_: None)
    rep = train_cls.test_report(model, train_cls.ScaledDataset(FakeEuroSAT(24, 2), cfg.scale, False), cfg,
                                tmp_path / "t.json")
    assert summ["epochs_run"] == 2 and 0 <= rep["accuracy"] <= 1 and len(rep["confusion_matrix"]) == 3


def test_active_fire_naming_and_pairing(tmp_path):
    import numpy as np, rasterio
    from rasterio.transform import from_origin
    from gstp.data import active_fire as AF

    def w(p, arr):
        p.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(p, "w", driver="GTiff", height=arr.shape[-2], width=arr.shape[-1], count=arr.shape[0],
                           dtype=arr.dtype, transform=from_origin(0, 0, 30, 30)) as d:
            d.write(arr)

    img = np.full((10, 8, 8), 20000, np.uint16)
    m = np.zeros((1, 8, 8), np.uint8); m[0, 2, 2] = 1
    w(tmp_path / "images/patches/LC08_A_p00001.tif", img)
    w(tmp_path / "images/patches/LC08_A_p00002.tif", img)
    w(tmp_path / "images/patches/LC08_T_p00001.tif", img)
    w(tmp_path / "masks/voting/LC08_A_Voting_p00001.tif", m)
    assert AF.strip_tokens("LC08_A_Kumar-Roy_p00001.tif") == "LC08_A_p00001.tif"
    assert AF.scene_of("LC08_A_v1_p00001.tif") == "LC08_A"
    s, rep = AF.build_training_index(tmp_path, nonfire_ratio=1.0, exclude_scenes={"LC08_T"})
    assert rep["fire_patches"] == 1 and rep["nonfire_patches"] == 1
    assert {x.meta["scene"] for x in s} == {"LC08_A"}
    assert abs(float(AF.l8_dn_to_toa(np.array([20000]))[0]) - 0.3) < 1e-6


def test_flood_v2_detectors():
    import numpy as np
    from gstp import flood_mapping as FM

    rng = np.random.default_rng(0)
    land = rng.normal(-12, 1.5, (512, 512)).astype(np.float32)
    after = land.copy(); after[:, :200] = rng.normal(-24, 1.5, (512, 200))
    valid = np.ones_like(after, bool)
    w, t, n = FM.split_based_otsu_water(after, valid)
    assert n > 0 and -22 < t < -14 and abs(w[:, :200].mean() - 1) < 0.02 and w[:, 220:].mean() < 0.02
    ch = FM.change_ratio_flood(after, land)
    assert ch[:, :200].mean() > 0.95 and ch[:, 220:].mean() < 0.05
    m = np.zeros((50, 50), bool); m[0:2, 0:2] = True; m[10:20, 10:20] = True
    assert FM.remove_small(m).sum() == 100
    assert abs(FM.gamma0_to_sigma0_db(np.zeros(1), 37.0)[0] + 0.977) < 0.01
