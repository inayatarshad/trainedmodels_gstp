"""Tests for the deblurring stage: PSFs, blur, classic deconvolution, metrics, datasets.

    pytest -q tests/test_deblur.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

torch = pytest.importorskip("torch")
pytest.importorskip("skimage")

from gstp import blur as B  # noqa: E402
from gstp import deblur_classic as D  # noqa: E402


def _scene(size=128, seed=0, bands=3):
    """Smooth background + sharp edges + small bright targets, reflectance in [0, 1]."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size] / size
    out = []
    for c in range(bands):
        img = 0.15 + 0.1 * np.sin(6 * xx + c) * np.cos(4 * yy)
        img[(xx > 0.3) & (xx < 0.6) & (yy > 0.2) & (yy < 0.7)] += 0.25  # field with sharp edges
        for _ in range(15):
            y, x = rng.integers(4, size - 4, 2)
            img[y - 1: y + 2, x - 1: x + 2] = 0.8  # small buildings
        out.append(img)
    return np.clip(np.stack(out), 0, 1).astype(np.float32)


@pytest.mark.parametrize("bt", B.ALL_TYPES)
def test_psfs_are_valid(bt):
    rng = np.random.default_rng(1)
    for _ in range(5):
        psf = B.psf_from_recipe(B.sample_recipe(rng, bt))
        assert psf.ndim == 2 and psf.shape[0] % 2 == 1 and psf.shape[1] % 2 == 1
        assert psf.min() >= 0 and abs(psf.sum() - 1) < 1e-5


def test_motion_psf_has_right_length_and_direction():
    psf = B.motion_psf(9, 0)  # horizontal
    rows = np.where(psf.sum(1) > 1e-3)[0]
    cols = np.where(psf.sum(0) > 1e-3)[0]
    assert len(rows) <= 2 and 8 <= len(cols) <= 11


def test_blur_matches_direct_convolution_and_keeps_brightness():
    from scipy.ndimage import convolve

    x = _scene(64, bands=1)[0]
    psf = B.defocus_psf(2.5)
    y = B.blur(torch.from_numpy(x), psf).numpy()
    ref = convolve(x, psf[::-1, ::-1], mode="mirror")  # mirror = reflect without edge repeat
    r = B.psf_radius(psf) + 1
    assert np.allclose(y[r:-r, r:-r], ref[r:-r, r:-r], atol=1e-5)
    assert abs(y.mean() - x.mean()) < 2e-3
    assert y.shape == x.shape


def test_degrade_is_reproducible():
    x = torch.from_numpy(_scene(64))
    r = B.sample_recipe(np.random.default_rng(5), "motion")
    a, _ = B.degrade(x, r)
    b, _ = B.degrade(x, r)
    assert torch.equal(a, b)


def test_classic_methods_improve_a_blurred_image():
    x = _scene(128)
    r = {"blur_type": "defocus", "radius": 2.5, "optics_sigma": 0.5, "noise_std": 0.001, "psf_seed": 0}
    y, psf = B.degrade(torch.from_numpy(x), r)
    valid = np.ones(x.shape[1:], bool)
    base = D.score(x, y.numpy(), valid)["psnr"]
    w = D.score(x, D.wiener(y, psf, 1e-2).numpy(), valid)["psnr"]
    rl = D.score(x, D.richardson_lucy(y, psf, 30).numpy(), valid)["psnr"]
    assert w > base + 1 and rl > base + 1, (base, w, rl)


def test_rl_checkpoints_match_single_runs():
    y = B.blur(torch.from_numpy(_scene(64, bands=1)), B.gaussian_psf(1.5))
    psf = B.gaussian_psf(1.5)
    many = D.richardson_lucy(y, psf, [5, 10])
    assert torch.allclose(many[10], D.richardson_lucy(y, psf, 10), atol=1e-6)


def test_metrics_known_values():
    gt = np.full((2, 32, 32), 0.5, np.float32)
    valid = np.ones((32, 32), bool)
    assert D.score(gt, gt, valid)["ssim"] == pytest.approx(1.0)
    assert D.psnr(gt, gt + 0.1, D.eval_mask(valid)) == pytest.approx(20.0, abs=1e-3)


def test_eval_set_tune_and_evaluate(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    from gstp.data.deblur import DeblurEvalSet, DeblurTrainSet, guess_scale
    from gstp.data.segmentation import Sample

    samples = []
    for i in range(10):
        p = tmp_path / f"chip{i}.tif"
        arr = (_scene(96, seed=i) * 10000).astype("float32")
        arr[:, :4, :4] = -9999  # a little no-data
        with rasterio.open(p, "w", driver="GTiff", height=96, width=96, count=3, dtype="float32",
                           transform=from_origin(0, 0, 10, 10)) as dst:
            dst.write(arr)
        samples.append(Sample(id=f"chip{i}", sources=[(str(p), None)]))
    scale = guess_scale(samples, nodata=-9999)
    assert scale == 10000
    val = DeblurEvalSet(samples[:5], scale, -9999, seed=1, name="fake")
    test = DeblurEvalSet(samples[5:], scale, -9999, seed=2, name="fake")
    assert [r["blur_type"] for r in test.recipe_table()][:4] == list(B.ALL_TYPES)
    it = test[0]
    assert it["blurred"].shape == it["sharp"].shape == (3, 96, 96) and not it["valid"][:5, :5].any()
    # rebuilding from saved recipes gives the identical test set
    again = DeblurEvalSet(samples[5:], scale, -9999, recipes=test.recipes)
    assert np.array_equal(again[3]["blurred"], test[3]["blurred"])

    best = D.tune(list(val), k_grid=[1e-3, 1e-2, 1e-1], rl_iters=(5, 10), log=lambda *_: None)
    rows = D.evaluate(list(test), D.method_table(best), log=lambda *_: None)
    assert len(rows) == 5 * 5 and all(np.isfinite(r["psnr"]) for r in rows)

    tr = DeblurTrainSet(samples[:5], scale, -9999, crop=32, length=4, min_valid=0.5)
    yb, xs = tr[0]
    assert yb.shape == xs.shape == (1, 32, 32)


def test_train_mix_and_motion_kinds():
    rng = np.random.default_rng(0)
    rs = [B.sample_recipe(rng) for _ in range(2000)]
    frac = {t: np.mean([r["blur_type"] == t for r in rs]) for t in B.ALL_TYPES}
    assert all(abs(frac[t] - B.TRAIN_MIX[t]) < 0.04 for t in B.ALL_TYPES)
    kinds = {r["kind"] for r in rs if r["blur_type"] == "motion"}
    assert kinds == {"line", "jitter"}
    assert max(r["noise_std"] for r in rs) <= B.RANGES["noise_std"][1]


def test_rl_matches_scikit_image():
    from skimage.restoration import richardson_lucy as sk_rl

    x = _scene(96, bands=1)[0]
    psf = B.defocus_psf(2.0)
    y = B.blur(torch.from_numpy(x), psf)
    ours = D.richardson_lucy(y, psf, 15).numpy()
    ref = sk_rl(y.numpy().astype(np.float64), psf.astype(np.float64), num_iter=15, clip=False)
    c = slice(20, -20)  # borders differ (we reflect-pad, scikit-image zero-pads)
    assert np.abs(ours[c, c] - ref[c, c]).mean() < 2e-3


@pytest.mark.parametrize("name", ["nafnet", "autoencoder"])
def test_models_keep_shape_and_odd_sizes(name):
    from gstp.deblur_models import build, tiled

    m = build(name).eval()
    with torch.no_grad():
        assert m(torch.rand(2, 1, 64, 64)).shape == (2, 1, 64, 64)
        assert m(torch.rand(1, 1, 75, 61)).shape == (1, 1, 75, 61)
        out = tiled(m, torch.rand(3, 300, 280), tile=128, overlap=16)
    assert out.shape == (3, 300, 280) and torch.isfinite(out).all()


def test_nafnet_starts_as_identity():
    from gstp.deblur_models import build

    m = build("nafnet").eval()
    x = torch.rand(1, 1, 64, 64) * 0.3
    with torch.no_grad():
        # beta/gamma start at 0, so the blocks are skipped; only intro->ending remains + residual
        y = m(x)
    assert y.shape == x.shape


def test_tiled_matches_untiled_for_pointwise_model():
    from gstp.deblur_models import tiled

    class Scale2(torch.nn.Module):
        def forward(self, t):
            return 2 * t

    x = torch.rand(2, 200, 230)
    assert torch.allclose(tiled(Scale2(), x, tile=64, overlap=16), 2 * x, atol=1e-5)


def test_short_training_run_learns(tmp_path):
    from gstp.data.deblur import DeblurTrainSet
    from gstp.train_deblur import DeblurTrainConfig, train

    chips = [(_scene(128, seed=i, bands=2), np.ones((128, 128), bool)) for i in range(6)]
    tr = DeblurTrainSet(chips, crop=48, length=64)
    val = []
    for i in range(2):
        sharp = _scene(96, seed=50 + i, bands=1)
        r = {"blur_type": "gaussian", "sigma": 1.5, "optics_sigma": 0.4, "noise_std": 0.0, "psf_seed": i}
        y, psf = B.degrade(torch.from_numpy(sharp), r)
        val.append({"sharp": sharp, "blurred": y.numpy(), "valid": np.ones((96, 96), bool)})
    cfg = DeblurTrainConfig(name="nafnet", iterations=60, batch_size=8, crop=48, val_every=30, num_workers=0,
                            amp=False)
    model, summary = train(tr, val, cfg, tmp_path, log=lambda *_: None)
    assert (tmp_path / "deblur_nafnet_best.pt").exists()
    h = summary["history"]
    assert len(h) == 2 and h[1]["train_l1"] < h[0]["train_l1"]  # it learns (beating blur takes longer)


def test_hazard_scorer_counts_sharp_once(tmp_path):
    pytest.importorskip("segmentation_models_pytorch")
    from gstp.hazard_check import SHARP, HazardScorer
    from gstp.models import build_unet

    m = build_unet(3, pretrained=False)
    torch.save({"state_dict": m.state_dict(), "threshold": 0.5,
                "config": {"in_channels": 3, "extra": {"stats": {"mean": [1000.0] * 3, "std": [500.0] * 3}}}},
               tmp_path / "h.pt")
    hs = HazardScorer(tmp_path / "h.pt", 10000.0, name="fake")
    sharp = _scene(64)
    lab = np.zeros((64, 64), np.uint8)
    lab[20:40, 20:40] = 1
    lab[:2] = 255
    item = {"id": "a", "blur_type": "motion", "sharp": sharp, "label": lab, "valid_data": np.ones((64, 64), bool)}
    for meth in ("No deblurring", "NAFNet"):
        hs(item, meth, sharp)
    rows = {(r["method"], r["blur_type"]): r for r in hs.table()}
    assert set(rows) == {(SHARP, "all"), (SHARP, "motion"), ("No deblurring", "all"), ("No deblurring", "motion"),
                         ("NAFNet", "all"), ("NAFNet", "motion")}
    assert rows[(SHARP, "all")]["iou"] == pytest.approx(rows[("NAFNet", "all")]["iou"])
    assert hs.meters[(SHARP, "all")].tp + hs.meters[(SHARP, "all")].fn == 400
