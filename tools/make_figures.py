"""Presentation figures rendered from the metric files in results/<module>/ (real run outputs)."""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path(__file__).resolve().parents[1] / "results"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
INK, ACCENT, MUTED = "#1f2a44", "#2f7d4f", "#8a94a6"


def agriculture():
    t = json.loads((R / "agriculture/test_landuse.json").read_text())
    h = json.loads((R / "agriculture/train_summary.json").read_text())
    names = t["class_names"]
    pretty = ["Annual crop", "Forest", "Herbaceous veg.", "Highway", "Industrial", "Pasture", "Permanent crop",
              "Residential", "River", "Sea / lake"]
    out = R / "agriculture/figures"
    out.mkdir(parents=True, exist_ok=True)

    cm = np.asarray(t["confusion_matrix"], float)
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8.6, 7.4))
    im = ax.imshow(cmn, cmap="Greens", vmin=0, vmax=1)
    for i in range(10):
        for j in range(10):
            if cm[i, j]:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=9,
                        color="white" if cmn[i, j] > 0.5 else INK)
    ax.set_xticks(range(10), pretty, rotation=40, ha="right")
    ax.set_yticks(range(10), pretty)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    ax.set_title(f"Land-use classifier — test set (n = {t['n_images']:,})\n"
                 f"accuracy {100*t['accuracy']:.2f} %   ·   macro-F1 {100*t['macro_f1']:.2f} %", color=INK)
    for s in ax.spines.values(): s.set_visible(False)
    fig.colorbar(im, fraction=0.046, label="row-normalised")
    fig.tight_layout(); fig.savefig(out / "confusion_matrix.png", dpi=200); plt.close(fig)

    ep = [x["epoch"] for x in h["history"]]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(ep, [x["train_loss"] for x in h["history"]], color=INK, lw=2)
    ax[0].set(title="Training loss", xlabel="epoch")
    ax[1].plot(ep, [100 * x["val_accuracy"] for x in h["history"]], color=ACCENT, lw=2, label="validation accuracy")
    b = h["best_epoch"]
    ax[1].scatter([b], [100 * h["history"][b - 1]["val_accuracy"]], color=ACCENT, zorder=3, s=60)
    ax[1].annotate(f"best epoch {b}\n{100*h['history'][b-1]['val_accuracy']:.2f} %", (b, 100 * h["history"][b - 1]["val_accuracy"]),
                   (b - 11, 91), arrowprops=dict(arrowstyle="->", color=MUTED), color=INK)
    ax[1].set(title="Validation accuracy (%)", xlabel="epoch", ylim=(88, 100.5))
    fig.suptitle(f"ResNet-18 (SSL4EO-S12 pretrained) on EuroSAT — {h['epochs_run']} epochs, "
                 f"{h['train_time_s']/60:.0f} min on a {h['gpu']}", color=INK)
    fig.tight_layout(); fig.savefig(out / "training_curves.png", dpi=200); plt.close(fig)

    pc = t["per_class"]
    f1 = [100 * pc[n]["f1"] for n in names]
    order = np.argsort(f1)
    agri = {"AnnualCrop", "PermanentCrop", "Pasture", "HerbaceousVegetation"}
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.barh([pretty[i] for i in order], [f1[i] for i in order],
            color=[ACCENT if names[i] in agri else "#b8c2d1" for i in order])
    for k, i in enumerate(order):
        ax.text(f1[i] + 0.03, k, f"{f1[i]:.1f}", va="center", fontsize=10, color=INK)
    ax.set_xlim(97, 100.4); ax.set_xlabel("F1 score on test set (%)")
    ax.set_title("Per-class F1 — agricultural classes in green", color=INK)
    fig.tight_layout(); fig.savefig(out / "per_class_f1.png", dpi=200); plt.close(fig)
    print("agriculture figures ->", out)


def wildfires():
    t = json.loads((R / "wildfires/test_burn_scar.json").read_text())["test"]["at_chosen_threshold"]
    b = json.loads((R / "wildfires/baselines.json").read_text())["NBR threshold"]
    h = json.loads((R / "wildfires/train_summary.json").read_text())
    out = R / "wildfires/figures"; out.mkdir(parents=True, exist_ok=True)

    keys = [("iou", "IoU (burn scar)"), ("f1", "F1"), ("precision", "Precision"), ("recall", "Recall")]
    x = np.arange(len(keys)); w = 0.36
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    m = [100 * t[k] for k, _ in keys]; n = [100 * b["test"][k] for k, _ in keys]
    ax.bar(x - w / 2, m, w, color="#c2410c", label="U-Net (6 bands)")
    ax.bar(x + w / 2, n, w, color="#b8c2d1", label=f"NBR threshold (< {b['threshold']:.2f})")
    for i in range(len(keys)):
        ax.text(x[i] - w / 2, m[i] + 1.2, f"{m[i]:.1f}", ha="center", fontsize=10, color=INK)
        ax.text(x[i] + w / 2, n[i] + 1.2, f"{n[i]:.1f}", ha="center", fontsize=10, color=MUTED)
    ax.set_xticks(x, [l for _, l in keys]); ax.set_ylim(0, 112); ax.set_ylabel("% on held-out test set")
    ax.set_title("Burn-scar mapping: learned model vs classic NBR threshold\n264 unseen test chips (HLS Burn Scars)", color=INK)
    ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(); fig.savefig(out / "model_vs_nbr.png", dpi=200); plt.close(fig)

    ep = [x["epoch"] for x in h["history"]]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(ep, [x["train_loss"] for x in h["history"]], color=INK, lw=2); ax[0].set(title="Training loss (BCE + Dice)", xlabel="epoch")
    ax[1].plot(ep, [x["val_iou"] for x in h["history"]], color="#c2410c", lw=2)
    ax[1].set(title="Validation IoU (burn scar)", xlabel="epoch", ylim=(0, 1))
    fig.suptitle(f"U-Net / ResNet-34 on HLS Burn Scars — {h['epochs_run']} epochs, {h['train_time_s']/60:.0f} min on a {h['gpu']}", color=INK)
    fig.tight_layout(); fig.savefig(out / "training_curves.png", dpi=200); plt.close(fig)
    print("wildfire figures ->", out)


def floods():
    d = json.loads((R / "floods/test_floods.json").read_text())["models"]
    b = json.loads((R / "floods/baselines.json").read_text())
    vi = json.loads((R / "floods/val_iou_history.json").read_text())
    out = R / "floods/figures"; out.mkdir(parents=True, exist_ok=True)
    rows = [("U-Net radar + optical", d["s1s2"], "#1d4ed8"), ("U-Net optical (S2)", d["s2"], "#3b82f6"),
            ("MNDWI threshold (optical)", b["MNDWI threshold (S2)"], "#b8c2d1"),
            ("U-Net radar (S1)", d["s1"], "#0f766e"), ("VV threshold (radar)", b["VV backscatter threshold (S1)"], "#d5dae3")]
    fig, ax = plt.subplots(figsize=(9.5, 4.6)); x = np.arange(2); w = 0.16
    for i, (name, r, c) in enumerate(rows):
        vals = [100 * r["test"]["iou"], 100 * r["bolivia"]["iou"]]
        bars = ax.bar(x + (i - 2) * w, vals, w, color=c, label=name)
        for xx, v in zip(x + (i - 2) * w, vals):
            ax.text(xx, v + 1, f"{v:.0f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x, ["Test split (90 chips, 10 events)", "Bolivia — event never seen (15 chips)"])
    ax.set_ylim(0, 105); ax.set_ylabel("Water IoU (%)")
    ax.set_title("Flood-water segmentation: learned models vs classic index thresholds", color=INK)
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(); fig.savefig(out / "models_vs_thresholds.png", dpi=200); plt.close(fig)

    ev = sorted(d["s1s2"]["per_event_test_iou"])
    fig, ax = plt.subplots(figsize=(9.5, 3.8)); x = np.arange(len(ev)); w = 0.27
    for i, (m, lab, c) in enumerate([("s1s2", "radar + optical", "#1d4ed8"), ("s2", "optical", "#3b82f6"), ("s1", "radar", "#0f766e")]):
        ax.bar(x + (i - 1) * w, [100 * d[m]["per_event_test_iou"][e] for e in ev], w, color=c, label=lab)
    ax.set_xticks(x, ev, rotation=0, fontsize=9); ax.set_ylim(0, 100); ax.set_ylabel("Water IoU (%)")
    ax.set_title("Test IoU per flood event", color=INK); ax.legend(frameon=False, fontsize=9, ncol=3)
    fig.tight_layout(); fig.savefig(out / "per_event_iou.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for m, lab, c in [("s1s2", "radar + optical", "#1d4ed8"), ("s2", "optical", "#3b82f6"), ("s1", "radar", "#0f766e")]:
        ax.plot(range(1, len(vi[m]) + 1), vi[m], color=c, lw=2, label=lab)
    ax.set(title="Validation IoU per epoch (early stopping)", xlabel="epoch", ylim=(0, 1)); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out / "training_curves.png", dpi=200); plt.close(fig)
    print("flood figures ->", out)


if __name__ == "__main__":
    for name in sys.argv[1:] or ["agriculture"]:
        globals()[name]()
