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


if __name__ == "__main__":
    for name in sys.argv[1:] or ["agriculture"]:
        globals()[name]()
