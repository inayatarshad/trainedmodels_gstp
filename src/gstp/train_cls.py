"""Training / evaluation loop for the image-classification (land-use) model."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .metrics import classification_report
from .utils import get_device, git_commit, save_json, utc_now


@dataclass
class ClsTrainConfig:
    name: str
    class_names: list
    bands: list
    epochs: int = 30
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8
    label_smoothing: float = 0.05
    scale: float = 10000.0  # Sentinel-2 DN -> reflectance, as used by the SSL4EO weights
    num_workers: int = 2
    amp: bool = True
    seed: int = 42


class ScaledDataset(Dataset):
    """Wraps a TorchGeo-style dataset (dict with 'image', 'label') -> (x, y).

    Applies the reflectance scaling and, for training, random flips/rotations.
    """

    def __init__(self, base, scale: float, train: bool):
        self.base, self.scale, self.train = base, scale, train

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        s = self.base[i]
        x = s["image"].float() / self.scale
        if self.train:
            k = int(torch.randint(4, (1,)))
            x = torch.rot90(x, k, dims=(1, 2))
            if torch.rand(1) < 0.5:
                x = torch.flip(x, dims=(2,))
        return x, int(s["label"])


@torch.no_grad()
def predict(model, loader, device, amp=True):
    model.eval()
    ys, ps, probs = [], [], []
    for x, y in loader:
        with torch.autocast(device_type="cuda", enabled=amp and device == "cuda"):
            logits = model(x.to(device, non_blocking=True))
        pr = torch.softmax(logits.float(), 1).cpu()
        probs.append(pr.numpy())
        ps.append(pr.argmax(1).numpy())
        ys.append(np.asarray(y))
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(probs)


def train(model, train_ds, val_ds, cfg: ClsTrainConfig, out_dir, log=print) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    model = model.to(device)
    tl = DataLoader(train_ds, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True,
                    pin_memory=device == "cuda", persistent_workers=cfg.num_workers > 0)
    vl = DataLoader(val_ds, cfg.batch_size * 2, shuffle=False, num_workers=cfg.num_workers)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=max(cfg.epochs * len(tl), 50), pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device == "cuda")
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    best, best_epoch, bad, history = -1.0, -1, 0, []
    ckpt = out_dir / f"{cfg.name}_best.pt"
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        for x, y in tl:
            x, y = x.to(device, non_blocking=True), torch.as_tensor(y).to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=cfg.amp and device == "cuda"):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        yv, pv, _ = predict(model, vl, device, cfg.amp)
        rep = classification_report(yv, pv, cfg.class_names)
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_accuracy": rep["accuracy"],
               "val_macro_f1": rep["macro_f1"], "elapsed_s": round(time.time() - t0, 1)}
        history.append(rec)
        improved = rep["macro_f1"] > best
        if improved:
            best, best_epoch, bad = rep["macro_f1"], epoch, 0
            torch.save({"state_dict": model.state_dict(), "config": asdict(cfg)}, ckpt)
        else:
            bad += 1
        log(f"[{cfg.name}] epoch {epoch:3d}  loss {rec['train_loss']:.4f}  val acc {rep['accuracy']:.4f}"
            f"  macro-F1 {rep['macro_f1']:.4f}{'  *' if improved else ''}")
        if bad >= cfg.patience:
            log(f"[{cfg.name}] early stop")
            break
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["state_dict"])
    summary = {"name": cfg.name, "best_epoch": best_epoch, "epochs_run": len(history), "best_val_macro_f1": best,
               "train_time_s": round(time.time() - t0, 1), "history": history, "config": asdict(cfg),
               "git_commit": git_commit(), "finished_utc": utc_now()}
    save_json(summary, out_dir / f"{cfg.name}_train_summary.json")
    return summary


def test_report(model, test_ds, cfg: ClsTrainConfig, out_path) -> dict:
    device = get_device()
    model = model.to(device)
    loader = DataLoader(test_ds, cfg.batch_size * 2, shuffle=False, num_workers=cfg.num_workers)
    y, p, _ = predict(model, loader, device, cfg.amp)
    rep = classification_report(y, p, cfg.class_names)
    rep["n_images"] = int(len(y))
    save_json(rep, out_path)
    return rep
