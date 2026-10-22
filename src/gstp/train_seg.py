"""Training / evaluation loop for the binary segmentation models.

Protocol (identical for floods, wildfires and forestry):
  1. Train on the training split with dihedral augmentation.
  2. After each epoch, score the validation split; keep the best checkpoint by
     validation IoU of the positive class; stop early if it stops improving.
  3. Choose the decision threshold on the VALIDATION split.
  4. Score the untouched TEST split(s) once, at that threshold.
The test set is never used for any choice, so the reported test numbers are an
unbiased estimate of performance on similar imagery.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data.raster import IGNORE_INDEX
from .losses import BCEDiceLoss
from .metrics import BinarySegMeter
from .utils import get_device, git_commit, save_json, utc_now

THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)


@dataclass
class SegTrainConfig:
    name: str
    in_channels: int
    bands: list
    epochs: int = 60
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 12
    crop_size: int | None = 256
    encoder: str = "resnet34"
    pretrained: bool = True
    bce_weight: float = 0.5
    pos_weight: float | None = None
    num_workers: int = 2
    amp: bool = True
    seed: int = 42
    extra: dict = field(default_factory=dict)


def _worker_init(worker_id: int):
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(torch.initial_seed() % (2**32))


@torch.no_grad()
def predict_probs(model, loader, device, amp=True):
    """Yield (prob [B,H,W] numpy, mask [B,H,W] numpy) batches."""
    model.eval()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=amp and device == "cuda"):
            logits = model(x)
        yield torch.sigmoid(logits.float()).squeeze(1).cpu().numpy(), y.numpy()


def evaluate(model, dataset, device=None, batch_size=4, thresholds=THRESHOLDS, group_key=None, amp=True, num_workers=2):
    """Score a dataset. Returns (meter over all pixels, {group: meter})."""
    device = device or get_device()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    meter = BinarySegMeter(thresholds)
    groups: dict[str, BinarySegMeter] = {}
    i = 0
    for prob, mask in predict_probs(model, loader, device, amp):
        for b in range(prob.shape[0]):
            valid = mask[b] != IGNORE_INDEX
            tgt = mask[b] == 1
            meter.update(prob[b], tgt, valid)
            if group_key is not None:
                g = group_key(dataset.samples[i])
                groups.setdefault(g, BinarySegMeter(thresholds)).update(prob[b], tgt, valid)
            i += 1
    return meter, groups


def _idx(thresholds, t):
    return int(np.argmin(np.abs(np.asarray(thresholds) - t)))


def train(model, train_ds, val_ds, cfg: SegTrainConfig, out_dir: str | Path, log=print) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    model = model.to(device)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=len(train_ds) > cfg.batch_size,
        pin_memory=device == "cuda",
        worker_init_fn=_worker_init if cfg.num_workers > 0 else None,
        persistent_workers=cfg.num_workers > 0,
    )
    loss_fn = BCEDiceLoss(cfg.bce_weight, cfg.pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=max(cfg.epochs * len(train_loader), 50), pct_start=0.1
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device == "cuda")

    best_iou, best_epoch, bad = -1.0, -1, 0
    history = []
    ckpt_path = out_dir / f"{cfg.name}_best.pt"
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=cfg.amp and device == "cuda"):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())

        meter, _ = evaluate(model, val_ds, device, thresholds=[0.5], amp=cfg.amp, num_workers=cfg.num_workers)
        val = meter.compute()
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_iou": val["iou"], "val_f1": val["f1"],
               "lr": sched.get_last_lr()[0], "elapsed_s": round(time.time() - t0, 1)}
        history.append(rec)
        improved = val["iou"] > best_iou
        if improved:
            best_iou, best_epoch, bad = val["iou"], epoch, 0
            torch.save({"state_dict": model.state_dict(), "config": asdict(cfg)}, ckpt_path)
        else:
            bad += 1
        log(f"[{cfg.name}] epoch {epoch:3d}  loss {rec['train_loss']:.4f}  val IoU {val['iou']:.4f}"
            f"  F1 {val['f1']:.4f}{'  *' if improved else ''}")
        if bad >= cfg.patience:
            log(f"[{cfg.name}] early stop: no val improvement for {cfg.patience} epochs")
            break

    # Reload best weights, choose the threshold on validation.
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["state_dict"])
    val_meter, _ = evaluate(model, val_ds, device, amp=cfg.amp, num_workers=cfg.num_workers)
    best_val = val_meter.best("iou")
    summary = {
        "name": cfg.name,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "train_time_s": round(time.time() - t0, 1),
        "chosen_threshold": best_val["threshold"],
        "val_at_0.5": val_meter.compute(_idx(THRESHOLDS, 0.5)),
        "val_at_chosen": best_val,
        "history": history,
        "config": asdict(cfg),
        "git_commit": git_commit(),
        "finished_utc": utc_now(),
    }
    state["threshold"] = best_val["threshold"]
    torch.save(state, ckpt_path)
    save_json(summary, out_dir / f"{cfg.name}_train_summary.json")
    return summary


def test_report(model, datasets: dict, threshold: float, out_path, group_key=None, amp=True, num_workers=2) -> dict:
    """Score each named test dataset once, at the validation-chosen threshold."""
    device = get_device()
    model = model.to(device)
    report = {"threshold": threshold}
    for name, ds in datasets.items():
        meter, groups = evaluate(model, ds, device, group_key=group_key, amp=amp, num_workers=num_workers)
        i = _idx(meter.thresholds, threshold)
        report[name] = {
            "n_chips": len(ds),
            "at_chosen_threshold": meter.compute(i),
            "at_0.5": meter.compute(_idx(meter.thresholds, 0.5)),
            "per_group": {g: m.compute(i) for g, m in sorted(groups.items())},
        }
    save_json(report, out_path)
    return report
