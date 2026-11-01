"""Training loop for the learned deblurring models (autoencoder and NAFNet).

Iteration-based (the training set is an endless stream of random blurs), L1
loss against the sharp image, AdamW with a cosine learning-rate schedule.
Validation PSNR on fixed blurred validation chips decides the best checkpoint;
test images are never touched here.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .deblur_classic import score
from .deblur_models import build, count_parameters, tiled
from .utils import get_device, git_commit, save_json, utc_now


@dataclass
class DeblurTrainConfig:
    name: str                 # "nafnet" or "autoencoder"
    iterations: int = 30000
    batch_size: int = 16
    crop: int = 128
    lr: float = 1e-3
    min_lr: float = 1e-6
    weight_decay: float = 0.0
    val_every: int = 2000
    num_workers: int = 2
    amp: bool = True
    seed: int = 42
    extra: dict = field(default_factory=dict)


def predict(model, blurred: np.ndarray, device) -> np.ndarray:
    x = torch.as_tensor(blurred, device=device)
    with torch.autocast("cuda", enabled=(device == "cuda")):
        return tiled(model, x).float().cpu().numpy()


def validate(model, val_items, device) -> float:
    model.eval()
    vals = [score(it["sharp"], predict(model, it["blurred"], device), it["valid"], ssim=False)["psnr"]
            for it in val_items]
    model.train()
    return float(np.mean(vals))


def train(train_ds, val_items, cfg: DeblurTrainConfig, out_dir, log=print) -> tuple[torch.nn.Module, dict]:
    device = get_device()
    torch.manual_seed(cfg.seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = build(cfg.name).to(device)
    n_params = count_parameters(model)
    log(f"[{cfg.name}] {n_params / 1e6:.2f} M parameters, device {device}")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.9))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.iterations, eta_min=cfg.min_lr)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device == "cuda")
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                        drop_last=True, persistent_workers=cfg.num_workers > 0)
    ckpt = out_dir / f"deblur_{cfg.name}_best.pt"
    history, best, it, t0, run_loss = [], -1.0, 0, time.time(), []
    base_psnr = float(np.mean([score(v["sharp"], v["blurred"], v["valid"], ssim=False)["psnr"] for v in val_items]))
    log(f"[{cfg.name}] validation PSNR of the blurred input (no fix): {base_psnr:.2f} dB")
    model.train()
    while it < cfg.iterations:
        train_ds.epoch += 1
        for yb, xs in loader:
            yb, xs = yb.to(device, non_blocking=True), xs.to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=scaler.is_enabled()):
                loss = torch.nn.functional.l1_loss(model(yb).float(), xs)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss.append(float(loss.detach()))
            it += 1
            if it % cfg.val_every == 0 or it == cfg.iterations:
                v = validate(model, val_items, device)
                improved = v > best
                if improved:
                    best = v
                    torch.save({"state_dict": model.state_dict(), "config": asdict(cfg), "iteration": it,
                                "val_psnr": v}, ckpt)
                rec = {"iteration": it, "train_l1": float(np.mean(run_loss)), "val_psnr": v,
                       "lr": sched.get_last_lr()[0], "minutes": round((time.time() - t0) / 60, 1)}
                history.append(rec)
                run_loss = []
                log(f"[{cfg.name}] it {it:6d}  L1 {rec['train_l1']:.5f}  val PSNR {v:.2f} dB"
                    f"  ({rec['minutes']} min){'  *' if improved else ''}")
            if it >= cfg.iterations:
                break
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["state_dict"])
    model.eval()
    summary = {"name": cfg.name, "parameters": n_params, "best_iteration": state["iteration"],
               "best_val_psnr": state["val_psnr"], "val_psnr_blurred_input": base_psnr,
               "train_minutes": round((time.time() - t0) / 60, 1),
               "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
               "history": history, "config": asdict(cfg), "git_commit": git_commit(), "finished_utc": utc_now()}
    save_json(summary, out_dir / f"deblur_{cfg.name}_train_summary.json")
    return model, summary


def load(path, device=None):
    device = device or get_device()
    ck = torch.load(path, map_location=device, weights_only=False)
    m = build(ck["config"]["name"]).to(device)
    m.load_state_dict(ck["state_dict"])
    return m.eval()


def benchmark(model, device, size=512, runs=20) -> float:
    """Milliseconds to deblur one size x size band (tiled), after warm-up."""
    x = torch.rand(1, size, size, device=device) * 0.3
    for _ in range(3):
        tiled(model, x)
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.time()
    for _ in range(runs):
        tiled(model, x)
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t) / runs * 1000
