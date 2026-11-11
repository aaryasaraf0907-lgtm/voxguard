"""Config-driven, resumable, device-agnostic training.

    python -m voxguard.train --config configs/rawnet2.yaml
    python -m voxguard.train --config configs/rawnet2_smoke.yaml --max-samples 100 --no-wandb

A checkpoint (``last.pt``) is written atomically after every epoch, and ``best.pt`` (weights and
config only, no optimizer state) whenever dev EER improves. ``--resume`` continues from
``last.pt`` (free GPU sessions die).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, RandomSampler

from voxguard.data.datasets import NUM_SAMPLES, ManifestDataset
from voxguard.metrics import summarize
from voxguard.models import build_model

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    train_manifests: list[str] = field(default_factory=list)
    dev_manifests: list[str] = field(default_factory=list)
    roots: list[str] = field(default_factory=lambda: ["data/raw", "data/processed"])
    num_samples: int = NUM_SAMPLES
    codecs: list[str] | None = None  # filter training rows by codec column (None = all)
    max_samples: int | None = None  # smoke mode: seeded random subset
    dev_max_samples: int | None = None  # fixed seeded dev subset for per-epoch model selection
    # Clips drawn (without replacement, reshuffled each epoch) per training epoch. Lets a larger
    # codec-augmented manifest train for the same number of steps as the clean one.
    train_samples_per_epoch: int | None = None


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-4
    frontend_lr: float | None = None  # separate LR for an SSL front-end (None = use lr)
    weight_decay: float = 1e-4
    # [bona fide, fake]: up-weight the minority bona fide class (labels: 1 = fake)
    class_weights: list[float] = field(default_factory=lambda: [0.9, 0.1])
    seed: int = 1234
    num_workers: int = 0
    amp: bool = True  # mixed precision, CUDA only
    out_dir: str = "runs/rawnet2"
    device: str = "auto"  # auto | cuda | mps | cpu
    multi_gpu: bool = True  # nn.DataParallel over all visible CUDA GPUs when there are several


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "voxguard"
    name: str | None = None


@dataclass
class Config:
    model: dict[str, Any] = field(default_factory=lambda: {"name": "rawnet2"})
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


def _build(cls: type, raw: dict[str, Any]) -> Any:
    names = {f.name for f in fields(cls)}
    unknown = set(raw) - names
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**raw)


def load_config(path: Path) -> Config:
    raw = yaml.safe_load(path.read_text()) or {}
    unknown = set(raw) - {"model", "data", "train", "wandb"}
    if unknown:
        raise ValueError(f"unknown top-level config keys: {sorted(unknown)}")
    return Config(
        model=raw.get("model", {"name": "rawnet2"}),
        data=_build(DataConfig, raw.get("data", {})),
        train=_build(TrainConfig, raw.get("train", {})),
        wandb=_build(WandbConfig, raw.get("wandb", {})),
    )


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic: a killed session never leaves a half-written file


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    scores, labels = [], []
    for x, y in loader:
        logits = model(x.to(device, non_blocking=True))
        scores.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
        labels.append(y.numpy())
    s, y = np.concatenate(scores), np.concatenate(labels)
    try:
        return summarize(s, y)
    except ValueError as exc:  # e.g. a tiny dev subset with a single class
        logger.warning("dev metrics unavailable: %s", exc)
        return {"eer": float("nan"), "n": int(len(y))}


def _make_loaders(cfg: Config, pin_memory: bool = False) -> tuple[DataLoader, DataLoader]:
    d, t = cfg.data, cfg.train
    kwargs = dict(roots=[Path(r) for r in d.roots], num_samples=d.num_samples, seed=t.seed)
    train_ds = ManifestDataset(
        [Path(m) for m in d.train_manifests],
        train=True,
        codecs=d.codecs,
        max_samples=d.max_samples,
        **kwargs,
    )
    dev_max = d.dev_max_samples
    if d.max_samples is not None:
        dev_max = max(d.max_samples // 4, 16) if dev_max is None else min(dev_max, d.max_samples)
    dev_ds = ManifestDataset(
        [Path(m) for m in d.dev_manifests], train=False, max_samples=dev_max, **kwargs
    )
    loader_kw: dict[str, Any] = dict(
        batch_size=t.batch_size, num_workers=t.num_workers, pin_memory=pin_memory
    )
    # Train workers are re-created each epoch so they see set_epoch() (seeded crops); the dev
    # set has no per-epoch state, so its workers can persist.
    n_epoch = d.train_samples_per_epoch
    if n_epoch is not None and n_epoch < len(train_ds):
        sampler = RandomSampler(train_ds, num_samples=n_epoch)
        train_dl = DataLoader(train_ds, sampler=sampler, drop_last=False, **loader_kw)
    else:
        train_dl = DataLoader(train_ds, shuffle=True, drop_last=False, **loader_kw)
    dev_dl = DataLoader(dev_ds, shuffle=False, persistent_workers=t.num_workers > 0, **loader_kw)
    return train_dl, dev_dl


def seed_epoch(loader: DataLoader, seed: int, epoch: int) -> None:
    """Make the shuffle order (and worker seeds) depend only on ``seed`` and ``epoch``.

    The shuffler keeps the generator it was built with, so both it and the loader's are set;
    a resumed run then sees the same batches as an uninterrupted one.
    """
    g = torch.Generator().manual_seed(seed + epoch)
    loader.generator = g
    if hasattr(loader.sampler, "generator"):
        loader.sampler.generator = g


def _param_groups(model: nn.Module, t: TrainConfig) -> list[dict[str, Any]]:
    """One group, or front-end/back-end groups when ``frontend_lr`` is set and supported."""
    if t.frontend_lr is not None and hasattr(model, "frontend_parameters"):
        groups = [
            {"params": model.frontend_parameters(), "lr": t.frontend_lr},
            {"params": model.backend_parameters(), "lr": t.lr},
        ]
        return [g for g in groups if g["params"]]
    return [{"params": [p for p in model.parameters() if p.requires_grad]}]


def train(cfg: Config, resume: bool = False, use_wandb: bool = False) -> dict[str, Any]:
    t = cfg.train
    out = Path(t.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(t.seed)
    device = pick_device(t.device)
    amp = t.amp and device.type == "cuda"
    logger.info("device=%s amp=%s seed=%d out=%s", device, amp, t.seed, out)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # fixed input length: let cuDNN pick fast kernels
    train_dl, dev_dl = _make_loaders(cfg, pin_memory=device.type == "cuda")
    logger.info("train=%d dev=%d clips", len(train_dl.dataset), len(dev_dl.dataset))
    model_params = {k: v for k, v in cfg.model.items() if k != "name"}
    model = build_model(cfg.model["name"], **model_params).to(device)
    n_all = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "model=%s params=%.1fM trainable=%.1fM", cfg.model["name"], n_all / 1e6, n_train / 1e6
    )
    opt = torch.optim.Adam(_param_groups(model, t), lr=t.lr, weight_decay=t.weight_decay)
    core = model  # unwrapped module: checkpoints keep plain (non-"module.") keys
    n_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    if t.multi_gpu and n_gpus > 1:
        model = nn.DataParallel(core)
        logger.info("DataParallel over %d GPUs (batch %d split across them)", n_gpus, t.batch_size)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(t.class_weights, device=device))

    start_epoch, best_eer = 0, float("inf")
    last = out / "last.pt"
    if resume and last.exists():
        ckpt = torch.load(last, map_location=device, weights_only=False)
        core.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch, best_eer = ckpt["epoch"] + 1, ckpt["best_eer"]
        logger.info("resumed from %s at epoch %d", last, start_epoch)
    elif resume:
        logger.info("--resume given but %s not found; starting fresh", last)

    run = None
    if use_wandb:
        import wandb

        run = wandb.init(project=cfg.wandb.project, name=cfg.wandb.name, config=asdict(cfg))

    history = []
    for epoch in range(start_epoch, t.epochs):
        t0 = time.perf_counter()
        model.train()
        train_dl.dataset.set_epoch(epoch)
        seed_epoch(train_dl, t.seed, epoch)
        total, n = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=amp):
                loss = loss_fn(model(x).float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(y)
            n += len(y)
        dev = evaluate(model, dev_dl, device)
        rec = {
            "epoch": epoch,
            "train_loss": total / max(n, 1),
            "dev_eer": dev["eer"],
            "dev_acc_at_0.5": dev.get("accuracy_at_0.5"),
            "seconds": time.perf_counter() - t0,
        }
        history.append(rec)
        logger.info("epoch %d: %s", epoch, json.dumps(rec))
        with (out / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")
        if run is not None:
            run.log(rec)
        improved = dev["eer"] < best_eer
        best_eer = dev["eer"] if improved else best_eer
        state = {
            "model": core.state_dict(),
            "opt": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_eer": best_eer,
            "config": asdict(cfg),
        }
        save_checkpoint(last, state)
        if improved:  # weights only: best.pt is for inference, last.pt carries optimizer state
            best = {k: v for k, v in state.items() if k not in ("opt", "scaler")}
            save_checkpoint(out / "best.pt", best)
    if run is not None:
        run.finish()
    return {"history": history, "best_eer": best_eer, "out_dir": str(out)}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--max-samples", type=int, default=None, help="smoke mode: seeded subset")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", default=None, help="auto | cuda | mps | cpu")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    if args.max_samples is not None:
        cfg.data.max_samples = args.max_samples
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.device is not None:
        cfg.train.device = args.device
    if args.out_dir is not None:
        cfg.train.out_dir = args.out_dir
    result = train(cfg, resume=args.resume, use_wandb=cfg.wandb.enabled and not args.no_wandb)
    logger.info("done: best dev EER %.4f, checkpoints in %s", result["best_eer"], result["out_dir"])


if __name__ == "__main__":
    main()
