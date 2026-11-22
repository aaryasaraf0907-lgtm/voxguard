"""Shared pieces of the head-only experiment: the back-end, cached features, the metrics."""

from __future__ import annotations

import copy
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from voxguard.metrics import compute_eer  # noqa: E402

X = Path.home() / "voxguard-private/headonly"
FEATS = X / "features"
P3 = Path.home() / (
    ".cache/huggingface/hub/models--voxguard-team--voxguard-ssl-aasist-codec-aug/snapshots/"
    "f376f4fde40fa8ac5feff4996e413bf8ff944b13/best.pt"
)
PHONE = ("g711_mulaw", "g711_alaw", "opus")  # dev/eval phone conditions (GSM stays unseen)


class Head(nn.Module):
    """SSL-AASIST after the front-end: features (B, 201, 1024) -> logits (B, 2)."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.proj = copy.deepcopy(model.proj)
        self.first_bn = copy.deepcopy(model.first_bn)
        self.encoder = copy.deepcopy(model.encoder)
        self.backend = copy.deepcopy(model.backend)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        z = self.proj(f.float())
        z = F.max_pool2d(z.transpose(1, 2).unsqueeze(1), (3, 3))
        z = F.selu(self.first_bn(z))
        return self.backend(self.encoder(z))


def load_p3() -> tuple[nn.Module, dict]:
    from voxguard.models import load_from_checkpoint

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return load_from_checkpoint(P3, "cpu")


def load_set(name: str) -> tuple[np.ndarray, list[dict]]:
    d = FEATS / name
    return np.load(d / "feats.npy", mmap_mode="r"), list(csv.DictReader((d / "meta.csv").open()))


@torch.no_grad()
def logits(head: nn.Module, feats: np.ndarray, device: str = "cuda", bs: int = 256) -> np.ndarray:
    head.eval().to(device)
    out = []
    for i in range(0, len(feats), bs):
        out.append(head(torch.from_numpy(np.asarray(feats[i : i + bs])).to(device)).float().cpu())
    return torch.cat(out).numpy()


def fake_prob(lg: np.ndarray) -> np.ndarray:
    return torch.softmax(torch.from_numpy(lg), -1)[:, 1].numpy()


def miss_at_far(scores: np.ndarray, labels: np.ndarray, far: float = 0.01) -> tuple[float, float]:
    """Missed fakes when the threshold flags ``far`` of the genuine clips; and that threshold."""
    thr = float(np.quantile(scores[labels == 0], 1.0 - far))
    return float(np.mean(scores[labels == 1] < thr)), thr


def summary(scores: np.ndarray, labels: np.ndarray, threshold: float | None = None) -> dict:
    eer, eer_thr = compute_eer(scores, labels)
    miss1, thr1 = miss_at_far(scores, labels)
    out = {"n": int(len(labels)), "eer": eer, "eer_thr": eer_thr, "miss_at_1far": miss1}
    if threshold is not None:
        out["far_at_thr"] = float(np.mean(scores[labels == 0] >= threshold))
        out["miss_at_thr"] = float(np.mean(scores[labels == 1] < threshold))
    return out


def phone_mask(meta: list[dict]) -> np.ndarray:
    return np.array([m["condition"] in PHONE for m in meta])


class TopHead(nn.Module):
    """Round 2d: XLS-R's last ``top`` layers + final layer norm + head, on cached layer inputs."""

    def __init__(self, model: nn.Module, top: int = 2) -> None:
        super().__init__()
        enc = model.frontend.model.encoder
        self.layers = nn.ModuleList(copy.deepcopy(enc.layers[-top:]))
        self.layer_norm = copy.deepcopy(enc.layer_norm)
        self.head = Head(model)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = h.float()
        for layer in self.layers:
            out = layer(h)
            h = out[0] if isinstance(out, tuple) else out
        return self.head(self.layer_norm(h))
