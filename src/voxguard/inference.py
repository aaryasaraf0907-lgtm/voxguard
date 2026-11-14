"""Detector contract (CLAUDE.md section 7): Prediction, Detector, StubDetector, RealDetector."""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

TARGET_SR = 16_000
NUM_SAMPLES = 64_600  # ~4 s at 16 kHz
DEFAULT_THRESHOLD = 0.5


@dataclass(frozen=True)
class Prediction:
    fake_score: float  # 0.0-1.0, HIGHER = more likely AI-generated
    label: Literal["real", "fake"]
    threshold: float
    latency_ms: float
    model_version: str


class Detector(Protocol):
    model_version: str
    threshold: float

    def predict(self, waveform: np.ndarray, sample_rate: int) -> Prediction: ...


def threshold_from_env(default: float = DEFAULT_THRESHOLD) -> float:
    raw = os.environ.get("VOXGUARD_THRESHOLD", "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"VOXGUARD_THRESHOLD must be a number in [0, 1], got {raw!r}") from None
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"VOXGUARD_THRESHOLD must be in [0, 1], got {raw!r}")
    return value


MIN_TAIL = NUM_SAMPLES // 4  # a trailing partial window shorter than ~1 s is dropped

Resampler = Callable[[np.ndarray, int, int], np.ndarray]


def to_mono(waveform: np.ndarray) -> np.ndarray:
    x = np.asarray(waveform, dtype=np.float32)
    if x.ndim == 2:  # (samples, channels) or (channels, samples) -> mono
        x = x.mean(axis=1 if x.shape[0] >= x.shape[1] else 0)
    if x.ndim != 1:
        raise ValueError(f"waveform must be 1-D or 2-D, got shape {np.shape(waveform)}")
    return x


def linear_resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Dependency-free linear-interpolation resampler (good enough for the stub)."""
    n_out = int(round(len(x) * sr_out / sr_in))
    grid = np.linspace(0.0, len(x) - 1, num=max(n_out, 1))
    return np.interp(grid, np.arange(len(x)), x).astype(np.float32)


def to_windows(
    waveform: np.ndarray, sample_rate: int, resampler: Resampler = linear_resample
) -> list[np.ndarray]:
    """Resample to 16 kHz mono float32 and split into 64,600-sample windows.

    Windows are non-overlapping. A short window is repeat-padded (as in training, see
    ``voxguard.data.datasets.fix_length``); a trailing partial window shorter than
    ``MIN_TAIL`` is dropped when there is at least one full window.
    """
    x = to_mono(waveform)
    if sample_rate != TARGET_SR and len(x):
        x = resampler(x, sample_rate, TARGET_SR)
    if len(x) == 0:
        return [np.zeros(NUM_SAMPLES, dtype=np.float32)]
    windows = []
    for start in range(0, len(x), NUM_SAMPLES):
        w = x[start : start + NUM_SAMPLES]
        if len(w) < NUM_SAMPLES:
            if windows and len(w) < MIN_TAIL:
                break
            w = np.tile(w, int(np.ceil(NUM_SAMPLES / len(w))))[:NUM_SAMPLES]
        windows.append(np.ascontiguousarray(w, dtype=np.float32))
    return windows


class StubDetector:
    """Deterministic fake_score from a hash of the waveform. For CI and API/UI development."""

    def __init__(self, threshold: float | None = None, model_version: str = "stub-0") -> None:
        self.model_version = model_version
        self.threshold = threshold_from_env() if threshold is None else threshold

    def predict(self, waveform: np.ndarray, sample_rate: int) -> Prediction:
        t0 = time.perf_counter()
        scores = []
        for w in to_windows(waveform, sample_rate):
            digest = hashlib.sha256(w.tobytes()).digest()
            scores.append(int.from_bytes(digest[:8], "big") / 2**64)
        score = float(np.mean(scores))
        return Prediction(
            fake_score=score,
            label="fake" if score >= self.threshold else "real",
            threshold=self.threshold,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            model_version=self.model_version,
        )


def torch_resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Band-limited resampling via torchaudio (used by the real detector)."""
    import torch
    import torchaudio.functional as AF

    return AF.resample(torch.from_numpy(x), sr_in, sr_out).numpy().astype(np.float32)


def resolve_checkpoint(model_repo: str, filename: str = "best.pt") -> Path:
    """A local file, a local directory holding ``filename``, or a Hugging Face repo id.

    Hub downloads use ``huggingface_hub`` (cached; a private repo needs ``HF_TOKEN`` in the
    environment, which the library reads itself).
    """
    local = Path(model_repo).expanduser()
    if local.is_file():
        return local
    if local.is_dir():
        if not (local / filename).is_file():
            raise FileNotFoundError(f"{local} has no {filename}")
        return local / filename
    if model_repo.count("/") != 1 or model_repo.startswith((".", "/", "~")):
        raise FileNotFoundError(f"{model_repo!r} is neither a local path nor an HF repo id")
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=model_repo, filename=filename))


def model_version(info: dict, source: str) -> str:
    """``<name>[/<run>]:<source>:epoch<N>``.

    E.g. ``ssl_aasist/ssl-aasist-codec-aug-v2:best.pt:epoch7``.

    The run name keeps a clean and a codec-augmented checkpoint of the same architecture and
    epoch apart; ``<source>`` is the load location (file or directory name, or HF repo id).
    """
    name = f"{info['name']}/{info['run']}" if info.get("run") else info["name"]
    return f"{name}:{Path(source).name or source}:epoch{info['epoch']}"


class RealDetector:
    """Trained RawNet2 / SSL-AASIST checkpoint behind the Detector contract.

    ``model_repo``: local ``.pt`` file, a directory containing ``filename``, or a Hugging Face
    repo id (default: ``VOXGUARD_MODEL_REPO``). Threshold precedence: argument, then
    ``VOXGUARD_THRESHOLD``, then 0.5. Long clips are scored per 64,600-sample window and the
    window scores averaged.
    """

    def __init__(
        self,
        model_repo: str | None = None,
        threshold: float | None = None,
        device: str = "cpu",
        filename: str = "best.pt",
        batch_size: int = 8,
    ) -> None:
        import torch

        from voxguard.models import load_from_checkpoint

        source = model_repo or os.environ.get("VOXGUARD_MODEL_REPO", "").strip()
        if not source:
            raise ValueError("RealDetector needs a model path or HF repo id (VOXGUARD_MODEL_REPO)")
        if device == "auto":
            from voxguard.train import pick_device

            device = str(pick_device("auto"))
        self.device = device
        self.batch_size = batch_size
        ckpt_path = resolve_checkpoint(source, filename)
        self.model, info = load_from_checkpoint(ckpt_path, device)
        self.model_version = model_version(info, source)
        self.threshold = threshold_from_env() if threshold is None else threshold
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}")
        self._torch = torch

    def predict(self, waveform: np.ndarray, sample_rate: int) -> Prediction:
        torch = self._torch
        t0 = time.perf_counter()
        windows = np.stack(to_windows(waveform, sample_rate, resampler=torch_resample))
        scores = []
        with torch.inference_mode():
            for i in range(0, len(windows), self.batch_size):
                x = torch.from_numpy(windows[i : i + self.batch_size]).to(self.device)
                logits = self.model(x).float()
                scores.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
        score = float(np.clip(np.mean(np.concatenate(scores)), 0.0, 1.0))
        return Prediction(
            fake_score=score,
            label="fake" if score >= self.threshold else "real",
            threshold=self.threshold,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            model_version=self.model_version,
        )


def get_detector() -> Detector:
    """Select the detector from VOXGUARD_DETECTOR=stub|real (default: stub)."""
    kind = os.environ.get("VOXGUARD_DETECTOR", "stub").lower()
    if kind == "stub":
        return StubDetector()
    if kind == "real":
        return RealDetector(os.environ.get("VOXGUARD_MODEL_REPO"))
    raise ValueError(f"VOXGUARD_DETECTOR must be 'stub' or 'real', got {kind!r}")
