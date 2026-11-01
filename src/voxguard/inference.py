"""Detector contract (CLAUDE.md section 7): Prediction, Detector, StubDetector, RealDetector."""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
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
    raw = os.environ.get("VOXGUARD_THRESHOLD")
    return float(raw) if raw else default


def to_windows(waveform: np.ndarray, sample_rate: int) -> list[np.ndarray]:
    """Resample to 16 kHz mono float32 and split into 64,600-sample windows.

    Short clips are zero-padded (repeat-padding is left to the real model code); clips longer
    than one window are split into non-overlapping windows, the last one padded.
    """
    x = np.asarray(waveform, dtype=np.float32)
    if x.ndim == 2:  # (samples, channels) or (channels, samples) -> mono
        x = x.mean(axis=1 if x.shape[0] >= x.shape[1] else 0)
    if x.ndim != 1:
        raise ValueError(f"waveform must be 1-D or 2-D, got shape {waveform.shape}")
    if sample_rate != TARGET_SR:
        n_out = int(round(len(x) * TARGET_SR / sample_rate))
        x = np.interp(np.linspace(0.0, len(x) - 1, num=max(n_out, 1)), np.arange(len(x)), x).astype(
            np.float32
        )
    if len(x) == 0:
        return [np.zeros(NUM_SAMPLES, dtype=np.float32)]
    windows = []
    for start in range(0, len(x), NUM_SAMPLES):
        w = x[start : start + NUM_SAMPLES]
        if len(w) < NUM_SAMPLES:
            w = np.pad(w, (0, NUM_SAMPLES - len(w)))
        windows.append(w)
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


class RealDetector:
    """Loads weights from a local path or HF Hub repo id. Implemented in a later PR."""

    def __init__(self, model_repo: str | None = None, threshold: float | None = None) -> None:
        raise NotImplementedError(
            "RealDetector is not implemented yet (planned for workstream 3g). "
            "Use VOXGUARD_DETECTOR=stub until a checkpoint exists."
        )


def get_detector() -> Detector:
    """Select the detector from VOXGUARD_DETECTOR=stub|real (default: stub)."""
    kind = os.environ.get("VOXGUARD_DETECTOR", "stub").lower()
    if kind == "stub":
        return StubDetector()
    if kind == "real":
        return RealDetector(os.environ.get("VOXGUARD_MODEL_REPO"))
    raise ValueError(f"VOXGUARD_DETECTOR must be 'stub' or 'real', got {kind!r}")
