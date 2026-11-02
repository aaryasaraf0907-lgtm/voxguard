"""Metrics (CLAUDE.md section 10). numpy only, so they are tested in CI.

Scores are P(fake): higher = more likely fake. Labels: 1 = fake/spoof, 0 = bona fide.
A clip is predicted fake when ``score >= threshold``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


def _validate(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels).ravel().astype(np.int64)
    if s.shape != y.shape:
        raise ValueError(f"scores {s.shape} and labels {y.shape} differ in length")
    if not np.isfinite(s).all():
        raise ValueError("scores contain NaN or infinite values")
    if not set(np.unique(y)) <= {0, 1}:
        raise ValueError("labels must be 0 (bona fide) or 1 (fake)")
    if y.sum() == 0 or y.sum() == len(y):
        raise ValueError("need at least one fake and one bona fide sample")
    return s, y


def roc_points(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(thresholds, fpr, fnr)`` with one point per distinct score plus the (+inf) start.

    Ties are handled: all samples sharing a score flip together.
    """
    s, y = _validate(scores, labels)
    order = np.argsort(-s, kind="stable")
    s, y = s[order], y[order]
    last = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
    tp = np.cumsum(y)[last]
    fp = np.cumsum(1 - y)[last]
    fpr = np.r_[0.0, fp / (len(y) - y.sum())]
    fnr = np.r_[1.0, 1.0 - tp / y.sum()]
    thresholds = np.r_[np.inf, s[last]]
    return thresholds, fpr, fnr


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Equal error rate and its threshold, linearly interpolated where FPR = FNR."""
    thr, fpr, fnr = roc_points(scores, labels)
    d = fnr - fpr  # starts at 1, ends at -1; FNR falls and FPR rises as the threshold drops
    i = int(np.argmax(d <= 0))  # first point at or past the crossing
    if d[i] == 0 or i == 0:
        return float(fpr[i]), float(thr[i] if np.isfinite(thr[i]) else thr[i + 1])
    w = d[i - 1] / (d[i - 1] - d[i])
    eer = fpr[i - 1] + w * (fpr[i] - fpr[i - 1])
    t_prev = thr[i - 1] if np.isfinite(thr[i - 1]) else thr[i]
    return float(eer), float(t_prev + w * (thr[i] - t_prev))


def confusion_matrix(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, int]:
    s, y = _validate(scores, labels)
    pred = s >= threshold
    return {
        "tp": int(np.sum(pred & (y == 1))),
        "fp": int(np.sum(pred & (y == 0))),
        "tn": int(np.sum(~pred & (y == 0))),
        "fn": int(np.sum(~pred & (y == 1))),
    }


def accuracy_at(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    c = confusion_matrix(scores, labels, threshold)
    return (c["tp"] + c["tn"]) / sum(c.values())


def summarize(scores: np.ndarray, labels: np.ndarray) -> dict[str, float | dict[str, int]]:
    """EER, accuracy at the EER threshold and at 0.5, confusion matrix at the EER threshold."""
    eer, thr = compute_eer(scores, labels)
    return {
        "eer": eer,
        "eer_threshold": thr,
        "accuracy_at_eer": accuracy_at(scores, labels, thr),
        "accuracy_at_0.5": accuracy_at(scores, labels, 0.5),
        "confusion_at_eer": confusion_matrix(scores, labels, thr),
        "n": int(np.size(labels)),
    }


@dataclass(frozen=True)
class LatencyStats:
    n: int
    mean_ms: float
    p50_ms: float
    p95_ms: float


def latency_stats(latencies_ms: list[float] | np.ndarray) -> LatencyStats:
    x = np.asarray(latencies_ms, dtype=np.float64)
    if x.size == 0:
        raise ValueError("no latencies")
    return LatencyStats(
        n=int(x.size),
        mean_ms=float(x.mean()),
        p50_ms=float(np.percentile(x, 50)),
        p95_ms=float(np.percentile(x, 95)),
    )


def measure_latency(fn: Callable[[], object], repeats: int = 20, warmup: int = 3) -> LatencyStats:
    """Wall-clock latency of ``fn()`` after ``warmup`` untimed calls."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return latency_stats(times)
