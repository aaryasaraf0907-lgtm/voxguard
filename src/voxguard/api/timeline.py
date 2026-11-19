"""Live-monitoring timeline: what a call monitor would have shown, window by window.

The detector scores a clip in 64,600-sample (about 4 s) windows and averages them
(``inference.to_windows``, CLAUDE.md section 7). Here every window is scored on its own with the
same public ``Detector.predict``, so the response can show each window's score and the running
average after it. The clip's score is the mean of the window scores, which is how the detector
averages them anyway, so the verdict is the same as without a timeline.

``first_flag_s`` is the amount of audio after which the running average first reached the
threshold: the moment a live monitor using the same rule would have flagged the call. It is a
property of this recording at this threshold, not a measured early-detection rate. The recording
is analysed after upload; it is not streamed from a phone line.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from voxguard.api.prevention import is_flagged
from voxguard.inference import (
    MIN_TAIL,
    NUM_SAMPLES,
    TARGET_SR,
    Detector,
    Prediction,
    RealDetector,
    Resampler,
    linear_resample,
    to_mono,
    to_windows,
    torch_resample,
)
from voxguard.schemas import TimelinePoint


@dataclass(frozen=True)
class WindowScore:
    start_s: float
    end_s: float
    fake_score: float


def window_bounds(n_samples: int) -> list[tuple[int, int]]:
    """(start, end) in samples of each window ``to_windows`` makes from a 16 kHz signal of
    ``n_samples``: full windows, then an overlapping last one for a tail of at least MIN_TAIL. A
    clip shorter than one window is one window covering the whole clip (repeat-padded inside)."""
    if n_samples < NUM_SAMPLES:
        return [(0, n_samples)]
    n_full, tail = divmod(n_samples, NUM_SAMPLES)
    bounds = [(i * NUM_SAMPLES, (i + 1) * NUM_SAMPLES) for i in range(n_full)]
    if tail >= MIN_TAIL:
        bounds.append((n_samples - NUM_SAMPLES, n_samples))
    return bounds


def _resampler_for(detector: Detector) -> Resampler:
    """The resampler the detector itself would use, so the windows are the ones it would score."""
    return torch_resample if isinstance(detector, RealDetector) else linear_resample


def score_windows(
    detector: Detector, waveform: np.ndarray, sample_rate: int
) -> tuple[Prediction, list[WindowScore]]:
    """Score each window separately. Returns the clip's prediction (mean of the window scores,
    detector latency summed) and the per-window scores in time order."""
    x = to_mono(waveform)
    if sample_rate != TARGET_SR:
        x = _resampler_for(detector)(x, sample_rate, TARGET_SR)
    windows = to_windows(x, TARGET_SR)  # already 16 kHz: windowing only
    bounds = window_bounds(len(x))
    if len(windows) != len(bounds):  # to_windows changed: never show misaligned times
        raise RuntimeError("timeline windows do not match the detector's windowing")
    preds = [detector.predict(w, TARGET_SR) for w in windows]
    scores = [WindowScore(s / TARGET_SR, e / TARGET_SR, p.fake_score)
              for (s, e), p in zip(bounds, preds, strict=True)]  # fmt: skip
    # summed in time order, as build_timeline does, so its last running average equals this
    total = 0.0
    for p in preds:
        total += p.fake_score
    score = min(1.0, max(0.0, total / len(preds)))
    clip = Prediction(
        fake_score=score,
        label="fake" if score >= detector.threshold else "real",
        threshold=detector.threshold,
        latency_ms=sum(p.latency_ms for p in preds),
        model_version=preds[0].model_version,
    )
    return clip, scores


def build_timeline(
    scores: list[WindowScore], threshold: float
) -> tuple[list[TimelinePoint], float | None]:
    """Points with the running average, and when it first reached ``threshold`` (or None).
    Uses the prevention rule (``prevention.is_flagged``)."""
    points, total, first = [], 0.0, None
    for i, w in enumerate(scores, start=1):
        total += w.fake_score
        running = total / i
        flagged = is_flagged(running, threshold)
        if flagged and first is None:
            first = w.end_s
        points.append(TimelinePoint(start_s=round(w.start_s, 3), end_s=round(w.end_s, 3),
                                    fake_score=w.fake_score, running_score=running,
                                    flagged=flagged))  # fmt: skip
    return points, first
