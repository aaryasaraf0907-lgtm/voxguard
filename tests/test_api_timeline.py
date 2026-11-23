"""Live-monitoring timeline (timeline=true on /v1/analyze): the detector's own windows, the same
verdict as without a timeline, and when the running average first reached the threshold."""

import numpy as np
import pytest
from sqlalchemy import select

from conftest import make_wav
from voxguard.api import codec_sim
from voxguard.api.db import SessionRow
from voxguard.api.timeline import WindowScore, build_timeline, window_bounds
from voxguard.data import codecs
from voxguard.inference import MIN_TAIL, NUM_SAMPLES, Prediction, StubDetector, to_windows


class ScriptedDetector:
    """Returns the given scores for successive windows (and the first for anything else)."""

    model_version = "scripted-0"
    threshold = 0.5

    def __init__(self, scores: list[float]) -> None:
        self.scores, self.calls = list(scores), 0

    def predict(self, waveform: np.ndarray, sample_rate: int) -> Prediction:
        score = self.scores[min(self.calls, len(self.scores) - 1)]
        self.calls += 1
        return Prediction(score, "fake" if score >= 0.5 else "real", 0.5, 2.0, self.model_version)


@pytest.mark.parametrize(
    "n",
    [
        1,
        NUM_SAMPLES - 1,
        NUM_SAMPLES,
        NUM_SAMPLES + 1,
        NUM_SAMPLES + MIN_TAIL - 1,
        2 * NUM_SAMPLES + MIN_TAIL - 1,
        2 * NUM_SAMPLES + MIN_TAIL,
        5 * NUM_SAMPLES + 123,
    ],
)
def test_window_bounds_match_the_detectors_windows(n: int) -> None:
    x = np.random.default_rng(n).standard_normal(n).astype(np.float32)
    windows, bounds = to_windows(x, 16_000), window_bounds(n)
    assert len(windows) == len(bounds)
    for w, (start, end) in zip(windows, bounds, strict=True):
        if n >= NUM_SAMPLES:  # full or overlapping last window: exactly that slice
            assert end - start == NUM_SAMPLES and np.array_equal(w, x[start:end])
        else:  # short clip: one window over the whole clip, repeat-padded inside
            assert (start, end) == (0, n) and np.array_equal(w[:n], x)


def test_a_short_tail_is_dropped_and_a_long_one_overlaps() -> None:
    assert window_bounds(2 * NUM_SAMPLES + MIN_TAIL - 1) == [
        (0, NUM_SAMPLES),
        (NUM_SAMPLES, 2 * NUM_SAMPLES),
    ]
    n = 2 * NUM_SAMPLES + MIN_TAIL
    assert window_bounds(n)[-1] == (n - NUM_SAMPLES, n)


def test_running_average_and_first_flag() -> None:
    scores = [WindowScore(0, 4, 0.1), WindowScore(4, 8, 0.9), WindowScore(8, 12, 0.9)]
    points, first = build_timeline(scores, threshold=0.5)
    assert [p.running_score for p in points] == pytest.approx([0.1, 0.5, 0.633333], abs=1e-6)
    assert [p.flagged for p in points] == [False, True, True]  # >=, the prevention rule
    assert first == 8


def test_never_reaching_the_threshold_gives_no_flag_time() -> None:
    points, first = build_timeline([WindowScore(0, 4, 0.1), WindowScore(4, 8, 0.2)], 0.5)
    assert first is None and not any(p.flagged for p in points)


def _post(client, data: bytes, **form: str):
    return client.post("/v1/analyze", files={"file": ("a.wav", data)}, data=form)


@pytest.mark.parametrize(
    ("seconds", "sr"), [(10.5, 16_000), (10.5, 44_100), (2.0, 16_000), (17.0, 8000)]
)
def test_same_verdict_and_score_with_and_without_the_timeline(client_factory, seconds, sr) -> None:
    wav = make_wav(seconds, sr=sr, seed=3)
    with client_factory(StubDetector(threshold=0.5)) as c:
        plain, timed = _post(c, wav).json(), _post(c, wav, timeline="true").json()
    assert plain["timeline"] is None and plain["first_flag_s"] is None  # off by default
    for key in ("fake_score", "label", "flagged", "threshold"):
        assert timed[key] == plain[key]
    points = timed["timeline"]
    assert points[-1]["running_score"] == pytest.approx(timed["fake_score"], abs=1e-12)
    # ends where the detector's last window ends: a leftover under MIN_TAIL is not scored
    assert points[-1]["end_s"] == pytest.approx(
        window_bounds(round(seconds * 16_000))[-1][1] / 16_000, abs=1e-3
    )
    assert points[-1]["flagged"] == timed["flagged"]


def test_flag_time_and_stored_session(client_factory) -> None:
    det = ScriptedDetector([0.1, 0.9, 0.9])
    with client_factory(det) as c:
        body = _post(c, make_wav(12.5), timeline="true", caller_id="x").json()
        with c.app.state.session_factory() as db:
            (row,) = db.scalars(select(SessionRow)).all()
    assert det.calls == 3 and body["first_flag_s"] == pytest.approx(2 * NUM_SAMPLES / 16_000)
    assert body["fake_score"] == pytest.approx((0.1 + 0.9 + 0.9) / 3)
    assert body["latency_ms"] == pytest.approx(6.0)  # detector time of all windows
    assert row.fake_score == body["fake_score"] and row.flagged is True  # stored as usual


def _fake_resample_to_16k(x: np.ndarray, sample_rate: int) -> np.ndarray:
    """Right output length, no ffmpeg."""
    return np.resize(x, round(len(x) * 16_000 / sample_rate)).astype(np.float32)


def test_timeline_follows_the_phone_line_simulation(client_factory, monkeypatch) -> None:
    seen = []

    def fake_apply(x: np.ndarray, codec: str, bitrate=None) -> np.ndarray:
        seen.append(len(x))
        return (x * 0.5).astype(np.float32)

    monkeypatch.setattr(codec_sim, "ffmpeg_path", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(codec_sim, "available_codecs", lambda: {c: True for c in codecs.CODECS})
    monkeypatch.setattr(codec_sim, "apply_codec", fake_apply)
    # No real ffmpeg (its path differs per OS: apt, Homebrew); the resampler is tested apart.
    monkeypatch.setattr(codec_sim, "resample_to_16k", _fake_resample_to_16k)
    with client_factory(StubDetector(threshold=0.5)) as c:
        body = _post(c, make_wav(9.5, sr=44_100), codec="g711_mulaw", timeline="true").json()
    assert body["codec_applied"] == "g711_mulaw" and seen == [152_000]  # 9.5 s at 16 kHz
    assert len(body["timeline"]) == 3 and body["timeline"][-1]["end_s"] == pytest.approx(9.5)
