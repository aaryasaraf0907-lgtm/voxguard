import numpy as np
import pytest

from voxguard.inference import (
    NUM_SAMPLES,
    Prediction,
    RealDetector,
    StubDetector,
    get_detector,
    to_windows,
)


def _wave(n: int = NUM_SAMPLES, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n).astype(np.float32) * 0.1


def test_same_input_same_score() -> None:
    det = StubDetector(threshold=0.5)
    w = _wave()
    assert det.predict(w, 16_000).fake_score == det.predict(w.copy(), 16_000).fake_score


def test_different_input_different_score() -> None:
    det = StubDetector(threshold=0.5)
    assert (
        det.predict(_wave(seed=1), 16_000).fake_score
        != det.predict(_wave(seed=2), 16_000).fake_score
    )


def test_prediction_fields_and_ranges() -> None:
    p = StubDetector(threshold=0.5).predict(_wave(), 16_000)
    assert isinstance(p, Prediction)
    assert 0.0 <= p.fake_score <= 1.0
    assert p.latency_ms >= 0.0
    assert p.label in ("real", "fake")
    assert p.model_version == "stub-0"


@pytest.mark.parametrize("threshold", [0.0, 0.3, 0.5, 0.9, 1.0])
def test_label_matches_threshold(threshold: float) -> None:
    p = StubDetector(threshold=threshold).predict(_wave(), 16_000)
    assert p.threshold == threshold
    assert (p.label == "fake") == (p.fake_score >= threshold)


def test_predict_does_not_depend_on_threshold_for_score() -> None:
    w = _wave()
    a = StubDetector(threshold=0.1).predict(w, 16_000).fake_score
    b = StubDetector(threshold=0.9).predict(w, 16_000).fake_score
    assert a == b


def test_windowing_pad_crop_and_split() -> None:
    short = _wave(1000)
    (w,) = to_windows(short, 16_000)
    assert len(w) == NUM_SAMPLES and np.array_equal(w[1000:2000], short)  # repeat-padded
    # a tiny tail (< 1/4 window) is dropped, a long tail becomes its own padded window
    assert len(to_windows(_wave(NUM_SAMPLES * 2 + 5), 16_000)) == 2
    assert len(to_windows(_wave(NUM_SAMPLES * 2 + NUM_SAMPLES // 2), 16_000)) == 3
    assert len(to_windows(np.zeros(0, dtype=np.float32), 16_000)) == 1


def test_resample_and_stereo() -> None:
    wins = to_windows(np.zeros((8000, 2), dtype=np.float32), 8_000)
    assert len(wins) == 1 and wins[0].ndim == 1
    p = StubDetector(threshold=0.5).predict(np.zeros((8000, 2), dtype=np.float32), 8_000)
    assert 0.0 <= p.fake_score <= 1.0


def test_threshold_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOXGUARD_THRESHOLD", "0.25")
    assert StubDetector().threshold == 0.25
    for bad in ("abc", "1.5", "-0.1"):
        monkeypatch.setenv("VOXGUARD_THRESHOLD", bad)
        with pytest.raises(ValueError, match="VOXGUARD_THRESHOLD"):
            StubDetector()


def test_get_detector_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOXGUARD_DETECTOR", raising=False)
    assert isinstance(get_detector(), StubDetector)
    monkeypatch.setenv("VOXGUARD_DETECTOR", "bogus")
    with pytest.raises(ValueError):
        get_detector()


def test_real_detector_needs_a_model(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    monkeypatch.delenv("VOXGUARD_MODEL_REPO", raising=False)
    with pytest.raises(ValueError, match="VOXGUARD_MODEL_REPO"):
        RealDetector()
