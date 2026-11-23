import numpy as np
import pytest

from voxguard.inference import (
    MAX_RESAMPLE_ELEMENTS,
    MIN_TAIL,
    NUM_SAMPLES,
    Prediction,
    RealDetector,
    StubDetector,
    get_detector,
    resample_elements,
    to_windows,
    torch_resample,
    window_starts,
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
    # a tiny tail (< 1/4 window) is dropped
    assert len(to_windows(_wave(NUM_SAMPLES * 2 + 5), 16_000)) == 2
    # a longer tail is scored as the clip's last 4 s (overlapping), never repeat-padded
    long = _wave(NUM_SAMPLES * 2 + NUM_SAMPLES // 2)
    wins = to_windows(long, 16_000)
    assert len(wins) == 3
    assert np.array_equal(wins[0], long[:NUM_SAMPLES])
    assert np.array_equal(wins[1], long[NUM_SAMPLES : 2 * NUM_SAMPLES])
    assert np.array_equal(wins[2], long[-NUM_SAMPLES:])
    # exactly one window: no tail window at all
    assert len(to_windows(_wave(NUM_SAMPLES), 16_000)) == 1
    # 4.0-5.05 s: the last 4 s is scored too, whatever the leftover, not the first 4 s alone
    for n in (NUM_SAMPLES + 1, NUM_SAMPLES + MIN_TAIL - 1):
        clip = _wave(n)
        first, last = to_windows(clip, 16_000)
        assert np.array_equal(first, clip[:NUM_SAMPLES]) and np.array_equal(
            last, clip[-NUM_SAMPLES:]
        )
    # from two windows on, a leftover under MIN_TAIL is still dropped
    assert window_starts(2 * NUM_SAMPLES + MIN_TAIL - 1) == [0, NUM_SAMPLES]
    assert window_starts(NUM_SAMPLES + MIN_TAIL) == [0, MIN_TAIL]
    assert window_starts(1) == window_starts(NUM_SAMPLES) == [0]
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


@pytest.mark.parametrize("value", ["stub ", " STUB", "\tstub\n", ""])
def test_get_detector_ignores_surrounding_whitespace(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("VOXGUARD_DETECTOR", value)
    assert isinstance(get_detector(), StubDetector)


@pytest.mark.parametrize("sr", [44_101, 16_001, 191_999, 2003])
def test_torch_resample_refuses_rates_needing_a_huge_filter(sr: int) -> None:
    """Checked before torch is imported, so nothing is allocated (and no torch is needed)."""
    assert resample_elements(sr) > MAX_RESAMPLE_ELEMENTS
    with pytest.raises(ValueError, match="standard sample rate"):
        torch_resample(np.zeros(sr // 100, dtype=np.float32), sr, 16_000)


@pytest.mark.parametrize(
    "sr", [7350, 8000, 11_025, 12_000, 22_050, 24_000, 32_000, 44_100, 48_000, 96_000, 192_000]
)
def test_standard_rates_are_within_the_resample_cap(sr: int) -> None:
    assert resample_elements(sr) <= MAX_RESAMPLE_ELEMENTS


def test_torch_resample_still_resamples_a_standard_rate() -> None:
    pytest.importorskip("torchaudio")
    y = torch_resample(_wave(44_100), 44_100, 16_000)
    assert y.dtype == np.float32 and len(y) == 16_000


def test_real_detector_needs_a_model(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    monkeypatch.delenv("VOXGUARD_MODEL_REPO", raising=False)
    with pytest.raises(ValueError, match="VOXGUARD_MODEL_REPO"):
        RealDetector()
