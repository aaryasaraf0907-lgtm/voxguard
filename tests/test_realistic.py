import numpy as np
import pytest

from voxguard.data.realistic import (
    SAMPLE_RATE,
    RealisticPolicy,
    add_noise,
    add_reverb,
    apply_realistic,
    random_eq,
    synthetic_rir,
)


def _speechlike(seconds: float = 1.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
    return (0.3 * env * np.sin(2 * np.pi * 220 * t) + 0.01 * rng.standard_normal(len(t))).astype(
        np.float32
    )


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def test_rir_has_direct_path_and_decays() -> None:
    rir = synthetic_rir(0.4, np.random.default_rng(0))
    assert rir[0] == 1.0 and np.abs(rir[1:]).max() <= 0.5
    n = len(rir)
    early, late = _rms(rir[1 : n // 4]), _rms(rir[3 * n // 4 :])
    assert late < early / 10  # exponential decay


def test_reverb_keeps_length_and_loudness_but_changes_signal() -> None:
    x = _speechlike().astype(np.float64)
    y = add_reverb(x, 0.6, np.random.default_rng(1))
    assert y.shape == x.shape
    assert _rms(y) == pytest.approx(_rms(x), rel=1e-6)
    assert not np.allclose(x, y)


def test_eq_changes_spectrum_within_bounds() -> None:
    x = np.random.default_rng(2).standard_normal(SAMPLE_RATE)
    y = random_eq(x, 8.0, np.random.default_rng(3))
    ratio = np.abs(np.fft.rfft(y))[10:] / (np.abs(np.fft.rfft(x))[10:] + 1e-12)
    db = 20 * np.log10(ratio)
    assert db.max() <= 8.0 + 1e-6 and db.min() >= -8.0 - 1e-6
    assert db.max() - db.min() > 1.0  # not flat


def test_noise_hits_requested_snr() -> None:
    x = _speechlike(2.0).astype(np.float64)
    y = add_noise(x, 20.0, np.random.default_rng(4))
    snr = 20 * np.log10(_rms(x) / _rms(y - x))
    assert snr == pytest.approx(20.0, abs=0.01)
    silent = np.zeros(100)
    assert np.array_equal(add_noise(silent, 20.0, np.random.default_rng(0)), silent)


def test_apply_is_seeded_bounded_and_respects_probability() -> None:
    x = _speechlike()
    p = RealisticPolicy(probability=1.0)
    a = apply_realistic(x, p, np.random.default_rng(7))
    b = apply_realistic(x, p, np.random.default_rng(7))
    assert np.array_equal(a, b) and a.dtype == np.float32 and a.shape == x.shape
    assert np.abs(a).max() <= 1.0
    assert not np.array_equal(a, apply_realistic(x, p, np.random.default_rng(8)))
    off = RealisticPolicy(probability=0.0)
    assert apply_realistic(x, off, np.random.default_rng(7)) is x
    assert apply_realistic(np.zeros(0, np.float32), p, np.random.default_rng(0)).size == 0


def test_share_of_augmented_clips_follows_probability() -> None:
    x = _speechlike(0.25)
    p = RealisticPolicy(probability=0.5)
    changed = sum(
        not np.array_equal(apply_realistic(x, p, np.random.default_rng(i)), x) for i in range(400)
    )
    assert 160 < changed < 240


def test_policy_from_dict() -> None:
    p = RealisticPolicy.from_dict({"probability": 0.3, "rt60_s": [0.2, 0.5]})
    assert p.probability == 0.3 and p.rt60_s == (0.2, 0.5)
    with pytest.raises(ValueError, match="unknown"):
        RealisticPolicy.from_dict({"reverb": 1})
