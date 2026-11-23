"""On-the-fly "realistic recording" augmentation: room reverb, microphone EQ, noise, gain.

Codec augmentation made the model robust to phone codecs but not to other real-world audio
(In-the-Wild, 15 Nov: 19 % of real clips flagged, false alarms clustering in old, reverberant
or music-heavy recordings). This adds the recording-chain variation that ASVspoof lacks, applied
to bona fide and spoofed clips alike so it carries no label information.

Everything is plain numpy and seeded by the caller's generator, so a training run stays
reproducible and resumable. Room responses are synthetic (exponentially decaying noise with a
direct path), so no impulse-response dataset is needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class RealisticPolicy:
    probability: float = 0.5  # share of training clips that get any of this augmentation
    reverb_probability: float = 0.5
    rt60_s: tuple[float, float] = (0.15, 0.8)  # reverberation time range
    eq_probability: float = 0.5
    eq_max_db: float = 8.0  # max boost/cut of the random microphone response
    noise_probability: float = 0.5
    snr_db: tuple[float, float] = (10.0, 35.0)
    gain_db: tuple[float, float] = (-6.0, 6.0)

    @classmethod
    def from_dict(cls, raw: dict) -> RealisticPolicy:
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown realistic augmentation keys: {sorted(unknown)}")
        kw = {k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()}
        return cls(**kw)


def synthetic_rir(rt60_s: float, rng: np.random.Generator, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Direct path plus exponentially decaying noise tail (-60 dB at ``rt60_s``), peak 1."""
    n = max(int(min(rt60_s, 1.0) * sr), 2)
    t = np.arange(n) / sr
    tail = rng.standard_normal(n) * 10.0 ** (-3.0 * t / rt60_s)  # amplitude: -60 dB at rt60
    tail[0] = 0.0
    delay = int(rng.integers(int(0.002 * sr), int(0.02 * sr)))  # first reflections 2-20 ms
    tail[:delay] = 0.0
    rir = 0.5 * tail / (np.abs(tail).max() + 1e-12)
    rir[0] = 1.0
    return rir


def add_reverb(x: np.ndarray, rt60_s: float, rng: np.random.Generator) -> np.ndarray:
    rir = synthetic_rir(rt60_s, rng)
    n = len(x) + len(rir) - 1
    y = np.fft.irfft(np.fft.rfft(x, n) * np.fft.rfft(rir, n), n)[: len(x)]
    return y * (np.sqrt(np.mean(x**2)) / (np.sqrt(np.mean(y**2)) + 1e-12))  # keep loudness


def random_eq(x: np.ndarray, max_db: float, rng: np.random.Generator) -> np.ndarray:
    """A smooth random frequency response (6 points on a log-frequency axis, +-max_db)."""
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1.0 / SAMPLE_RATE)
    knots_f = np.geomspace(50.0, SAMPLE_RATE / 2, 6)
    knots_db = rng.uniform(-max_db, max_db, size=6)
    gain_db = np.interp(np.log(np.maximum(freqs, 50.0)), np.log(knots_f), knots_db)
    return np.fft.irfft(spec * 10.0 ** (gain_db / 20.0), len(x))


def coloured_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """White, pink or brown noise (power ~ 1/f^beta, beta in {0, 1, 2}), unit RMS."""
    beta = float(rng.choice([0.0, 1.0, 2.0]))
    spec = rng.standard_normal(n // 2 + 1) + 1j * rng.standard_normal(n // 2 + 1)
    f = np.fft.rfftfreq(n)
    f[0] = f[1] if n > 1 else 1.0
    noise = np.fft.irfft(spec / f ** (beta / 2.0), n)
    return noise / (np.sqrt(np.mean(noise**2)) + 1e-12)


def add_noise(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    rms = np.sqrt(np.mean(x**2))
    if rms == 0.0:
        return x
    return x + coloured_noise(len(x), rng) * rms / 10.0 ** (snr_db / 20.0)


def apply_realistic(x: np.ndarray, policy: RealisticPolicy, rng: np.random.Generator) -> np.ndarray:
    """Return a float32 copy of ``x`` with a random subset of the augmentations, in [-1, 1]."""
    if len(x) == 0 or rng.random() >= policy.probability:
        return x
    y = x.astype(np.float64)
    if rng.random() < policy.reverb_probability:
        y = add_reverb(y, float(rng.uniform(*policy.rt60_s)), rng)
    if rng.random() < policy.eq_probability:
        y = random_eq(y, policy.eq_max_db, rng)
    if rng.random() < policy.noise_probability:
        y = add_noise(y, float(rng.uniform(*policy.snr_db)), rng)
    y = y * 10.0 ** (float(rng.uniform(*policy.gain_db)) / 20.0)
    return np.clip(y, -1.0, 1.0).astype(np.float32)
