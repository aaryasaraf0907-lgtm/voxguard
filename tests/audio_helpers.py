"""Shared synthetic-audio helper."""

import numpy as np

SAMPLE_RATE = 16_000


def synthetic(seconds: float = 1.0, seed: int = 0) -> np.ndarray:
    """440 Hz + 3 kHz sine plus light noise, 16 kHz float32."""
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    rng = np.random.default_rng(seed)
    x = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 3000 * t)
    return (x + 0.005 * rng.standard_normal(len(t))).astype(np.float32)
