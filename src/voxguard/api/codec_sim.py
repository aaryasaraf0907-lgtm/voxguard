"""Phone-line simulation for /v1/analyze: reuses ``voxguard.data.codecs.apply_codec`` (ffmpeg)."""

from __future__ import annotations

import logging

import numpy as np

from voxguard.data.codecs import (
    CODECS,
    SAMPLE_RATE,
    CodecError,
    apply_codec,
    available_codecs,
    ffmpeg_path,
)
from voxguard.inference import linear_resample

logger = logging.getLogger(__name__)

_SILENCE = 1e-4  # peak below this counts as silent input


class CodecUnavailableError(Exception):
    """ffmpeg or the configured encoder is missing on this host."""


class CodecSimulationError(Exception):
    """ffmpeg ran but the transformation failed or produced unusable audio."""


def ensure_available(codec: str) -> None:
    if codec not in CODECS:
        raise CodecUnavailableError(f"unknown codec {codec!r}")
    if ffmpeg_path() is None:
        raise CodecUnavailableError("ffmpeg was not found on PATH")
    if not available_codecs()[codec]:
        raise CodecUnavailableError(
            f"this ffmpeg has no encoder for {codec} ({CODECS[codec].encoder})"
        )


def simulate_phone_line(
    waveform: np.ndarray, sample_rate: int, codec: str
) -> tuple[np.ndarray, int]:
    """Encode and decode ``waveform`` with ``codec``; returns (16 kHz mono float32, 16000).

    ``apply_codec`` expects 16 kHz mono input, so other rates are resampled first (linear
    interpolation, the same resampler the stub detector uses; it does not low-pass, so
    downsampling from rates above 16 kHz can alias). Never falls back to the original audio.
    """
    ensure_available(codec)
    x = np.asarray(waveform, dtype=np.float32)
    if sample_rate != SAMPLE_RATE:
        x = linear_resample(x, sample_rate, SAMPLE_RATE)
    try:
        y = apply_codec(x, codec)
    except CodecError as exc:
        logger.error("codec simulation failed (%s): %s", codec, exc)
        raise CodecSimulationError(f"{codec} processing failed") from None
    if y.shape != x.shape or not np.isfinite(y).all():
        raise CodecSimulationError(f"{codec} produced malformed output")
    if float(np.abs(x).max()) > _SILENCE and float(np.abs(y).max()) == 0.0:
        raise CodecSimulationError(f"{codec} produced silent output for non-silent input")
    return y, SAMPLE_RATE
