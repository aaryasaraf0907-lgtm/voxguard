"""Phone-line simulation for /v1/analyze: reuses ``voxguard.data.codecs.apply_codec`` (ffmpeg)."""

from __future__ import annotations

import logging

import numpy as np

from voxguard.data.codecs import (
    CODECS,
    SAMPLE_RATE,
    CodecError,
    _run,
    apply_codec,
    available_codecs,
    ffmpeg_path,
)

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


_FLUSH_S = 0.02  # zero tail fed to the resampler so very short clips are not swallowed by its delay


def resample_to_16k(x: np.ndarray, sample_rate: int) -> np.ndarray:
    """Band-limited resampling of a mono float32 signal to 16 kHz with ffmpeg's resampler.

    ffmpeg (libswresample) low-pass filters before decimation, so content above 8 kHz does not
    alias into the band. ffmpeg is already required for codec simulation, so this adds no
    dependency. The output has exactly ``round(len(x) * 16000 / sample_rate)`` samples (at least
    1); input already at 16 kHz is returned unchanged.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1 or len(x) == 0:
        raise CodecSimulationError("resampling needs a non-empty mono signal")
    if sample_rate == SAMPLE_RATE:
        return x
    exe = ffmpeg_path()
    if exe is None:
        raise CodecUnavailableError("ffmpeg was not found on PATH")
    n_out = max(1, round(len(x) * SAMPLE_RATE / sample_rate))
    padded = np.concatenate([x, np.zeros(int(sample_rate * _FLUSH_S) + 1, dtype=np.float32)])
    cmd = [
        exe, "-v", "error", "-nostdin",
        "-f", "f32le", "-ar", str(sample_rate), "-ac", "1", "-i", "-",
        "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "f32le", "-",
    ]  # fmt: skip
    try:
        raw = _run(cmd, np.ascontiguousarray(padded, dtype="<f4").tobytes())
    except CodecError as exc:
        logger.error("resampling %d Hz -> 16 kHz failed: %s", sample_rate, exc)
        raise CodecSimulationError("resampling failed") from None
    y = np.frombuffer(raw[: len(raw) // 4 * 4], dtype="<f4").astype(np.float32)
    if len(y) < n_out or not np.isfinite(y).all():
        raise CodecSimulationError("resampling produced malformed output")
    return y[:n_out]


def simulate_phone_line(
    waveform: np.ndarray, sample_rate: int, codec: str
) -> tuple[np.ndarray, int]:
    """Encode and decode ``waveform`` with ``codec``; returns (16 kHz mono float32, 16000).

    ``apply_codec`` expects 16 kHz mono input, so other rates are resampled first with
    ``resample_to_16k`` (band-limited). Never falls back to the original audio.
    """
    ensure_available(codec)
    x = np.asarray(waveform, dtype=np.float32)
    x = resample_to_16k(x, sample_rate)
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
