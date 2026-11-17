"""Upload decoding: bytes -> (mono float32 waveform, sample rate).

libsndfile decodes in memory first, at the native rate with no resampling (the Detector does its
own preprocessing, CLAUDE.md section 7). Only if libsndfile fails does the ffmpeg fallback
(``audio_ffmpeg``) run, for MP4/M4A (AAC) and FLAC files that libsndfile rejects. It returns
16 kHz mono, so its resampling is done by ffmpeg rather than by the Detector.

Memory bound: decoding is limited by duration (VOXGUARD_MAX_AUDIO_SECONDS, default 60 s),
MAX_SAMPLE_RATE and MAX_CHANNELS. Worst case at 60 s, 192 kHz, 2 channels is
60 * 192_000 * 2 * 4 B = ~92 MB for the decoded float32 array plus ~46 MB for the mono
downmix. Header checks run before any sample is read. Raising the limits raises this bound.
"""

from __future__ import annotations

import io

import numpy as np
import soundfile as sf
from fastapi import HTTPException

from voxguard.api import audio_ffmpeg
from voxguard.api.deps import DEFAULT_MAX_UPLOAD_MB

MAX_CHANNELS = 2
MAX_SAMPLE_RATE = 192_000
DEFAULT_MAX_BYTES = int(DEFAULT_MAX_UPLOAD_MB * 1024 * 1024)


class AudioError(HTTPException):
    """Client-side audio problem, rendered as {"detail": ...} with the given status."""


def _fallback(
    data: bytes, max_seconds: float, max_bytes: int, original: AudioError
) -> tuple[np.ndarray, int]:
    """ffmpeg for what libsndfile could not read. Files it does not recognise keep the
    libsndfile-path 415 status, with its fixed error text."""
    if audio_ffmpeg.sniff(data) is None:
        raise original
    limits = audio_ffmpeg.Limits(max_seconds, MAX_CHANNELS, MAX_SAMPLE_RATE, max_bytes)
    try:
        return audio_ffmpeg.decode_with_ffmpeg(data, limits)
    except audio_ffmpeg.FallbackError as exc:
        raise AudioError(exc.status, exc.detail) from None


def decode_audio(
    data: bytes, max_seconds: float, max_bytes: int = DEFAULT_MAX_BYTES
) -> tuple[np.ndarray, int]:
    try:
        f = sf.SoundFile(io.BytesIO(data))
    except (sf.LibsndfileError, RuntimeError):
        # fixed text: libsndfile's message names an in-memory object address
        error = AudioError(415, audio_ffmpeg.MSG_UNSUPPORTED)
        return _fallback(data, max_seconds, max_bytes, error)
    with f:
        sample_rate, channels = f.samplerate, f.channels
        if channels > MAX_CHANNELS:
            raise AudioError(422, f"At most {MAX_CHANNELS} channels are supported")
        if sample_rate > MAX_SAMPLE_RATE:
            raise AudioError(422, f"Sample rate above {MAX_SAMPLE_RATE} Hz is not supported")
        max_frames = int(max_seconds * sample_rate)
        if f.frames == 0:
            raise AudioError(422, "Audio contains no samples")
        if f.frames > max_frames:
            raise AudioError(422, f"Audio is longer than {max_seconds:g} s")
        try:
            # +1 so a file that decodes to more frames than its header declared is caught
            samples = f.read(frames=max_frames + 1, dtype="float32", always_2d=True)
        except sf.LibsndfileError:
            error = AudioError(415, audio_ffmpeg.MSG_UNDECODABLE)
            read_failed = True
        else:
            read_failed = False
    if read_failed:  # outside the `with`, so libsndfile has released the file
        return _fallback(data, max_seconds, max_bytes, error)
    if len(samples) == 0:
        raise AudioError(422, "Audio contains no samples")
    if len(samples) > max_frames:
        raise AudioError(422, f"Audio is longer than {max_seconds:g} s")
    waveform = samples.mean(axis=1, dtype=np.float32)
    if not np.isfinite(waveform).all():
        raise AudioError(422, "Audio contains NaN or infinite samples")
    return waveform, sample_rate
