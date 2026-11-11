"""Upload decoding: bytes -> (mono float32 waveform, native sample rate).

No resampling here: the Detector does its own preprocessing (CLAUDE.md section 7).

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

MAX_CHANNELS = 2
MAX_SAMPLE_RATE = 192_000


class AudioError(HTTPException):
    """Client-side audio problem, rendered as {"detail": ...} with the given status."""


def decode_audio(data: bytes, max_seconds: float) -> tuple[np.ndarray, int]:
    try:
        f = sf.SoundFile(io.BytesIO(data))
    except (sf.LibsndfileError, RuntimeError) as exc:
        raise AudioError(415, f"Unsupported or undecodable audio: {exc}") from exc
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
        except sf.LibsndfileError as exc:
            raise AudioError(415, f"Audio could not be decoded: {exc}") from exc
    if len(samples) == 0:
        raise AudioError(422, "Audio contains no samples")
    if len(samples) > max_frames:
        raise AudioError(422, f"Audio is longer than {max_seconds:g} s")
    waveform = samples.mean(axis=1, dtype=np.float32)
    if not np.isfinite(waveform).all():
        raise AudioError(422, "Audio contains NaN or infinite samples")
    return waveform, sample_rate
