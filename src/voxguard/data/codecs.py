"""Telephony codec simulation via ffmpeg subprocess (CLAUDE.md section 8).

Chain: 16 kHz float32 mono -> resample to the codec rate -> encode -> decode -> 16 kHz.
Audio moves through pipes as raw float32, so this module needs only numpy and ffmpeg.
G.711 is always required; other codecs are skipped with a warning when ffmpeg lacks the encoder.
"""

from __future__ import annotations

import functools
import logging
import shutil
import subprocess
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000


class CodecError(RuntimeError):
    pass


@dataclass(frozen=True)
class CodecSpec:
    name: str
    encoder: str  # ffmpeg encoder name
    codec_rate: int  # Hz the codec runs at (8 kHz narrowband, 16 kHz wideband)
    container: str  # ffmpeg muxer used for the pipe
    bitrates: tuple[int, ...] = ()  # bit/s; empty = fixed-rate codec
    required: bool = False


CODECS: dict[str, CodecSpec] = {
    c.name: c
    for c in (
        CodecSpec("g711_mulaw", "pcm_mulaw", 8000, "wav", required=True),
        CodecSpec("g711_alaw", "pcm_alaw", 8000, "wav", required=True),
        CodecSpec(
            "amr_nb",
            "libopencore_amrnb",
            8000,
            "amr",
            (4750, 5150, 5900, 6700, 7400, 7950, 10200, 12200),
        ),  # fmt: skip
        CodecSpec(
            "amr_wb",
            "libvo_amrwbenc",
            16000,
            "amr",
            (6600, 8850, 12650, 14250, 15850, 18250, 19850, 23050, 23850),
        ),  # fmt: skip
        CodecSpec("gsm", "libgsm", 8000, "gsm", (13000,)),
        CodecSpec("opus", "libopus", 16000, "ogg", (6000, 8000, 12000, 16000, 24000)),
    )
}


def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


@functools.lru_cache(maxsize=1)
def _encoder_names() -> frozenset[str]:
    exe = ffmpeg_path()
    if exe is None:
        return frozenset()
    out = subprocess.run(
        [exe, "-hide_banner", "-encoders"], capture_output=True, text=True, check=False
    ).stdout
    names = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("A"):  # audio encoders only
            names.add(parts[1])
    return frozenset(names)


def available_codecs() -> dict[str, bool]:
    names = _encoder_names()
    return {name: spec.encoder in names for name, spec in CODECS.items()}


def codec_report() -> str:
    exe = ffmpeg_path()
    lines = [f"ffmpeg: {exe or 'NOT FOUND'}"]
    for name, ok in available_codecs().items():
        lines.append(f"  {name:<11} {CODECS[name].encoder:<19} {'OK' if ok else 'missing'}")
    return "\n".join(lines)


def usable_codecs(requested: list[str] | None = None) -> list[str]:
    """Names of codecs that can run now. Warns about skipped ones; G.711 must be present."""
    if ffmpeg_path() is None:
        raise CodecError("ffmpeg not found on PATH; install it (e.g. `brew install ffmpeg`)")
    avail = available_codecs()
    names = requested if requested is not None else list(CODECS)
    unknown = [n for n in names if n not in CODECS]
    if unknown:
        raise CodecError(f"unknown codec(s) {unknown}; known: {sorted(CODECS)}")
    usable = []
    for n in names:
        if avail[n]:
            usable.append(n)
        elif CODECS[n].required:
            raise CodecError(f"required codec {n} ({CODECS[n].encoder}) unavailable in ffmpeg")
        else:
            logger.warning("skipping codec %s: ffmpeg lacks encoder %s", n, CODECS[n].encoder)
    return usable


FFMPEG_TIMEOUT_S = 30.0  # per ffmpeg process; a hung encoder must not block a caller forever


def _run(cmd: list[str], data: bytes, timeout: float = FFMPEG_TIMEOUT_S) -> bytes:
    try:
        proc = subprocess.run(cmd, input=data, capture_output=True, check=False, timeout=timeout)
    except subprocess.TimeoutExpired:  # subprocess.run kills the child before raising
        raise CodecError(f"ffmpeg timed out after {timeout:g} s") from None
    if proc.returncode != 0:
        raise CodecError(f"ffmpeg failed: {proc.stderr.decode(errors='replace').strip()[:300]}")
    return proc.stdout


def apply_codec(waveform: np.ndarray, codec: str, bitrate: int | None = None) -> np.ndarray:
    """Pass a 16 kHz mono float32 waveform through ``codec``; output has the same length."""
    if codec not in CODECS:
        raise CodecError(f"unknown codec {codec!r}; known: {sorted(CODECS)}")
    spec = CODECS[codec]
    if spec.bitrates and bitrate is not None and bitrate not in spec.bitrates:
        raise CodecError(f"{codec}: bitrate {bitrate} not in {spec.bitrates}")
    exe = ffmpeg_path()
    if exe is None:
        raise CodecError("ffmpeg not found on PATH")
    x = np.ascontiguousarray(waveform, dtype="<f4")
    if x.ndim != 1 or len(x) == 0:
        raise CodecError("waveform must be a non-empty 1-D array")
    base = [exe, "-v", "error", "-nostdin"]
    enc = [
        *base, "-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", "-",
        "-ar", str(spec.codec_rate), "-ac", "1", "-c:a", spec.encoder,
    ]  # fmt: skip
    if spec.bitrates and bitrate is not None:
        enc += ["-b:a", str(bitrate)]
    enc += ["-f", spec.container, "-"]
    dec = [*base, "-i", "-", "-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-"]
    y = np.frombuffer(_run(dec, _run(enc, x.tobytes())), dtype="<f4").astype(np.float32)
    n = len(x)
    if len(y) < n:  # codec frame padding/delay: align to the input length
        y = np.pad(y, (0, n - len(y)))
    return y[:n]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(codec_report())


if __name__ == "__main__":
    main()
