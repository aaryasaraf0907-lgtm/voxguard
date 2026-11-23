"""ffmpeg fallback for uploads libsndfile cannot read (MP4/M4A with AAC, some FLAC files).

Only ``voxguard.api.audio.decode_audio`` calls this, and only after libsndfile has failed.
Security measures (reviewed with Aarya, 16 Nov 2026):

- The container is chosen from the upload's first bytes (FLAC or an allow-listed MP4 brand) and
  forced with ``-f``. Nothing else, including HLS or concat playlists, reaches ffmpeg. File
  names and client content types are never used.
- The bytes are written to a 0600 ``mkstemp`` file in a private (0700) ``TemporaryDirectory``,
  because an MP4 with its index (``moov``) at the end needs a seekable input. This is the only
  ffmpeg input path. ffmpeg reads exactly that file: ``-protocol_whitelist file``,
  ``-enable_drefs 0`` for MP4, and no shell (argument lists only).
- ffprobe checks the container, the codec and the limits before anything is decoded. Output is
  read in chunks under a hard byte cap, and every process has a timeout.
- Every process is killed (if still running) and reaped before the directory is removed. The
  removal runs on every path.
- Clients get fixed messages. Logs get a category, the exit code and the elapsed time, never
  ffmpeg's own output, audio bytes or paths.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import IO, Any, Literal

import numpy as np

logger = logging.getLogger(__name__)

OUTPUT_RATE = 16_000
TIMEOUT_S = 30.0  # per process, the same limit as the codec simulation
CHUNK_BYTES = 64 * 1024
PROBE_MAX_BYTES = 1024 * 1024
# AAC priming and end padding can make the decoded audio longer than the probed duration:
# measured up to 32 ms with ffmpeg's encoder, and up to two untrimmed 1024-sample frames
# (~0.28 s at AAC's lowest rate, 7350 Hz) from encoders that write no edit list.
DECODE_SLACK_S = 0.5
MP4_BRANDS = frozenset({b"M4A ", b"mp42", b"isom", b"iso2", b"mp41"})

Demuxer = Literal["flac", "mov"]
# demuxer -> (ffprobe format_name, the only accepted audio codec)
EXPECTED: dict[Demuxer, tuple[str, str]] = {
    "flac": ("flac", "flac"),
    "mov": ("mov,mp4,m4a,3gp,3g2,mj2", "aac"),
}

MSG_UNSUPPORTED = "Unsupported or undecodable audio"
MSG_UNDECODABLE = "Audio could not be decoded"
MSG_UNAVAILABLE = "Audio decoding for this file format is unavailable on this server."
MSG_FAILED = "Audio decoding failed"


class FallbackError(Exception):
    """A fallback failure with the HTTP status and fixed client message it maps to."""

    def __init__(self, status: int, detail: str, category: str, code: int | None = None):
        super().__init__(detail)
        self.status = status
        self.detail = detail
        self.category = category
        self.code = code  # ffmpeg/ffprobe exit code, when one ran to completion


@dataclass(frozen=True)
class Limits:
    max_seconds: float
    max_channels: int
    max_sample_rate: int
    max_bytes: int


def sniff(data: bytes) -> Demuxer | None:
    """The demuxer for these bytes, from their signature alone; None if not allow-listed."""
    if data[:4] == b"fLaC":
        return "flac"
    if len(data) >= 16 and data[4:8] == b"ftyp":
        size = int.from_bytes(data[:4], "big")
        if size < 16 or size > len(data):
            return None
        brands = {data[8:12]} | {data[i : i + 4] for i in range(16, size - 3, 4)}
        if brands & MP4_BRANDS:
            return "mov"
    return None


def _drain(stream: IO[bytes], cap: int, chunks: list[bytes], state: dict[str, Any]) -> None:
    """Reader thread: read until EOF, or stop as soon as more than ``cap`` bytes have arrived."""
    try:
        while chunk := stream.read(CHUNK_BYTES):
            chunks.append(chunk)
            state["total"] += len(chunk)
            if state["total"] > cap:
                state["over"] = True
                return
    except (OSError, ValueError):  # pipe closed by the kill
        return


def _run(cmd: list[str], *, cap: int, timeout: float) -> tuple[int, bytes]:
    """Run ``cmd`` (argument list, no shell). Returns (exit code, stdout).

    Raises FallbackError on a timeout, an output over ``cap`` bytes or a missing binary. On every
    path the process is killed if still running, then reaped, before this returns or raises.
    """
    t0 = time.monotonic()
    try:
        proc = subprocess.Popen(  # noqa: S603 (fixed argument list, no shell)
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,  # never logged or returned, and it cannot fill up
            shell=False,
        )
    except OSError:
        raise FallbackError(503, MSG_UNAVAILABLE, "missing_binary") from None
    chunks: list[bytes] = []
    state: dict[str, Any] = {"total": 0, "over": False}
    assert proc.stdout is not None
    reader: threading.Thread | None = None
    try:
        reader = threading.Thread(
            target=_drain, args=(proc.stdout, cap, chunks, state), daemon=True
        )
        reader.start()
        reader.join(timeout)
        if reader.is_alive():
            raise FallbackError(500, MSG_FAILED, "timeout")
        if state["over"]:
            raise FallbackError(422, "", "output_cap")  # caller fills in the limit message
        remaining = max(0.0, timeout - (time.monotonic() - t0))
        try:
            code = proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            raise FallbackError(500, MSG_FAILED, "timeout") from None
        return code, b"".join(chunks)
    finally:
        # The process is dead once wait() returns, so the reader then sees EOF and ends. No
        # other process holds the pipe: Popen closes inherited descriptors (close_fds=True).
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        if reader is not None and reader.ident is not None:  # started
            reader.join()
        proc.stdout.close()


def _log(category: str, code: int | None, t0: float) -> None:
    ms = (time.monotonic() - t0) * 1000.0
    logger.warning("ffmpeg fallback: %s (exit=%s, %.0f ms)", category, code, ms)


def _probe(exe: str, demuxer: Demuxer, url: str, limits: Limits) -> None:
    cmd = [
        exe,
        "-v",
        "error",
        "-protocol_whitelist",
        "file",
        "-f",
        demuxer,
        *(["-enable_drefs", "0"] if demuxer == "mov" else []),
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        url,
    ]
    try:
        code, out = _run(cmd, cap=PROBE_MAX_BYTES, timeout=TIMEOUT_S)
    except FallbackError as exc:
        if exc.category == "output_cap":  # a sane probe report is a few KB
            raise FallbackError(500, MSG_FAILED, "malformed") from None
        raise
    if code != 0:
        raise FallbackError(415, MSG_UNSUPPORTED, "probe_rejected", code)
    try:
        info = json.loads(out)
        fmt = info["format"]
        audio = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
    except (ValueError, KeyError, TypeError):
        raise FallbackError(500, MSG_FAILED, "malformed") from None
    container, codec = EXPECTED[demuxer]
    if fmt.get("format_name") != container or len(audio) != 1:
        raise FallbackError(415, MSG_UNSUPPORTED, "probe_rejected")
    stream = audio[0]
    if stream.get("codec_name") != codec:
        raise FallbackError(415, MSG_UNSUPPORTED, "probe_rejected")
    try:
        channels = int(stream["channels"])
        sample_rate = int(stream["sample_rate"])
    except (KeyError, TypeError, ValueError):
        raise FallbackError(415, MSG_UNSUPPORTED, "probe_rejected") from None
    if channels > limits.max_channels:
        raise FallbackError(422, f"At most {limits.max_channels} channels are supported", "limit")
    if sample_rate > limits.max_sample_rate:
        msg = f"Sample rate above {limits.max_sample_rate} Hz is not supported"
        raise FallbackError(422, msg, "limit")
    duration = stream.get("duration", fmt.get("duration"))
    try:
        seconds = float(duration) if duration is not None else None
    except (TypeError, ValueError):
        seconds = None
    # A missing duration is allowed (policy): the output cap and the trim in _decode still keep
    # the audio within max_seconds.
    if seconds is not None and math.isfinite(seconds) and seconds > limits.max_seconds:
        raise FallbackError(422, f"Audio is longer than {limits.max_seconds:g} s", "limit")


def _decode(exe: str, demuxer: Demuxer, url: str, limits: Limits) -> np.ndarray:
    cmd = [exe, "-nostdin", "-hide_banner", "-loglevel", "error", "-protocol_whitelist", "file"]
    cmd += ["-f", demuxer]
    if demuxer == "mov":
        cmd += ["-enable_drefs", "0"]  # never follow MP4 references to other files
    cmd += ["-i", url, "-map", "0:a:0", "-vn", "-sn", "-dn"]
    cmd += ["-ac", "1", "-ar", str(OUTPUT_RATE), "-c:a", "pcm_f32le", "-f", "f32le", "pipe:1"]
    max_samples = math.ceil((limits.max_seconds + DECODE_SLACK_S) * OUTPUT_RATE)
    too_long = f"Audio is longer than {limits.max_seconds:g} s"
    try:
        code, out = _run(cmd, cap=max_samples * 4, timeout=TIMEOUT_S)
    except FallbackError as exc:
        if exc.category == "output_cap":
            raise FallbackError(422, too_long, "output_cap") from None
        raise
    if code != 0:
        raise FallbackError(415, MSG_UNDECODABLE, "decode_failed", code)
    if not out or len(out) % 4:
        raise FallbackError(500, MSG_FAILED, "malformed")
    waveform = np.frombuffer(out, dtype="<f4").astype(np.float32)
    if not np.isfinite(waveform).all():
        raise FallbackError(422, "Audio contains NaN or infinite samples", "non_finite")
    # Same frame limit as the libsndfile path. Anything past it is at most DECODE_SLACK_S (the
    # cap above stops longer output), so the detector never gets more than max_seconds.
    max_frames = int(limits.max_seconds * OUTPUT_RATE)
    return waveform[:max_frames]


def _private_dir() -> tempfile.TemporaryDirectory[str]:
    tmp = tempfile.TemporaryDirectory(prefix="voxguard-decode-")
    try:
        if os.name == "posix":
            os.chmod(tmp.name, 0o700)
            if os.stat(tmp.name).st_mode & 0o777 != 0o700:
                raise FallbackError(500, MSG_FAILED, "permissions")
    except BaseException:
        tmp.cleanup()  # removed now, not left for the garbage collector
        raise
    return tmp


def decode_with_ffmpeg(data: bytes, limits: Limits) -> tuple[np.ndarray, int]:
    """Decode ``data`` to (mono float32 waveform, 16000) with ffmpeg. Raises FallbackError."""
    t0 = time.monotonic()
    try:
        if len(data) > limits.max_bytes:  # before anything touches the disk
            raise FallbackError(413, "Upload too large", "too_large")
        demuxer = sniff(data)
        if demuxer is None:
            raise FallbackError(415, MSG_UNSUPPORTED, "sniff_rejected")
        ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
        if ffmpeg is None or ffprobe is None:
            raise FallbackError(503, MSG_UNAVAILABLE, "missing_binary")
        tmp = _private_dir()
        try:
            fd, path = tempfile.mkstemp(dir=tmp.name)  # 0600, O_EXCL
            try:
                view = memoryview(data)
                while view:
                    view = view[os.write(fd, view) :]
            finally:
                os.close(fd)  # closed before ffmpeg starts
            url = "file:" + path
            _probe(ffprobe, demuxer, url, limits)
            waveform = _decode(ffmpeg, demuxer, url, limits)
        finally:
            # every process has been reaped by now (see _run)
            try:
                tmp.cleanup()
            except OSError:
                _log("cleanup_failed", None, t0)
    except FallbackError as exc:
        _log(exc.category, exc.code, t0)
        raise
    except Exception:
        _log("unexpected", None, t0)
        raise FallbackError(500, MSG_FAILED, "unexpected") from None
    return waveform, OUTPUT_RATE
