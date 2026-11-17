"""The ffmpeg fallback in ``voxguard.api.audio``: M4A/AAC and libsndfile-rejected FLAC.

All audio is generated here (numpy, soundfile, ffmpeg). No recordings are used as fixtures.
Tests that run ffmpeg are skipped where it is missing. Process behaviour (timeouts, endless
output, failures) is exercised with real child processes (``python -c``) standing in for ffmpeg.
"""

import io
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi import HTTPException
from fastapi.testclient import TestClient

from conftest import make_wav
from voxguard.api import audio, audio_ffmpeg
from voxguard.api.audio import decode_audio
from voxguard.api.audio_ffmpeg import FallbackError, Limits, decode_with_ffmpeg, sniff

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="needs ffmpeg and ffprobe")
LIMITS = Limits(max_seconds=60, max_channels=2, max_sample_rate=192_000, max_bytes=25 << 20)
REAL_POPEN = subprocess.Popen
REAL_SOUNDFILE = sf.SoundFile
NAN_BYTES = np.full(8, np.nan, dtype="<f4").tobytes()
FLAC_SIGNATURE_ONLY = b"fLaC" + bytes(64)  # libsndfile rejects it; the signature picks FLAC


def _ffmpeg_audio(
    tmp: Path, seconds: float, *, codec: str = "aac", faststart: bool = False, ext: str = "m4a"
) -> bytes:
    """A generated tone encoded by ffmpeg. Default muxing puts the MP4 index (moov) at the end."""
    out = tmp / f"gen-{codec}-{faststart}.{ext}"
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
    cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}:sample_rate=44100"]
    cmd += ["-c:a", codec] + (["-b:a", "64k"] if codec == "aac" else [])
    cmd += (["-movflags", "+faststart"] if faststart else []) + [str(out)]
    with REAL_POPEN(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) as proc:
        assert proc.wait(timeout=60) == 0  # REAL_POPEN: unaffected by the Recorder's patch
    return out.read_bytes()


def _top_level_boxes(data: bytes) -> list[bytes]:
    boxes, i = [], 0
    while i + 8 <= len(data):
        size = int.from_bytes(data[i : i + 4], "big")
        boxes.append(data[i + 4 : i + 8])
        if size < 8:
            break
        i += size
    return boxes


def _flac(seconds: float = 1.0, sr: int = 16_000) -> bytes:
    x = (np.random.default_rng(0).standard_normal(int(seconds * sr)) * 0.1).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="FLAC")
    return buf.getvalue()


class Recorder:
    """Records temp dirs, processes and the order of reaping vs cleanup."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.dirs: list[str] = []
        self.procs: list[subprocess.Popen] = []
        self.cmds: list[list[str]] = []
        self.events: list[str] = []
        self.modes: list[tuple[int, int]] = []
        rec = self

        class TrackedDir(tempfile.TemporaryDirectory):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                rec.dirs.append(self.name)

            def cleanup(self):
                alive = [p for p in rec.procs if p.returncode is None]
                rec.events.append("cleanup-with-live-process" if alive else "cleanup")
                super().cleanup()

        monkeypatch.setattr(audio_ffmpeg.tempfile, "TemporaryDirectory", TrackedDir)
        self.swaps: dict[str, list[str]] = {}

        def popen(cmd, **kw):
            rec.cmds.append(list(cmd))
            url = next((c for c in cmd if c.startswith("file:")), None)
            if url and os.name == "posix":
                path = url.removeprefix("file:")
                rec.modes.append(
                    (os.stat(os.path.dirname(path)).st_mode & 0o777, os.stat(path).st_mode & 0o777)
                )
            tool = os.path.basename(cmd[0])
            proc = REAL_POPEN(rec.swaps.get(tool, cmd), **kw)
            rec.procs.append(proc)
            return proc

        monkeypatch.setattr(audio_ffmpeg.subprocess, "Popen", popen)

    def swap(self, tool: str, code: str) -> None:
        """Replace ffmpeg or ffprobe with a python child process running ``code``."""
        self.swaps[tool] = [sys.executable, "-c", code]

    def assert_clean(self) -> None:
        assert self.dirs, "no temp dir was created"
        assert all(not os.path.exists(d) for d in self.dirs)
        assert all(p.returncode is not None for p in self.procs)  # every process reaped
        assert "cleanup-with-live-process" not in self.events


@pytest.fixture
def rec(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    return Recorder(monkeypatch)


# --- Aarya's required tests ---------------------------------------------------------------


@needs_ffmpeg
def test_moov_at_end_m4a_decodes(tmp_path: Path, rec: Recorder) -> None:
    data = _ffmpeg_audio(tmp_path, 3.0)
    boxes = _top_level_boxes(data)
    assert boxes.index(b"mdat") < boxes.index(b"moov")  # index at the end
    x, sr = decode_audio(data, 60)
    assert sr == 16_000 and x.dtype == np.float32 and x.ndim == 1
    assert abs(len(x) / sr - 3.0) <= 0.05
    assert np.abs(x).max() > 0.1  # the tone, not silence
    rec.assert_clean()  # temp file gone after a success


@needs_ffmpeg
def test_pipe_input_cannot_read_moov_at_end(tmp_path: Path) -> None:
    """The reason for the temp file: from a pipe, ffmpeg cannot seek back to the samples."""
    data = _ffmpeg_audio(tmp_path, 10.0)
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-f", "mov", "-i", "pipe:0"]
    cmd += ["-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"]
    out = subprocess.run(cmd, input=data, capture_output=True, timeout=30).stdout
    if len(out) >= 9.5 * 16_000 * 4:
        pytest.skip("this ffmpeg build decodes the pipe anyway; nothing to reproduce")
    x, _ = decode_audio(data, 60)
    assert abs(len(x) / 16_000 - 10.0) <= 0.05


@needs_ffmpeg
def test_faststart_m4a_decodes(tmp_path: Path) -> None:
    data = _ffmpeg_audio(tmp_path, 2.0, faststart=True)
    boxes = _top_level_boxes(data)
    assert boxes.index(b"moov") < boxes.index(b"mdat")
    x, sr = decode_audio(data, 60)
    assert sr == 16_000 and abs(len(x) / sr - 2.0) <= 0.05


@needs_ffmpeg
@pytest.mark.parametrize(
    ("tool", "code", "status"),
    [
        ("ffmpeg", "import sys; sys.exit(1)", 415),  # decode failure
        ("ffmpeg", "import time; time.sleep(60)", 500),  # timeout
        ("ffmpeg", "import sys\nwhile True: sys.stdout.buffer.write(bytes(65536))", 422),  # cap
        ("ffmpeg", "import sys; sys.stdout.buffer.write(b'abc')", 500),  # malformed
        ("ffmpeg", "import sys; sys.stdout.buffer.write(b'')", 500),  # no samples
        ("ffmpeg", f"import sys; sys.stdout.buffer.write({NAN_BYTES!r})", 422),  # non-finite
        ("ffprobe", "import sys; sys.exit(1)", 415),  # probe rejects
        ("ffprobe", "print('not json')", 500),
        ("ffprobe", "import sys\nwhile True: sys.stdout.write('x' * 65536)", 500),  # probe cap
    ],
)
def test_temp_file_gone_after_failure(
    tmp_path: Path,
    rec: Recorder,
    monkeypatch: pytest.MonkeyPatch,
    tool: str,
    code: str,
    status: int,
) -> None:
    monkeypatch.setattr(audio_ffmpeg, "TIMEOUT_S", 0.5)
    rec.swap(tool, code)
    data = _ffmpeg_audio(tmp_path, 1.0)
    with pytest.raises(HTTPException) as exc:
        decode_audio(data, 2.0)
    assert exc.value.status_code == status
    rec.assert_clean()


@needs_ffmpeg
def test_temp_file_gone_after_limit_missing_binary_and_unexpected_error(
    tmp_path: Path, rec: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _ffmpeg_audio(tmp_path, 3.0)
    with pytest.raises(HTTPException) as exc:  # probe: longer than the limit
        decode_audio(data, 1.0)
    assert exc.value.status_code == 422
    rec.swaps["ffmpeg"] = [str(tmp_path / "no-such-ffmpeg")]  # Popen fails after the write
    with pytest.raises(HTTPException) as exc:
        decode_audio(data, 60)
    assert exc.value.status_code == 503
    rec.swaps.clear()

    def boom(*a, **kw):
        raise RuntimeError("injected")

    monkeypatch.setattr(audio_ffmpeg, "_decode", boom)
    with pytest.raises(HTTPException) as exc:
        decode_audio(data, 60)
    assert exc.value.status_code == 500 and exc.value.detail == "Audio decoding failed"
    assert len(rec.dirs) == 3
    rec.assert_clean()


@pytest.mark.parametrize(
    "payload",
    [
        "#EXTM3U\n#EXT-X-VERSION:3\n#EXTINF:1.0,\nfile:{wav}\n#EXT-X-ENDLIST\n",  # HLS
        "ffconcat version 1.0\nfile '{wav}'\n",  # concat demuxer script
        "file '{wav}'\nfile '{wav}'\n",  # plain concat list
        "v=0\no=- 0 0 IN IP4 127.0.0.1\ns=x\nc=IN IP4 127.0.0.1\nm=audio 5004 RTP/AVP 0\n",  # SDP
    ],
)
def test_playlist_input_rejected_before_any_file_or_process(
    tmp_path: Path, rec: Recorder, payload: str
) -> None:
    wav = tmp_path / "real.wav"
    wav.write_bytes(make_wav(1.0))
    data = payload.format(wav=wav).encode()
    with pytest.raises(HTTPException) as exc:
        decode_audio(data, 60)
    assert exc.value.status_code == 415
    with pytest.raises(FallbackError) as fexc:
        decode_with_ffmpeg(data, LIMITS)
    assert fexc.value.status == 415 and fexc.value.category == "sniff_rejected"
    assert rec.dirs == [] and rec.cmds == []  # nothing written, nothing started


# --- routing: libsndfile first, the fallback only on failure -------------------------------


def test_supported_formats_never_reach_the_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*a, **kw):
        raise AssertionError("fallback called")

    monkeypatch.setattr(audio_ffmpeg, "decode_with_ffmpeg", forbidden)
    x = (np.random.default_rng(1).standard_normal(8000) * 0.1).astype(np.float32)
    for fmt in ["WAV", "FLAC", "OGG", "MP3"]:
        if fmt not in sf.available_formats():
            continue
        buf = io.BytesIO()
        try:
            sf.write(buf, x, 16_000, format=fmt)
        except (sf.LibsndfileError, TypeError, ValueError):
            continue  # this libsndfile cannot write it; nothing to check
        _, sr = decode_audio(buf.getvalue(), 60)
        assert sr == 16_000


READS: list[int] = []


class _ReadFails:
    """A SoundFile whose header opens but whose samples fail, as on a FLAC that lost sync."""

    def __init__(self, src):
        self._f = REAL_SOUNDFILE(src)  # not sf.SoundFile, which the test patches to this class
        self.samplerate, self.channels, self.frames = (
            self._f.samplerate,
            self._f.channels,
            self._f.frames,
        )

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._f.close()

    def read(self, **kw):
        READS.append(1)
        raise sf.LibsndfileError(1, "injected: ")


@needs_ffmpeg
@pytest.mark.parametrize("stage", ["open", "read"])
def test_flac_libsndfile_rejects_matches_libsndfile(
    monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    data = _flac(1.0)
    expected, _ = sf.read(io.BytesIO(data), dtype="float32")

    def refuse(src):
        raise RuntimeError("injected")

    monkeypatch.setattr(audio.sf, "SoundFile", refuse if stage == "open" else _ReadFails)
    READS.clear()
    x, sr = decode_audio(data, 60)
    assert len(READS) == (stage == "read")  # the read path really was the one that failed
    assert sr == 16_000 and x.shape == expected.shape
    assert np.abs(x - expected).max() <= 1e-4


def test_sniff_signatures() -> None:
    assert sniff(b"fLaC" + bytes(40)) == "flac"
    ftyp = (24).to_bytes(4, "big") + b"ftypM4A " + bytes(4) + b"M4A isom"
    assert sniff(ftyp + bytes(16)) == "mov"
    compat_only = (24).to_bytes(4, "big") + b"ftypXXXX" + bytes(4) + b"YYYYmp42"
    assert sniff(compat_only) == "mov"
    assert sniff(ftyp[:12]) is None  # truncated
    assert sniff((24).to_bytes(4, "big") + b"ftypqt  " + bytes(4) + b"qt  qt  ") is None
    assert sniff((4096).to_bytes(4, "big") + b"ftypM4A " + bytes(8)) is None  # box past the end
    for data in [b"", b"RIFF" + bytes(40), b"#EXTM3U\n", os.urandom(64), b"ID3\x04" + bytes(40)]:
        assert sniff(data) is None


def _only_brand(data: bytes, brand: bytes) -> bytes:
    """``data`` with every brand in its ftyp box (major and compatible) replaced by ``brand``,
    as a recorder that lists nothing else would write it. The box size is unchanged."""
    size = int.from_bytes(data[:4], "big")
    assert data[4:8] == b"ftyp" and size % 4 == 0
    brands = brand + data[12:16] + brand * ((size - 16) // 4)  # major, minor version, compatible
    return data[:8] + brands + data[size:]


def test_sniff_accepts_m4a_as_the_only_brand() -> None:
    """Review follow-up (#73): pins `M4A ` itself, not just the brands ffmpeg also writes."""
    only_m4a = (24).to_bytes(4, "big") + b"ftypM4A " + bytes(4) + b"M4A M4A "
    assert sniff(only_m4a + bytes(16)) == "mov"
    compat_only = (20).to_bytes(4, "big") + b"ftypXXXX" + bytes(4) + b"M4A "
    assert sniff(compat_only + bytes(16)) == "mov"


@needs_ffmpeg
def test_m4a_with_only_the_m4a_brand_decodes(tmp_path: Path) -> None:
    """Review follow-up (#73): a real file whose ftyp lists only `M4A `."""
    data = _only_brand(_ffmpeg_audio(tmp_path, 1.0), b"M4A ")
    size = int.from_bytes(data[:4], "big")
    assert {data[i : i + 4] for i in range(16, size, 4)} == {b"M4A "} and data[8:12] == b"M4A "
    x, sr = decode_audio(data, 60)
    assert sr == 16_000 and abs(len(x) / sr - 1.0) <= 0.05


@needs_ffmpeg
def test_routing_ignores_names_and_content_types(tmp_path: Path, client: TestClient) -> None:
    m4a = _ffmpeg_audio(tmp_path, 1.0)
    r = client.post("/v1/analyze", files={"file": ("clip.flac", m4a, "audio/flac")})
    assert r.status_code == 200
    r = client.post("/v1/analyze", files={"file": ("clip.m4a", b"hello world", "audio/mp4")})
    assert r.status_code == 415


# --- probe checks ----------------------------------------------------------------------------


def _fake_probe(**overrides) -> str:
    stream = {"codec_type": "audio", "codec_name": "aac", "channels": 1, "sample_rate": "44100"}
    stream |= overrides.pop("stream", {})
    info = {"format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "1.0"}}
    info["format"] |= overrides.pop("format", {})
    info["streams"] = overrides.pop("streams", [stream])
    return "import sys; sys.stdout.write(" + repr(json.dumps(info)) + ")"


@needs_ffmpeg
@pytest.mark.parametrize(
    ("probe", "status"),
    [
        (_fake_probe(format={"format_name": "flac"}), 415),  # container != signature
        (_fake_probe(stream={"codec_name": "alac"}), 415),
        (_fake_probe(streams=[]), 415),
        (_fake_probe(streams=[{"codec_type": "audio", "codec_name": "aac"}] * 2), 415),
        (_fake_probe(stream={"channels": 3}), 422),
        (_fake_probe(stream={"sample_rate": "200000"}), 422),
        (_fake_probe(format={"duration": "100.0"}), 422),
    ],
)
def test_probe_mismatch_or_limit_stops_before_decoding(
    tmp_path: Path, rec: Recorder, probe: str, status: int
) -> None:
    rec.swap("ffprobe", probe)
    with pytest.raises(HTTPException) as exc:
        decode_audio(_ffmpeg_audio(tmp_path, 1.0), 60)
    assert exc.value.status_code == status
    assert [os.path.basename(c[0]) for c in rec.cmds] == ["ffprobe"]  # no decode started
    rec.assert_clean()


@needs_ffmpeg
def test_real_alac_in_mp4_is_rejected(tmp_path: Path, rec: Recorder) -> None:
    data = _ffmpeg_audio(tmp_path, 1.0, codec="alac")
    assert sniff(data) == "mov"
    with pytest.raises(HTTPException) as exc:
        decode_audio(data, 60)
    assert exc.value.status_code == 415
    assert [os.path.basename(c[0]) for c in rec.cmds] == ["ffprobe"]


# --- limits, command, permissions, processes ------------------------------------------------


@needs_ffmpeg
def test_size_checked_before_anything_touches_disk(tmp_path: Path, rec: Recorder) -> None:
    data = _ffmpeg_audio(tmp_path, 1.0)
    with pytest.raises(HTTPException) as exc:
        decode_audio(data, 60, max_bytes=len(data) - 1)
    assert exc.value.status_code == 413
    assert rec.dirs == [] and rec.cmds == []


@needs_ffmpeg
def test_command_shape_and_permissions(tmp_path: Path, rec: Recorder) -> None:
    decode_audio(_ffmpeg_audio(tmp_path, 1.0), 60)
    probe, dec = rec.cmds
    (tmpdir,) = rec.dirs
    for cmd in (probe, dec):
        assert all(isinstance(a, str) for a in cmd)
        urls = [a for a in cmd if a.startswith("file:")]
        assert len(urls) == 1 and os.path.dirname(urls[0].removeprefix("file:")) == tmpdir
        i = cmd.index("-protocol_whitelist")
        assert cmd[i + 1] == "file"
        assert cmd[cmd.index("-f") + 1] == "mov"  # forced, the first -f is the input's
    assert "-nostdin" in dec
    assert all(c[c.index("-enable_drefs") + 1] == "0" for c in (probe, dec))
    assert dec.count("-i") == 1 and dec[dec.index("-i") + 1].startswith("file:")
    assert dec[-9:] == ["-ac", "1", "-ar", "16000", "-c:a", "pcm_f32le", "-f", "f32le", "pipe:1"]
    if os.name == "posix":
        assert rec.modes == [(0o700, 0o600), (0o700, 0o600)]


def test_popen_is_called_without_a_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}

    def popen(cmd, **kw):
        seen.update(kw, cmd=cmd)
        return REAL_POPEN([sys.executable, "-c", "pass"], **kw)

    monkeypatch.setattr(audio_ffmpeg.subprocess, "Popen", popen)
    audio_ffmpeg._run(["ffmpeg", "-version"], cap=10, timeout=5)
    assert seen["shell"] is False and isinstance(seen["cmd"], list)
    assert seen["stdin"] is subprocess.DEVNULL and seen["stderr"] is subprocess.DEVNULL


def test_output_cap_is_enforced_while_reading(monkeypatch: pytest.MonkeyPatch) -> None:
    totals: list[int] = []
    drain = audio_ffmpeg._drain

    def tracked(stream, cap, chunks, state):
        drain(stream, cap, chunks, state)
        totals.append(state["total"])

    monkeypatch.setattr(audio_ffmpeg, "_drain", tracked)
    endless = "import sys\nwhile True: sys.stdout.buffer.write(bytes(65536))"
    with pytest.raises(FallbackError) as exc:
        audio_ffmpeg._run([sys.executable, "-c", endless], cap=200_000, timeout=10)
    assert exc.value.category == "output_cap"
    assert totals[0] <= 200_000 + audio_ffmpeg.CHUNK_BYTES


def test_timeout_kills_and_reaps(monkeypatch: pytest.MonkeyPatch) -> None:
    procs: list[subprocess.Popen] = []

    def popen(cmd, **kw):
        procs.append(REAL_POPEN(cmd, **kw))
        return procs[-1]

    monkeypatch.setattr(audio_ffmpeg.subprocess, "Popen", popen)
    for code in ["import time; time.sleep(60)", "import os, time; os.close(1); time.sleep(60)"]:
        with pytest.raises(FallbackError) as exc:  # hangs while writing, or after closing stdout
            audio_ffmpeg._run([sys.executable, "-c", code], cap=10, timeout=0.5)
        assert exc.value.status == 500 and exc.value.category == "timeout"
    assert all(p.returncode is not None for p in procs)


@needs_ffmpeg
def test_exception_mid_decode_still_reaps_before_cleanup(
    tmp_path: Path, rec: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    rec.swap("ffmpeg", "import time; time.sleep(60)")

    class Interrupted(audio_ffmpeg.threading.Thread):
        def join(self, timeout=None):
            if timeout is not None and len(rec.procs) == 2:  # the decode, not the probe
                raise RuntimeError("injected while ffmpeg runs")
            return super().join(timeout)

    monkeypatch.setattr(audio_ffmpeg.threading, "Thread", Interrupted)
    with pytest.raises(HTTPException) as exc:
        decode_audio(_ffmpeg_audio(tmp_path, 1.0), 60)
    assert exc.value.status_code == 500
    rec.assert_clean()
    assert rec.events == ["cleanup"]


def test_missing_binary_is_503(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audio_ffmpeg.shutil, "which", lambda name: None)
    with pytest.raises(HTTPException) as exc:
        decode_audio(FLAC_SIGNATURE_ONLY, 60)
    assert exc.value.status_code == 503
    assert exc.value.detail == "Audio decoding for this file format is unavailable on this server."


@needs_ffmpeg
def test_nothing_leaks_to_logs_or_clients(
    tmp_path: Path, rec: Recorder, caplog: pytest.LogCaptureFixture
) -> None:
    leak = "import sys; sys.stderr.write('SECRET-ERR'); print('SECRET-OUT'); sys.exit(1)"
    rec.swap("ffmpeg", leak)
    data = _ffmpeg_audio(tmp_path, 1.0)
    with caplog.at_level(logging.DEBUG, logger="voxguard"), pytest.raises(HTTPException) as exc:
        decode_audio(data, 60)
    assert exc.value.detail == "Audio could not be decoded"
    rec.swaps.clear()
    with caplog.at_level(logging.DEBUG, logger="voxguard"), pytest.raises(HTTPException):
        decode_audio(data[: len(data) // 3], 60)  # real ffprobe on a truncated file
    text = caplog.text
    assert "decode_failed" in text and "exit=1" in text
    for secret in ["SECRET", "voxguard-decode-", tempfile.gettempdir(), "moov", "Invalid data"]:
        assert secret not in text


# --- API and limits end to end ---------------------------------------------------------------


@needs_ffmpeg
def test_api_analyzes_m4a_and_stores_it(tmp_path: Path, client: TestClient) -> None:
    r = client.post("/v1/analyze", files={"file": ("rec.m4a", _ffmpeg_audio(tmp_path, 2.0))})
    assert r.status_code == 200
    assert len(client.get("/v1/sessions").json()) == 1


@needs_ffmpeg
def test_api_errors_store_nothing(
    tmp_path: Path, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _ffmpeg_audio(tmp_path, 3.0)
    monkeypatch.setattr(audio_ffmpeg, "TIMEOUT_S", 0.5)
    cases: list[tuple[str | None, int]] = [(None, 503), ("import time; time.sleep(60)", 500)]
    for code, status in cases:
        with monkeypatch.context() as m:
            if code is None:
                m.setattr(audio_ffmpeg.shutil, "which", lambda name: None)
            else:
                Recorder(m).swap("ffmpeg", code)
            r = client.post("/v1/analyze", files={"file": ("a.m4a", data)})
        assert r.status_code == status
    assert client.get("/v1/sessions").json() == []


@needs_ffmpeg
def test_api_limit_from_probe_is_422(tmp_path: Path, client_factory) -> None:
    with client_factory(max_audio_seconds=1.0) as c:
        r = c.post("/v1/analyze", files={"file": ("a.m4a", _ffmpeg_audio(tmp_path, 3.0))})
        assert r.status_code == 422 and r.json()["detail"] == "Audio is longer than 1 s"
        assert c.get("/v1/sessions").json() == []


# --- review follow-ups (16 Nov): fixed 415 text, AAC margin, reaping on rare paths -----------


@pytest.mark.parametrize("stage", ["open", "read"])
def test_libsndfile_415_has_fixed_text(monkeypatch: pytest.MonkeyPatch, stage: str) -> None:
    data, expected = b"hello world", "Unsupported or undecodable audio"
    if stage == "read":  # header opens, samples fail, and the bytes are not FLAC or MP4
        data, expected = make_wav(0.1), "Audio could not be decoded"
        monkeypatch.setattr(audio.sf, "SoundFile", _ReadFails)
    READS.clear()
    with pytest.raises(HTTPException) as exc:
        decode_audio(data, 60)
    assert exc.value.status_code == 415 and exc.value.detail == expected
    assert len(READS) == (stage == "read")


@needs_ffmpeg
@pytest.mark.parametrize("rate", [8000, 44100])
def test_aac_at_exactly_the_limit_is_accepted(tmp_path: Path, rate: int) -> None:
    out = tmp_path / f"limit-{rate}.m4a"
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi"]
    cmd += ["-i", f"sine=frequency=300:duration=2:sample_rate={rate}", "-c:a", "aac", str(out)]
    with REAL_POPEN(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) as proc:
        assert proc.wait(timeout=60) == 0
    x, sr = decode_audio(out.read_bytes(), 2.0)  # padding past 2 s is within the allowance
    assert sr == 16_000 and len(x) == 2 * 16_000  # and trimmed off: exactly the limit


@needs_ffmpeg
def test_reader_thread_failure_still_reaps_and_cleans_up(
    tmp_path: Path, rec: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _ffmpeg_audio(tmp_path, 1.0)

    class NoStart(audio_ffmpeg.threading.Thread):
        def start(self):
            raise RuntimeError("cannot start thread")

    monkeypatch.setattr(audio_ffmpeg.threading, "Thread", NoStart)
    with pytest.raises(HTTPException) as exc:
        decode_audio(data, 60)
    assert exc.value.status_code == 500 and exc.value.detail == "Audio decoding failed"
    rec.assert_clean()


def test_permission_failure_removes_the_dir_at_once(
    rec: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(path, mode):
        raise PermissionError("injected")

    monkeypatch.setattr(audio_ffmpeg.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(audio_ffmpeg.os, "chmod", refuse)
    with pytest.raises(HTTPException) as exc:
        decode_audio(FLAC_SIGNATURE_ONLY, 60)
    assert exc.value.status_code == 500
    assert rec.dirs and not any(os.path.exists(d) for d in rec.dirs) and rec.cmds == []


# --- duration allowance: trim to the limit, reject beyond it (16 Nov) ------------------------


def _write_seconds(seconds: float) -> str:
    """Python code for a stand-in ffmpeg that writes ``seconds`` of 16 kHz float32 output."""
    return f"import sys; sys.stdout.buffer.write(bytes(4 * round({seconds} * 16000)))"


NO_DURATION_PROBE = (
    "import sys; sys.stdout.write("
    + repr(
        json.dumps(
            {
                "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2"},
                "streams": [
                    {
                        "codec_type": "audio",
                        "codec_name": "aac",
                        "channels": 1,
                        "sample_rate": "8000",
                    }
                ],
            }
        )
    )
    + ")"
)


@needs_ffmpeg
def test_output_within_the_allowance_is_trimmed_to_the_limit(tmp_path: Path, rec: Recorder) -> None:
    rec.swap("ffmpeg", _write_seconds(2.3))  # the real probe sees 1 s, under the 2 s limit
    x, sr = decode_audio(_ffmpeg_audio(tmp_path, 1.0), 2.0)
    assert sr == 16_000 and len(x) == int(2.0 * 16_000)
    rec.assert_clean()


@needs_ffmpeg
def test_output_beyond_the_allowance_is_422(tmp_path: Path, rec: Recorder) -> None:
    rec.swap("ffmpeg", _write_seconds(2.0 + audio_ffmpeg.DECODE_SLACK_S + 0.2))
    with pytest.raises(HTTPException) as exc:
        decode_audio(_ffmpeg_audio(tmp_path, 1.0), 2.0)
    assert exc.value.status_code == 422 and exc.value.detail == "Audio is longer than 2 s"
    rec.assert_clean()


@needs_ffmpeg
def test_probe_without_duration_is_decoded_within_cap_and_trim(
    tmp_path: Path, rec: Recorder
) -> None:
    """Policy: a missing probe duration is not rejected. The output cap and the trim still bound
    the audio, so such a file can never give the detector more than the limit."""
    data = _ffmpeg_audio(tmp_path, 1.0)
    rec.swap("ffprobe", NO_DURATION_PROBE)
    rec.swap("ffmpeg", _write_seconds(1.5))  # under the limit: kept whole
    x, _ = decode_audio(data, 2.0)
    assert len(x) == int(1.5 * 16_000)
    rec.swap("ffmpeg", _write_seconds(2.4))  # over the limit, within the allowance: trimmed
    x, _ = decode_audio(data, 2.0)
    assert len(x) == int(2.0 * 16_000)
    rec.swap("ffmpeg", _write_seconds(3.0))  # beyond the allowance: rejected
    with pytest.raises(HTTPException) as exc:
        decode_audio(data, 2.0)
    assert exc.value.status_code == 422
    assert [os.path.basename(c[0]) for c in rec.cmds].count("ffmpeg") == 3  # all were decoded
    rec.assert_clean()
