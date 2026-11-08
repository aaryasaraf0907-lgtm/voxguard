import subprocess
import tempfile

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from conftest import FixedDetector, make_wav
from voxguard.api import codec_sim
from voxguard.api.db import AlertRow, BlocklistRow, SessionRow
from voxguard.api.deps import Settings
from voxguard.data import codecs
from voxguard.data.codecs import CodecError, available_codecs, ffmpeg_path
from voxguard.inference import Prediction

CODEC = "g711_mulaw"


class SpyDetector(FixedDetector):
    def __init__(self, score: float = 0.9) -> None:
        super().__init__(score)
        self.seen: list[tuple[np.ndarray, int]] = []

    def predict(self, waveform: np.ndarray, sample_rate: int) -> Prediction:
        self.seen.append((np.array(waveform), sample_rate))
        return super().predict(waveform, sample_rate)


def _post(client: TestClient, data: bytes | None = None, **form: str):
    data = data if data is not None else make_wav(0.5)
    return client.post("/v1/analyze", files={"file": ("a.wav", data)}, data=form)


def _count(client: TestClient, model) -> int:
    with client.app.state.session_factory() as db:
        return db.scalar(select(func.count()).select_from(model))


def _counts(client: TestClient) -> tuple[int, int, int]:
    return tuple(_count(client, m) for m in (SessionRow, AlertRow, BlocklistRow))


@pytest.fixture
def sim_client(client_factory):
    """Client configured with a codec; availability checks pass (ffmpeg itself is mocked)."""

    def make(detector=None, **kw):
        return client_factory(detector or SpyDetector(), simulate_codec=CODEC, **kw)

    return make


@pytest.fixture
def fake_codec(monkeypatch):
    """Pretend ffmpeg is present and make apply_codec a visible, cheap transform."""
    calls = []

    def fake_apply(x: np.ndarray, codec: str, bitrate=None) -> np.ndarray:
        calls.append((np.array(x), codec))
        return (x * 0.5).astype(np.float32)

    monkeypatch.setattr(codec_sim, "ffmpeg_path", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(codec_sim, "available_codecs", lambda: {c: True for c in codecs.CODECS})
    monkeypatch.setattr(codec_sim, "apply_codec", fake_apply)
    return calls


# --- off by default / configuration ----------------------------------------------------


def test_without_simulation_audio_is_untouched_and_codec_is_null(sim_client, monkeypatch) -> None:
    def boom(*a, **k):
        raise AssertionError("codec must not run")

    monkeypatch.setattr(codec_sim, "apply_codec", boom)
    det = SpyDetector()
    with sim_client(det) as c:
        r = _post(c, make_wav(0.5, sr=8_000), simulate_codec="false")
        assert r.status_code == 200 and r.json()["codec_applied"] is None
        assert _post(c).json()["codec_applied"] is None  # field omitted
    assert [sr for _, sr in det.seen] == [8_000, 16_000]  # native rate, no resampling here


def test_true_without_configured_codec_is_501_and_stores_nothing(client_factory) -> None:
    with client_factory(SpyDetector()) as c:
        r = _post(c, simulate_codec="true")
        assert r.status_code == 501 and "VOXGUARD_SIMULATE_CODEC" in r.json()["detail"]
        assert _counts(c) == (0, 0, 0)


def test_unknown_configured_codec_fails_startup() -> None:
    with pytest.raises(ValueError, match="VOXGUARD_SIMULATE_CODEC"):
        Settings(simulate_codec="mp3_magic")


def test_env_setting(monkeypatch) -> None:
    monkeypatch.delenv("VOXGUARD_SIMULATE_CODEC", raising=False)
    assert Settings.from_env().simulate_codec is None
    monkeypatch.setenv("VOXGUARD_SIMULATE_CODEC", " opus ")
    assert Settings.from_env().simulate_codec == "opus"
    monkeypatch.setenv("VOXGUARD_SIMULATE_CODEC", "nope")
    with pytest.raises(ValueError, match="VOXGUARD_SIMULATE_CODEC"):
        Settings.from_env()
    monkeypatch.setenv("VOXGUARD_SIMULATE_CODEC", "  ")
    assert Settings.from_env().simulate_codec is None


# --- simulation happens before inference -----------------------------------------------


def test_detector_receives_transformed_audio_and_codec_is_reported(sim_client, fake_codec) -> None:
    det = SpyDetector()
    wav = make_wav(1.0, sr=16_000)
    original, _ = sf.read(__import__("io").BytesIO(wav), dtype="float32")
    with sim_client(det) as c:
        r = _post(c, wav, simulate_codec="true", caller_id="alice")
        assert r.status_code == 200 and r.json()["codec_applied"] == CODEC
        (row,) = c.app.state.session_factory().scalars(select(SessionRow)).all()
        assert row.codec_applied == CODEC
    ((received, sr),) = det.seen
    assert sr == 16_000 and received.dtype == np.float32 and received.ndim == 1
    assert received.shape == original.shape
    np.testing.assert_allclose(received, original * 0.5, atol=1e-6)  # the transformed signal
    assert not np.allclose(received, original)
    assert fake_codec[0][1] == CODEC


def test_other_sample_rates_are_resampled_to_16k_before_the_codec(sim_client, fake_codec) -> None:
    det = SpyDetector()
    with sim_client(det) as c:
        assert (
            _post(c, make_wav(1.0, sr=8_000, channels=2), simulate_codec="true").status_code == 200
        )
    assert fake_codec[0][0].shape == (16_000,)  # 8 kHz stereo -> mono 16 kHz into the codec
    assert det.seen[0][1] == 16_000 and det.seen[0][0].shape == (16_000,)


def test_simulated_flagged_analysis_keeps_prevention_behavior(sim_client, fake_codec) -> None:
    with sim_client(SpyDetector(0.9)) as c:
        body = _post(c, simulate_codec="true", caller_id=" bob ").json()
        assert body["flagged"] is True and body["codec_applied"] == CODEC
        assert _counts(c) == (1, 1, 1)
        (s,) = c.get("/v1/sessions?flagged_only=true").json()
        assert s["codec_applied"] == CODEC and s["caller_id"] == "bob"
        assert c.get("/v1/blocklist").json()[0]["caller_id"] == "bob"


def test_simulated_unflagged_analysis_creates_no_alert(sim_client, fake_codec) -> None:
    with sim_client(SpyDetector(0.1)) as c:
        assert _post(c, simulate_codec="true", caller_id="bob").json()["flagged"] is False
        assert _counts(c) == (1, 0, 0)


# --- failures never fall back to the original audio ------------------------------------


def _assert_nothing_scored_or_stored(client, det) -> None:
    assert det.seen == [] and _counts(client) == (0, 0, 0)


def test_codec_failure_is_500_and_nothing_is_scored_or_stored(sim_client, monkeypatch) -> None:
    monkeypatch.setattr(codec_sim, "ffmpeg_path", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(codec_sim, "available_codecs", lambda: {c: True for c in codecs.CODECS})

    def fail(*a, **k):
        raise CodecError("ffmpeg failed: secret stderr /home/x")

    monkeypatch.setattr(codec_sim, "apply_codec", fail)
    det = SpyDetector()
    with sim_client(det) as c:
        r = _post(c, simulate_codec="true", caller_id="a")
        assert r.status_code == 500 and r.json()["detail"] == "Codec simulation failed"
        assert "secret" not in r.text
        _assert_nothing_scored_or_stored(c, det)


@pytest.mark.parametrize(
    "bad",
    [
        lambda x: x[:-1],  # wrong length
        lambda x: np.full_like(x, np.nan),
        lambda x: np.full_like(x, np.inf),
        lambda x: np.zeros_like(x),  # silent output for non-silent input
    ],
    ids=["short", "nan", "inf", "silent"],
)
def test_malformed_codec_output_is_rejected(sim_client, monkeypatch, bad) -> None:
    monkeypatch.setattr(codec_sim, "ffmpeg_path", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(codec_sim, "available_codecs", lambda: {c: True for c in codecs.CODECS})
    monkeypatch.setattr(codec_sim, "apply_codec", lambda x, codec, bitrate=None: bad(x))
    det = SpyDetector()
    with sim_client(det) as c:
        assert _post(c, simulate_codec="true").status_code == 500
        _assert_nothing_scored_or_stored(c, det)


def test_silent_input_may_produce_silent_output(sim_client, fake_codec) -> None:
    silence = __import__("io").BytesIO()
    sf.write(silence, np.zeros(8_000, dtype=np.float32), 16_000, format="WAV")
    with sim_client(SpyDetector(0.1)) as c:
        assert _post(c, silence.getvalue(), simulate_codec="true").status_code == 200


def test_missing_ffmpeg_is_503_before_decoding(sim_client, monkeypatch) -> None:
    monkeypatch.setattr(codec_sim, "ffmpeg_path", lambda: None)
    det = SpyDetector()
    with sim_client(det) as c:
        r = _post(c, simulate_codec="true")
        assert r.status_code == 503 and "unavailable" in r.json()["detail"]
        _assert_nothing_scored_or_stored(c, det)
        assert _post(c).status_code == 200  # normal analysis still works


def test_missing_encoder_is_503(sim_client, monkeypatch) -> None:
    monkeypatch.setattr(codec_sim, "ffmpeg_path", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(codec_sim, "available_codecs", lambda: {c: False for c in codecs.CODECS})
    det = SpyDetector()
    with sim_client(det) as c:
        assert _post(c, simulate_codec="true").status_code == 503
        _assert_nothing_scored_or_stored(c, det)


def test_unavailable_check_raises_typed_errors(monkeypatch) -> None:
    monkeypatch.setattr(codec_sim, "ffmpeg_path", lambda: None)
    with pytest.raises(codec_sim.CodecUnavailableError, match="ffmpeg"):
        codec_sim.ensure_available(CODEC)
    with pytest.raises(codec_sim.CodecUnavailableError, match="unknown"):
        codec_sim.ensure_available("nope")


# --- subprocess handling and resources -------------------------------------------------


def test_ffmpeg_is_called_with_a_timeout_and_pipes_only(monkeypatch) -> None:
    runs = []

    def fake_run(cmd, **kw):
        runs.append((cmd, kw))
        out = np.zeros(1600, dtype="<f4").tobytes() if len(runs) == 2 else b"encoded"
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr=b"")

    def no_temp_files(*a, **k):
        raise AssertionError("codec simulation must not create temp files")

    for name in ("mkstemp", "mkdtemp", "NamedTemporaryFile", "TemporaryFile"):
        monkeypatch.setattr(tempfile, name, no_temp_files)
    monkeypatch.setattr(codecs, "ffmpeg_path", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(codecs.subprocess, "run", fake_run)
    codecs.apply_codec(np.zeros(1600, dtype=np.float32), CODEC)
    assert len(runs) == 2
    for cmd, kw in runs:
        assert kw["timeout"] == codecs.FFMPEG_TIMEOUT_S and kw["input"] is not None
        assert isinstance(cmd, list)  # argument list: no shell, no string interpolation
        assert "shell" not in kw


def test_ffmpeg_timeout_becomes_a_clean_500(sim_client, monkeypatch) -> None:
    def hang(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw["timeout"])

    monkeypatch.setattr(codec_sim, "ffmpeg_path", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(codec_sim, "available_codecs", lambda: {c: True for c in codecs.CODECS})
    monkeypatch.setattr(codecs, "ffmpeg_path", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(codecs.subprocess, "run", hang)
    det = SpyDetector()
    with sim_client(det) as c:
        r = _post(c, simulate_codec="true")
        assert r.status_code == 500 and r.json()["detail"] == "Codec simulation failed"
        _assert_nothing_scored_or_stored(c, det)


def test_run_timeout_message() -> None:
    with pytest.raises(CodecError, match="timed out"):
        codecs._run(["sleep", "5"], b"", timeout=0.2)


# --- real ffmpeg (skipped when ffmpeg or the encoder is missing) ------------------------


def _needs(codec: str):
    ok = ffmpeg_path() is not None and available_codecs().get(codec, False)
    return pytest.mark.skipif(not ok, reason=f"ffmpeg with {codec} not available")


def _wideband_wav(sr: int = 16_000, channels: int = 1) -> bytes:
    import io

    t = np.arange(sr) / sr
    x = 0.4 * np.sin(2 * np.pi * 1000 * t) + 0.3 * np.sin(2 * np.pi * 6000 * t)
    x = np.stack([x] * channels, axis=1).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV", subtype="FLOAT")
    return buf.getvalue()


def _energy(x: np.ndarray, lo: float, hi: float, sr: int = 16_000) -> float:
    f = np.fft.rfftfreq(len(x), 1 / sr)
    return float(np.sum(np.abs(np.fft.rfft(x))[(f >= lo) & (f <= hi)] ** 2))


@_needs("g711_mulaw")
def test_real_g711_removes_wideband_content_before_inference(client_factory) -> None:
    det = SpyDetector()
    with client_factory(det, simulate_codec="g711_mulaw") as c:
        plain = _post(c, _wideband_wav())
        sim = _post(c, _wideband_wav(), simulate_codec="true")
        assert plain.status_code == sim.status_code == 200
        assert sim.json()["codec_applied"] == "g711_mulaw" and plain.json()["codec_applied"] is None
    original, simulated = det.seen[0][0], det.seen[1][0]
    assert simulated.shape == original.shape and simulated.dtype == np.float32
    assert _energy(original, 5500, 6500) > 100 * _energy(simulated, 5500, 6500)  # 6 kHz tone gone
    assert _energy(simulated, 900, 1100) > 0.5 * _energy(original, 900, 1100)  # 1 kHz survives
    assert not np.array_equal(original, simulated)


@_needs("g711_mulaw")
def test_real_g711_with_44k_stereo_input(client_factory) -> None:
    det = SpyDetector()
    with client_factory(det, simulate_codec="g711_mulaw") as c:
        r = _post(c, _wideband_wav(44_100, 2), simulate_codec="true")
        assert r.status_code == 200
    x, sr = det.seen[0]
    assert sr == 16_000 and x.shape == (16_000,) and np.isfinite(x).all() and np.abs(x).max() > 0


@_needs("g711_mulaw")
def test_real_simulation_is_deterministic(client_factory) -> None:
    det = SpyDetector()
    with client_factory(det, simulate_codec="g711_mulaw") as c:
        for _ in range(2):
            _post(c, _wideband_wav(), simulate_codec="true")
    assert np.array_equal(det.seen[0][0], det.seen[1][0])


@pytest.mark.parametrize("codec", [c for c in codecs.CODECS if c != "g711_mulaw"])
def test_real_other_codecs_when_available(codec: str, client_factory) -> None:
    if not (ffmpeg_path() and available_codecs().get(codec)):
        pytest.skip(f"ffmpeg lacks {codec}")
    det = SpyDetector()
    with client_factory(det, simulate_codec=codec) as c:
        r = _post(c, _wideband_wav(), simulate_codec="true")
        assert r.status_code == 200 and r.json()["codec_applied"] == codec
    x, _ = det.seen[0]
    assert np.isfinite(x).all() and x.shape == (16_000,)
