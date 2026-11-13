"""Channel-matched thresholds from thresholds.json (agreed with Aarya, 13 Nov 2026).

All threshold values here are DUMMY test values, not thresholds of any trained checkpoint.
"""

import json
import sys
import types

import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import FixedDetector, make_wav
from voxguard.api import codec_sim, detector_loader, prevention
from voxguard.api.deps import Settings
from voxguard.api.main import create_app
from voxguard.api.thresholds import ThresholdsError, load_thresholds, parse_thresholds
from voxguard.data import codecs
from voxguard.inference import Prediction, StubDetector

VERSION = "fixed-0"  # FixedDetector.model_version
DUMMY = {  # far apart so each assertion shows which value was used
    "model_version": VERSION,
    "selected_on": "dev",
    "clean": 0.11,
    "phone": 0.33,
    "low_far": {"clean": 0.05, "phone": 0.2},
    "per_codec": {"opus": 0.99, "gsm": 0.01},
}
DUMMY_SET = parse_thresholds(DUMMY, VERSION)


def _client(thresholds=DUMMY_SET, **settings):
    cfg = Settings(db_url="sqlite://", **settings)
    return TestClient(create_app(FixedDetector(0.5), cfg, thresholds))


def _post(c, **form):
    return c.post("/v1/analyze", files={"file": ("a.wav", make_wav(0.5))}, data=form).json()


@pytest.fixture
def fake_codec(monkeypatch):
    monkeypatch.setattr(codec_sim, "ffmpeg_path", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(codec_sim, "available_codecs", lambda: {c: True for c in codecs.CODECS})
    monkeypatch.setattr(
        codec_sim, "apply_codec", lambda x, codec, bitrate=None: np.asarray(x, np.float32)
    )


# --- selection ------------------------------------------------------------------------------


def test_unknown_origin_upload_uses_the_phone_threshold() -> None:
    with _client() as c:
        body = _post(c)
    assert body["codec_applied"] is None and body["threshold"] == 0.33
    assert body["flagged"] is True  # score 0.5 >= phone 0.33 (it would be too at 0.5)


def test_codec_simulated_audio_uses_the_phone_threshold_not_per_codec(fake_codec) -> None:
    with _client() as c:
        body = _post(c, codec="opus")  # per_codec opus is 0.99: must not be used
    assert body["codec_applied"] == "opus" and body["threshold"] == 0.33 and body["flagged"]


def test_known_clean_audio_uses_the_clean_threshold() -> None:
    pred = Prediction(0.2, "real", 0.5, 1.0, VERSION)
    cfg = Settings(db_url="sqlite://")
    clean = prevention.decide(pred, cfg, DUMMY_SET, prevention.channel_for(True, None))
    assert clean.threshold == 0.11 and clean.flagged  # 0.2 >= clean 0.11
    phone = prevention.decide(pred, cfg, DUMMY_SET, prevention.channel_for(False, None))
    assert phone.threshold == 0.33 and not phone.flagged


def test_simulated_audio_is_phone_even_if_known_clean() -> None:
    assert prevention.channel_for(known_clean=True, codec_applied="g711_mulaw") == "phone"
    assert prevention.channel_for(known_clean=False, codec_applied=None) == "phone"
    assert prevention.channel_for(known_clean=True, codec_applied=None) == "clean"


def test_explicit_threshold_overrides_metadata(fake_codec) -> None:
    with _client(threshold_override=0.6) as c:
        assert _post(c)["threshold"] == 0.6 and _post(c, codec="opus")["threshold"] == 0.6


def test_low_far_mode_keeps_its_explicit_threshold() -> None:
    with _client(threshold_mode="low_far", low_far_threshold=0.7) as c:
        assert _post(c)["threshold"] == 0.7


def test_no_metadata_keeps_the_detectors_threshold() -> None:
    with _client(thresholds=None) as c:
        assert _post(c)["threshold"] == 0.5  # FixedDetector's own, as before


# --- file validation ------------------------------------------------------------------------


def test_parse_keeps_reporting_fields() -> None:
    assert DUMMY_SET.low_far == {"clean": 0.05, "phone": 0.2} and DUMMY_SET.per_codec["gsm"] == 0.01
    minimal = parse_thresholds({"model_version": VERSION, "clean": 0, "phone": 1}, VERSION)
    assert minimal.per_codec == {} and minimal.low_far == {} and minimal.selected_on is None


@pytest.mark.parametrize(
    "change",
    [
        {"phone": None},
        {"clean": 1.5},
        {"phone": -0.1},
        {"clean": "0.3"},
        {"phone": True},
        {"clean": float("nan")},
        {"model_version": ""},
        {"per_codec": {"opus": 2}},
        {"low_far": [0.1]},
        {"selected_on": 3},
    ],
)
def test_invalid_metadata_is_rejected(change) -> None:
    with pytest.raises(ThresholdsError):
        parse_thresholds({**DUMMY, **change}, VERSION)


def test_model_version_mismatch_is_rejected() -> None:
    with pytest.raises(ThresholdsError, match="ssl_aasist:best.pt:epoch8"):
        parse_thresholds(DUMMY, "ssl_aasist:best.pt:epoch8")


LOCAL = "ssl_aasist:best.pt:epoch8"  # as evaluate.py writes it after evaluating a local best.pt


@pytest.mark.parametrize(
    "loaded",
    [
        LOCAL,
        "ssl_aasist:voxguard-ssl-aasist:epoch8",  # the same checkpoint loaded from its HF repo
        "ssl_aasist:models:epoch8",  # ... or from a mounted /models directory
        "ssl_aasist:a:b:epoch8",  # the load location may itself contain ":"
    ],
)
def test_same_model_and_epoch_match_wherever_it_was_loaded_from(loaded) -> None:
    ts = parse_thresholds({**DUMMY, "model_version": LOCAL}, loaded)
    assert ts.model_version == LOCAL and ts.phone == 0.33


@pytest.mark.parametrize(
    "loaded",
    [
        "ssl_aasist:voxguard-ssl-aasist:epoch7",  # other epoch
        "rawnet2:voxguard-ssl-aasist:epoch8",  # other model
        "ssl_aasist:epoch8",  # not the <name>:<source>:epoch<N> shape: compared whole
    ],
)
def test_other_model_or_epoch_is_rejected(loaded) -> None:
    with pytest.raises(ThresholdsError, match="model name and epoch must match"):
        parse_thresholds({**DUMMY, "model_version": LOCAL}, loaded)


def test_load_from_a_local_checkpoint_directory_or_file(tmp_path) -> None:
    assert load_thresholds(str(tmp_path), VERSION) is None  # no file: legacy behaviour
    (tmp_path / "best.pt").write_bytes(b"x")
    (tmp_path / "thresholds.json").write_text(json.dumps(DUMMY))
    assert load_thresholds(str(tmp_path), VERSION) == DUMMY_SET
    assert load_thresholds(str(tmp_path / "best.pt"), VERSION) == DUMMY_SET  # sibling file
    (tmp_path / "thresholds.json").write_text("{not json")
    with pytest.raises(ThresholdsError, match="cannot read"):
        load_thresholds(str(tmp_path), VERSION)


# --- startup with the configured (real) detector ------------------------------------------


def _start_real(monkeypatch, repo, version=VERSION, **settings) -> TestClient:
    import voxguard.inference as inference

    class FakeReal(StubDetector):  # model_version must match the file's
        def __init__(self, repo: str, device: str = "cpu") -> None:
            super().__init__(threshold=0.5, model_version=version)

    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))  # no ml extra in CI
    monkeypatch.setattr(inference, "RealDetector", FakeReal)
    monkeypatch.setattr(detector_loader, "_check_device", lambda device: None)
    monkeypatch.setenv("VOXGUARD_DETECTOR", "real")
    monkeypatch.setenv("VOXGUARD_MODEL_REPO", str(repo))
    monkeypatch.delenv("VOXGUARD_THRESHOLD", raising=False)
    return TestClient(create_app(settings=Settings(db_url="sqlite://", **settings)))


def test_startup_loads_thresholds_next_to_the_checkpoint(monkeypatch, tmp_path, caplog) -> None:
    (tmp_path / "thresholds.json").write_text(json.dumps(DUMMY))
    with _start_real(monkeypatch, tmp_path) as c:
        assert c.app.state.thresholds == DUMMY_SET
        assert _post(c)["threshold"] == 0.33
    assert "VOXGUARD_THRESHOLD is the default" not in caplog.text


def test_startup_accepts_local_thresholds_for_a_hub_loaded_checkpoint(monkeypatch, tmp_path):
    """Regression: evaluate.py writes the local model_version; the API loads from the Hub."""
    (tmp_path / "thresholds.json").write_text(json.dumps({**DUMMY, "model_version": LOCAL}))
    hub_version = "ssl_aasist:voxguard-ssl-aasist:epoch8"
    with _start_real(monkeypatch, tmp_path, version=hub_version) as c:
        assert c.app.state.thresholds.model_version == LOCAL
        assert _post(c)["threshold"] == 0.33


@pytest.mark.parametrize(
    "content", ['{"model_version": "other:best.pt:epoch1", "clean": 0.1, "phone": 0.2}', "[]"]
)
def test_startup_refuses_unusable_thresholds(monkeypatch, tmp_path, content) -> None:
    (tmp_path / "thresholds.json").write_text(content)
    with (
        pytest.raises(RuntimeError, match="unusable thresholds"),
        _start_real(monkeypatch, tmp_path),
    ):
        pass


def test_startup_without_thresholds_file_warns_and_keeps_legacy(monkeypatch, tmp_path, caplog):
    with _start_real(monkeypatch, tmp_path) as c:
        assert c.app.state.thresholds is None and _post(c)["threshold"] == 0.5
    assert "has no thresholds.json" in caplog.text
    assert "VOXGUARD_THRESHOLD is the default" in caplog.text


def test_stub_never_loads_thresholds(monkeypatch) -> None:
    monkeypatch.setenv("VOXGUARD_DETECTOR", "stub")
    with TestClient(create_app(settings=Settings(db_url="sqlite://"))) as c:
        assert c.app.state.thresholds is None
