"""RealDetector through the API. Tests that need torch build a small, untrained RawNet2
checkpoint with Aarya's ``build_model``; no trained checkpoint is involved."""

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import make_wav
from voxguard.api import detector_loader
from voxguard.api.deps import Settings
from voxguard.api.main import create_app
from voxguard.inference import StubDetector


def _settings(**kw) -> Settings:
    return Settings(db_url="sqlite://", **kw)


def _start(monkeypatch, device: str = "cpu", **env: str):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return TestClient(create_app(settings=_settings(device=device)))


# --- configuration and errors (no torch needed) -------------------------------------------


def test_stub_is_default_and_unchanged(monkeypatch) -> None:
    monkeypatch.delenv("VOXGUARD_DETECTOR", raising=False)
    with TestClient(create_app(settings=_settings())) as c:
        assert isinstance(c.app.state.detector, StubDetector)
        assert c.get("/health").json() == {
            "status": "ok",
            "detector": "stub",
            "model_version": "stub-0",
        }


def test_real_without_model_repo_fails_clearly(monkeypatch) -> None:
    monkeypatch.delenv("VOXGUARD_MODEL_REPO", raising=False)
    with (
        pytest.raises(RuntimeError, match="VOXGUARD_MODEL_REPO"),
        _start(monkeypatch, VOXGUARD_DETECTOR="real"),
    ):
        pass


@pytest.mark.parametrize("value", ["gpu", "tpu", "CUDA:0"])
def test_invalid_device_setting_rejected(monkeypatch, value: str) -> None:
    monkeypatch.setenv("VOXGUARD_DEVICE", value)
    with pytest.raises(ValueError, match="VOXGUARD_DEVICE"):
        Settings.from_env()


def test_device_setting_parsed(monkeypatch) -> None:
    monkeypatch.delenv("VOXGUARD_DEVICE", raising=False)
    assert Settings.from_env().device == "cpu"
    monkeypatch.setenv("VOXGUARD_DEVICE", " Auto ")
    assert Settings.from_env().device == "auto"


def test_no_silent_fallback_to_stub(monkeypatch) -> None:
    def boom(*a, **k):
        raise FileNotFoundError("/models/best.pt")

    import voxguard.inference as inference

    monkeypatch.setattr(inference, "RealDetector", boom)
    with (
        pytest.raises(RuntimeError, match="checkpoint not found"),
        _start(monkeypatch, VOXGUARD_DETECTOR="real", VOXGUARD_MODEL_REPO="/models/best.pt"),
    ):
        pass


def test_hub_errors_are_explained(monkeypatch) -> None:
    class RepositoryNotFoundError(Exception):
        __module__ = "huggingface_hub.errors"

    def boom(*a, **k):
        raise RepositoryNotFoundError("404 Client Error: Repository Not Found")

    import voxguard.inference as inference

    monkeypatch.setattr(inference, "RealDetector", boom)
    with (
        pytest.raises(RuntimeError, match="Hugging Face Hub") as exc,
        _start(monkeypatch, VOXGUARD_DETECTOR="real", VOXGUARD_MODEL_REPO="someone/missing"),
    ):
        pass
    assert "someone/missing" in str(exc.value) and "RepositoryNotFoundError" in str(exc.value)


def test_detector_is_loaded_once_per_process(monkeypatch) -> None:
    built = []

    def factory():
        built.append(1)
        return StubDetector(threshold=0.5)

    monkeypatch.setattr(detector_loader, "get_detector", factory)
    with TestClient(create_app(settings=_settings())) as c:
        for _ in range(3):
            assert (
                c.post("/v1/analyze", files={"file": ("a.wav", make_wav(0.5))}).status_code == 200
            )
    assert built == [1]


# --- with torch: an untrained RawNet2 checkpoint ------------------------------------------


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchaudio")
    from voxguard.models import build_model

    torch.manual_seed(0)
    params = {"gru_hidden": 16, "gru_layers": 1}
    model = build_model("rawnet2", **params)
    path = tmp_path / "best.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": 1,
            "config": {"model": {"name": "rawnet2", **params}},
        },
        path,
    )
    return path


def test_real_detector_serves_analyze_and_prevention(monkeypatch, checkpoint: Path) -> None:
    with _start(
        monkeypatch, VOXGUARD_DETECTOR="real", VOXGUARD_MODEL_REPO=str(checkpoint),
        VOXGUARD_THRESHOLD="0.0",  # untrained weights: force a flag to exercise prevention
    ) as c:  # fmt: skip
        health = c.get("/health").json()
        assert health["detector"] == "real"
        assert health["model_version"] == f"rawnet2:{checkpoint.name}:epoch1"
        r = c.post(
            "/v1/analyze", files={"file": ("a.wav", make_wav(5.0, sr=44_100, channels=2))},
            data={"caller_id": "x"},
        )  # fmt: skip
        body = r.json()
        assert r.status_code == 200 and 0.0 <= body["fake_score"] <= 1.0
        assert body["model_version"] == health["model_version"] and body["threshold"] == 0.0
        assert body["flagged"] is True and body["label"] == "fake"
        assert c.get("/v1/blocklist").json()[0]["caller_id"] == "x"


def test_api_score_equals_direct_detector_score(monkeypatch, checkpoint: Path) -> None:
    """The API adds no preprocessing of its own beyond decoding and downmixing."""
    import io

    import soundfile as sf

    from voxguard.inference import RealDetector

    wav = make_wav(3.0, sr=22_050, channels=2, seed=5)
    x, sr = sf.read(io.BytesIO(wav), dtype="float32", always_2d=True)
    direct = RealDetector(str(checkpoint)).predict(x.mean(axis=1), sr).fake_score
    with _start(monkeypatch, VOXGUARD_DETECTOR="real", VOXGUARD_MODEL_REPO=str(checkpoint)) as c:
        via_api = c.post("/v1/analyze", files={"file": ("a.wav", wav)}).json()["fake_score"]
    assert via_api == pytest.approx(direct, abs=1e-6)


def test_corrupt_checkpoint_fails_clearly(monkeypatch, tmp_path: Path) -> None:
    pytest.importorskip("torch")
    bad = tmp_path / "best.pt"
    bad.write_bytes(b"not a checkpoint")
    with (
        pytest.raises(RuntimeError, match="could not load it"),
        _start(monkeypatch, VOXGUARD_DETECTOR="real", VOXGUARD_MODEL_REPO=str(bad)),
    ):
        pass


def test_checkpoint_without_config_fails_clearly(monkeypatch, tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    path = tmp_path / "best.pt"
    torch.save({"model": {}, "epoch": 1}, path)
    with (
        pytest.raises(RuntimeError, match="KeyError"),
        _start(monkeypatch, VOXGUARD_DETECTOR="real", VOXGUARD_MODEL_REPO=str(path)),
    ):
        pass


def test_missing_checkpoint_path(monkeypatch, tmp_path: Path) -> None:
    pytest.importorskip("torch")
    with (
        pytest.raises(RuntimeError, match="neither a local path nor an HF repo id|not found"),
        _start(
            monkeypatch,
            VOXGUARD_DETECTOR="real",
            VOXGUARD_MODEL_REPO=str(tmp_path / "nope" / "x.pt"),
        ),
    ):
        pass


def test_cuda_requested_without_gpu_fails_clearly(monkeypatch, checkpoint: Path) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with (
        pytest.raises(RuntimeError, match="VOXGUARD_DEVICE=cuda"),
        _start(monkeypatch, "cuda", VOXGUARD_DETECTOR="real", VOXGUARD_MODEL_REPO=str(checkpoint)),
    ):
        pass


def _cuda() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


@pytest.mark.skipif(not _cuda(), reason="no CUDA device")
def test_real_detector_on_gpu(monkeypatch, checkpoint: Path) -> None:
    with _start(
        monkeypatch, "cuda", VOXGUARD_DETECTOR="real", VOXGUARD_MODEL_REPO=str(checkpoint)
    ) as c:
        assert str(c.app.state.detector.device).startswith("cuda")
        r = c.post("/v1/analyze", files={"file": ("a.wav", make_wav(4.0))})
        assert r.status_code == 200 and np.isfinite(r.json()["fake_score"])


@pytest.mark.skipif(not _cuda(), reason="no CUDA device")
def test_gpu_and_cpu_scores_agree(checkpoint: Path) -> None:
    from voxguard.inference import RealDetector

    x = np.random.default_rng(0).standard_normal(64_600).astype(np.float32) * 0.1
    cpu = RealDetector(str(checkpoint), device="cpu").predict(x, 16_000).fake_score
    gpu = RealDetector(str(checkpoint), device="cuda").predict(x, 16_000).fake_score
    assert gpu == pytest.approx(cpu, abs=1e-4)
