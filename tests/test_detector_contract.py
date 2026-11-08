"""The section-7 contract: every Detector must pass these, the stub and the real one."""

from pathlib import Path

import numpy as np
import pytest

from voxguard.inference import NUM_SAMPLES, Detector, Prediction, StubDetector, resolve_checkpoint

SR = 16_000


def _wave(seconds: float, seed: int = 0) -> np.ndarray:
    return (np.random.default_rng(seed).standard_normal(int(seconds * SR)) * 0.1).astype(np.float32)


def _tiny_checkpoint(path: Path) -> Path:
    torch = pytest.importorskip("torch")
    from voxguard.models import build_model

    torch.manual_seed(0)
    params = {"gru_hidden": 16, "gru_layers": 1}
    model = build_model("rawnet2", **params)
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": 3,
            "config": {"model": {"name": "rawnet2", **params}},
        },
        path,
    )
    return path


@pytest.fixture(params=["stub", "real"])
def make_detector(request: pytest.FixtureRequest, tmp_path: Path):
    def make(threshold: float | None = 0.5) -> Detector:
        if request.param == "stub":
            return StubDetector(threshold=threshold)
        pytest.importorskip("torchaudio")
        from voxguard.inference import RealDetector

        return RealDetector(str(_tiny_checkpoint(tmp_path / "best.pt")), threshold=threshold)

    return make


def test_prediction_type_and_ranges(make_detector) -> None:
    p = make_detector().predict(_wave(4.0), SR)
    assert isinstance(p, Prediction)
    assert 0.0 <= p.fake_score <= 1.0 and p.latency_ms >= 0.0
    assert p.label in ("real", "fake") and isinstance(p.model_version, str) and p.model_version


def test_same_input_same_score(make_detector) -> None:
    det = make_detector()
    w = _wave(4.0)
    assert det.predict(w, SR).fake_score == det.predict(w.copy(), SR).fake_score


@pytest.mark.parametrize("threshold", [0.0, 0.3, 0.5, 0.9, 1.0])
def test_label_matches_threshold(make_detector, threshold: float) -> None:
    p = make_detector(threshold).predict(_wave(2.0), SR)
    assert p.threshold == threshold
    assert (p.label == "fake") == (p.fake_score >= threshold)


def test_threshold_from_env(make_detector, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOXGUARD_THRESHOLD", "0.37")
    assert make_detector(None).threshold == 0.37


@pytest.mark.parametrize(
    ("seconds", "sr", "stereo"),
    [(0.5, SR, False), (4.0, SR, False), (12.3, SR, False), (3.0, 8_000, False),
     (3.0, 44_100, True), (0.0, SR, False)],
)  # fmt: skip
def test_accepts_any_length_rate_and_channels(make_detector, seconds, sr, stereo) -> None:
    w = _wave(seconds) if sr == SR else np.zeros(int(seconds * sr), np.float32) + 0.01
    if stereo:
        w = np.stack([w, w], axis=1)
    p = make_detector().predict(w, sr)
    assert 0.0 <= p.fake_score <= 1.0


def test_real_detector_specifics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torchaudio")
    from voxguard.inference import RealDetector, get_detector, to_windows

    ckpt = _tiny_checkpoint(tmp_path / "best.pt")
    det = RealDetector(str(tmp_path))  # a directory holding best.pt works too
    assert det.model_version == f"rawnet2:{tmp_path.name}:epoch3"

    # long clip score == mean of its window scores
    long = _wave(10.0, seed=4)  # 2 full windows + a 1.9 s tail
    wins = to_windows(long, SR)
    assert len(wins) == 3 and all(len(w) == NUM_SAMPLES for w in wins)
    per_window = [det.predict(w, SR).fake_score for w in wins]
    assert det.predict(long, SR).fake_score == pytest.approx(np.mean(per_window), abs=1e-6)

    monkeypatch.setenv("VOXGUARD_DETECTOR", "real")
    monkeypatch.setenv("VOXGUARD_MODEL_REPO", str(ckpt))
    assert isinstance(get_detector(), RealDetector)
    with pytest.raises(ValueError, match="threshold"):
        RealDetector(str(ckpt), threshold=1.5)


def test_resolve_checkpoint_local_and_hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = tmp_path / "best.pt"
    f.write_bytes(b"x")
    assert resolve_checkpoint(str(f)) == f
    assert resolve_checkpoint(str(tmp_path)) == f
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(str(tmp_path / "missing" / "deeper.pt"))
    hub = pytest.importorskip("huggingface_hub")
    calls = {}

    def fake_download(repo_id: str, filename: str) -> str:
        calls.update(repo_id=repo_id, filename=filename)
        return str(f)

    monkeypatch.setattr(hub, "hf_hub_download", fake_download)
    assert resolve_checkpoint("someuser/voxguard-rawnet2", "last.pt") == f
    assert calls == {"repo_id": "someuser/voxguard-rawnet2", "filename": "last.pt"}
