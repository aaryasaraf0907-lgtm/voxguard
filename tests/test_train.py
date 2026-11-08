from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("soundfile")

from voxguard.data.synthetic import write_synthetic  # noqa: E402
from voxguard.train import Config, DataConfig, TrainConfig, load_config, pick_device, train  # noqa: E402


def _cfg(tmp_path: Path, epochs: int, out: str = "run") -> Config:
    m = write_synthetic(tmp_path / "data", n=16, seed=0)
    return Config(
        model={"name": "rawnet2", "gru_hidden": 16, "gru_layers": 1},
        data=DataConfig(
            train_manifests=[str(m["train"])],
            dev_manifests=[str(m["dev"])],
            roots=[str(tmp_path / "data")],
            num_samples=16_000,
        ),
        train=TrainConfig(
            epochs=epochs, batch_size=8, lr=1e-3, class_weights=[0.5, 0.5],
            out_dir=str(tmp_path / out), device="cpu",
        ),
    )  # fmt: skip


def test_train_writes_checkpoints_and_metrics(tmp_path: Path) -> None:
    res = train(_cfg(tmp_path, 2))
    out = tmp_path / "run"
    assert (out / "last.pt").exists() and (out / "best.pt").exists()
    assert len((out / "metrics.jsonl").read_text().splitlines()) == 2
    assert len(res["history"]) == 2 and all(r["train_loss"] > 0 for r in res["history"])
    assert not list(out.glob("*.tmp"))


def test_resume_continues_from_last_epoch(tmp_path: Path) -> None:
    train(_cfg(tmp_path, 1))
    res = train(_cfg(tmp_path, 3), resume=True)
    assert [r["epoch"] for r in res["history"]] == [1, 2]
    ckpt = torch.load(tmp_path / "run" / "last.pt", weights_only=False)
    assert ckpt["epoch"] == 2


def test_max_samples_smoke_mode(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, 1)
    cfg.data.max_samples = 8
    train(cfg)
    assert (tmp_path / "run" / "last.pt").exists()


def test_load_config_roundtrip_and_validation(tmp_path: Path) -> None:
    cfg = load_config(Path(__file__).parents[1] / "configs" / "rawnet2.yaml")
    assert cfg.model["name"] == "rawnet2" and cfg.train.class_weights == [0.1, 0.9]
    bad = tmp_path / "bad.yaml"
    bad.write_text("train:\n  epochz: 3\n")
    with pytest.raises(ValueError, match="epochz"):
        load_config(bad)


def test_pick_device() -> None:
    assert pick_device("cpu").type == "cpu"
    assert pick_device("auto").type in {"cuda", "mps", "cpu"}
