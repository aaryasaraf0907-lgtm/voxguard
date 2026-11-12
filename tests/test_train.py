import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("soundfile")

from voxguard.data.synthetic import write_synthetic  # noqa: E402
from voxguard.train import (  # noqa: E402
    Config,
    DataConfig,
    TrainConfig,
    _make_loaders,
    load_config,
    pick_device,
    seed_epoch,
    train,
    train_step,
)


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
    assert all(r["skipped_batches"] == 0 for r in res["history"])
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
    assert cfg.model["name"] == "rawnet2" and cfg.train.class_weights == [0.9, 0.1]
    bad = tmp_path / "bad.yaml"
    bad.write_text("train:\n  epochz: 3\n")
    with pytest.raises(ValueError, match="epochz"):
        load_config(bad)


def test_pick_device() -> None:
    assert pick_device("cpu").type == "cpu"
    assert pick_device("auto").type in {"cuda", "mps", "cpu"}


def _order(loader) -> list[float]:
    return [float(x[:, 0].sum()) for x, _ in loader]


def test_shuffle_order_depends_only_on_seed_and_epoch(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, 1)
    a, _ = _make_loaders(cfg)
    b, _ = _make_loaders(cfg)
    torch.manual_seed(99)  # global RNG state must not matter (it differs after a resume)
    seed_epoch(a, 1234, 3)
    first = _order(a)
    torch.manual_seed(7)
    seed_epoch(b, 1234, 3)
    assert _order(b) == first
    seed_epoch(b, 1234, 4)
    assert _order(b) != first


def test_dev_max_samples_limits_dev_set(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, 1)
    full = len(_make_loaders(cfg)[1].dataset)
    cfg.data.dev_max_samples = 4
    assert len(_make_loaders(cfg)[1].dataset) == 4 < full


def test_train_samples_per_epoch_caps_epoch_and_reshuffles(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, 1)
    cfg.data.train_samples_per_epoch = 4
    train_dl, _ = _make_loaders(cfg)
    assert len(train_dl.dataset) > 4
    seed_epoch(train_dl, 1234, 0)
    first = [float(x[:, 0].sum()) for x, _ in train_dl]
    assert sum(len(y) for _, y in train_dl) == 4
    seed_epoch(train_dl, 1234, 1)
    assert [float(x[:, 0].sum()) for x, _ in train_dl] != first  # a different subset per epoch
    cfg.data.train_samples_per_epoch = 10_000  # larger than the set: plain shuffle, all clips
    assert sum(len(y) for _, y in _make_loaders(cfg)[0]) == len(train_dl.dataset)


def test_best_checkpoint_is_weights_only(tmp_path: Path) -> None:
    train(_cfg(tmp_path, 1))
    best = torch.load(tmp_path / "run" / "best.pt", weights_only=False)
    last = torch.load(tmp_path / "run" / "last.pt", weights_only=False)
    assert "opt" not in best and "scaler" not in best
    assert {"model", "config", "epoch", "best_eer"} <= set(best) and "opt" in last
    from voxguard.models import load_from_checkpoint

    _, info = load_from_checkpoint(tmp_path / "run" / "best.pt", "cpu")
    assert info["epoch"] == 0


@pytest.mark.parametrize("name", ["ssl_aasist.yaml", "rawnet2.yaml"])
def test_config_upweights_bonafide(name: str) -> None:
    # labels are 1 = fake; ASVspoof is ~90 % fake, so the minority bona fide class gets the
    # larger weight (the reference code's [0.1, 0.9] assumes bona fide = 1).
    cfg = load_config(Path(__file__).parents[1] / "configs" / name)
    assert cfg.train.class_weights[0] > cfg.train.class_weights[1]
    assert TrainConfig().class_weights[0] > TrainConfig().class_weights[1]


class _BNNet(torch.nn.Module):
    """Linear -> BatchNorm1d -> 2 logits; enough to see running stats move."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(4, 4)
        self.bn = torch.nn.BatchNorm1d(4)
        self.out = torch.nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.bn(self.fc(x)))


def _step_parts():
    torch.manual_seed(0)
    model = _BNNet().train()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    return model, opt, scaler, torch.nn.CrossEntropyLoss()


def test_train_step_trains_on_a_finite_batch() -> None:
    model, opt, scaler, loss_fn = _step_parts()
    before = model.bn.running_mean.clone()
    loss = train_step(
        model, model, torch.randn(8, 4), torch.randint(0, 2, (8,)), loss_fn, opt, scaler, False
    )
    assert loss is not None and math.isfinite(loss)
    assert not torch.equal(model.bn.running_mean, before)


def test_train_step_skips_non_finite_input_without_forward() -> None:
    model, opt, scaler, loss_fn = _step_parts()
    x = torch.randn(8, 4)
    x[3, 1] = float("inf")
    before = {k: v.clone() for k, v in model.state_dict().items()}
    assert (
        train_step(model, model, x, torch.zeros(8, dtype=torch.long), loss_fn, opt, scaler, False)
        is None
    )
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_train_step_nan_loss_restores_batchnorm_stats() -> None:
    # finite input, but a NaN weight makes the forward pass (and the BN stats update) NaN
    model, opt, scaler, loss_fn = _step_parts()
    with torch.no_grad():
        model.fc.weight[0, 0] = float("nan")
    stats = {k: v.clone() for k, v in model.named_buffers()}
    y = torch.randint(0, 2, (8,))
    assert train_step(model, model, torch.randn(8, 4), y, loss_fn, opt, scaler, False) is None
    for k, v in model.named_buffers():
        assert torch.equal(v, stats[k]), k  # still finite: eval-mode scoring is not poisoned
