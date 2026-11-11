from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from voxguard.models import build_model  # noqa: E402
from voxguard.models.aasist import AASISTBackend, GraphPool, HeteroGraphAttention  # noqa: E402
from voxguard.models.ssl_aasist import SSLAASIST  # noqa: E402

# Tiny random wav2vec2 so tests never download XLS-R.
TINY = dict(
    hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
    conv_dim=(32,) * 7, feat_extract_norm="layer", do_stable_layer_norm=True,
)  # fmt: skip


def tiny(**kw: object) -> SSLAASIST:
    return build_model("ssl_aasist", pretrained=None, ssl_config_overrides=TINY, **kw)


@pytest.mark.parametrize("n_samples", [16_000, 64_600])
def test_forward_shapes_and_scores(n_samples: int) -> None:
    model = tiny().eval()
    x = torch.randn(2, n_samples)
    assert model(x).shape == (2, 2)
    s = model.fake_score(x)
    assert ((s >= 0) & (s <= 1)).all()


def test_finetune_modes() -> None:
    def front_trainable(m: SSLAASIST) -> int:
        return sum(p.numel() for p in m.frontend_parameters())

    full = front_trainable(tiny(finetune="full"))
    last1 = front_trainable(tiny(finetune="last_n", last_n=1))
    assert full > last1 > 0
    assert front_trainable(tiny(finetune="frozen")) == 0
    assert tiny(finetune="frozen").backend_parameters()  # back-end still trains
    with pytest.raises(ValueError):
        tiny(finetune="last_n", last_n=99)
    with pytest.raises(ValueError):
        tiny(finetune="partial")


def test_frozen_frontend_gets_no_gradient() -> None:
    model = tiny(finetune="frozen")
    loss = torch.nn.functional.cross_entropy(model(torch.randn(2, 16_000)), torch.tensor([0, 1]))
    loss.backward()
    assert all(p.grad is None for p in model.frontend.parameters())
    assert any(p.grad is not None for p in model.backend_parameters())


def test_backend_standalone_and_node_check() -> None:
    backend = AASISTBackend(in_channels=64, n_spectral_nodes=42).eval()
    assert backend(torch.randn(3, 64, 42, 20)).shape == (3, 2)
    with pytest.raises(ValueError, match="spectral nodes"):
        backend(torch.randn(1, 64, 40, 20))


def test_graph_pool_keeps_top_ratio() -> None:
    pool = GraphPool(0.5, 8).eval()
    assert pool(torch.randn(2, 10, 8)).shape == (2, 5, 8)
    assert pool(torch.randn(2, 1, 8)).shape == (2, 1, 8)  # never empty


def test_hetero_attention_shapes() -> None:
    layer = HeteroGraphAttention(16, 8, temperature=100.0).eval()
    t, s, m = layer(torch.randn(2, 5, 16), torch.randn(2, 7, 16), torch.randn(2, 1, 16))
    assert t.shape == (2, 5, 8) and s.shape == (2, 7, 8) and m.shape == (2, 1, 8)


def test_trains_through_train_loop(tmp_path: Path) -> None:
    pytest.importorskip("soundfile")
    from voxguard.data.synthetic import write_synthetic
    from voxguard.train import Config, DataConfig, TrainConfig, train

    m = write_synthetic(tmp_path / "data", n=8, seed=0)
    cfg = Config(
        model={"name": "ssl_aasist", "pretrained": None, "finetune": "last_n", "last_n": 1,
               "ssl_config_overrides": TINY},
        data=DataConfig(train_manifests=[str(m["train"])], dev_manifests=[str(m["dev"])],
                        roots=[str(tmp_path / "data")], num_samples=16_000),
        train=TrainConfig(epochs=1, batch_size=4, lr=1e-3, frontend_lr=1e-4,
                          class_weights=[0.5, 0.5], out_dir=str(tmp_path / "run"), device="cpu"),
    )  # fmt: skip
    res = train(cfg)
    assert (tmp_path / "run" / "last.pt").exists() and res["history"][0]["train_loss"] > 0


def test_config_file_loads() -> None:
    from voxguard.train import load_config

    cfg = load_config(Path(__file__).parents[1] / "configs" / "ssl_aasist.yaml")
    assert cfg.model["name"] == "ssl_aasist" and cfg.model["finetune"] == "full"
    assert cfg.train.batch_size == 14 and cfg.train.lr == 1e-6


def test_checkpoint_loader_does_not_redownload_xlsr(tmp_path: Path, monkeypatch) -> None:
    import transformers

    from voxguard.models import load_from_checkpoint

    model = tiny(finetune="last_n", last_n=1)
    params = {"name": "ssl_aasist", "pretrained": "facebook/wav2vec2-xls-r-300m",
              "finetune": "last_n", "last_n": 1}  # fmt: skip
    torch.save(
        {"model": model.state_dict(), "epoch": 0, "config": {"model": params}}, tmp_path / "c.pt"
    )
    fetched = []

    def fake_config(name: str):
        fetched.append(name)
        return transformers.Wav2Vec2Config(**TINY)

    monkeypatch.setattr(transformers.Wav2Vec2Config, "from_pretrained", staticmethod(fake_config))
    monkeypatch.setattr(
        transformers.Wav2Vec2Model, "from_pretrained",
        staticmethod(lambda *a, **k: pytest.fail("weights must not be downloaded")),
    )  # fmt: skip
    loaded, info = load_from_checkpoint(tmp_path / "c.pt")
    assert fetched == ["facebook/wav2vec2-xls-r-300m"] and info["name"] == "ssl_aasist"
    x = torch.randn(1, 16_000)
    assert torch.allclose(loaded(x), model.eval()(x))


def test_frontend_never_masks_time_steps() -> None:
    # Even if a config asks for heavy time masking, fine-tuning must not mask (reference:
    # mask=False). With every dropout off, train and eval mode must then give the same features.
    off = dict(hidden_dropout=0.0, attention_dropout=0.0, activation_dropout=0.0,
               feat_proj_dropout=0.0, layerdrop=0.0)  # fmt: skip
    overrides = {**TINY, **off, "mask_time_prob": 0.9}
    model = build_model("ssl_aasist", pretrained=None, ssl_config_overrides=overrides)
    assert model.frontend.model.config.apply_spec_augment is False
    torch.manual_seed(0)
    x = torch.randn(2, 16_000)
    train_feats = model.frontend.train()(x)
    eval_feats = model.frontend.eval()(x)
    assert torch.allclose(train_feats, eval_feats, atol=1e-5)
