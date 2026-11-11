import pytest

torch = pytest.importorskip("torch")

from voxguard.models import build_model  # noqa: E402
from voxguard.models.rawnet2 import FMS, RawNet2, SincConv  # noqa: E402

SMALL = dict(gru_hidden=32, gru_layers=1)


def test_sinc_filters_are_bandpass() -> None:
    sinc = SincConv(out_channels=4, kernel_size=256)
    assert sinc.kernel_size == 257
    t = torch.arange(16_000) / 16_000
    low = torch.sin(2 * torch.pi * 150 * t)[None]
    high = torch.sin(2 * torch.pi * 6000 * t)[None]
    e_low, e_high = sinc(low).pow(2).mean(dim=-1)[0], sinc(high).pow(2).mean(dim=-1)[0]
    assert e_low.argmax() < e_high.argmax()  # filters are ordered by centre frequency
    assert sinc(low).shape == (1, 4, 16_000 - 257 + 1)


def test_fms_preserves_shape() -> None:
    x = torch.randn(2, 8, 50)
    assert FMS(8)(x).shape == x.shape


def test_forward_shape_and_scores() -> None:
    model = RawNet2(**SMALL).eval()
    x = torch.randn(2, 64_600)
    logits = model(x)
    assert logits.shape == (2, 2) and torch.isfinite(logits).all()
    s = model.fake_score(x)
    assert s.shape == (2,) and ((s >= 0) & (s <= 1)).all()


def test_gradients_flow_and_loss_decreases() -> None:
    torch.manual_seed(0)
    model = RawNet2(**SMALL)
    x = torch.randn(4, 64_600)
    y = torch.tensor([0, 1, 0, 1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = None
    for _ in range(15):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert loss.item() < first


def test_sinc_fixed_by_default_learnable_on_request() -> None:
    assert not any(p.requires_grad for p in RawNet2(**SMALL).sinc.parameters())
    assert any(p.requires_grad for p in RawNet2(sinc_learnable=True, **SMALL).sinc.parameters())


def test_registry() -> None:
    assert isinstance(build_model("rawnet2", **SMALL), RawNet2)
    with pytest.raises(ValueError):
        build_model("nope")
