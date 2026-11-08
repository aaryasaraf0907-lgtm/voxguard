"""SSL-AASIST: XLS-R (wav2vec 2.0) front-end + AASIST back-end (Tak et al., Odyssey 2022).

Front-end weights: ``facebook/wav2vec2-xls-r-300m`` (Apache-2.0, ~1.27 GB, downloaded by
Transformers on first use). ``finetune`` selects what trains:

- ``full``: the whole SSL model (paper setting; needs the most GPU memory)
- ``last_n``: only the top ``last_n`` transformer layers (+ final layer norm); the fallback
  for free GPUs (CLAUDE.md section 9)
- ``frozen``: SSL model fixed, back-end only
"""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from voxguard.models.aasist import AASISTBackend, build_encoder

Finetune = Literal["full", "last_n", "frozen"]
DEFAULT_SSL = "facebook/wav2vec2-xls-r-300m"


class SSLFrontEnd(nn.Module):
    """Waveform (B, T) at 16 kHz -> frame features (B, frames, hidden) at ~50 Hz."""

    def __init__(
        self,
        pretrained: str | None = DEFAULT_SSL,
        finetune: Finetune = "full",
        last_n: int = 4,
        gradient_checkpointing: bool = False,
        config_overrides: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        from transformers import Wav2Vec2Config, Wav2Vec2Model

        if pretrained is None:  # random init, for tests (no download)
            self.model = Wav2Vec2Model(Wav2Vec2Config(**(config_overrides or {})))
        else:
            self.model = Wav2Vec2Model.from_pretrained(pretrained)
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.hidden_size = self.model.config.hidden_size
        self.set_finetune(finetune, last_n)

    def set_finetune(self, mode: Finetune, last_n: int = 4) -> None:
        if mode not in ("full", "last_n", "frozen"):
            raise ValueError(f"finetune must be full, last_n or frozen, got {mode!r}")
        for p in self.model.parameters():
            p.requires_grad = mode == "full"
        if mode == "last_n":
            layers = self.model.encoder.layers
            if not 1 <= last_n <= len(layers):
                raise ValueError(f"last_n must be in [1, {len(layers)}], got {last_n}")
            for module in [*layers[-last_n:], self.model.encoder.layer_norm]:
                for p in module.parameters():
                    p.requires_grad = True

    def forward(self, x: Tensor) -> Tensor:
        # per-utterance zero mean / unit variance, as the XLS-R feature extractor does
        x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-7)
        return self.model(x).last_hidden_state


class SSLAASIST(nn.Module):
    """``forward(waveform[B, T]) -> logits[B, 2]`` (class 1 = fake)."""

    def __init__(
        self,
        pretrained: str | None = DEFAULT_SSL,
        finetune: Finetune = "full",
        last_n: int = 4,
        gradient_checkpointing: bool = False,
        proj_dim: int = 128,
        encoder_channels: list[list[int]] | None = None,
        gat_dims: list[int] | None = None,
        pool_ratios: list[float] | None = None,
        temperatures: list[float] | None = None,
        ssl_config_overrides: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.frontend = SSLFrontEnd(
            pretrained, finetune, last_n, gradient_checkpointing, ssl_config_overrides
        )
        channels = [tuple(c) for c in (encoder_channels or [[1, 32], [32, 32], [32, 64], [64, 64]])]
        self.proj = nn.Linear(self.frontend.hidden_size, proj_dim)
        self.first_bn = nn.BatchNorm2d(1)
        self.encoder = build_encoder(channels)  # no pooling: keeps proj_dim // 3 spectral nodes
        self.backend = AASISTBackend(
            in_channels=channels[-1][1],
            n_spectral_nodes=proj_dim // 3,
            gat_dims=tuple(gat_dims or (64, 32)),
            pool_ratios=tuple(pool_ratios or (0.5, 0.5, 0.5, 0.5)),
            temperatures=tuple(temperatures or (2.0, 2.0, 100.0, 100.0)),
        )

    def frontend_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.frontend.parameters() if p.requires_grad]

    def backend_parameters(self) -> list[nn.Parameter]:
        front = {id(p) for p in self.frontend.parameters()}
        return [p for p in self.parameters() if p.requires_grad and id(p) not in front]

    def forward(self, x: Tensor) -> Tensor:
        feats = self.proj(self.frontend(x))  # (B, frames, proj_dim)
        x = F.max_pool2d(feats.transpose(1, 2).unsqueeze(1), (3, 3))  # (B, 1, proj/3, frames/3)
        x = F.selu(self.first_bn(x))
        return self.backend(self.encoder(x))

    @torch.no_grad()
    def fake_score(self, x: Tensor) -> Tensor:
        return torch.softmax(self.forward(x), dim=-1)[:, 1]
