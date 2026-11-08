"""RawNet2 baseline: raw waveform -> sinc filterbank -> residual blocks with FMS -> GRU.

Architecture follows Tak et al., "End-to-end anti-spoofing with RawNet2" (ICASSP 2021), as
used for the ASVspoof 2021 LA baseline. Implemented from the published description (no code
copied). Output is 2-class logits with class 1 = fake/spoof, matching ``fake_score``.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def _to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10 ** (mel / 2595.0) - 1.0)


class SincConv(nn.Module):
    """Band-pass sinc filterbank with mel-spaced initial cut-offs (SincNet, Ravanelli 2018).

    Filters are fixed by default (as in the RawNet2 baseline); ``learnable=True`` trains the
    cut-off frequencies.
    """

    def __init__(
        self,
        out_channels: int = 20,
        kernel_size: int = 1024,
        sample_rate: int = 16_000,
        min_low_hz: float = 50.0,
        min_band_hz: float = 50.0,
        learnable: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size  # odd
        self.sample_rate = sample_rate
        self.min_low_hz, self.min_band_hz = min_low_hz, min_band_hz
        hz = _to_hz(
            np.linspace(
                _to_mel(30.0),
                _to_mel(sample_rate / 2 - (min_low_hz + min_band_hz)),
                out_channels + 1,
            )
        )
        low = torch.tensor(hz[:-1], dtype=torch.float32).view(-1, 1)
        band = torch.tensor(np.diff(hz), dtype=torch.float32).view(-1, 1)
        if learnable:
            self.low_hz, self.band_hz = nn.Parameter(low), nn.Parameter(band)
        else:
            self.register_buffer("low_hz", low)
            self.register_buffer("band_hz", band)
        half = self.kernel_size // 2
        self.register_buffer(
            "window", 0.54 - 0.46 * torch.cos(2 * math.pi * torch.arange(half) / self.kernel_size)
        )
        self.register_buffer("n", 2 * math.pi * torch.arange(-half, 0).view(1, -1) / sample_rate)

    def forward(self, x: Tensor) -> Tensor:  # x: (B, T) -> (B, C, T - K + 1)
        low = self.min_low_hz + torch.abs(self.low_hz)
        high = torch.clamp(
            low + self.min_band_hz + torch.abs(self.band_hz), self.min_low_hz, self.sample_rate / 2
        )
        band = (high - low)[:, 0]
        left = (torch.sin(high @ self.n) - torch.sin(low @ self.n)) / (self.n / 2) * self.window
        centre = 2 * band.view(-1, 1)
        filt = torch.cat([left, centre, left.flip(1)], dim=1) / (2 * band[:, None])
        return F.conv1d(x.unsqueeze(1), filt.view(-1, 1, self.kernel_size))


class FMS(nn.Module):
    """Filter-wise feature map scaling: per-channel sigmoid gate, applied as x * s + s."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.fc = nn.Linear(channels, channels)

    def forward(self, x: Tensor) -> Tensor:
        s = torch.sigmoid(self.fc(x.mean(dim=-1))).unsqueeze(-1)
        return x * s + s


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, first: bool = False) -> None:
        super().__init__()
        self.bn1 = None if first else nn.BatchNorm1d(in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.shortcut = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.pool = nn.MaxPool1d(3)
        self.fms = FMS(out_ch)

    def forward(self, x: Tensor) -> Tensor:
        out = x if self.bn1 is None else F.selu(self.bn1(x))
        out = self.conv2(F.selu(self.bn2(self.conv1(out))))
        out = out + (x if self.shortcut is None else self.shortcut(x))
        return self.fms(self.pool(out))


class RawNet2(nn.Module):
    """``forward(waveform[B, 64600]) -> logits[B, 2]``; use :meth:`fake_score` for P(fake)."""

    def __init__(
        self,
        sinc_channels: int = 20,
        sinc_kernel: int = 1024,
        res_channels: int = 128,
        gru_hidden: int = 1024,
        gru_layers: int = 3,
        sinc_learnable: bool = False,
    ) -> None:
        super().__init__()
        self.sinc = SincConv(sinc_channels, sinc_kernel, learnable=sinc_learnable)
        self.bn0 = nn.BatchNorm1d(sinc_channels)
        widths = [(sinc_channels, sinc_channels)] * 2 + [(sinc_channels, res_channels)]
        widths += [(res_channels, res_channels)] * 3
        self.blocks = nn.Sequential(
            *[ResBlock(i, o, first=(k == 0)) for k, (i, o) in enumerate(widths)]
        )
        self.bn_gru = nn.BatchNorm1d(res_channels)
        self.gru = nn.GRU(res_channels, gru_hidden, gru_layers, batch_first=True)
        self.fc1 = nn.Linear(gru_hidden, gru_hidden)
        self.fc2 = nn.Linear(gru_hidden, 2)

    def forward(self, x: Tensor) -> Tensor:
        x = F.selu(self.bn0(F.max_pool1d(self.sinc(x), 3)))
        x = F.selu(self.bn_gru(self.blocks(x)))
        out, _ = self.gru(x.transpose(1, 2))
        return self.fc2(self.fc1(out[:, -1]))

    @torch.no_grad()
    def fake_score(self, x: Tensor) -> Tensor:
        return torch.softmax(self.forward(x), dim=-1)[:, 1]
