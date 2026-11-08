"""AASIST back-end: spectro-temporal graph attention over a 2-D encoder feature map.

Follows Jung et al., "AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph
Attention Networks" (ICASSP 2022) and its use in SSL-AASIST (Tak et al., Odyssey 2022).
Written from the papers; the reference implementations (clovaai/aasist and
TakHemlata/SSL_Anti-spoofing, both MIT) were used only to check structure and defaults.

Input to :class:`AASISTBackend` is ``e`` with shape (B, C, F, T); output is 2-class logits
with class 1 = fake/spoof.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Residual2d(nn.Module):
    """2-D residual block. ``pool`` is the max-pool kernel; (1, 1) disables pooling."""

    def __init__(
        self, in_ch: int, out_ch: int, first: bool = False, pool: tuple[int, int] = (1, 1)
    ) -> None:
        super().__init__()
        self.bn1 = None if first else nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, (2, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, (2, 3), padding=(0, 1))
        self.shortcut = (
            nn.Conv2d(in_ch, out_ch, (1, 3), padding=(0, 1)) if in_ch != out_ch else None
        )
        self.pool = nn.MaxPool2d(pool) if pool != (1, 1) else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        out = x if self.bn1 is None else F.selu(self.bn1(x))
        out = self.conv2(F.selu(self.bn2(self.conv1(out))))
        return self.pool(out + (x if self.shortcut is None else self.shortcut(x)))


def build_encoder(channels: list[tuple[int, int]], pool: tuple[int, int] = (1, 1)) -> nn.Sequential:
    return nn.Sequential(
        *[Residual2d(i, o, first=(k == 0), pool=pool) for k, (i, o) in enumerate(channels)]
    )


def _pairwise(x: Tensor) -> Tensor:
    """(B, N, D) -> (B, N, N, D) element-wise products of every node pair."""
    return x.unsqueeze(2) * x.unsqueeze(1)


class GraphAttention(nn.Module):
    """Graph attention over a fully connected graph of N nodes (one node type)."""

    def __init__(self, in_dim: int, out_dim: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = nn.Parameter(torch.empty(out_dim, 1))
        nn.init.xavier_normal_(self.att_weight)
        self.proj_att = nn.Linear(in_dim, out_dim)
        self.proj_self = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.drop = nn.Dropout(0.2)
        self.temperature = temperature

    def forward(self, x: Tensor) -> Tensor:  # (B, N, in) -> (B, N, out)
        x = self.drop(x)
        scores = torch.tanh(self.att_proj(_pairwise(x))) @ self.att_weight  # (B, N, N, 1)
        att = torch.softmax(scores.squeeze(-1) / self.temperature, dim=-1)
        out = self.proj_att(att @ x) + self.proj_self(x)
        return F.selu(self.bn(out.transpose(1, 2)).transpose(1, 2))


class HeteroGraphAttention(nn.Module):
    """Heterogeneous graph attention over two node types plus a master node (HS-GAL)."""

    def __init__(self, in_dim: int, out_dim: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_proj_m = nn.Linear(in_dim, out_dim)
        self.w11, self.w22, self.w12, self.wm = (
            nn.Parameter(nn.init.xavier_normal_(torch.empty(out_dim, 1))) for _ in range(4)
        )
        self.proj_att = nn.Linear(in_dim, out_dim)
        self.proj_self = nn.Linear(in_dim, out_dim)
        self.proj_att_m = nn.Linear(in_dim, out_dim)
        self.proj_self_m = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.drop = nn.Dropout(0.2)
        self.temperature = temperature

    def forward(self, x1: Tensor, x2: Tensor, master: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        n1 = x1.size(1)
        x = self.drop(torch.cat([self.proj_type1(x1), self.proj_type2(x2)], dim=1))

        # master node attends over every node
        m_scores = torch.tanh(self.att_proj_m(x * master)) @ self.wm  # (B, N, 1)
        m_att = torch.softmax(m_scores / self.temperature, dim=1)
        master = self.proj_att_m((m_att * x).sum(dim=1, keepdim=True)) + self.proj_self_m(master)

        # node attention with separate weights for type1-type1, type2-type2 and cross pairs
        a = torch.tanh(self.att_proj(_pairwise(x)))  # (B, N, N, out)
        s11, s22, s12 = ((a @ w).squeeze(-1) for w in (self.w11, self.w22, self.w12))
        type1 = torch.zeros(x.size(1), dtype=torch.bool, device=x.device)
        type1[:n1] = True
        same1 = type1[:, None] & type1[None, :]
        same2 = ~type1[:, None] & ~type1[None, :]
        scores = torch.where(same1, s11, torch.where(same2, s22, s12))
        att = torch.softmax(scores / self.temperature, dim=-1)
        out = self.proj_att(att @ x) + self.proj_self(x)
        out = F.selu(self.bn(out.transpose(1, 2)).transpose(1, 2))
        return out[:, :n1], out[:, n1:], master


class GraphPool(nn.Module):
    """Keep the top ``ratio`` of nodes by a learned score (at least one node)."""

    def __init__(self, ratio: float, dim: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.ratio = ratio
        self.proj = nn.Linear(dim, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: Tensor) -> Tensor:
        scores = torch.sigmoid(self.proj(self.drop(h)))  # (B, N, 1)
        k = max(int(h.size(1) * self.ratio), 1)
        idx = scores.topk(k, dim=1).indices.expand(-1, -1, h.size(-1))
        return torch.gather(h * scores, 1, idx)


class AASISTBackend(nn.Module):
    """Spectral and temporal graphs, two HS-GAL branches with master nodes, max/avg readout."""

    def __init__(
        self,
        in_channels: int = 64,
        n_spectral_nodes: int = 42,
        gat_dims: tuple[int, int] = (64, 32),
        pool_ratios: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 0.5),
        temperatures: tuple[float, float, float, float] = (2.0, 2.0, 100.0, 100.0),
    ) -> None:
        super().__init__()
        d0, d1 = gat_dims
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 128, 1),
            nn.SELU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, in_channels, 1),
        )
        self.pos_s = nn.Parameter(torch.randn(1, n_spectral_nodes, in_channels))
        self.master1 = nn.Parameter(torch.randn(1, 1, d0))
        self.master2 = nn.Parameter(torch.randn(1, 1, d0))
        self.gat_s = GraphAttention(in_channels, d0, temperatures[0])
        self.gat_t = GraphAttention(in_channels, d0, temperatures[1])
        self.st11 = HeteroGraphAttention(d0, d1, temperatures[2])
        self.st12 = HeteroGraphAttention(d1, d1, temperatures[2])
        self.st21 = HeteroGraphAttention(d0, d1, temperatures[3])
        self.st22 = HeteroGraphAttention(d1, d1, temperatures[3])
        self.pool_s = GraphPool(pool_ratios[0], d0)
        self.pool_t = GraphPool(pool_ratios[1], d0)
        self.pool_hs1 = GraphPool(pool_ratios[2], d1)
        self.pool_ht1 = GraphPool(pool_ratios[2], d1)
        self.pool_hs2 = GraphPool(pool_ratios[3], d1)
        self.pool_ht2 = GraphPool(pool_ratios[3], d1)
        self.drop_way = nn.Dropout(0.2)
        self.drop = nn.Dropout(0.5)
        self.out = nn.Linear(5 * d1, 2)

    def _branch(
        self, t: Tensor, s: Tensor, master: Tensor, first: nn.Module, second: nn.Module,
        pool_t: nn.Module, pool_s: nn.Module,
    ) -> tuple[Tensor, Tensor, Tensor]:  # fmt: skip
        t1, s1, m1 = first(t, s, master)
        t1, s1 = pool_t(t1), pool_s(s1)
        t2, s2, m2 = second(t1, s1, m1)
        return t1 + t2, s1 + s2, m1 + m2

    def forward(self, e: Tensor) -> Tensor:  # e: (B, C, F, T)
        if e.size(2) != self.pos_s.size(1):
            raise ValueError(f"expected {self.pos_s.size(1)} spectral nodes, got {e.size(2)}")
        w = self.attention(e)
        spectral = (e * torch.softmax(w, dim=-1)).sum(dim=-1).transpose(1, 2) + self.pos_s
        temporal = (e * torch.softmax(w, dim=-2)).sum(dim=-2).transpose(1, 2)
        s = self.pool_s(self.gat_s(spectral))  # (B, Fk, d0)
        t = self.pool_t(self.gat_t(temporal))  # (B, Tk, d0)
        b = e.size(0)
        t1, s1, m1 = self._branch(
            t, s, self.master1.expand(b, -1, -1), self.st11, self.st12, self.pool_ht1, self.pool_hs1
        )
        t2, s2, m2 = self._branch(
            t, s, self.master2.expand(b, -1, -1), self.st21, self.st22, self.pool_ht2, self.pool_hs2
        )
        t = torch.maximum(self.drop_way(t1), self.drop_way(t2))
        s = torch.maximum(self.drop_way(s1), self.drop_way(s2))
        m = torch.maximum(self.drop_way(m1), self.drop_way(m2))
        hidden = torch.cat(
            [t.abs().amax(dim=1), t.mean(dim=1), s.abs().amax(dim=1), s.mean(dim=1), m.squeeze(1)],
            dim=1,
        )
        return self.out(self.drop(hidden))
