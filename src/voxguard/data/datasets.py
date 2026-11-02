"""PyTorch datasets over manifests (needs the ``ml`` extra: torch + soundfile)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from voxguard.data.manifests import ManifestRow, read_manifest

NUM_SAMPLES = 64_600  # ~4 s at 16 kHz
SAMPLE_RATE = 16_000


def fix_length(
    x: np.ndarray, num_samples: int = NUM_SAMPLES, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Crop (random start if ``rng`` else from 0) or repeat-pad ``x`` to ``num_samples``."""
    if len(x) == 0:
        return np.zeros(num_samples, dtype=np.float32)
    if len(x) >= num_samples:
        start = int(rng.integers(0, len(x) - num_samples + 1)) if rng is not None else 0
        return x[start : start + num_samples].astype(np.float32, copy=False)
    reps = int(np.ceil(num_samples / len(x)))
    return np.tile(x, reps)[:num_samples].astype(np.float32, copy=False)


def load_rows(
    manifests: Sequence[Path],
    codecs: Sequence[str] | None = None,
    max_samples: int | None = None,
    seed: int = 0,
) -> list[ManifestRow]:
    """Concatenate manifests, optionally filter by codec and take a seeded random subset."""
    rows = [r for m in manifests for r in read_manifest(m)]
    if codecs is not None:
        rows = [r for r in rows if r.codec in codecs]
    if max_samples is not None and max_samples < len(rows):
        keep = np.random.default_rng(seed).choice(len(rows), size=max_samples, replace=False)
        rows = [rows[i] for i in sorted(keep)]
    return rows


try:  # torch is an optional (ml extra) dependency; helpers above work without it
    from torch.utils.data import Dataset as _Base
except ImportError:  # pragma: no cover - exercised only when torch is absent
    _Base = object  # type: ignore[assignment,misc]


class ManifestDataset(_Base):  # type: ignore[misc,valid-type]
    """Yields ``(waveform[num_samples] float32, label int64)``.

    Paths in a manifest are resolved against ``roots`` in order (clean audio lives under the
    raw data root, degraded audio under the processed root). Train mode takes a random crop
    seeded by ``(seed, epoch, index)``, so runs are reproducible; eval mode takes the start.
    """

    def __init__(
        self,
        manifests: Sequence[Path],
        roots: Sequence[Path],
        num_samples: int = NUM_SAMPLES,
        train: bool = False,
        codecs: Sequence[str] | None = None,
        max_samples: int | None = None,
        seed: int = 0,
    ) -> None:
        if _Base is object:
            raise ImportError("ManifestDataset needs torch: install with `uv sync --extra ml`")
        self.rows = load_rows(manifests, codecs, max_samples, seed)
        self.roots = [Path(r) for r in roots]
        self.num_samples = num_samples
        self.train = train
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    def resolve(self, rel: str) -> Path:
        for root in self.roots:
            if (root / rel).is_file():
                return root / rel
        raise FileNotFoundError(f"{rel} not found under any of {[str(r) for r in self.roots]}")

    def __getitem__(self, idx: int):
        import soundfile as sf
        import torch

        row = self.rows[idx]
        data, sr = sf.read(self.resolve(row.path), dtype="float32", always_2d=True)
        if sr != SAMPLE_RATE:
            raise ValueError(f"{row.path}: expected {SAMPLE_RATE} Hz, got {sr}")
        rng = np.random.default_rng([self.seed, self.epoch, idx]) if self.train else None
        x = fix_length(data.mean(axis=1), self.num_samples, rng)
        return torch.from_numpy(x), torch.tensor(row.label, dtype=torch.int64)
