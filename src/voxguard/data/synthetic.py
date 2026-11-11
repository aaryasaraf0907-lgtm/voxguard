"""Tiny synthetic dataset for smoke tests (mechanics only: proves nothing about accuracy).

"Fake" clips are harmonic tones, "real" clips are band-limited noise, so a working training
loop can separate them. Variable lengths exercise cropping and padding.

    python -m voxguard.data.synthetic --out data/processed/synthetic --n 100
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from voxguard.data.manifests import ManifestRow, write_manifest

logger = logging.getLogger(__name__)
SR = 16_000


def make_clip(label: int, rng: np.random.Generator) -> np.ndarray:
    n = int(rng.uniform(1.5, 5.0) * SR)
    t = np.arange(n) / SR
    if label == 1:
        f0 = rng.uniform(100, 300)
        x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6))
        x = 0.2 * x + 0.01 * rng.standard_normal(n)
    else:
        noise = np.fft.rfft(rng.standard_normal(n))
        freqs = np.fft.rfftfreq(n, 1 / SR)
        lo = rng.uniform(200, 1500)
        noise[(freqs < lo) | (freqs > lo + 2500)] = 0
        x = np.fft.irfft(noise, n)
        x = 0.2 * x / (np.abs(x).max() + 1e-9)
    return x.astype(np.float32)


def write_synthetic(out: Path, n: int = 100, seed: int = 0) -> dict[str, Path]:
    import soundfile as sf

    rng = np.random.default_rng(seed)
    manifests = {}
    for split, count in (("train", n), ("dev", max(n // 4, 8))):
        rows = []
        for i in range(count):
            label = i % 2
            rel = f"synthetic/{split}/{split}_{i:04d}.flac"
            path = out / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, make_clip(label, rng), SR, format="FLAC")
            rows.append(ManifestRow(rel, label, split, source="synthetic"))
        manifests[split] = out / "manifests" / f"synthetic_{split}.csv"
        write_manifest(rows, manifests[split])
    return manifests


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("data/processed/synthetic"))
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for split, path in write_synthetic(args.out, args.n, args.seed).items():
        logger.info("%s manifest: %s", split, path)


if __name__ == "__main__":
    main()
