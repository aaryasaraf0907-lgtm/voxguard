"""Seeded codec/noise degradation and offline precompute of degraded variants + manifests.

Two modes (CLAUDE.md section 8):
- ``variants``: per input file, the clean row plus N random (codec, bitrate, noise) variants.
  Used for training sets.
- ``per-codec``: per input file, one deterministic copy per codec at its default bitrate and
  no noise. Used for eval sets so numbers are reproducible.

Randomness is derived from ``sha256(seed, file path, index)``, so output does not depend on
processing order or worker count.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from voxguard.data.codecs import CODECS, SAMPLE_RATE, apply_codec, usable_codecs
from voxguard.data.manifests import ManifestRow, read_manifest, write_manifest

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DegradeSpec:
    codec: str
    bitrate: int | None
    snr_db: float | None = None  # None = no added noise


@dataclass(frozen=True)
class Policy:
    codecs: dict[str, float]  # codec name -> sampling weight
    noise_probability: float = 0.0
    snr_db: tuple[float, float] = (10.0, 30.0)

    @classmethod
    def from_yaml(cls, path: Path) -> Policy:
        cfg = yaml.safe_load(path.read_text())
        noise = cfg.get("noise", {})
        lo, hi = noise.get("snr_db", [10, 30])
        return cls(
            codecs={k: float(v) for k, v in cfg["codecs"].items()},
            noise_probability=float(noise.get("probability", 0.0)),
            snr_db=(float(lo), float(hi)),
        )


def derive_rng(seed: int, key: str, index: int) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{key}:{index}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def add_noise(waveform: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add white Gaussian noise at the given SNR (relative to the signal's RMS)."""
    power = float(np.mean(waveform.astype(np.float64) ** 2))
    if power == 0.0:
        return waveform.copy()
    sigma = np.sqrt(power / (10 ** (snr_db / 10)))
    return (waveform + rng.normal(0.0, sigma, size=waveform.shape)).astype(np.float32)


def sample_spec(rng: np.random.Generator, policy: Policy, codecs: Sequence[str]) -> DegradeSpec:
    weights = np.array([policy.codecs.get(c, 0.0) for c in codecs], dtype=np.float64)
    if weights.sum() <= 0:
        raise ValueError("no usable codec has a positive weight in the policy")
    name = codecs[int(rng.choice(len(codecs), p=weights / weights.sum()))]
    rates = CODECS[name].bitrates
    bitrate = int(rates[int(rng.integers(len(rates)))]) if rates else None
    snr = None
    if rng.random() < policy.noise_probability:
        snr = float(rng.uniform(*policy.snr_db))
    return DegradeSpec(name, bitrate, snr)


def default_spec(codec: str) -> DegradeSpec:
    rates = CODECS[codec].bitrates
    return DegradeSpec(codec, rates[len(rates) // 2] if rates else None)


def degrade(waveform: np.ndarray, spec: DegradeSpec, rng: np.random.Generator) -> np.ndarray:
    x = waveform
    if spec.snr_db is not None:
        x = add_noise(x, spec.snr_db, rng)
    return apply_codec(x, spec.codec, spec.bitrate)


def _read_audio(path: Path) -> np.ndarray:
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE:
        raise ValueError(f"{path}: expected {SAMPLE_RATE} Hz, got {sr}")
    return data.mean(axis=1)


def _write_audio(path: Path, waveform: np.ndarray) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, waveform, SAMPLE_RATE, format="FLAC", subtype="PCM_16")


def plan_specs(
    row: ManifestRow,
    mode: str,
    n_variants: int,
    seed: int,
    policy: Policy,
    codecs: Sequence[str],
) -> list[tuple[int, DegradeSpec]]:
    if mode == "per-codec":
        return [(i, default_spec(c)) for i, c in enumerate(codecs)]
    if mode == "variants":
        return [
            (i, sample_spec(derive_rng(seed, row.path, i), policy, codecs))
            for i in range(n_variants)
        ]
    raise ValueError(f"mode must be 'variants' or 'per-codec', got {mode!r}")


def _process(
    row: ManifestRow,
    data_root: Path,
    out_root: Path,
    plan: list[tuple[int, DegradeSpec]],
    seed: int,
) -> list[ManifestRow]:
    clean = _read_audio(data_root / row.path)
    out_rows = []
    for i, spec in plan:
        noisy_rng = derive_rng(seed, row.path, 1000 + i)  # separate stream for noise
        y = degrade(clean, spec, noisy_rng)
        tag = spec.codec + (f"_{spec.bitrate}" if spec.bitrate else "")
        if spec.snr_db is not None:
            tag += f"_snr{spec.snr_db:.0f}"
        rel = Path("degraded") / row.split / spec.codec / f"{Path(row.path).stem}__{tag}_{i}.flac"
        _write_audio(out_root / rel, y)
        out_rows.append(
            ManifestRow(
                path=rel.as_posix(),
                label=row.label,
                split=row.split,
                codec=spec.codec,
                bitrate=str(spec.bitrate or ""),
                source=row.source,
            )
        )
    return out_rows


def precompute(
    manifest: Path,
    data_root: Path,
    out_root: Path,
    out_manifest: Path,
    mode: str = "variants",
    n_variants: int = 2,
    seed: int = 0,
    policy: Policy | None = None,
    codecs: list[str] | None = None,
    include_clean: bool = True,
    max_files: int | None = None,
    workers: int = 4,
) -> int:
    """Write degraded audio under ``out_root`` and a manifest at ``out_manifest``.

    Clean rows keep their original path (relative to ``data_root``); degraded rows are
    relative to ``out_root``. Use the same directory for both, or pass a combined root.
    """
    rows = read_manifest(manifest)[:max_files]
    policy = policy or Policy(codecs={c: 1.0 for c in CODECS})
    names = usable_codecs(codecs if codecs is not None else list(policy.codecs))
    logger.info("mode=%s codecs=%s files=%d seed=%d", mode, names, len(rows), seed)
    plans = [plan_specs(r, mode, n_variants, seed, policy, names) for r in rows]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(
            ex.map(
                lambda rp: _process(rp[0], data_root, out_root, rp[1], seed),
                zip(rows, plans, strict=True),
            )
        )
    out: list[ManifestRow] = []
    for row, degraded in zip(rows, results, strict=True):
        if include_clean:
            out.append(row)
        out.extend(degraded)
    return write_manifest(out, out_manifest)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path("data/raw"))
    p.add_argument("--out-root", type=Path, default=Path("data/processed"))
    p.add_argument("--out-manifest", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/codecs.yaml"))
    p.add_argument("--mode", choices=["variants", "per-codec"], default="variants")
    p.add_argument("--n-variants", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--codecs", nargs="*", help="restrict to these codecs (default: from config)")
    p.add_argument("--no-clean", action="store_true", help="omit clean rows from the manifest")
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    n = precompute(
        args.manifest, args.data_root, args.out_root, args.out_manifest, args.mode,
        args.n_variants, args.seed, Policy.from_yaml(args.config), args.codecs,
        not args.no_clean, args.max_files, args.workers,
    )  # fmt: skip
    logger.info("wrote %d rows to %s", n, args.out_manifest)


if __name__ == "__main__":
    main()
