"""Evaluation: the 2x2 matrix, per-codec breakdown, latency, JSON + docs/results.md.

    python -m voxguard.evaluate --config configs/eval.yaml
    python -m voxguard.evaluate --config configs/eval.yaml --render-only   # rebuild the doc
    python -m voxguard.evaluate --select-thresholds runs/x/best.pt \\
        --dev-clean data/processed/manifests/asvspoof2019_la_dev.csv \\
        --dev-phone data/processed/manifests/dev_per_codec.csv    # -> runs/x/thresholds.json

For each model family, the ``clean`` and ``codec_aug`` checkpoints are scored on the ``clean``
and ``degraded`` test sets. Degraded rows are grouped by the manifest ``codec`` column for the
per-codec breakdown. An optional ``heldout`` test set holds codecs that no model saw in training
(the unseen-codec test); it is reported separately and never enters the 2x2 matrix. Missing
checkpoints are skipped and shown as "not run", so partial results still render.
Numbers in docs/results.md come only from results/*.json (never hand-typed).
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from voxguard.metrics import compute_eer, latency_stats, summarize

logger = logging.getLogger(__name__)

TRAIN_CONDITIONS = ("clean", "codec_aug")
MATRIX_TESTS = ("clean", "degraded")  # the 2x2 columns
TEST_CONDITIONS = (*MATRIX_TESTS, "heldout")  # heldout: codecs unseen in training (optional)
CODEC_TESTS = ("degraded", "heldout")  # sets whose clean rows are dropped
TRAIN_LABELS = {"clean": "Clean only", "codec_aug": "Codec-augmented"}
THRESHOLD_EXCLUDED_CODECS = ("gsm",)  # held out as the unseen codec: never used to pick one
LOW_FAR_MAX_MISSED = 0.01  # low_far: at most this share of dev fakes scored below the threshold


@dataclass
class EvalConfig:
    models: dict[str, dict[str, str]] = field(default_factory=dict)  # family -> condition -> ckpt
    test_sets: dict[str, list[str]] = field(default_factory=dict)  # test condition -> manifests
    roots: list[str] = field(default_factory=lambda: ["data/raw", "data/processed"])
    max_samples: int | None = None
    batch_size: int = 32
    device: str = "auto"
    latency_clips: int = 50
    latency_devices: list[str] = field(default_factory=lambda: ["cpu", "auto"])
    seed: int = 0
    out_dir: str = "results"
    docs_path: str = "docs/results.md"


def load_eval_config(path: Path) -> EvalConfig:
    raw = yaml.safe_load(path.read_text()) or {}
    known = set(EvalConfig.__dataclass_fields__)
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown eval config keys: {sorted(unknown)}")
    cfg = EvalConfig(**raw)
    bad = {c for conds in cfg.models.values() for c in conds} - set(TRAIN_CONDITIONS)
    if bad:
        raise ValueError(f"train conditions must be {TRAIN_CONDITIONS}, got {sorted(bad)}")
    if set(cfg.test_sets) - set(TEST_CONDITIONS):
        raise ValueError(f"test sets must be {TEST_CONDITIONS}, got {sorted(cfg.test_sets)}")
    return cfg


# --- pure metrics (numpy only, tested in CI) ----------------------------------------------


def evaluate_condition(scores: np.ndarray, labels: np.ndarray, codecs: list[str]) -> dict[str, Any]:
    """Pooled metrics plus a per-codec breakdown (codec groups need both classes)."""
    out: dict[str, Any] = {"pooled": summarize(scores, labels), "per_codec": {}}
    codec_arr = np.asarray(codecs)
    for codec in sorted(set(codecs)):
        mask = codec_arr == codec
        try:
            out["per_codec"][codec] = summarize(scores[mask], labels[mask])
        except ValueError as exc:
            logger.warning("codec %s skipped: %s", codec, exc)
    return out


def low_far_threshold(scores: np.ndarray, labels: np.ndarray, max_missed: float) -> float:
    """The highest threshold that misses at most ``max_missed`` of the fakes (label 1).

    A fake is missed when its score is below the threshold, so this is the ``max_missed``
    quantile of the fake scores (lower, i.e. conservative). Real clips are not considered: this
    operating point trades more false alarms for fewer missed fakes.
    """
    fake = np.sort(np.asarray(scores, dtype=np.float64)[np.asarray(labels) == 1])
    if fake.size == 0:
        raise ValueError("low_far threshold needs at least one fake clip")
    k = int(np.floor(max_missed * fake.size))  # this many fakes may fall below
    return float(fake[min(k, fake.size - 1)])


def select_thresholds(
    clean: tuple[np.ndarray, np.ndarray],
    phone: tuple[np.ndarray, np.ndarray, list[str]],
    model_version: str,
    max_missed: float = LOW_FAR_MAX_MISSED,
) -> dict[str, Any]:
    """thresholds.json contents (CLAUDE.md section 7) from dev-set scores.

    ``clean`` / ``phone`` are EER thresholds on clean and pooled codec-degraded dev audio;
    ``per_codec`` and ``low_far`` are for reporting. Excluded codecs (GSM) are dropped first.
    """
    p_scores, p_labels, p_codecs = phone
    keep = ~np.isin(np.asarray(p_codecs), THRESHOLD_EXCLUDED_CODECS)
    p_scores, p_labels = np.asarray(p_scores)[keep], np.asarray(p_labels)[keep]
    p_codecs = [c for c, k in zip(p_codecs, keep, strict=True) if k]
    if p_scores.size == 0:
        raise ValueError("no phone-degraded dev clips left after excluding held-out codecs")
    clean_eer, clean_thr = compute_eer(*clean)
    phone_eer, phone_thr = compute_eer(p_scores, p_labels)
    per_codec: dict[str, float] = {}
    per_codec_eer: dict[str, float] = {}
    codec_arr = np.asarray(p_codecs)
    for codec in sorted(set(p_codecs)):
        mask = codec_arr == codec
        try:
            per_codec_eer[codec], per_codec[codec] = compute_eer(p_scores[mask], p_labels[mask])
        except ValueError as exc:
            logger.warning("codec %s: no threshold (%s)", codec, exc)
    clip = lambda t: float(min(max(t, 0.0), 1.0))  # noqa: E731 - scores are P(fake) in [0, 1]
    return {
        "model_version": model_version,
        "selected_on": "dev",
        "clean": clip(clean_thr),
        "phone": clip(phone_thr),
        "low_far": {
            "clean": clip(low_far_threshold(*clean, max_missed)),
            "phone": clip(low_far_threshold(p_scores, p_labels, max_missed)),
        },
        "per_codec": {c: clip(t) for c, t in per_codec.items()},
        "low_far_max_missed": max_missed,
        "excluded_codecs": list(THRESHOLD_EXCLUDED_CODECS),
        "dev_eer": {"clean": clean_eer, "phone": phone_eer, "per_codec": per_codec_eer},
        "dev_clips": {"clean": int(len(clean[1])), "phone": int(p_labels.size)},
    }


# --- scoring (torch, lazy imports) --------------------------------------------------------


def load_model(ckpt_path: Path, device: str):
    """Rebuild the model from the config stored in a train.py checkpoint."""
    from voxguard.models import load_from_checkpoint

    return load_from_checkpoint(ckpt_path, device)


def score_manifests(
    model, manifests: list[Path], cfg: EvalConfig, device: str, codec_filter_clean: bool
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (P(fake) scores, labels, codec per row). Crops from the start (eval mode)."""
    import torch
    from torch.utils.data import DataLoader

    from voxguard.data.datasets import ManifestDataset

    ds = ManifestDataset(
        manifests, [Path(r) for r in cfg.roots], train=False,
        max_samples=cfg.max_samples, seed=cfg.seed,
    )  # fmt: skip
    if codec_filter_clean:  # codec test sets must not include clean rows
        ds.rows = [r for r in ds.rows if r.codec != "clean"]
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)
    scores, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device)).float()
            scores.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(scores), np.concatenate(labels), [r.codec for r in ds.rows]


def measure_model_latency(model, device: str, clips: int, num_samples: int = 64_600):
    """Per-clip (batch 1, ~4 s) forward latency with device synchronisation."""
    import time

    import torch

    model = model.to(device).eval()
    x = torch.randn(1, num_samples, device=device)

    def sync() -> None:
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    times = []
    with torch.no_grad():
        for i in range(clips + 3):  # first 3 are warm-up
            sync()
            t0 = time.perf_counter()
            model(x)
            sync()
            if i >= 3:
                times.append((time.perf_counter() - t0) * 1000.0)
    return latency_stats(times)


def evaluate_family(family: str, ckpts: dict[str, str], cfg: EvalConfig) -> dict[str, Any]:
    from voxguard.train import pick_device

    device = str(pick_device(cfg.device))
    result: dict[str, Any] = {"family": family, "matrix": {}, "models": {}, "latency": {}}
    latency_model = None
    for train_cond in TRAIN_CONDITIONS:
        path = ckpts.get(train_cond)
        if not path or not Path(path).exists():
            logger.warning("%s/%s: checkpoint %s missing, skipping", family, train_cond, path)
            continue
        model, info = load_model(Path(path), device)
        result["models"][train_cond] = {"checkpoint": path, **info}
        result["matrix"][train_cond] = {}
        for test_cond in TEST_CONDITIONS:
            manifests = cfg.test_sets.get(test_cond)
            if not manifests:
                continue
            s, y, codecs = score_manifests(
                model, [Path(m) for m in manifests], cfg, device, test_cond in CODEC_TESTS
            )
            result["matrix"][train_cond][test_cond] = evaluate_condition(s, y, codecs)
            logger.info(
                "%s train=%s test=%s EER=%.4f (n=%d)", family, train_cond, test_cond,
                result["matrix"][train_cond][test_cond]["pooled"]["eer"], len(y),
            )  # fmt: skip
        latency_model = latency_model or model
    if latency_model is not None:
        for dev in dict.fromkeys(str(pick_device(d)) for d in cfg.latency_devices):
            stats = measure_model_latency(latency_model, dev, cfg.latency_clips)
            result["latency"][dev] = stats.__dict__
    return result


def run_eval(cfg: EvalConfig) -> dict[str, Any]:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "max_samples": cfg.max_samples,
        "test_sets": cfg.test_sets,
    }
    families = {}
    for family, ckpts in cfg.models.items():
        res = evaluate_family(family, ckpts, cfg)
        res["meta"] = meta
        (out / f"{family}.json").write_text(json.dumps(res, indent=2))
        families[family] = res
    summary = {"meta": meta, "families": sorted(families)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return families


# --- markdown rendering (pure, tested in CI) ----------------------------------------------


def _pct(x: float | None) -> str:
    return "not run" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.2f}"


def _get(res: dict[str, Any], train: str, test: str) -> dict[str, Any] | None:
    return res.get("matrix", {}).get(train, {}).get(test)


def _per_codec_table(res: dict[str, Any], test: str, title: str) -> list[str]:
    codecs = sorted(
        {c for t in TRAIN_CONDITIONS for c in ((_get(res, t, test) or {}).get("per_codec", {}))}
    )
    if not codecs:
        return []
    lines = ["", f"### {title}", ""]
    lines += ["| Train | " + " | ".join(codecs) + " |", "|---" * (len(codecs) + 1) + "|"]
    for train in TRAIN_CONDITIONS:
        per = (_get(res, train, test) or {}).get("per_codec", {})
        cells = [_pct(per.get(c, {}).get("eer")) for c in codecs]
        lines.append(f"| {TRAIN_LABELS[train]} | " + " | ".join(cells) + " |")
    return lines


def render_family(res: dict[str, Any]) -> str:
    lines = [f"## {res['family']}", "", "### 2x2 matrix: EER (%)", ""]
    lines += ["| Train \\ Test | Clean | Phone-degraded (pooled) |", "|---|---|---|"]
    for train in TRAIN_CONDITIONS:
        cells = [
            _pct((_get(res, train, t) or {}).get("pooled", {}).get("eer")) for t in MATRIX_TESTS
        ]
        lines.append(f"| {TRAIN_LABELS[train]} | {cells[0]} | {cells[1]} |")

    lines += _per_codec_table(res, "degraded", "Phone-degraded EER (%) per codec")
    lines += _per_codec_table(
        res, "heldout", "Unseen-codec EER (%): codecs not used in any training run"
    )

    lines += ["", "### Accuracy (%) at the EER threshold / at 0.5", ""]
    lines += ["| Train \\ Test | Clean | Phone-degraded |", "|---|---|---|"]
    for train in TRAIN_CONDITIONS:
        cells = []
        for test in MATRIX_TESTS:
            pooled = (_get(res, train, test) or {}).get("pooled")
            cells.append(
                "not run"
                if pooled is None
                else f"{_pct(pooled['accuracy_at_eer'])} / {_pct(pooled['accuracy_at_0.5'])}"
            )
        lines.append(f"| {TRAIN_LABELS[train]} | {cells[0]} | {cells[1]} |")

    if res.get("latency"):
        lines += ["", "### Latency per ~4 s clip (ms, batch 1)", ""]
        lines += ["| Device | p50 | p95 | mean | clips |", "|---|---|---|---|---|"]
        for dev, s in res["latency"].items():
            lines.append(
                f"| {dev} | {s['p50_ms']:.1f} | {s['p95_ms']:.1f} | {s['mean_ms']:.1f} | {s['n']} |"
            )

    sizes = []
    for train in TRAIN_CONDITIONS:
        for test in TEST_CONDITIONS:
            pooled = (_get(res, train, test) or {}).get("pooled")
            if pooled:
                sizes.append(f"{TRAIN_LABELS[train]} on {test}: n={pooled['n']}")
    if sizes:
        lines += ["", "Clips scored: " + "; ".join(sizes) + "."]
    return "\n".join(lines)


def render_markdown(families: dict[str, dict[str, Any]]) -> str:
    meta = next(iter(families.values()), {}).get("meta", {})
    head = [
        "# Results",
        "",
        "> Auto-generated by `python -m voxguard.evaluate` from `results/*.json`. "
        "Do not edit by hand.",
        f"> Generated: {meta.get('generated_at', 'n/a')}.",
        "",
        "EER: equal error rate (lower is better). Scores are P(fake); a clip is called fake "
        "when its score is at or above the threshold. Eval crops the first ~4 s of each clip.",
    ]
    if meta.get("max_samples"):
        head += ["", f"**Smoke/partial run: max_samples={meta['max_samples']} per test set.**"]
    body = [render_family(families[f]) for f in sorted(families)]
    return "\n".join(head) + "\n\n" + "\n\n".join(body) + "\n"


def load_results(out_dir: Path) -> dict[str, dict[str, Any]]:
    return {
        p.stem: json.loads(p.read_text())
        for p in sorted(out_dir.glob("*.json"))
        if p.name != "summary.json"
    }


def write_thresholds(
    checkpoint: Path, dev_clean: list[Path], dev_phone: list[Path], cfg: EvalConfig
) -> Path:
    """Score the dev sets with ``checkpoint`` and write ``thresholds.json`` next to it."""
    from voxguard.inference import model_version
    from voxguard.train import pick_device

    device = str(pick_device(cfg.device))
    model, info = load_model(checkpoint, device)
    version = model_version(info, str(checkpoint))
    clean = score_manifests(model, dev_clean, cfg, device, codec_filter_clean=False)[:2]
    phone = score_manifests(model, dev_phone, cfg, device, codec_filter_clean=True)
    data = select_thresholds(clean, phone, version)
    out = checkpoint.parent / "thresholds.json"
    out.write_text(json.dumps(data, indent=2) + "\n")
    logger.info(
        "%s: clean %.4g (dev EER %.2f%%), phone %.4g (dev EER %.2f%%) -> %s", version,
        data["clean"], 100 * data["dev_eer"]["clean"], data["phone"],
        100 * data["dev_eer"]["phone"], out,
    )  # fmt: skip
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--render-only", action="store_true", help="rebuild docs from results/*.json")
    p.add_argument(
        "--select-thresholds",
        type=Path,
        metavar="CHECKPOINT",
        help="pick dev-set thresholds for CHECKPOINT and write thresholds.json",
    )
    p.add_argument("--dev-clean", type=Path, nargs="+", help="clean dev manifest(s)")
    p.add_argument("--dev-phone", type=Path, nargs="+", help="codec-degraded dev manifest(s)")
    p.add_argument("--roots", nargs="+", help="audio roots (default: the eval config's)")
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.select_thresholds is not None:
        if not args.dev_clean or not args.dev_phone:
            p.error("--select-thresholds needs --dev-clean and --dev-phone")
        cfg = load_eval_config(args.config) if args.config else EvalConfig()
        cfg.max_samples = args.max_samples if args.max_samples is not None else cfg.max_samples
        cfg.roots = args.roots or cfg.roots
        cfg.device = args.device or cfg.device
        write_thresholds(args.select_thresholds, args.dev_clean, args.dev_phone, cfg)
        return
    if args.config is None:
        p.error("--config is required")

    cfg = load_eval_config(args.config)
    if args.max_samples is not None:
        cfg.max_samples = args.max_samples
    families = load_results(Path(cfg.out_dir)) if args.render_only else run_eval(cfg)
    if not families:
        raise SystemExit(f"no results in {cfg.out_dir}")
    docs = Path(cfg.docs_path)
    docs.parent.mkdir(parents=True, exist_ok=True)
    docs.write_text(render_markdown(families))
    logger.info("wrote %s and %d result file(s) in %s", docs, len(families), cfg.out_dir)


if __name__ == "__main__":
    main()
