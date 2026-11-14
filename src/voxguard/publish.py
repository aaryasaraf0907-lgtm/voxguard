"""Publish a trained checkpoint to the private Hugging Face org, with its thresholds and a card.

    python -m voxguard.publish runs/kaggle/<run>/runs/<run>/best.pt --repo voxguard-team/<name>
    python -m voxguard.publish ... --dry-run      # print the model card, upload nothing

Uploads ``best.pt``, ``thresholds.json`` (required: from ``evaluate --select-thresholds``) and a
generated ``README.md`` model card. The repo is created **private** and the upload is refused
if it is (or becomes) public: CLAUDE.md section 15 keeps every model repo private until the IPR
cell confirms the filing. The token is read by ``huggingface_hub`` itself (``HF_TOKEN`` or the
cached login) and never printed.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.2f} %"


def model_card(
    model_version: str,
    info: dict[str, Any],
    thresholds: dict[str, Any],
    history: list[dict[str, Any]],
) -> str:
    """The README.md for the model repo (pure; tested in CI)."""
    dev = thresholds.get("dev_eer", {})
    per_codec = thresholds.get("per_codec", {})
    lines = [
        "---",
        "license: other",
        "tags: [audio, anti-spoofing, deepfake-detection, private]",
        "---",
        "",
        f"# VoxGuard: {info.get('run') or info['name']}",
        "",
        "> **Confidential.** Private VoxGuard team model; do not make this repository public or "
        "describe the method publicly before the IPR cell confirms the patent filing.",
        "",
        f"- **model_version:** `{model_version}`",
        f"- **Architecture:** `{info['name']}`; **epoch:** {info['epoch']} (0-based, best on dev)",
        f"- **Training run:** `{info.get('run') or 'unknown'}`",
        "- **Output:** P(fake) in [0, 1] for 16 kHz mono audio, 64,600-sample (~4 s) windows.",
        "- **Data:** ASVspoof 2019 LA (Todisco et al., Interspeech 2019; ODC-By).",
        "",
        "## Thresholds (chosen on the dev set)",
        "",
        "The API reads `thresholds.json` from this repo and applies the **phone** threshold.",
        "",
        "| | Clean | Phone (codec-degraded) |",
        "|---|---|---|",
        f"| EER threshold | {thresholds['clean']:.4g} | {thresholds['phone']:.4g} |",
        f"| Dev EER | {_pct(dev.get('clean'))} | {_pct(dev.get('phone'))} |",
    ]
    if per_codec:
        codecs = ", ".join(f"{c} {t:.4g}" for c, t in sorted(per_codec.items()))
        lines += ["", f"Per-codec dev thresholds (reporting only): {codecs}."]
    excluded = thresholds.get("excluded_codecs") or []
    if excluded:
        lines += ["", f"Never used to choose a threshold (held out): {', '.join(excluded)}."]
    if history:
        lines += ["", "## Training curve (dev subset)", "", "| Epoch | Train loss | Dev EER |",
                  "|---|---|---|"]  # fmt: skip
        lines += [f"| {r['epoch']} | {r['train_loss']:.4f} | {_pct(r.get('dev_eer'))} |"
                  for r in history]  # fmt: skip
    lines += ["", "Full evaluation (2x2 matrix, per codec, latency): `docs/results.md` in the "
              "VoxGuard repository."]  # fmt: skip
    return "\n".join(lines) + "\n"


def _history(checkpoint: Path) -> list[dict[str, Any]]:
    path = checkpoint.parent / "metrics.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def publish(checkpoint: Path, repo: str, dry_run: bool = False) -> str:
    from voxguard.api.thresholds import parse_thresholds
    from voxguard.inference import model_version
    from voxguard.models import load_from_checkpoint

    thr_path = checkpoint.parent / "thresholds.json"
    if not thr_path.is_file():
        raise SystemExit(f"{thr_path} missing: run `evaluate --select-thresholds` first")
    thresholds = json.loads(thr_path.read_text())
    _, info = load_from_checkpoint(checkpoint, "cpu")
    version = model_version(info, repo)
    parse_thresholds(thresholds, version)  # the API must accept it once it is on the Hub
    card = model_card(version, info, thresholds, _history(checkpoint))
    if dry_run:
        print(card)  # noqa: T201 - the dry run's output is the card itself
        return version

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    if not api.repo_info(repo).private:
        raise SystemExit(f"{repo} is public: refusing to upload (CLAUDE.md section 15)")
    card_path = checkpoint.parent / "README.md"
    card_path.write_text(card)
    for local, remote in ((checkpoint, "best.pt"), (thr_path, "thresholds.json"),
                          (card_path, "README.md")):  # fmt: skip
        api.upload_file(path_or_fileobj=str(local), path_in_repo=remote, repo_id=repo,
                        commit_message=f"Upload {remote} for {version}")  # fmt: skip
        logger.info("uploaded %s -> %s/%s", local, repo, remote)
    return version


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--repo", required=True, help="e.g. voxguard-team/voxguard-ssl-aasist-codec-aug")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    version = publish(args.checkpoint, args.repo, args.dry_run)
    logger.info("%s %s", "checked" if args.dry_run else "published", version)


if __name__ == "__main__":
    main()
