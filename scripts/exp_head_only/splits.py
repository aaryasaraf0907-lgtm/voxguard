"""Fixed, seeded data splits for the head-only experiment, written before any training.

Writes ``experiments/head_only/splits/``:
- ``la_train.txt``: every ASVspoof 2019 LA train file (training; codec and realistic copies).
- ``la_dev_gate.txt``: every 2019 LA dev bona fide file plus 1,000 per attack (A01–A06), seeded.
  The gate: phone EER, missed fakes at 1 % false alarms, dev-threshold misses.
- ``la_eval_report.txt``: 2019 LA eval, 2,000 bona fide plus 400 per attack (A07–A19), seeded.
  Report only (per attack; A10, A11, A17, A18 watched), never used to choose anything.
- ``itw_gate_speakers.txt`` / ``itw_report_speakers.txt``: In-the-Wild speakers split in two by a
  seeded shuffle (no speaker in both), and ``itw_gate.txt`` / ``itw_report.txt``: up to 1,500
  bona fide and 1,500 spoof clips from each part, seeded.

Never used for training or gating: the Hindi set, the same-speaker clone test, Set C and the
ASVspoof 2021 LA calls (test only). Usage: ``python splits.py PROTOCOL_DIR ITW_META OUT_DIR``.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

SEED = 20261121
DEV_PER_ATTACK, EVAL_PER_ATTACK, EVAL_BONAFIDE, ITW_PER_LABEL = 1000, 400, 2000, 1500


def protocol(path: Path) -> list[tuple[str, str]]:
    """(file id, attack or 'bonafide')."""
    out = []
    for line in path.read_text().splitlines():
        _spk, file_id, _, attack, label = line.split()
        out.append((file_id, "bonafide" if label == "bonafide" else attack))
    return out


def sample(rows: list[tuple[str, str]], per: dict[str, int | None], rng: random.Random) -> list:
    groups: dict[str, list[str]] = defaultdict(list)
    for file_id, group in rows:
        groups[group].append(file_id)
    picked = []
    for group in sorted(groups):
        ids = sorted(groups[group])
        n = per.get(group, per.get("*"))
        picked += [(i, group) for i in (ids if n is None else rng.sample(ids, min(n, len(ids))))]
    return sorted(picked)


def main(argv: list[str]) -> int:
    proto_dir, itw_meta, out = Path(argv[0]), Path(argv[1]), Path(argv[2])
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    p = "ASVspoof2019.LA.cm."
    train = protocol(proto_dir / f"{p}train.trn.txt")
    dev_rows = protocol(proto_dir / f"{p}dev.trl.txt")
    dev = sample(dev_rows, {"bonafide": None, "*": DEV_PER_ATTACK}, rng)
    ev = sample(protocol(proto_dir / f"{p}eval.trl.txt"),
                {"bonafide": EVAL_BONAFIDE, "*": EVAL_PER_ATTACK}, rng)  # fmt: skip

    def write(name: str, rows: list[tuple[str, str]]) -> None:
        (out / name).write_text("".join(f"{a}\t{b}\n" for a, b in rows))

    write("la_train.txt", sorted(train))
    write("la_dev_gate.txt", dev)
    write("la_eval_report.txt", ev)

    meta = list(csv.DictReader(itw_meta.open()))
    speakers = sorted({r["speaker"] for r in meta})
    rng.shuffle(speakers)
    half = len(speakers) // 2
    parts = {"gate": sorted(speakers[:half]), "report": sorted(speakers[half:])}
    counts = {}
    for part, spk in parts.items():
        (out / f"itw_{part}_speakers.txt").write_text("\n".join(spk) + "\n")
        mine = set(spk)
        rows = [(r["file"], r["label"]) for r in meta if r["speaker"] in mine]
        picked = sample(rows, {"*": ITW_PER_LABEL}, rng)
        write(f"itw_{part}.txt", picked)
        counts[part] = {"speakers": len(spk), "clips": len(picked),
                        "bona-fide": sum(lab == "bona-fide" for _, lab in picked)}  # fmt: skip
    summary = {"seed": SEED, "la_train": len(train), "la_dev_gate": len(dev),
               "la_eval_report": len(ev), "itw": counts}  # fmt: skip
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
