"""Manifest CSVs: ``path,label,split,codec,bitrate,source`` (CLAUDE.md section 8).

``label`` is 1 for spoof/fake and 0 for bona fide, matching ``fake_score`` (higher = fake).
``path`` is relative to the dataset root so manifests stay valid when the data moves.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, fields
from pathlib import Path

logger = logging.getLogger(__name__)

COLUMNS = ("path", "label", "split", "codec", "bitrate", "source")
LABELS = {"bonafide": 0, "spoof": 1}


@dataclass(frozen=True)
class ManifestRow:
    path: str
    label: int
    split: str
    codec: str = "clean"
    bitrate: str = ""
    source: str = ""


assert tuple(f.name for f in fields(ManifestRow)) == COLUMNS


def write_manifest(rows: Iterable[ManifestRow], out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
            n += 1
    return n


def read_manifest(path: Path) -> list[ManifestRow]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError(f"{path}: expected columns {COLUMNS}, got {reader.fieldnames}")
        return [
            ManifestRow(
                path=r["path"],
                label=int(r["label"]),
                split=r["split"],
                codec=r["codec"],
                bitrate=r["bitrate"],
                source=r["source"],
            )
            for r in reader
        ]


def parse_asvspoof_protocol(lines: Iterable[str]) -> Iterator[tuple[str, int]]:
    """Yield ``(utterance_id, label)`` from an ASVspoof 2019 LA CM protocol.

    Line format: ``SPEAKER_ID UTT_ID - ATTACK_ID KEY`` with KEY in {bonafide, spoof}.
    """
    for lineno, line in enumerate(lines, 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5 or parts[4] not in LABELS:
            raise ValueError(f"protocol line {lineno}: unexpected format: {line.strip()!r}")
        yield parts[1], LABELS[parts[4]]


def build_asvspoof2019_la_manifests(
    la_root: Path, data_root: Path, out_dir: Path, source: str = "asvspoof2019_la"
) -> dict[str, int]:
    """Write ``train``/``dev``/``eval`` manifests for an extracted ``ASVspoof2019/LA`` tree.

    ``la_root`` holds ``ASVspoof2019_LA_cm_protocols`` and the per-split audio folders;
    manifest paths are stored relative to ``data_root``. Missing audio is an error.
    """
    protocols = {
        "train": "ASVspoof2019.LA.cm.train.trn.txt",
        "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
        "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
    }
    counts: dict[str, int] = {}
    for split, proto_name in protocols.items():
        proto = la_root / "ASVspoof2019_LA_cm_protocols" / proto_name
        audio_dir = la_root / f"ASVspoof2019_LA_{split}" / "flac"
        with proto.open() as f:
            items = list(parse_asvspoof_protocol(f))
        rows = []
        for utt, label in items:
            wav = audio_dir / f"{utt}.flac"
            if not wav.is_file():
                raise FileNotFoundError(f"{split}: {wav} listed in {proto.name} but missing")
            rows.append(
                ManifestRow(
                    path=wav.relative_to(data_root).as_posix(),
                    label=label,
                    split=split,
                    source=source,
                )
            )
        counts[split] = write_manifest(rows, out_dir / f"asvspoof2019_la_{split}.csv")
        logger.info("wrote %s manifest: %d rows", split, counts[split])
    return counts
