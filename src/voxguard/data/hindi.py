"""Hindi generalisation test set (stretch, CLAUDE.md section 11): real FLEURS vs MMS-TTS speech.

    python -m voxguard.data.hindi --fleurs-tsv data/raw/hindi/download/data/hi_in/test.tsv \\
        --fleurs-audio data/raw/hindi/fleurs_hi_test/test --tts data/raw/hindi/mms-tts-hin

Real: Google FLEURS ``hi_in`` test split (read speech, CC-BY-4.0). Fake: ``facebook/mms-tts-hin``
(VITS, CC-BY-NC-4.0, academic use only) speaking the *same* transcripts, so real and fake match
in content and only the voice source differs; each fake is scaled to its real clip's RMS so
loudness (TTS comes out ~5x louder) is not a give-away. One TTS voice only: this is a small
generalisation check, not evidence that the detector "supports Hindi". Out-of-domain test only
(split ``eval``).
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np

from voxguard.data.manifests import ManifestRow, write_manifest

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
Synthesizer = Callable[[str], np.ndarray]  # text -> 16 kHz mono float waveform


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2)))


def read_fleurs_tsv(path: Path) -> list[tuple[str, str]]:
    """``(wav file name, normalised transcript)`` per row of a FLEURS split tsv (no header)."""
    items = []
    with path.open(newline="") as f:
        for lineno, row in enumerate(csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE), 1):
            if len(row) < 4 or not row[1].endswith(".wav") or not row[3].strip():
                raise ValueError(f"{path} line {lineno}: unexpected FLEURS row")
            items.append((row[1], row[3].strip()))
    return items


def build_hindi_set(
    fleurs_tsv: Path,
    fleurs_audio: Path,
    synthesize: Synthesizer,
    fake_dir: Path,
    data_root: Path,
    manifest_out: Path,
) -> int:
    """Write one real and one TTS row per transcript; existing fake wavs are reused.

    Each fake is scaled to the RMS of the real clip with the same transcript (peak-limited to 1).
    """
    import soundfile as sf

    rows = []
    fake_dir.mkdir(parents=True, exist_ok=True)
    for name, text in read_fleurs_tsv(fleurs_tsv):
        real = fleurs_audio / name
        if not real.is_file():
            raise FileNotFoundError(f"{real} listed in {fleurs_tsv.name} but missing")
        fake = fake_dir / name
        if not fake.is_file():
            wav = np.asarray(synthesize(text), dtype=np.float64)
            if wav.ndim != 1 or len(wav) == 0 or not np.any(wav):
                raise ValueError(f"synthesizer returned no audio for {name}")
            ref, _ = sf.read(real, dtype="float64", always_2d=True)
            wav *= _rms(ref.mean(axis=1)) / _rms(wav)
            wav /= max(1.0, float(np.abs(wav).max()))
            sf.write(fake, wav.astype(np.float32), SAMPLE_RATE)
        for path, label, source in ((real, 0, "fleurs_hi_in"), (fake, 1, "mms_tts_hin")):
            rows.append(
                ManifestRow(
                    path=path.relative_to(data_root).as_posix(),
                    label=label,
                    split="eval",
                    source=source,
                )
            )
    n = write_manifest(rows, manifest_out)
    logger.info("wrote Hindi manifest %s: %d rows", manifest_out, n)
    return n


def mms_synthesizer(model_dir: Path, seed: int = 0) -> Synthesizer:  # pragma: no cover - ml extra
    """VITS text-to-speech from a local ``facebook/mms-tts-hin`` snapshot, seeded per call."""
    import torch
    from transformers import AutoTokenizer, VitsModel

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = VitsModel.from_pretrained(model_dir).eval()
    if model.config.sampling_rate != SAMPLE_RATE:
        raise ValueError(f"expected a {SAMPLE_RATE} Hz model, got {model.config.sampling_rate}")

    def synthesize(text: str) -> np.ndarray:
        torch.manual_seed(seed)  # VITS samples noise: same text -> same audio
        with torch.no_grad():
            out = model(**tok(text, return_tensors="pt")).waveform
        return out[0].numpy()

    return synthesize


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--fleurs-tsv", type=Path, required=True)
    p.add_argument("--fleurs-audio", type=Path, required=True)
    p.add_argument("--tts", type=Path, required=True, help="local mms-tts-hin snapshot")
    p.add_argument("--fake-dir", type=Path, default=Path("data/raw/hindi/mms_tts_fake"))
    p.add_argument("--data-root", type=Path, default=Path("data/raw"))
    p.add_argument("--out", type=Path, default=Path("data/processed/manifests/hindi.csv"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_hindi_set(
        args.fleurs_tsv, args.fleurs_audio, mms_synthesizer(args.tts, args.seed),
        args.fake_dir, args.data_root, args.out,
    )  # fmt: skip


if __name__ == "__main__":
    main()
