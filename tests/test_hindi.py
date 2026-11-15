from pathlib import Path

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")

from voxguard.data.hindi import build_hindi_set, read_fleurs_tsv  # noqa: E402
from voxguard.data.manifests import read_manifest  # noqa: E402

TSV = (
    "1\ta.wav\tपहला वाक्य।\tपहला वाक्य\tप ह ल ा\t16000\tFEMALE\n"
    "2\tb.wav\tदूसरा (वाक्य)\tदूसरा वाक्य\tद ू स र ा\t16000\tMALE\n"
)


def _tree(tmp_path: Path) -> tuple[Path, Path]:
    audio = tmp_path / "raw" / "fleurs" / "test"
    audio.mkdir(parents=True)
    t = np.arange(16000) / 16000
    for name, amp in (("a.wav", 0.02), ("b.wav", 0.05)):
        sf.write(audio / name, amp * np.sin(2 * np.pi * 200 * t), 16000)
    tsv = tmp_path / "test.tsv"
    tsv.write_text(TSV)
    return tsv, audio


def _rms(path: Path) -> float:
    x, _ = sf.read(path)
    return float(np.sqrt(np.mean(x**2)))


def test_read_fleurs_tsv_takes_normalised_transcript(tmp_path: Path) -> None:
    tsv, _ = _tree(tmp_path)
    assert read_fleurs_tsv(tsv) == [("a.wav", "पहला वाक्य"), ("b.wav", "दूसरा वाक्य")]
    bad = tmp_path / "bad.tsv"
    bad.write_text("1\tnot-audio\tx\tx\n")
    with pytest.raises(ValueError, match="unexpected FLEURS row"):
        read_fleurs_tsv(bad)


def test_build_hindi_set_pairs_real_and_loudness_matched_fakes(tmp_path: Path) -> None:
    tsv, audio = _tree(tmp_path)
    said: list[str] = []

    def tts(text: str) -> np.ndarray:  # loud, like MMS-TTS
        said.append(text)
        return 0.6 * np.sin(2 * np.pi * 300 * np.arange(12000) / 16000)

    fake_dir, out = tmp_path / "raw" / "fake", tmp_path / "hindi.csv"
    assert build_hindi_set(tsv, audio, tts, fake_dir, tmp_path / "raw", out) == 4
    rows = read_manifest(out)
    assert [(r.path, r.label, r.split, r.source) for r in rows] == [
        ("fleurs/test/a.wav", 0, "eval", "fleurs_hi_in"),
        ("fake/a.wav", 1, "eval", "mms_tts_hin"),
        ("fleurs/test/b.wav", 0, "eval", "fleurs_hi_in"),
        ("fake/b.wav", 1, "eval", "mms_tts_hin"),
    ]
    assert said == ["पहला वाक्य", "दूसरा वाक्य"]
    for name in ("a.wav", "b.wav"):
        assert _rms(fake_dir / name) == pytest.approx(_rms(audio / name), rel=1e-3)
    build_hindi_set(tsv, audio, tts, fake_dir, tmp_path / "raw", out)  # reuses existing fakes
    assert len(said) == 2


def test_build_hindi_set_rejects_missing_audio_and_empty_tts(tmp_path: Path) -> None:
    tsv, audio = _tree(tmp_path)
    root = tmp_path / "raw"
    with pytest.raises(ValueError, match="no audio"):
        build_hindi_set(tsv, audio, lambda _: np.zeros(10), root / "f1", root, tmp_path / "x.csv")
    (audio / "b.wav").unlink()
    with pytest.raises(FileNotFoundError):
        build_hindi_set(tsv, audio, lambda _: np.ones(10), root / "f2", root, tmp_path / "y.csv")
