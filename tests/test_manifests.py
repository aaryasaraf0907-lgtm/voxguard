from pathlib import Path

import pytest

from voxguard.data.download import find_la_root, safe_extract
from voxguard.data.manifests import (
    COLUMNS,
    ManifestRow,
    build_asvspoof2019_la_manifests,
    parse_asvspoof_protocol,
    read_manifest,
    write_manifest,
)

PROTO = {
    "train": ["LA_0079 LA_T_1138215 - - bonafide", "LA_0079 LA_T_1271820 - A01 spoof"],
    "dev": ["LA_0069 LA_D_1047731 - - bonafide"],
    "eval": ["LA_0040 LA_E_2834763 - A11 spoof", "LA_0040 LA_E_1665632 - - bonafide"],
}
NAMES = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
    "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
}


def _fake_tree(root: Path, skip_audio: bool = False) -> Path:
    la = root / "LA"
    (la / "ASVspoof2019_LA_cm_protocols").mkdir(parents=True)
    for split, lines in PROTO.items():
        (la / "ASVspoof2019_LA_cm_protocols" / NAMES[split]).write_text("\n".join(lines) + "\n")
        audio = la / f"ASVspoof2019_LA_{split}" / "flac"
        audio.mkdir(parents=True)
        for line in lines:
            if not skip_audio:
                (audio / f"{line.split()[1]}.flac").write_bytes(b"x")
    return la


def test_parse_protocol() -> None:
    got = list(parse_asvspoof_protocol(PROTO["train"] + [""]))
    assert got == [("LA_T_1138215", 0), ("LA_T_1271820", 1)]


def test_parse_protocol_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        list(parse_asvspoof_protocol(["only three fields"]))
    with pytest.raises(ValueError):
        list(parse_asvspoof_protocol(["a b - - maybe"]))


def test_manifest_roundtrip(tmp_path: Path) -> None:
    rows = [
        ManifestRow("a.flac", 0, "train", source="s"),
        ManifestRow("b.flac", 1, "dev", "gsm", "13"),
    ]
    out = tmp_path / "m.csv"
    assert write_manifest(rows, out) == 2
    assert out.read_text().splitlines()[0] == ",".join(COLUMNS)
    assert read_manifest(out) == rows


def test_read_manifest_bad_header(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("path,label\nx,0\n")
    with pytest.raises(ValueError):
        read_manifest(bad)


def test_build_manifests(tmp_path: Path) -> None:
    la = _fake_tree(tmp_path)
    assert find_la_root(tmp_path) == la
    counts = build_asvspoof2019_la_manifests(la, tmp_path, tmp_path / "out")
    assert counts == {"train": 2, "dev": 1, "eval": 2}
    rows = read_manifest(tmp_path / "out" / "asvspoof2019_la_eval.csv")
    assert rows[0].path == "LA/ASVspoof2019_LA_eval/flac/LA_E_2834763.flac"
    assert [r.label for r in rows] == [1, 0]
    assert {r.codec for r in rows} == {"clean"}


def test_build_manifests_missing_audio(tmp_path: Path) -> None:
    la = _fake_tree(tmp_path, skip_audio=True)
    with pytest.raises(FileNotFoundError):
        build_asvspoof2019_la_manifests(la, tmp_path, tmp_path / "out")


def test_safe_extract_blocks_zip_slip(tmp_path: Path) -> None:
    import zipfile

    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("../escape.txt", "x")
    with pytest.raises(ValueError):
        safe_extract(z, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


def test_forbidden_fails_fast(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import urllib.error

    from voxguard.data import download

    calls = []

    def boom(*args: object, **kwargs: object) -> None:
        calls.append(1)
        raise urllib.error.HTTPError("u", 403, "Forbidden", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr(download.urllib.request, "urlopen", boom)
    with pytest.raises(OSError, match="--from-zip"):
        download.download_resumable("http://x", tmp_path / "a.zip", 10)
    assert len(calls) == 1  # no retries


def test_from_zip_builds_manifests(tmp_path: Path) -> None:
    import zipfile

    from voxguard.data import download

    src = tmp_path / "src"
    _fake_tree(src)
    z = tmp_path / "LA.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for f in (src / "LA").rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(src).as_posix())
    root, out = tmp_path / "raw", tmp_path / "man"
    download.main(["--from-zip", str(z), "--root", str(root), "--manifests-dir", str(out)])
    assert z.exists()  # never deleted
    rows = read_manifest(out / "asvspoof2019_la_eval.csv")
    assert rows[0].path == "LA/ASVspoof2019_LA_eval/flac/LA_E_2834763.flac"
    assert (root / rows[0].path).is_file()
