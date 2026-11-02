from pathlib import Path

import numpy as np
import pytest

from audio_helpers import synthetic
from voxguard.data.datasets import NUM_SAMPLES, fix_length, load_rows
from voxguard.data.manifests import ManifestRow, write_manifest


def test_fix_length_pad_crop_exact() -> None:
    x = np.arange(10, dtype=np.float32)
    padded = fix_length(x, 25)
    assert len(padded) == 25 and np.array_equal(padded[:10], x) and np.array_equal(padded[10:20], x)
    assert np.array_equal(fix_length(x, 4), x[:4])  # eval: from the start
    assert np.array_equal(fix_length(x, 10), x)
    assert len(fix_length(np.zeros(0, np.float32), 7)) == 7
    assert padded.dtype == np.float32


def test_fix_length_random_crop_is_seeded() -> None:
    x = np.arange(1000, dtype=np.float32)
    a = fix_length(x, 100, np.random.default_rng(3))
    assert np.array_equal(a, fix_length(x, 100, np.random.default_rng(3)))
    assert a[-1] - a[0] == 99  # contiguous window


def _manifest(tmp_path: Path, n: int = 6) -> Path:
    rows = [
        ManifestRow(f"f{i}.flac", i % 2, "train", "clean" if i < 4 else "g711_mulaw", "", "s")
        for i in range(n)
    ]
    path = tmp_path / "m.csv"
    write_manifest(rows, path)
    return path


def test_load_rows_filter_and_subset(tmp_path: Path) -> None:
    m = _manifest(tmp_path)
    assert len(load_rows([m])) == 6
    assert {r.codec for r in load_rows([m], codecs=["g711_mulaw"])} == {"g711_mulaw"}
    sub = load_rows([m, m], max_samples=5, seed=1)
    assert len(sub) == 5 and sub == load_rows([m, m], max_samples=5, seed=1)
    assert sub != load_rows([m, m], max_samples=5, seed=2)


def test_manifest_dataset_end_to_end(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    sf = pytest.importorskip("soundfile")
    from voxguard.data.datasets import ManifestDataset

    raw, proc = tmp_path / "raw", tmp_path / "proc"
    raw.mkdir(), proc.mkdir()
    m = _manifest(tmp_path, 4)
    sf.write(raw / "f0.flac", synthetic(1.0), 16000)  # short -> padded
    sf.write(raw / "f1.flac", synthetic(6.0), 16000)  # long -> cropped
    sf.write(proc / "f2.flac", synthetic(1.0, 2), 16000)  # found via the second root
    sf.write(raw / "f3.flac", synthetic(5.0, 3), 16000)

    ds = ManifestDataset([m], [raw, proc], train=False)
    assert len(ds) == 4
    for i in range(4):
        x, y = ds[i]
        assert x.shape == (NUM_SAMPLES,) and x.dtype == torch.float32
        assert int(y) == i % 2
    assert torch.equal(ds[1][0], ds[1][0])  # eval is deterministic

    tr = ManifestDataset([m], [raw, proc], train=True, seed=5)
    a, b = tr[1][0], tr[1][0]
    assert torch.equal(a, b)  # same (seed, epoch, idx) -> same crop
    tr.set_epoch(1)
    assert not torch.equal(a, tr[1][0])  # new epoch -> new crop

    loader = torch.utils.data.DataLoader(ds, batch_size=2)
    xb, yb = next(iter(loader))
    assert xb.shape == (2, NUM_SAMPLES) and yb.tolist() == [0, 1]

    with pytest.raises(FileNotFoundError):
        ManifestDataset([m], [tmp_path / "nowhere"])[0]
