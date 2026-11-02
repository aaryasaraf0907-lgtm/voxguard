from pathlib import Path

import numpy as np
import pytest

from audio_helpers import synthetic
from voxguard.data.augment import (
    Policy,
    add_noise,
    derive_rng,
    precompute,
    sample_spec,
)
from voxguard.data.codecs import CODECS, ffmpeg_path
from voxguard.data.manifests import ManifestRow, read_manifest, write_manifest

POLICY = Policy(codecs={c: 1.0 for c in CODECS}, noise_probability=0.5, snr_db=(10, 30))


def test_derive_rng_is_deterministic_and_distinct() -> None:
    a = derive_rng(0, "f.flac", 0).random()
    assert a == derive_rng(0, "f.flac", 0).random()
    assert a != derive_rng(1, "f.flac", 0).random()
    assert a != derive_rng(0, "f.flac", 1).random()
    assert a != derive_rng(0, "g.flac", 0).random()


def test_add_noise_hits_requested_snr() -> None:
    x = synthetic(2.0)
    noisy = add_noise(x, 20.0, np.random.default_rng(0))
    noise = noisy - x
    snr = 10 * np.log10(np.mean(x**2) / np.mean(noise**2))
    assert snr == pytest.approx(20.0, abs=0.5)
    assert np.array_equal(
        add_noise(np.zeros(10, np.float32), 10.0, np.random.default_rng(0)), np.zeros(10)
    )


def test_sample_spec_seeded_and_valid() -> None:
    names = list(CODECS)
    specs = [sample_spec(derive_rng(3, "x", i), POLICY, names) for i in range(200)]
    assert specs == [sample_spec(derive_rng(3, "x", i), POLICY, names) for i in range(200)]
    assert {s.codec for s in specs} == set(names)
    for s in specs:
        assert s.bitrate in CODECS[s.codec].bitrates or (
            not CODECS[s.codec].bitrates and s.bitrate is None
        )
        assert s.snr_db is None or 10 <= s.snr_db <= 30
    assert any(s.snr_db is not None for s in specs) and any(s.snr_db is None for s in specs)


def test_sample_spec_respects_weights() -> None:
    pol = Policy(codecs={"g711_mulaw": 1.0, "opus": 0.0})
    specs = {
        sample_spec(derive_rng(0, "x", i), pol, ["g711_mulaw", "opus"]).codec for i in range(50)
    }
    assert specs == {"g711_mulaw"}
    with pytest.raises(ValueError):
        sample_spec(derive_rng(0, "x", 0), Policy(codecs={}), ["opus"])


def test_policy_from_yaml() -> None:
    pol = Policy.from_yaml(Path(__file__).parents[1] / "configs" / "codecs.yaml")
    assert pol.codecs["g711_mulaw"] == 1.0 and pol.snr_db == (10.0, 30.0)


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> tuple[Path, Path]:
    sf = pytest.importorskip("soundfile")
    root = tmp_path / "raw"
    rows = []
    for i in range(3):
        rel = f"LA/train/flac/f{i}.flac"
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        sf.write(root / rel, synthetic(1.0, seed=i), 16000, format="FLAC")
        rows.append(ManifestRow(rel, i % 2, "train", source="synthetic"))
    manifest = tmp_path / "in.csv"
    write_manifest(rows, manifest)
    return root, manifest


@pytest.mark.skipif(ffmpeg_path() is None, reason="ffmpeg not installed")
def test_precompute_variants_seeded_and_reproducible(
    tiny_dataset: tuple[Path, Path], tmp_path: Path
) -> None:
    root, manifest = tiny_dataset
    runs = []
    for k in range(2):
        out = tmp_path / f"out{k}"
        n = precompute(
            manifest,
            root,
            out,
            out / "m.csv",
            "variants",
            2,
            7,
            POLICY,
            ["g711_mulaw", "g711_alaw"],
            workers=2,
        )
        rows = read_manifest(out / "m.csv")
        assert n == len(rows) == 3 * (1 + 2)  # clean + 2 variants per file
        assert sum(r.codec == "clean" for r in rows) == 3
        for r in rows:
            if r.codec != "clean":
                assert (out / r.path).is_file()
        runs.append([(r.path, r.label, r.codec, r.bitrate) for r in rows])
    assert runs[0] == runs[1]  # same seed -> identical manifest
    other = tmp_path / "out_seed8"
    precompute(
        manifest,
        root,
        other,
        other / "m.csv",
        "variants",
        2,
        8,
        POLICY,
        ["g711_mulaw", "g711_alaw"],
        workers=2,
    )
    assert [(r.path) for r in read_manifest(other / "m.csv")] != [p for p, *_ in runs[0]]


@pytest.mark.skipif(ffmpeg_path() is None, reason="ffmpeg not installed")
def test_precompute_per_codec_is_deterministic(
    tiny_dataset: tuple[Path, Path], tmp_path: Path
) -> None:
    sf = pytest.importorskip("soundfile")
    root, manifest = tiny_dataset
    outs = []
    for k in range(2):
        out = tmp_path / f"o{k}"
        precompute(
            manifest,
            root,
            out,
            out / "m.csv",
            "per-codec",
            0,
            k,
            POLICY,
            ["g711_mulaw", "g711_alaw"],
            include_clean=False,
        )
        outs.append(out)
    r0, r1 = read_manifest(outs[0] / "m.csv"), read_manifest(outs[1] / "m.csv")
    assert len(r0) == 3 * 2 and [r.path for r in r0] == [r.path for r in r1]  # seed-independent
    a, _ = sf.read(outs[0] / r0[0].path)
    b, _ = sf.read(outs[1] / r1[0].path)
    assert np.array_equal(a, b)
    assert {r.split for r in r0} == {"train"}
    assert [r.label for r in r0[:2]] == [0, 0]  # labels carried over
