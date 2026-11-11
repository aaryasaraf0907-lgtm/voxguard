import json
from pathlib import Path

import pytest

from voxguard import kaggle_runner as kr


def spec(**kw: object) -> kr.RunSpec:
    base = {"run": "rawnet2-clean", "config": "configs/rawnet2.yaml",
            "train_manifest": "asvspoof2019_la_train.csv"}  # fmt: skip
    return kr.RunSpec(**{**base, **kw})  # type: ignore[arg-type]


def test_kernel_slug_validates() -> None:
    assert kr.kernel_slug("rawnet2-clean") == "voxguard-rawnet2-clean"
    for bad in ("Raw_Net", "x" * 60, "a b"):
        with pytest.raises(ValueError):
            kr.kernel_slug(bad)


def test_run_from_kernel() -> None:
    assert kr.run_from_kernel("me/voxguard-rawnet2-clean") == "rawnet2-clean"
    with pytest.raises(ValueError):
        kr.run_from_kernel("me/other-kernel")


def test_rendered_kernel_compiles_and_embeds_spec() -> None:
    code = kr.render_kernel(spec(epochs=3, resume_run="rawnet2-clean"))
    compile(code, "run.py", "exec")
    assert '"epochs": 3' in code and '"resume_run": "rawnet2-clean"' in code
    assert '("last.pt", "best.pt", "metrics.jsonl")' in code  # resume keeps the earlier best


def test_metadata_is_private_gpu_with_sources() -> None:
    meta = kr.kernel_metadata("me", spec(), resume_from="me/voxguard-old")
    assert meta["id"] == "me/voxguard-rawnet2-clean"
    assert meta["is_private"] is True and meta["enable_gpu"] is True
    assert meta["kernel_sources"] == ["me/voxguard-old"]
    assert "me/voxguard-src" in meta["dataset_sources"]  # type: ignore[operator]


def test_link_tree_hard_links(tmp_path: Path) -> None:
    src = tmp_path / "a"
    (src / "x").mkdir(parents=True)
    (src / "x" / "f.txt").write_text("hi")
    assert kr.link_tree(src, tmp_path / "b") == 1
    out = tmp_path / "b" / "x" / "f.txt"
    assert out.read_text() == "hi" and out.stat().st_ino == (src / "x" / "f.txt").stat().st_ino


def test_push_dry_run_stages_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kr, "STAGE", tmp_path)
    folder = kr.push("me", spec(), "NvidiaTeslaT4", None, dry_run=True)
    meta = json.loads((folder / "kernel-metadata.json").read_text())
    assert meta["code_file"] == "run.py" and (folder / "run.py").is_file()


def test_augment_kernel_is_cpu_and_compiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(kr, "STAGE", tmp_path)
    folder = kr.push_augment("me", kr.AugmentSpec(run="eval-heldout"), dry_run=True)
    meta = json.loads((folder / "kernel-metadata.json").read_text())
    assert meta["enable_gpu"] is False and meta["is_private"] is True
    code = (folder / "run.py").read_text()
    compile(code, "run.py", "exec")
    assert '"codecs": ["gsm", "amr_wb"]' in code and '"subset": 10000' in code


def test_internet_only_for_pretrained_configs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(__file__).parents[1] / "configs"
    assert kr.needs_internet(root / "ssl_aasist.yaml") is True  # XLS-R from HF Hub
    assert kr.needs_internet(root / "rawnet2.yaml") is False
    monkeypatch.setattr(kr, "STAGE", tmp_path)
    ssl = spec(run="ssl-aasist-clean", config=str(root / "ssl_aasist.yaml"))
    folder = kr.push("me", ssl, "NvidiaTeslaT4", None, dry_run=True)
    assert json.loads((folder / "kernel-metadata.json").read_text())["enable_internet"] is True
    assert kr.kernel_metadata("me", spec())["enable_internet"] is False
