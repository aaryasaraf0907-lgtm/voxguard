import json
from pathlib import Path

import numpy as np
import pytest

from voxguard.evaluate import (
    evaluate_condition,
    load_eval_config,
    load_results,
    render_markdown,
)


def _fake_family() -> dict:
    def block(eer: float, n: int = 100) -> dict:
        return {"eer": eer, "eer_threshold": 0.5, "accuracy_at_eer": 1 - eer,
                "accuracy_at_0.5": 0.9, "confusion_at_eer": {}, "n": n}  # fmt: skip

    return {
        "family": "rawnet2",
        "meta": {"generated_at": "2026-11-08T00:00:00+00:00", "max_samples": None},
        "matrix": {
            "clean": {
                "clean": {"pooled": block(0.0123), "per_codec": {}},
                "degraded": {
                    "pooled": block(0.2),
                    "per_codec": {"gsm": block(0.25), "opus": block(0.15)},
                },
            },
            "codec_aug": {
                "clean": {"pooled": block(0.015), "per_codec": {}},
                "degraded": {"pooled": block(0.05), "per_codec": {"gsm": block(0.06)}},
            },
        },
        "latency": {"cpu": {"n": 50, "mean_ms": 40.0, "p50_ms": 39.5, "p95_ms": 45.25}},
    }


def test_evaluate_condition_pooled_and_per_codec() -> None:
    scores = np.array([0.1, 0.9, 0.2, 0.8, 0.6, 0.4, 0.3, 0.7])
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    codecs = ["gsm"] * 4 + ["opus"] * 4
    out = evaluate_condition(scores, labels, codecs)
    assert out["per_codec"]["gsm"]["eer"] == 0.0  # perfectly separated
    assert out["per_codec"]["opus"]["eer"] == 0.5  # 0.6(R) > 0.4(F), 0.7(F) > 0.3(R)
    assert out["pooled"]["n"] == 8


def test_evaluate_condition_skips_single_class_codec() -> None:
    out = evaluate_condition(np.array([0.1, 0.9, 0.5]), np.array([0, 1, 1]), ["a", "a", "b"])
    assert set(out["per_codec"]) == {"a"}


def test_render_markdown_uses_json_numbers() -> None:
    md = render_markdown({"rawnet2": _fake_family()})
    assert "Do not edit by hand" in md
    assert "| Clean only | 1.23 | 20.00 |" in md
    assert "| Codec-augmented | 1.50 | 5.00 |" in md
    assert "| Train | gsm | opus |" in md
    assert "| Codec-augmented | 6.00 | not run |" in md  # opus missing for codec_aug
    assert "| cpu | 39.5 | 45.2 | 40.0 | 50 |" in md


def test_render_marks_missing_conditions() -> None:
    fam = _fake_family()
    del fam["matrix"]["codec_aug"]
    md = render_markdown({"rawnet2": fam})
    assert "| Codec-augmented | not run | not run |" in md


def test_render_heldout_section_separate_from_matrix() -> None:
    fam = _fake_family()
    assert "Unseen-codec" not in render_markdown({"rawnet2": fam})  # optional section
    block = fam["matrix"]["clean"]["degraded"]["per_codec"]["gsm"]
    fam["matrix"]["clean"]["heldout"] = {"pooled": block, "per_codec": {"amr_wb": block}}
    md = render_markdown({"rawnet2": fam})
    assert "### Unseen-codec EER (%)" in md and "| Train | amr_wb |" in md
    assert "| Clean only | 25.00 |" in md and "| Codec-augmented | not run |" in md
    assert "| Clean only | 1.23 | 20.00 |" in md  # 2x2 unchanged


def test_render_flags_smoke_runs() -> None:
    fam = _fake_family()
    fam["meta"]["max_samples"] = 100
    assert "Smoke/partial run" in render_markdown({"rawnet2": fam})


def test_load_results_ignores_summary(tmp_path: Path) -> None:
    (tmp_path / "rawnet2.json").write_text(json.dumps(_fake_family()))
    (tmp_path / "summary.json").write_text("{}")
    assert set(load_results(tmp_path)) == {"rawnet2"}


def test_eval_config_validation(tmp_path: Path) -> None:
    cfg = load_eval_config(Path(__file__).parents[1] / "configs" / "eval.yaml")
    assert set(cfg.models["rawnet2"]) == {"clean", "codec_aug"}
    bad = tmp_path / "bad.yaml"
    bad.write_text("models:\n  rawnet2:\n    noisy: x.pt\n")
    with pytest.raises(ValueError, match="train conditions"):
        load_eval_config(bad)
    bad.write_text("typo_key: 1\n")
    with pytest.raises(ValueError, match="typo_key"):
        load_eval_config(bad)


def test_end_to_end_with_tiny_checkpoints(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("soundfile")
    from voxguard.data.manifests import read_manifest, write_manifest
    from voxguard.data.synthetic import write_synthetic
    from voxguard.evaluate import EvalConfig, main, run_eval
    from voxguard.train import Config, DataConfig, TrainConfig, train

    data = tmp_path / "data"
    m = write_synthetic(data, n=16, seed=0)
    # "degraded" set: same clips tagged with two codecs (codec column only matters here)
    rows = read_manifest(m["dev"])
    degraded = tmp_path / "deg.csv"
    write_manifest(
        [
            r.__class__(r.path, r.label, r.split, c, "", r.source)
            for r in rows
            for c in ("gsm", "opus")
        ]
        + rows,  # clean rows must be excluded from the degraded set
        degraded,
    )
    ckpts = {}
    for cond in ("clean", "codec_aug"):
        out = tmp_path / cond
        train(Config(
            model={"name": "rawnet2", "gru_hidden": 16, "gru_layers": 1},
            data=DataConfig(train_manifests=[str(m["train"])], dev_manifests=[str(m["dev"])],
                            roots=[str(data)], num_samples=16_000),
            train=TrainConfig(epochs=1, batch_size=8, class_weights=[0.5, 0.5],
                              out_dir=str(out), device="cpu"),
        ))  # fmt: skip
        ckpts[cond] = str(out / "best.pt")

    cfg = EvalConfig(
        models={"rawnet2": ckpts, "missing": {"clean": str(tmp_path / "nope.pt")}},
        test_sets={"clean": [str(m["dev"])], "degraded": [str(degraded)]},
        roots=[str(data)], batch_size=4, device="cpu", latency_clips=3,
        latency_devices=["cpu"], out_dir=str(tmp_path / "results"),
        docs_path=str(tmp_path / "results.md"),
    )  # fmt: skip
    fams = run_eval(cfg)
    res = fams["rawnet2"]
    for train_cond in ("clean", "codec_aug"):
        for test_cond in ("clean", "degraded"):
            assert 0.0 <= res["matrix"][train_cond][test_cond]["pooled"]["eer"] <= 1.0
    deg = res["matrix"]["clean"]["degraded"]
    assert set(deg["per_codec"]) == {"gsm", "opus"} and "clean" not in deg["per_codec"]
    assert deg["pooled"]["n"] == 2 * len(rows)
    assert res["latency"]["cpu"]["n"] == 3
    assert fams["missing"]["matrix"] == {}  # skipped, not crashed
    assert json.loads((tmp_path / "results" / "rawnet2.json").read_text())["family"] == "rawnet2"
    assert torch.__version__  # torch was used

    cfg_file = tmp_path / "eval.yaml"
    import yaml

    cfg_file.write_text(yaml.safe_dump(cfg.__dict__))
    main(["--config", str(cfg_file), "--render-only"])
    md = (tmp_path / "results.md").read_text()
    assert "## rawnet2" in md and "## missing" in md
