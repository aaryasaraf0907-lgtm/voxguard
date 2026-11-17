from pathlib import Path

import numpy as np
import pytest

from voxguard.evaluate import render_markdown
from voxguard.plots import det_points, make_figures, scores_path


def test_det_points_trace_false_alarms_against_misses() -> None:
    fa, miss = det_points(np.array([0.9, 0.8, 0.3, 0.1]), np.array([1, 0, 1, 0]))
    assert fa.tolist() == [0, 0, 0.5, 0.5, 1]  # threshold lowered one clip at a time
    assert miss.tolist() == [1, 0.5, 0.5, 0, 0]
    with pytest.raises(ValueError):
        det_points(np.array([0.1, 0.2]), np.array([1, 1]))


def _family() -> dict:
    block = {"eer": 0.02, "n": 4, "accuracy_at_eer": 0.98, "accuracy_at_0.5": 0.9}
    return {
        "family": "fam",
        "meta": {},
        "matrix": {
            "clean": {"clean": {"pooled": block}, "degraded": {"pooled": block,
                      "per_codec": {"opus": block}}},
            "codec_aug": {"clean": {"pooled": block}, "heldout": {"pooled": block,
                          "per_codec": {"gsm": {**block, "eer": 0.03}}}},
        },
    }  # fmt: skip


def test_figures_written_and_linked_from_the_doc(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    rng = np.random.default_rng(0)
    for train in ("clean", "codec_aug"):
        for test in ("clean", "degraded"):
            path = scores_path(tmp_path / "scores", "fam", train, test)
            path.parent.mkdir(parents=True, exist_ok=True)
            labels = np.repeat([0, 1], 50)
            np.savez(path, scores=rng.random(100) * 0.5 + 0.5 * labels, labels=labels)
    figs = make_figures({"fam": _family()}, tmp_path / "scores", tmp_path / "figures", "figures/")
    assert [p for _, p in figs["fam"]] == [
        "figures/fam_eer_by_condition.png",
        "figures/fam_det.png",
    ]
    for _, rel in figs["fam"]:
        assert (tmp_path / rel).read_bytes()[:4] == b"\x89PNG"
    md = render_markdown({"fam": _family()}, figs)
    assert "![fam DET curves](figures/fam_det.png)" in md


def test_det_figure_skipped_without_cached_scores(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    figs = make_figures({"fam": _family()}, tmp_path / "none", tmp_path / "figures", "figures/")
    assert [p for _, p in figs["fam"]] == ["figures/fam_eer_by_condition.png"]
