"""scripts/operating_point.py and `real_audio_sanity.py advise`: exact bounds, the suggested
threshold, the trade-off with missed fakes, and the command's safety checks."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "operating_point", ROOT / "scripts/operating_point.py"
)
assert SPEC and SPEC.loader
op = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = op
SPEC.loader.exec_module(op)


@pytest.mark.parametrize(
    ("k", "n", "expected"),
    [(0, 10, 0.2589), (0, 59, 0.0495), (1, 20, 0.2161), (5, 100, 0.1023), (3, 3, 1.0)],
)
def test_clopper_pearson_upper_bounds(k: int, n: int, expected: float) -> None:
    assert op.far_upper_bound(k, n) == pytest.approx(expected, abs=5e-4)


def test_recordings_needed_follows_the_rule_of_three() -> None:
    assert op.clips_needed(0.05) == 59 and op.clips_needed(0.01) == 299
    assert op.far_upper_bound(0, op.clips_needed(0.05)) <= 0.05


def test_too_few_recordings_suggest_nothing() -> None:
    c = op.advise_condition([0.001] * 20, [], 0.01, (0.05,))
    (t,) = c["targets"]
    assert c["false_alarms"] == 0 and t["suggested"] is None and t["clips_needed"] == 59


def test_a_threshold_that_already_meets_the_target_is_kept() -> None:
    c = op.advise_condition([0.001] * 100, [0.5] * 10, 0.01, (0.05,))
    (t,) = c["targets"]
    assert t["suggested"] == 0.01 and t["keeps_current"] and t["missed_fakes"] == 0


def test_it_never_suggests_lowering_the_shipped_threshold() -> None:
    c = op.advise_condition([0.0001] * 400, [0.005], 0.01, (0.01, 0.05))
    assert all(t["suggested"] == 0.01 and t["keeps_current"] for t in c["targets"])


def test_a_higher_threshold_is_suggested_and_its_cost_shown() -> None:
    genuine = [0.001] * 95 + [0.02] * 3 + [0.03] * 2  # 5 of 100 over the current 0.01
    fakes = [0.015, 0.025, 0.9, 0.9]
    c = op.advise_condition(genuine, fakes, 0.01, (0.05,))
    (t,) = c["targets"]
    assert c["false_alarms"] == 5 and c["far_upper_95"] > 0.05 and c["missed_fakes"] == 0
    assert t["suggested"] > 0.03 and not t["keeps_current"] and t["false_alarms"] == 0
    assert t["missed_fakes"] == 2  # the trade-off is shown, not hidden
    assert op.far_upper_bound(t["false_alarms"], 100) <= 0.05


def _report(tmp_path: Path) -> Path:
    def row(gt: str, score: float, codec=None) -> dict:
        return {
            "file": "f",
            "ground_truth": gt,
            "outcome": "ok",
            "fake_score": score,
            "threshold": 0.01,
            "codec_requested": codec,
        }

    results = [row("genuine", 0.001) for _ in range(60)] + [row("spoof", 0.5)]
    results += [row("genuine", 0.02, "g711_mulaw")] + [{"outcome": "http_error"}]
    path = tmp_path / "sanity.json"
    path.write_text(
        json.dumps(
            {"run": {"started_utc": "t", "health": {"model_version": "m"}}, "results": results}
        )
    )
    return path


def test_advice_per_condition_and_report(tmp_path: Path) -> None:
    advice = op.advise(op.load_report(_report(tmp_path)), (0.05,))
    assert set(advice["conditions"]) == {"as uploaded", "g711_mulaw"}
    assert advice["conditions"]["as uploaded"]["targets"][0]["keeps_current"]
    md = op.render_markdown(advice)
    assert "recommendation only" in md and "VOXGUARD_THRESHOLD" in md and "## g711_mulaw" in md


def test_advise_command_writes_private_files_and_refuses_the_repo(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "real_audio_sanity", ROOT / "scripts/real_audio_sanity.py"
    )
    sanity = importlib.util.module_from_spec(spec)
    sys.modules["real_audio_sanity"] = sanity
    spec.loader.exec_module(sanity)
    report = _report(tmp_path)
    out = tmp_path / "advice"
    out.mkdir()
    assert sanity.main(["advise", "--report", str(report), "--out", str(out)]) == 0
    files = sorted(p.name.split("-")[0] + p.suffix for p in out.iterdir())
    assert files == ["advice.json", "advice.md"]
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in out.iterdir())
    assert sanity.main(["advise", "--report", str(report), "--out", str(ROOT)]) == 2
    assert sanity.main(["advise", "--report", str(report), "--out", str(out), "--target", "2"]) == 2
