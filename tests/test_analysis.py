"""voxguard.analysis: per-attack EER, Platt calibration and the sequential alert rule."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from voxguard import analysis as A
from voxguard.api.risk import ALERT_FACTOR, MIN_ALERT_WINDOWS


def test_file_id_handles_clean_and_degraded_paths() -> None:
    assert A.file_id("LA/ASVspoof2019_LA_eval/flac/LA_E_2834763.flac") == "LA_E_2834763"
    assert A.file_id("degraded/eval/gsm/LA_E_8877452__gsm_13000_0.flac") == "LA_E_8877452"


def test_load_protocol(tmp_path: Path) -> None:
    p = tmp_path / "proto.txt"
    p.write_text("LA_0039 LA_E_1 - A11 spoof\nLA_0014 LA_E_2 - - bonafide\n\n")
    assert A.load_protocol(p) == {"LA_E_1": ("LA_0039", "A11"), "LA_E_2": ("LA_0014", "bonafide")}


def test_per_attack_eer_scores_each_attack_against_all_bona_fide() -> None:
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    attacks = np.array(["bonafide"] * 4 + ["A07", "A07", "A10", "A10"])
    # A07 fakes all above every bona fide clip; A10 fakes inside the bona fide range
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.9, 0.95, 0.15, 0.35])
    out = A.per_attack_eer(scores, labels, attacks)
    assert set(out) == {"A07", "A10"} and out["A07"]["n_fake"] == 2
    assert out["A07"]["eer"] == pytest.approx(0.0)
    assert out["A10"]["eer"] > 0.2


def test_platt_fit_recovers_a_known_calibration() -> None:
    rng = np.random.default_rng(0)
    z = rng.normal(0, 3, 20_000)  # logit of the raw score
    a, b = 0.5, -1.0
    y = (rng.random(z.size) < 1 / (1 + np.exp(-(a * z + b)))).astype(int)
    scores = 1 / (1 + np.exp(-z))
    # the fit weights classes equally, so the intercept absorbs the class ratio
    prior = np.log((y == 0).sum() / (y == 1).sum())
    fa, fb = A.platt_fit(scores, y)
    assert fa == pytest.approx(a, abs=0.03) and fb == pytest.approx(b + prior, abs=0.08)


def test_calibration_is_monotonic_and_reduces_error_on_squashed_scores() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 5000)
    scores = np.where(y == 1, rng.beta(2, 30, y.size), rng.beta(1, 300, y.size))  # all near 0
    rep = A.calibration_report((scores, y), (scores, y), threshold=0.0127)
    cal = A.platt_apply(np.sort(scores), rep["a"], rep["b"])
    assert np.all(np.diff(cal) >= 0)  # order kept: no decision changes
    assert rep["after"]["ece"] < rep["before"]["ece"]
    assert 0.0127 < rep["threshold_calibrated"] < 1


def test_calibration_error_of_a_perfect_predictor_is_zero() -> None:
    y = np.array([0, 0, 1, 1])
    assert A.calibration_error(y.astype(float), y) == {"ece": 0.0, "brier": 0.0}


def test_simulated_calls_stay_within_one_group() -> None:
    scores = np.arange(30, dtype=float)
    groups = np.repeat(["a", "b", "c"], 10)
    calls = A.simulate_calls(scores, groups, n_calls=50, n_windows=4, seed=0)
    assert calls.shape == (50, 4)
    assert all(len({int(v) // 10 for v in c}) == 1 for c in calls)  # one group per call
    assert all(len(set(c)) == 4 for c in calls)  # different clips
    with pytest.raises(ValueError):
        A.simulate_calls(scores, groups, 5, 11, 0)


def test_first_alert_and_the_fixed_rule() -> None:
    running = np.array([[0.9, 0.9, 0.9], [0.1, 0.2, 0.6], [0.1, 0.1, 0.1]])
    fixed = A.fixed_rule(0.5, 3)
    assert np.isinf(fixed[0]) and list(fixed[1:]) == [0.5, 0.5]
    assert list(A.first_alert(running, fixed)) == [2, 3, 0]  # never on the first window


def test_sequential_rule_controls_whole_call_false_alarms() -> None:
    rng = np.random.default_rng(2)
    genuine = A.running_mean(rng.beta(2, 50, (4000, 8)))
    for alpha in (0.01, 0.05):
        t = A.sequential_rule(genuine, alpha)
        assert (A.first_alert(genuine, t) > 0).mean() <= alpha
        assert t[0] >= t[-1]  # a single window needs more evidence than eight
    fresh = A.running_mean(rng.beta(2, 50, (4000, 8)))  # held-out calls: close to the target
    assert (A.first_alert(fresh, A.sequential_rule(genuine, 0.05)) > 0).mean() < 0.08


def test_sequential_report_compares_rules_at_the_same_false_alarm_rate() -> None:
    rng = np.random.default_rng(3)

    def split(n: int) -> dict[str, np.ndarray]:
        labels = rng.integers(0, 2, n)
        scores = np.where(labels == 1, rng.beta(5, 20, n), rng.beta(2, 60, n))
        groups = [
            f"{s}|{c}"
            for s, c in zip(rng.choice(list("abcd"), n), rng.choice(["x", "y"], n), strict=True)
        ]
        return {"scores": scores, "labels": labels, "groups": np.array(groups)}

    rep = A.sequential_report(split(6000), split(6000), threshold=0.06)
    r = rep["rules"]
    assert set(r) == {"fixed", "fixed_at_threshold", "sequential_same_fa", "sequential_1pct"}
    assert rep["alert_factor"] == ALERT_FACTOR and rep["min_windows"] == MIN_ALERT_WINDOWS
    # the API's rule alerts at ALERT_FACTOR x the threshold, never before MIN_ALERT_WINDOWS
    assert r["fixed"]["thresholds"][MIN_ALERT_WINDOWS - 1] == pytest.approx(0.06 * ALERT_FACTOR)
    assert r["fixed"]["false_alarm_calls"] <= r["fixed_at_threshold"]["false_alarm_calls"]
    assert r["sequential_1pct"]["false_alarm_calls"] <= 0.03
    for name in r:
        assert 0 <= r[name]["fake_detected"] <= 1 and len(r[name]["thresholds"]) == A.CALL_WINDOWS
    assert set(rep["per_codec"]) == {"x", "y"}  # the codec is the group's last part
    for c in rep["per_codec"].values():
        assert c["genuine_clips"] > 0 and 0 <= c["false_alarm_calls"] <= 1


def test_analyse_model_refuses_misaligned_cached_scores() -> None:
    rows = [
        SimpleNamespace(path="x/LA_E_1.flac", label=0),
        SimpleNamespace(path="x/LA_E_2.flac", label=1),
    ]
    cache = {"clean": {"scores": np.array([0.1, 0.9]), "labels": np.array([1, 0])}}
    proto = ({"LA_E_1": ("s", "bonafide"), "LA_E_2": ("s", "A07")}, {})
    with pytest.raises(ValueError, match="line up"):
        A.analyse_model(cache, {"clean": rows}, None, proto, None)


def test_analyse_model_without_dev_scores_gives_per_attack_only() -> None:
    rows = [SimpleNamespace(path=f"x/LA_E_{i}.flac", label=int(i >= 2)) for i in range(4)]
    cache = {"clean": {"scores": np.array([0.1, 0.2, 0.8, 0.9]), "labels": np.array([0, 0, 1, 1])}}
    proto = ({f"LA_E_{i}": ("s", "bonafide" if i < 2 else "A07") for i in range(4)}, {})
    out = A.analyse_model(cache, {"clean": rows}, None, proto, 0.5)
    assert set(out) == {"per_attack"} and out["per_attack"]["clean"]["A07"]["eer"] == 0.0


# --- evaluate.py wiring ----------------------------------------------------------------------


def _manifest(path: Path, rows: list[tuple[str, int, str]]) -> Path:
    path.write_text(
        "path,label,split,codec,bitrate,source\n"
        + "".join(f"{p},{y},dev,{c},,asvspoof2019_la\n" for p, y, c in rows)
    )
    return path


def test_dev_scores_are_saved_with_their_files_and_must_line_up(tmp_path: Path) -> None:
    from voxguard.evaluate import EvalConfig, save_dev_scores

    clean = _manifest(
        tmp_path / "c.csv", [("LA_D_1.flac", 0, "clean"), ("LA_D_2.flac", 1, "clean")]
    )
    phone = _manifest(tmp_path / "p.csv", [("LA_D_1.flac", 0, "clean"),  # dropped, as in scoring
                                           ("d/LA_D_1__gsm.flac", 0, "gsm"),
                                           ("d/LA_D_2__opus.flac", 1, "opus")])  # fmt: skip
    out = tmp_path / "dev_scores.npz"
    save_dev_scores(out, (np.array([0.1, 0.9]), np.array([0, 1])),
                    (np.array([0.2, 0.8]), np.array([0, 1]), ["gsm", "opus"]),
                    [clean], [phone], EvalConfig())  # fmt: skip
    z = np.load(out)
    assert list(z["phone_files"]) == ["d/LA_D_1__gsm.flac", "d/LA_D_2__opus.flac"]
    assert list(z["clean_files"]) == ["LA_D_1.flac", "LA_D_2.flac"]
    with pytest.raises(ValueError, match="line up"):
        save_dev_scores(out, (np.array([0.1]), np.array([0])),
                        (np.array([0.2, 0.8]), np.array([0, 1]), ["gsm", "opus"]),
                        [clean], [phone], EvalConfig())  # fmt: skip


def test_results_markdown_renders_the_analysis_tables() -> None:
    from voxguard.evaluate import render_family

    rule = {"thresholds": [1.0] * 8, "false_alarm_calls": 0.01, "fake_detected": 0.97,
            "fake_median_windows": 2.0, "mixed_early_alert": 0.005, "mixed_detected": 0.9,
            "mixed_median_windows_after_switch": 2.0}  # fmt: skip
    res = {
        "family": "ssl_aasist", "matrix": {}, "models": {},
        "analysis": {"codec_aug": {
            "per_attack": {"clean": {"A10": {"eer": 0.032, "n_fake": 300}},
                           "degraded": {"A10": {"eer": 0.036, "n_fake": 1200}}},
            "calibration": {"a": 1.5, "b": 2.0, "before": {"ece": 0.4, "brier": 0.3},
                            "after": {"ece": 0.02, "brier": 0.03}, "threshold": 0.0127,
                            "threshold_calibrated": 0.41},
            "sequential": {"n_calls": 2000, "call_windows": 8, "switch": 4,
                           "fixed_dev_false_alarm_calls": 0.02,
                           "rules": {"fixed": rule, "sequential_1pct": rule},
                           "per_codec": {"opus": {**rule, "genuine_clips": 1066}}},
        }},
    }  # fmt: skip
    md = render_family(res)
    assert "| A10 | 3.20 | 3.60 |" in md
    assert "0.400 -> 0.020" in md and "0.0127 -> 0.410" in md
    assert "Fixed: 1.5 x the phone threshold, from 2 windows" in md
    assert "| 1.00 | 97.00 | 2.0 |" in md
    assert "| opus | 1066 | 1.00 | 97.00 | 0.50 | 90.00 |" in md
    assert "EER (%) per attack" not in render_family({**res, "analysis": {}})
