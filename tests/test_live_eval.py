"""scripts/live_eval.py: reading the inputs, building one-speaker calls, the measurements, the
report, the safety checks, and one run end to end against the real app (stub detector).
All audio is generated here."""

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("live_eval", ROOT / "scripts/live_eval.py")
assert SPEC and SPEC.loader
le = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = le
SPEC.loader.exec_module(le)


def _u(name: str, label: str, speaker: str = "S1") -> object:
    return le.Utterance(Path(f"/x/{name}.flac"), label, speaker)


def _measure(durations: dict[str, float]):
    return lambda u: durations.get(u.path.stem)


def test_asvspoof_keys_keep_only_untransmitted_eval_trials(tmp_path: Path) -> None:
    keys = tmp_path / "trial_metadata.txt"
    keys.write_text(
        "LA_0001 LA_E_1 none - - bonafide notrim eval\n"
        "LA_0001 LA_E_2 alaw ita_tx A07 spoof notrim eval\n"
        "LA_0002 LA_E_3 none - A10 spoof notrim eval\n"
        "LA_0002 LA_E_4 none - - bonafide notrim progress\n"
    )
    us = le.read_asvspoof(keys, tmp_path)
    assert [(u.path.name, u.label, u.speaker, u.attack) for u in us] == [
        ("LA_E_1.flac", "bonafide", "LA_0001", "-"),
        ("LA_E_3.flac", "spoof", "LA_0002", "A10"),
    ]


def test_calls_use_one_speaker_never_reuse_an_utterance_and_switch_once() -> None:
    us = [_u(f"b{i}", "bonafide", s) for s in ("S1", "S2") for i in range(12)]
    us += [_u(f"f{i}", "spoof", s) for s in ("S1", "S2") for i in range(12)]
    us = [le.Utterance(Path(f"/x/{u.speaker}-{u.path.stem}.flac"), u.label, u.speaker) for u in us]
    calls = le.build_calls(us, lambda u: 3.0, 2, 12.0, seed=1)
    assert [c.kind for c in calls] == ["genuine"] * 2 + ["fake"] * 2 + ["mixed"] * 2
    used = [s["file"] for c in calls for s in c.segments]
    assert len(used) == len(set(used))  # no utterance in two calls
    for c in calls:
        assert {s["file"].split("-")[0] for s in c.segments} == {c.speaker}
        assert c.duration_s >= 12.0 - le.PAUSE_S
        labels = [s["label"] for s in c.segments]
        if c.kind == "mixed":
            k = labels.index("spoof")
            assert set(labels[:k]) == {"bonafide"} and set(labels[k:]) == {"spoof"}
            assert c.switch_s == c.segments[k]["start_s"] and c.switch_s >= 6.0 - le.PAUSE_S
        else:
            assert len(set(labels)) == 1 and c.switch_s is None
    again = le.build_calls(us, lambda u: 3.0, 2, 12.0, seed=1)
    assert [c.segments for c in again] == [c.segments for c in calls]  # deterministic


def test_unreadable_utterances_are_skipped_and_short_speakers_give_fewer_calls(caplog) -> None:
    us = [_u(f"b{i}", "bonafide") for i in range(5)] + [_u("bad", "bonafide")]
    durations = {f"b{i}": 3.0 for i in range(5)}  # "bad" cannot be decoded
    calls = le.build_calls(us, _measure(durations), 3, 6.0, seed=0)
    genuine = [c for c in calls if c.kind == "genuine"]
    assert len(genuine) == 2 and "bad.flac" not in [s["file"] for c in calls for s in c.segments]
    assert "only 2 genuine calls" in caplog.text


def _measured(windows: list[float], alert_t: float | None, flagged: bool, thr: float = 0.5):
    scores = [
        {"t_s": 4.0 + 2 * i, "window": w, "running": w, "arrived_after_ms": 500.0}
        for i, w in enumerate(windows)
    ]
    return {
        "scores": scores,
        "alert": {"t_s": alert_t, "level": "high", "arrived_after_ms": 500.0} if alert_t else None,
        "final": {"type": "final", "threshold": thr, "flagged": flagged, "fake_score": 0.4},
        "ready": {"min_alert_windows": 2, "caution_factor": 1.5},
    }


def test_measurements_per_call_and_table() -> None:
    mixed = le.Call("mixed-00", "mixed", "S1", [{"end_s": 20.0}], switch_s=10.0)
    early = le.Call("mixed-01", "mixed", "S1", [{"end_s": 20.0}], switch_s=10.0)
    fake = le.Call("fake-00", "fake", "S1", [{"end_s": 20.0}])
    genuine = le.Call("genuine-00", "genuine", "S1", [{"end_s": 20.0}])
    results = [
        le.summarize_call(mixed, "gsm", _measured([0.1, 0.1, 0.1, 0.9], 12.5, True)),
        le.summarize_call(early, "gsm", _measured([0.9, 0.6], 6.0, True)),
        le.summarize_call(fake, "gsm", _measured([0.9, 0.9], 6.0, True)),
        le.summarize_call(genuine, "gsm", _measured([0.6, 0.1, 0.1], None, False)),
    ]
    assert results[0]["alert_after_switch_s"] == 2.5 and results[1]["alert_after_switch_s"] == -4.0
    assert results[3]["first_window_over"] and results[3]["alert_t_s"] is None
    rows = {r["kind"]: r for r in le.table(results, ["gsm"])}
    assert rows["mixed"]["alerted"] == 2 and rows["mixed"]["alerted_before_switch"] == 1
    assert rows["mixed"]["alert_after_switch_median_s"] == 2.5
    assert rows["fake"]["time_to_alert_s"] == 6.0 and rows["genuine"]["alerted"] == 0
    assert rows["genuine"]["first_window_over"] == 1 and rows["genuine"]["verdict_flagged"] == 0
    assert rows["genuine"]["score_arrival_median_ms"] == 500.0
    report = {
        "run": {
            "started": "t",
            "model_version": "m",
            "api_url": "ws://x",
            "commit": "c",
            "calls_per_type": 1,
            "call_seconds": 20,
            "source": "s",
            "speed": 1,
            "rule": "r",
            "codecs": ["gsm"],
            "unavailable": ["amr_nb"],
        },
        "table": le.table(results, ["gsm"]),
    }
    md = le.render_markdown(report)
    assert "| gsm | mixed | 2 | 2 (2) |" in md and "Not available on this server: amr_nb" in md


def test_reports_must_stay_outside_the_repo(tmp_path: Path) -> None:
    with pytest.raises(le.UsageError, match="outside the git tree"):
        le.check_out_dir(str(ROOT / "live-eval-out"))
    (tmp_path / "x").write_text("")
    with pytest.raises(le.UsageError, match="new or empty"):
        le.check_out_dir(str(tmp_path))


def test_remote_apis_are_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        le.main(
            [
                "--api-url",
                "ws://example.com:8000",
                "--out",
                str(tmp_path / "o"),
                "--labels",
                "x.csv",
                "--audio-dir",
                ".",
            ]
        )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="decodes with ffmpeg")
def test_end_to_end_against_the_real_app(live_server, tmp_path: Path) -> None:
    base, _ = live_server
    audio = tmp_path / "flac"
    audio.mkdir()
    lines = []
    rng = np.random.default_rng(0)
    for i in range(8):
        for label in ("bonafide", "spoof"):
            name = f"LA_E_{label[0]}{i}"
            sf.write(
                audio / f"{name}.flac", (rng.standard_normal(16_000 * 3) * 0.1).astype("f4"), 16_000
            )
            lines.append(
                f"LA_0001 {name} none - {'-' if label == 'bonafide' else 'A10'} {label} notrim eval"
            )
    keys = tmp_path / "keys.txt"
    keys.write_text("\n".join(lines) + "\n")
    out = tmp_path / "report"
    code = le.main(
        [
            "--api-url",
            base,
            "--out",
            str(out),
            "--asvspoof-keys",
            str(keys),
            "--asvspoof-audio",
            str(audio),
            "--calls-per-type",
            "1",
            "--call-seconds",
            "8",
            "--speed",
            "0",
            "--codecs",
            "none,nope",
        ]
    )
    report = json.loads(next(out.glob("*.json")).read_text())
    assert code == 0 and report["run"]["unavailable"] == ["nope"]
    assert [r["kind"] for r in report["results"]] == ["genuine", "fake", "mixed"]
    assert all(r["windows"] >= 2 and r["verdict_flagged"] is not None for r in report["results"])
    assert "2 windows" in report["run"]["rule"]
    assert not list(out.glob("*.wav")) and not list(out.glob("*.flac"))  # no audio written
    md = next(out.glob("*.md")).read_text()
    assert "| none | mixed | 1 |" in md
