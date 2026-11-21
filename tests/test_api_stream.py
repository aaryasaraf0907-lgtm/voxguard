"""Live call monitor (WS /v1/stream): scoring rhythm, alert, stored verdict, limits and errors.
Generated audio only; the phone line is faked where a codec is used (no ffmpeg needed)."""

import json
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from starlette.websockets import WebSocketDisconnect

from voxguard.api import codec_sim
from voxguard.api.db import AlertRow, BlocklistRow, SessionRow
from voxguard.api.deps import Settings
from voxguard.api.gate import Busy
from voxguard.api.main import create_app
from voxguard.api.stream import HOP_S, WindowCutter, origin_allowed
from voxguard.data import codecs
from voxguard.inference import NUM_SAMPLES, TARGET_SR, Prediction, StubDetector


class ScriptedDetector:
    """Returns the given scores for successive windows (then repeats the last)."""

    model_version = "scripted-0"
    threshold = 0.5

    def __init__(self, scores: list[float]) -> None:
        self.scores, self.calls, self.rates = list(scores), 0, []

    def predict(self, waveform: np.ndarray, sample_rate: int) -> Prediction:
        self.rates.append((len(waveform), sample_rate))
        score = self.scores[min(self.calls, len(self.scores) - 1)]
        self.calls += 1
        return Prediction(score, "fake" if score >= 0.5 else "real", 0.5, 10.0, self.model_version)


def _client(det=None, **settings) -> TestClient:
    return TestClient(
        create_app(det or StubDetector(threshold=0.5), Settings(db_url="sqlite://", **settings))
    )


def _audio(seconds: float, rate: int, seed: int = 0) -> np.ndarray:
    return (np.random.default_rng(seed).standard_normal(int(seconds * rate)) * 0.1).astype("<f4")


def _send(ws, x: np.ndarray, rate: int, frame_s: float = 0.25) -> None:
    n = int(frame_s * rate)
    for i in range(0, len(x), n):
        ws.send_bytes(x[i : i + n].tobytes())


def _until_final(ws) -> list[dict]:
    out = []
    while True:
        m = ws.receive_json()
        out.append(m)
        if m["type"] in ("final", "error"):
            return out


def _call(client, query: str, x: np.ndarray, rate: int, end: bool = True) -> list[dict]:
    with client.websocket_connect(f"/v1/stream?{query}") as ws:
        ready = ws.receive_json()
        _send(ws, x, rate)
        if end:
            ws.send_text(json.dumps({"type": "end"}))
        return [ready, *_until_final(ws)]


def _rows(client, model) -> list:
    with client.app.state.session_factory() as db:
        return list(db.scalars(select(model)))


# --- windowing ------------------------------------------------------------------------------


def test_cutter_waits_for_one_window_then_every_hop_and_keeps_only_one_window() -> None:
    c = WindowCutter(48_000)
    assert c.window_n == int(np.ceil(NUM_SAMPLES * 48_000 / TARGET_SR)) and c.hop_n == 96_000
    due = [c.feed(np.zeros(12_000, np.float32)) for _ in range(44)]  # 11 s in 0.25 s frames
    hits = [d for d in due if d is not None]
    assert [round(t, 2) for _, t, _ in hits] == [4.25, 6.25, 8.25, 10.25]
    assert all(len(w) == c.window_n and behind == 0 for w, _, behind in hits)
    assert len(c.buffer) == c.window_n  # only the latest window is held


def test_cutter_scores_only_the_newest_window_when_audio_arrives_in_a_burst() -> None:
    c = WindowCutter(16_000)
    window, t_s, behind = c.feed(np.zeros(16_000 * 12, np.float32))
    assert t_s == 12.0 and behind == 3 and len(window) == c.window_n  # 4.04, 6.04, 8.04 skipped


# --- a call ---------------------------------------------------------------------------------


def test_scores_alert_once_and_store_the_verdict() -> None:
    det = ScriptedDetector([0.9, 0.7, 0.9, 0.1])
    with _client(det) as c:
        msgs = _call(c, "sample_rate=48000&caller_id=%2B91-1", _audio(11, 48_000), 48_000)
        rows, alerts, blocked = _rows(c, SessionRow), _rows(c, AlertRow), _rows(c, BlocklistRow)
    ready, *rest = msgs
    assert ready["type"] == "ready" and ready["threshold"] == 0.5 and ready["hop_s"] == HOP_S
    scores = [m for m in rest if m["type"] == "score"]
    assert [round(m["t_s"], 2) for m in scores] == [4.25, 6.25, 8.25, 10.25]
    assert [m["running_score"] for m in scores] == pytest.approx([0.9, 0.8, 0.8333333, 0.65])
    alerts_sent = [m for m in rest if m["type"] == "alert"]
    assert len(alerts_sent) == 1 and alerts_sent[0]["t_s"] == pytest.approx(6.25)  # >= rule
    assert alerts_sent[0]["level"] == "high"  # 0.8 is over 1.5 x the threshold (the alert level)
    final = rest[-1]
    assert final["type"] == "final" and final["stored"] and final["flagged"]
    assert final["fake_score"] == pytest.approx(0.65) and final["first_flag_s"] == pytest.approx(
        6.25
    )
    assert det.rates[0] == (int(np.ceil(NUM_SAMPLES * 48_000 / TARGET_SR)), 48_000)  # native rate
    (row,) = rows
    assert row.session_id == ready["session_id"] and row.caller_id == "+91-1" and row.flagged
    assert len(alerts) == 1 and blocked[0].hit_count == 1  # the usual prevention layer


def test_a_call_shorter_than_one_window_is_scored_once_at_the_end() -> None:
    det = ScriptedDetector([0.2])
    with _client(det) as c:
        msgs = _call(c, "sample_rate=16000", _audio(2.5, 16_000), 16_000)
    assert [m["type"] for m in msgs] == ["ready", "final"]
    assert msgs[-1]["stored"] and msgs[-1]["windows"] == 1 and det.rates == [(40_000, 16_000)]


def test_a_call_under_a_second_stores_nothing() -> None:
    with _client() as c:
        msgs = _call(c, "sample_rate=16000", _audio(0.5, 16_000), 16_000)
        assert msgs[-1] == {**msgs[-1], "type": "final", "stored": False, "windows": 0}
        assert _rows(c, SessionRow) == []


def test_hanging_up_without_end_still_stores_the_call(live_server) -> None:
    """A real server (TestClient cancels the handler when the client leaves): a caller who
    hangs up mid-call still leaves a stored session."""
    from websockets.sync.client import connect

    base, factory = live_server
    with connect(f"{base}/v1/stream?sample_rate=16000&caller_id=gone") as ws:
        assert json.loads(ws.recv())["type"] == "ready"
        x = _audio(5, 16_000)
        for i in range(0, len(x), 4000):
            ws.send(x[i : i + 4000].tobytes())
        assert json.loads(ws.recv())["type"] == "score"
    for _ in range(100):
        with factory() as db:
            rows = list(db.scalars(select(SessionRow)))
        if rows:
            break
        time.sleep(0.05)
    assert [r.caller_id for r in rows] == ["gone"]


def test_phone_line_is_simulated_per_window(monkeypatch) -> None:
    seen = []

    def fake_sim(x, rate, codec):
        seen.append((len(x), rate, codec))
        return np.zeros(NUM_SAMPLES, np.float32), TARGET_SR

    monkeypatch.setattr(codec_sim, "ffmpeg_path", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(codec_sim, "available_codecs", lambda: {c: True for c in codecs.CODECS})
    monkeypatch.setattr(codec_sim, "simulate_phone_line", fake_sim)
    det = ScriptedDetector([0.3])
    with _client(det) as c:
        msgs = _call(c, "sample_rate=16000&codec=g711_mulaw", _audio(7, 16_000), 16_000)
        (row,) = _rows(c, SessionRow)
    assert seen == [(NUM_SAMPLES, 16_000, "g711_mulaw")] * 2 and det.rates[0] == (
        NUM_SAMPLES,
        TARGET_SR,
    )
    assert msgs[0]["codec"] == "g711_mulaw" and row.codec_applied == "g711_mulaw"
    assert all(m["codec_ms"] is not None for m in msgs if m["type"] == "score")


def test_a_busy_server_skips_the_window_and_says_so(monkeypatch) -> None:
    with _client(ScriptedDetector([0.3])) as c:

        def busy(*args, **kwargs):
            raise Busy("queue full")

        monkeypatch.setattr(c.app.state.gate, "run", busy)
        msgs = _call(c, "sample_rate=16000", _audio(7, 16_000), 16_000)
    assert [m["type"] for m in msgs[1:]] == ["busy", "busy", "final"]
    assert msgs[-1]["skipped"] == 2 and not msgs[-1]["stored"]


def test_the_length_limit_ends_the_call_and_keeps_the_verdict() -> None:
    with _client(ScriptedDetector([0.3]), max_stream_seconds=5.0) as c:
        msgs = _call(c, "sample_rate=16000", _audio(9, 16_000), 16_000, end=False)
    kinds = [m["type"] for m in msgs]
    assert "limit" in kinds and kinds[-1] == "final" and msgs[-1]["stored"]
    assert msgs[-1]["duration_s"] == pytest.approx(5.0)


# --- refusals -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "detail"),
    [
        ("", "sample_rate"),
        ("sample_rate=7999", "between"),
        ("sample_rate=96000", "between"),
        ("sample_rate=16000&codec=nope", "Unknown codec"),
        ("sample_rate=x", "whole number"),
        ("sample_rate=16000&caller_id=" + "x" * 129, "128"),
    ],
)
def test_bad_parameters_are_refused(query: str, detail: str) -> None:
    with _client() as c, c.websocket_connect(f"/v1/stream?{query}") as ws:
        m = ws.receive_json()
        assert m["type"] == "error" and detail in m["detail"]
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_json()
        assert exc.value.code == 1008


@pytest.mark.parametrize(
    ("frame", "detail"),
    [
        (b"abc", "whole float32"),
        (np.array([np.nan], "<f4").tobytes(), "NaN"),
        (bytes(1024 * 1024 + 4), "at most"),
        ('{"type": "hello"}', "binary float32"),
    ],
)
def test_bad_frames_are_refused_and_nothing_is_stored(frame, detail: str) -> None:
    with _client() as c:
        with c.websocket_connect("/v1/stream?sample_rate=16000") as ws:
            ws.receive_json()
            ws.send_text(frame) if isinstance(frame, str) else ws.send_bytes(frame)
            m = ws.receive_json()
            assert m["type"] == "error" and detail in m["detail"]
        assert _rows(c, SessionRow) == []


def test_too_many_live_calls_are_refused() -> None:
    with _client(max_streams=1) as c, c.websocket_connect("/v1/stream?sample_rate=16000") as first:
        assert first.receive_json()["type"] == "ready"
        with c.websocket_connect("/v1/stream?sample_rate=16000") as second:
            m = second.receive_json()
            assert m["type"] == "error" and "too many live calls" in m["detail"]
            with pytest.raises(WebSocketDisconnect) as exc:
                second.receive_json()
            assert exc.value.code == 1013


def test_another_sites_page_cannot_open_a_stream() -> None:
    with _client() as c:
        with (
            pytest.raises(WebSocketDisconnect) as exc,
            c.websocket_connect(
                "/v1/stream?sample_rate=16000", headers={"origin": "https://evil.example"}
            ),
        ):
            pass
        assert exc.value.code == 1008
        with c.websocket_connect(
            "/v1/stream?sample_rate=16000", headers={"origin": "http://testserver"}
        ) as ws:
            assert ws.receive_json()["type"] == "ready"  # the API's own /monitor page


def test_origin_rules() -> None:
    assert origin_allowed(None, "localhost:8000", False, ())  # not a browser
    assert origin_allowed("http://localhost:8000/", "localhost:8000", False, ())
    assert not origin_allowed("http://localhost:8501", "localhost:8000", False, ())
    assert origin_allowed(
        "http://localhost:8501", "localhost:8000", False, ("http://localhost:8501",)
    )
    assert not origin_allowed(
        "http://localhost:8000", "localhost:8000", True, ()
    )  # wss needs https


def test_stream_origins_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("VOXGUARD_STREAM_ORIGINS", " http://a:1/ ,http://b:2,")
    monkeypatch.setenv("VOXGUARD_MAX_STREAMS", "2")
    s = Settings.from_env()
    assert s.stream_origins == ("http://a:1", "http://b:2") and s.max_streams == 2


# --- the page -------------------------------------------------------------------------------


def test_monitor_page_is_served_with_both_languages() -> None:
    from voxguard.ui.i18n import EN, HI

    with _client() as c:
        r = c.get("/monitor")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert r.headers["cache-control"] == "no-store"
    body = r.text
    assert "__STRINGS__" not in body and "/v1/stream" in body
    assert EN["monitor_alert_advice"] in body and HI["monitor_alert_advice"] in body
    for key in ("monitor_consent", "monitor_guidance_report"):  # consent notice, 1930 guidance
        assert EN[key] in body and HI[key] in body
    assert 'id="consent"' in body and 'id="guidance" hidden' in body  # guidance only after alert
    assert "<script src=" not in body and "https://" not in body  # nothing from the network


def test_one_window_over_the_threshold_is_not_an_alert_yet() -> None:
    """Seen on the real model: a genuine recording's first window was just over the threshold
    and the rest below it. The alert waits for two windows."""
    with _client(ScriptedDetector([0.9, 0.05, 0.05])) as c:
        msgs = _call(c, "sample_rate=16000", _audio(9, 16_000), 16_000)
    assert msgs[0]["min_alert_windows"] == 2 and msgs[0]["caution_factor"] == 1.5
    assert [m["type"] for m in msgs].count("alert") == 0
    assert msgs[1]["type"] == "score" and msgs[1]["flagged"]  # the raw state is still reported
    assert msgs[-1]["first_flag_s"] is None


def test_just_over_the_threshold_is_not_a_live_alert_but_still_flags_the_call() -> None:
    """Genuine calls often run just over the threshold (35 of 100 streamed ASVspoof genuine calls
    did, 20 Nov): the live alert waits for ALERT_FACTOR x it, the stored verdict does not."""
    with _client(ScriptedDetector([0.6, 0.6, 0.6])) as c:  # threshold 0.5, alert level 0.75
        msgs = _call(c, "sample_rate=16000", _audio(9, 16_000), 16_000)
        alerts, rows = _rows(c, AlertRow), _rows(c, SessionRow)
    assert msgs[0]["alert_factor"] == 1.5
    assert [m["type"] for m in msgs].count("alert") == 0
    assert all(m["flagged"] for m in msgs if m["type"] == "score")  # raw state still reported
    assert msgs[-1]["flagged"] and msgs[-1]["first_flag_s"] is None
    assert rows[0].flagged and len(alerts) == 1  # the prevention rule is unchanged


def test_an_alert_is_always_at_or_above_the_caution_boundary() -> None:
    """An alert's level is always "high"; that is only true while ALERT_FACTOR >= CAUTION_FACTOR."""
    from voxguard.api.risk import ALERT_FACTOR, CAUTION_FACTOR

    assert ALERT_FACTOR >= CAUTION_FACTOR


def test_alert_level_is_high_well_over_the_threshold() -> None:
    with _client(ScriptedDetector([0.9, 0.9])) as c:
        msgs = _call(c, "sample_rate=16000", _audio(7, 16_000), 16_000)
    (alert,) = [m for m in msgs if m["type"] == "alert"]
    assert alert["level"] == "high" and alert["t_s"] == pytest.approx(6.25)  # the second window


def test_caution_band_matches_the_ui() -> None:
    import ast  # read the constant without importing the Streamlit app

    from voxguard.api import stream

    tree = ast.parse((Path(stream.__file__).parents[1] / "ui" / "app.py").read_text())
    ui_factor = next(
        n.value.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "CAUTION_FACTOR"
    )
    assert ui_factor == stream.CAUTION_FACTOR


def test_a_detector_failure_on_a_short_call_still_sends_final() -> None:
    class Broken(ScriptedDetector):
        def predict(self, waveform, sample_rate):
            raise RuntimeError("boom")

    with _client(Broken([0.5])) as c:
        msgs = _call(c, "sample_rate=16000", _audio(2.0, 16_000), 16_000)
    assert msgs[-1]["type"] == "final" and not msgs[-1]["stored"] and msgs[-1]["windows"] == 0
