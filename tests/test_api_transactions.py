"""Transfer checks against a call's risk (api/policy.py, api/transactions.py), the live call's
feature-level timeline, the incident report and the signed alert webhook. Generated audio only."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import FixedDetector, make_wav
from voxguard.api import policy
from voxguard.api.deps import Settings
from voxguard.api.main import create_app
from voxguard.api.webhook import signature
from voxguard.inference import Prediction

THR = 0.1  # FixedDetector's threshold below: low < 0.1 <= caution < 0.15 <= high


def _client(score: float = 0.05, **settings) -> TestClient:
    return TestClient(
        create_app(FixedDetector(score, threshold=THR), Settings(db_url="sqlite://", **settings))
    )


def _upload(c, caller: str | None = None) -> dict:
    data = {"caller_id": caller} if caller else {}
    return c.post("/v1/analyze", files={"file": ("a.wav", make_wav(1.0))}, data=data).json()


def _check(c, session_id: str, amount: float = 1000.0, **extra):
    return c.post(
        "/v1/transactions/check", json={"session_id": session_id, "amount_inr": amount, **extra}
    )


# --- policy ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("score", "windows", "tier"),
    [
        (0.05, None, "low"),
        (0.1, None, "caution"),
        (0.149, None, "caution"),
        (0.151, None, "high"),
        (0.9, 1, "unknown"),
        (0.9, 2, "high"),
        (0.0, 0, "unknown"),
    ],
)
def test_tiers(score: float, windows, tier: str) -> None:
    assert policy.tier_for(score, THR, windows) == tier


def test_a_caller_flagged_before_moves_one_tier_up() -> None:
    assert [policy.escalate(t, True) for t in policy.TIERS] == [
        "caution",
        "caution",
        "high",
        "high",
    ]
    assert [policy.escalate(t, False) for t in policy.TIERS] == list(policy.TIERS)


def test_default_table_and_high_value() -> None:
    table = policy.DEFAULT_POLICY
    assert policy.decide("caution", 49_999, 50_000, table) == ("step_up", False)
    assert policy.decide("caution", 50_000, 50_000, table) == ("hold", True)
    assert policy.decide("low", 10**6, 50_000, table) == ("allow", True)
    assert policy.decide("unknown", 10, 50_000, table) == ("step_up", False)


def test_release_rules() -> None:
    assert policy.released("step_up", {"otp"}) and not policy.released("step_up", set())
    assert not policy.released("hold", {"callback"})
    assert policy.released("hold", {"callback", "second_approver"})


@pytest.mark.parametrize(
    "raw",
    [
        "{",
        "[]",
        '{"low": ["allow", "allow"]}',
        json.dumps(
            {
                "unknown": ["x", "allow"],
                "low": ["allow", "allow"],
                "caution": ["hold", "hold"],
                "high": ["hold", "hold"],
            }
        ),
    ],
)
def test_invalid_policy_fails_at_startup(monkeypatch, raw: str) -> None:
    monkeypatch.setenv("VOXGUARD_TRANSACTION_POLICY", raw)
    with pytest.raises(ValueError, match="VOXGUARD_TRANSACTION_POLICY"):
        Settings.from_env()


def test_policy_and_limit_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("VOXGUARD_HIGH_VALUE_INR", "1000")
    monkeypatch.setenv(
        "VOXGUARD_TRANSACTION_POLICY",
        json.dumps(
            {
                "unknown": ["hold", "hold"],
                "low": ["allow", "step_up"],
                "caution": ["hold", "hold"],
                "high": ["hold", "hold"],
            }
        ),
    )
    s = Settings.from_env()
    assert s.high_value_inr == 1000 and dict(s.transaction_policy)["low"] == ("allow", "step_up")
    with TestClient(
        create_app(
            FixedDetector(0.05, threshold=THR), Settings(**{**s.__dict__, "db_url": "sqlite://"})
        )
    ) as c:
        assert _check(c, _upload(c)["session_id"], 1500).json()["action"] == "step_up"


# --- checks on a stored session --------------------------------------------------------------


@pytest.mark.parametrize(
    ("score", "amount", "tier", "action", "status"),
    [
        (0.05, 1000, "low", "allow", "allowed"),
        (0.12, 1000, "caution", "step_up", "pending"),
        (0.12, 75_000, "caution", "hold", "held"),
        (0.5, 10, "high", "hold", "held"),
    ],
)
def test_check_a_stored_session(score, amount, tier, action, status) -> None:
    with _client(score) as c:
        s = _upload(c)
        d = _check(c, s["session_id"], amount, beneficiary="  A/c 1234 ").json()
    assert (d["tier"], d["action"], d["status"]) == (tier, action, status)
    assert d["live"] is False and d["windows"] is None and d["score"] == pytest.approx(score)
    assert d["beneficiary"] == "A/c 1234" and d["high_value"] == (amount >= 50_000)
    assert d["reasons"][0] == f"risk_{tier}" and d["required_steps"] == policy.required_steps(
        action
    )
    assert [e["kind"] for e in d["events"]] == ["decided"]


def test_a_caller_flagged_on_an_earlier_call_raises_the_tier() -> None:
    with TestClient(create_app(_Scripted([0.5, 0.05]), Settings(db_url="sqlite://"))) as c:
        _upload(c, caller="+91-1")  # flagged: on the blocklist now
        first_low = _upload(c, caller="+91-1")
        d = _check(c, first_low["session_id"]).json()
        other = _check(c, _upload(c, caller="+91-2")["session_id"]).json()
    assert d["tier"] == "caution" and "caller_flagged_before" in d["reasons"]
    assert other["tier"] == "low" and "caller_flagged_before" not in other["reasons"]


def test_a_flagged_session_does_not_count_itself_as_an_earlier_flag() -> None:
    with _client(0.12) as c:
        d = _check(c, _upload(c, caller="+91-3")["session_id"]).json()
    assert d["tier"] == "caution" and "caller_flagged_before" not in d["reasons"]


@pytest.mark.parametrize(
    "body",
    [
        {"session_id": "x", "amount_inr": 0},
        {"session_id": "x", "amount_inr": 2e9},
        {"session_id": "", "amount_inr": 5},
        {"session_id": "x", "amount_inr": 5, "beneficiary": "b" * 65},
    ],
)
def test_bad_requests_are_refused(body) -> None:
    with _client() as c:
        assert c.post("/v1/transactions/check", json=body).status_code == 422


def test_unknown_session_is_404() -> None:
    with _client() as c:
        assert _check(c, "nope").status_code == 404


# --- verification -------------------------------------------------------------------------


def test_one_step_releases_a_step_up_and_the_trail_is_kept() -> None:
    with _client(0.12) as c:
        d = _check(c, _upload(c)["session_id"]).json()
        v = c.post(
            f"/v1/transactions/{d['decision_id']}/verify",
            json={"method": "otp", "verified_by": " Asha ", "note": "OTP read back"},
        ).json()
        listed = c.get("/v1/transactions").json()
    assert v["status"] == "released" and v["completed_steps"] == ["otp"]
    assert [(e["kind"], e["method"], e["verified_by"]) for e in v["events"]] == [
        ("decided", None, None),
        ("verified", "otp", "Asha"),
        ("released", None, None),
    ]
    assert listed[0]["status"] == "released"


def test_a_hold_needs_both_a_second_approver_and_a_call_back() -> None:
    with _client(0.5) as c:
        d = _check(c, _upload(c)["session_id"]).json()
        url = f"/v1/transactions/{d['decision_id']}/verify"
        one = c.post(url, json={"method": "callback", "verified_by": "Asha"}).json()
        wrong = c.post(url, json={"method": "otp", "verified_by": "Asha"})
        two = c.post(url, json={"method": "second_approver", "verified_by": "Manager"}).json()
        again = c.post(url, json={"method": "callback", "verified_by": "Asha"})
    assert one["status"] == "held" and wrong.status_code == 422  # OTP is not enough for a hold
    assert two["status"] == "released" and set(two["completed_steps"]) == {
        "callback",
        "second_approver",
    }
    assert again.status_code == 409


def test_nothing_to_verify_on_an_allowed_transfer_and_unknown_ids() -> None:
    with _client(0.05) as c:
        d = _check(c, _upload(c)["session_id"]).json()
        assert (
            c.post(
                f"/v1/transactions/{d['decision_id']}/verify",
                json={"method": "otp", "verified_by": "A"},
            ).status_code
            == 409
        )
        assert (
            c.post(
                "/v1/transactions/nope/verify", json={"method": "otp", "verified_by": "A"}
            ).status_code
            == 404
        )
        assert (
            c.post(
                f"/v1/transactions/{d['decision_id']}/verify",
                json={"method": "otp", "verified_by": ""},
            ).status_code
            == 422
        )


def test_list_filters_by_session() -> None:
    with _client(0.05) as c:
        a, b = _upload(c)["session_id"], _upload(c)["session_id"]
        _check(c, a), _check(c, b), _check(c, a)
        assert len(c.get(f"/v1/transactions?session_id={a}").json()) == 2
        assert len(c.get("/v1/transactions?limit=1").json()) == 1


# --- live calls -----------------------------------------------------------------------------


class _Scripted(FixedDetector):
    def __init__(self, scores: list[float]) -> None:
        super().__init__(scores[0], threshold=THR)
        self.scores, self.n = scores, 0

    def predict(self, waveform: np.ndarray, sample_rate: int) -> Prediction:
        s = self.scores[min(self.n, len(self.scores) - 1)]
        self.n += 1
        return Prediction(s, "fake" if s >= THR else "real", THR, 1.0, self.model_version)


_AUDIO = (np.random.default_rng(0).standard_normal(16_000 * 12) * 0.1).astype("<f4")


def _send(ws, start_s: float, end_s: float) -> None:
    """Send this stretch of the call in 0.25 s frames (the server scores as frames arrive)."""
    x = _AUDIO[int(start_s * 16_000) : int(end_s * 16_000)]
    for i in range(0, len(x), 4000):
        ws.send_bytes(x[i : i + 4000].tobytes())


def _read_scores(ws, n: int) -> None:
    got = 0
    while got < n:
        got += ws.receive_json()["type"] == "score"


def test_a_live_call_is_checked_on_its_running_risk_then_stored_with_its_timeline() -> None:
    with TestClient(create_app(_Scripted([0.5, 0.5, 0.5]), Settings(db_url="sqlite://"))) as c:
        with c.websocket_connect("/v1/stream?sample_rate=16000&caller_id=live") as ws:
            sid = ws.receive_json()["session_id"]
            _send(ws, 0, 5)  # one window (at 4.25 s)
            _read_scores(ws, 1)
            early = _check(c, sid, 75_000).json()
            _send(ws, 5, 7)  # the second (at 6.25 s)
            _read_scores(ws, 1)
            later = _check(c, sid, 75_000).json()
            ws.send_text(json.dumps({"type": "end"}))
            while ws.receive_json()["type"] != "final":
                pass
        timeline = c.get(f"/v1/sessions/{sid}/timeline").json()
        after = _check(c, sid, 10).json()
        listed = c.get(f"/v1/transactions?session_id={sid}").json()
    assert early["live"] and early["windows"] == 1 and early["tier"] == "unknown"
    assert "not_enough_audio" in early["reasons"] and early["action"] == "step_up"
    assert later["windows"] == 2 and later["tier"] == "high" and later["status"] == "held"
    assert [round(p["t_s"], 2) for p in timeline][:2] == [4.25, 6.25]
    assert all(p["running_score"] == pytest.approx(0.5) for p in timeline)
    assert after["live"] is False and after["tier"] == "high"  # from the stored session
    assert len(listed) == 3


def test_uploads_have_no_timeline_and_unknown_sessions_are_404() -> None:
    with _client() as c:
        sid = _upload(c)["session_id"]
        assert c.get(f"/v1/sessions/{sid}/timeline").json() == []
        assert c.get("/v1/sessions/nope/timeline").status_code == 404
        assert c.get("/v1/sessions/nope/report").status_code == 404


# --- incident report ------------------------------------------------------------------------


@pytest.mark.parametrize("lang", ["en", "hi"])
def test_incident_report(lang: str) -> None:
    from voxguard.ui.i18n import t

    with _client(0.5) as c:
        sid = _upload(c, caller="<script>alert(1)</script>")["session_id"]
        _check(c, sid, 75_000, beneficiary="A/c <b>1234</b>")
        r = c.get(f"/v1/sessions/{sid}/report?lang={lang}")
    body = r.text
    assert r.status_code == 200 and r.headers["cache-control"] == "no-store"
    assert t("report_flagged", lang) in body and t("report_no_timeline", lang) in body
    assert t("tx_status_held", lang) in body and "₹75,000.00" in body
    assert t("monitor_guidance_report", lang) in body and t("report_privacy", lang) in body
    assert "<script>alert" not in body and "&lt;script&gt;" in body and "<b>1234" not in body
    assert "<script" not in body and "src=" not in body and "https://" not in body


def test_report_of_a_live_call_has_its_chart(monkeypatch) -> None:
    with TestClient(create_app(_Scripted([0.05, 0.5]), Settings(db_url="sqlite://"))) as c:
        with c.websocket_connect("/v1/stream?sample_rate=16000") as ws:
            sid = ws.receive_json()["session_id"]
            _send(ws, 0, 7)
            _read_scores(ws, 2)
            ws.send_text(json.dumps({"type": "end"}))
            while ws.receive_json()["type"] != "final":
                pass
        body = c.get(f"/v1/sessions/{sid}/report").text
    assert "<svg" in body and "<polyline" in body and "6.2" in body


# --- signed webhook -------------------------------------------------------------------------


@pytest.fixture
def receiver():
    got: list[tuple[dict, bytes]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers["Content-Length"]))
            got.append(({k.lower(): v for k, v in self.headers.items()}, body))
            self.send_response(204)
            self.end_headers()

        def log_message(self, *args) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/hook", got
    server.shutdown()


def _wait(got: list, n: int) -> None:
    for _ in range(100):
        if len(got) >= n:
            return
        time.sleep(0.05)


SECRET = "s3cret-for-tests-0123"


def test_flagged_calls_and_held_transfers_are_posted_signed(receiver) -> None:
    url, got = receiver
    with _client(0.5, alert_webhook_url=url, alert_webhook_secret=SECRET) as c:
        sid = _upload(c, caller="+91-9")["session_id"]
        _check(c, sid, 75_000)
        _wait(got, 2)
    events = [(h["x-voxguard-event"], json.loads(b)) for h, b in got]
    assert [e for e, _ in events] == ["call_flagged", "transaction_held"]
    assert events[0][1]["session_id"] == sid and events[0][1]["source"] == "upload"
    for headers, body in got:  # the receiver can check every request
        assert headers["x-voxguard-signature"] == signature(SECRET, body)


def test_a_live_alert_is_posted(receiver) -> None:
    url, got = receiver
    with TestClient(
        create_app(
            _Scripted([0.5]),
            Settings(db_url="sqlite://", alert_webhook_url=url, alert_webhook_secret=SECRET),
        )
    ) as c:
        with c.websocket_connect("/v1/stream?sample_rate=16000") as ws:
            ws.receive_json()
            _send(ws, 0, 7)
            _read_scores(ws, 2)
            ws.send_text(json.dumps({"type": "end"}))
            while ws.receive_json()["type"] != "final":
                pass
        _wait(got, 2)
    kinds = [h["x-voxguard-event"] for h, _ in got]
    assert kinds == ["live_alert", "call_flagged"]
    assert json.loads(got[1][1])["source"] == "live"


def test_an_unreachable_webhook_does_not_break_requests(caplog) -> None:
    with _client(
        0.5, alert_webhook_url="http://127.0.0.1:9/nothing", alert_webhook_secret=SECRET
    ) as c:
        r = c.post("/v1/analyze", files={"file": ("a.wav", make_wav(1.0))})
        assert r.status_code == 200
    assert "not delivered" in caplog.text and "fake_score" not in caplog.text


@pytest.mark.parametrize(
    ("url", "secret"), [("ftp://x", SECRET), ("http://x", None), ("http://x", "short")]
)
def test_webhook_settings_are_checked(url, secret) -> None:
    with pytest.raises(ValueError, match="WEBHOOK"):
        Settings(alert_webhook_url=url, alert_webhook_secret=secret)
