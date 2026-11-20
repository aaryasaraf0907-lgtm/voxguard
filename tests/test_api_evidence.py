"""Tamper-evident evidence trail (api/evidence.py), the second-approver rule and the webhook's
redirect and backlog limits. Generated audio only."""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select, update

from conftest import FixedDetector, make_wav
from voxguard.api import evidence, webhook
from voxguard.api.db import (
    CallWindowRow,
    EvidenceRow,
    SessionRow,
    TransactionEventRow,
    TransactionRow,
)
from voxguard.api.deps import Settings
from voxguard.api.main import create_app

THR = 0.1


def _client(score: float = 0.5, **settings) -> TestClient:
    return TestClient(
        create_app(FixedDetector(score, threshold=THR), Settings(db_url="sqlite://", **settings))
    )


def _upload(c) -> str:
    return c.post("/v1/analyze", files={"file": ("a.wav", make_wav(1.0))}).json()["session_id"]


def _live_call(c) -> str:
    with c.websocket_connect("/v1/stream?sample_rate=16000") as ws:
        sid = ws.receive_json()["session_id"]
        x = (np.random.default_rng(0).standard_normal(16_000 * 7) * 0.1).astype("<f4")
        for i in range(0, len(x), 4000):
            ws.send_bytes(x[i : i + 4000].tobytes())
        ws.send_text(json.dumps({"type": "end"}))
        while ws.receive_json()["type"] != "final":
            pass
    return sid


def _activity(c) -> tuple[str, str, str]:
    """An upload with a held transfer released by two people, and a live call."""
    sid = _upload(c)
    d = c.post("/v1/transactions/check", json={"session_id": sid, "amount_inr": 75_000}).json()
    url = f"/v1/transactions/{d['decision_id']}/verify"
    c.post(url, json={"method": "callback", "verified_by": "Asha"})
    c.post(url, json={"method": "second_approver", "verified_by": "Manager"})
    return sid, d["decision_id"], _live_call(c)


def _verify(c, **q) -> dict:
    return c.get("/v1/evidence/verify", params=q).json()


def test_every_record_is_chained_in_order_and_checks_out() -> None:
    with _client() as c:
        sid, decision, live = _activity(c)
        v = _verify(c)
        mine = _verify(c, session_id=sid)["session_entries"]
        with c.app.state.session_factory() as db:
            rows = list(db.scalars(select(EvidenceRow).order_by(EvidenceRow.seq)))
    assert v["ok"] and v["problems"] == [] and v["entries"] == len(rows) == 8
    assert [r.kind for r in rows] == [
        "session",
        "decision",
        "event",
        "event",
        "event",
        "event",
        "session",
        "windows",
    ]
    assert [r.seq for r in rows] == list(range(1, 9)) and rows[0].prev_hash == evidence.GENESIS
    assert all(b.prev_hash == a.hash for a, b in zip(rows, rows[1:], strict=False))
    assert v["head"] == rows[-1].hash
    assert [e["kind"] for e in mine] == ["session", "decision", "event", "event", "event", "event"]


def test_an_empty_log_is_intact() -> None:
    with _client() as c:
        v = _verify(c)
    assert v == {**v, "ok": True, "entries": 0, "head": evidence.GENESIS}


@pytest.mark.parametrize(
    "tamper",
    [
        "edit_session",
        "edit_window",
        "delete_window",
        "edit_decision",
        "edit_step",
        "edit_entry",
        "delete_entry",
    ],
)
def test_tampering_is_detected(tamper: str) -> None:
    with _client() as c:
        sid, decision, live = _activity(c)
        with c.app.state.session_factory() as db:
            if tamper == "edit_session":  # e.g. lower a flagged score afterwards
                db.execute(
                    update(SessionRow).where(SessionRow.session_id == sid).values(fake_score=0.01)
                )
            elif tamper == "edit_window":
                db.execute(
                    update(CallWindowRow)
                    .where(CallWindowRow.session_id == live)
                    .values(window_score=0.0)
                )
            elif tamper == "delete_window":
                first = db.scalars(
                    select(CallWindowRow.id).where(CallWindowRow.session_id == live)
                ).first()
                db.execute(delete(CallWindowRow).where(CallWindowRow.id == first))
            elif tamper == "edit_decision":
                db.execute(update(TransactionRow).values(action="allow"))
            elif tamper == "edit_step":  # e.g. change who verified it
                db.execute(
                    update(TransactionEventRow)
                    .where(TransactionEventRow.verified_by == "Manager")
                    .values(verified_by="Asha")
                )
            elif tamper == "edit_entry":
                db.execute(
                    update(EvidenceRow).where(EvidenceRow.seq == 3).values(record_digest="0" * 64)
                )
            elif tamper == "delete_entry":
                db.execute(delete(EvidenceRow).where(EvidenceRow.seq == 4))
            db.commit()
        v = _verify(c)
        report = c.get(f"/v1/sessions/{sid}/report").text
    assert not v["ok"] and v["problems"]
    expected = {
        "edit_session": "record was altered",
        "edit_window": "record was altered",
        "delete_window": "record was altered",
        "edit_decision": "record was altered",
        "edit_step": "record was altered",
        "edit_entry": "entry was altered",
        "delete_entry": "entries missing",
    }[tamper]
    assert any(expected in p["problem"] for p in v["problems"])
    assert "does not match the records" in report


def test_the_report_shows_the_chain() -> None:
    with _client() as c:
        sid, _, _ = _activity(c)
        head = _verify(c)["head"]
        body = c.get(f"/v1/sessions/{sid}/report").text
    assert "Intact" in body and head in body and "/v1/evidence/verify?session_id=" in body


def test_unknown_session_is_404() -> None:
    with _client() as c:
        assert c.get("/v1/evidence/verify", params={"session_id": "nope"}).status_code == 404


def test_concurrent_writes_keep_one_unbroken_chain(tmp_path) -> None:
    settings = Settings(
        db_url=f"sqlite:///{tmp_path / 'e.db'}", max_running_analyses=4, max_waiting_analyses=64
    )
    with TestClient(create_app(FixedDetector(0.5, threshold=THR), settings)) as c:
        with ThreadPoolExecutor(12) as pool:
            sids = list(pool.map(lambda _: _upload(c), range(60)))
        with ThreadPoolExecutor(8) as pool:
            list(
                pool.map(
                    lambda s: c.post(
                        "/v1/transactions/check", json={"session_id": s, "amount_inr": 10}
                    ),
                    sids[:30],
                )
            )
        v = _verify(c)
    assert v["ok"] and v["entries"] == 60 + 30 * 2


# --- second approver ------------------------------------------------------------------------


def test_a_hold_needs_two_different_people() -> None:
    with _client() as c:
        sid = _upload(c)
        d = c.post("/v1/transactions/check", json={"session_id": sid, "amount_inr": 75_000}).json()
        url = f"/v1/transactions/{d['decision_id']}/verify"
        assert c.post(url, json={"method": "callback", "verified_by": "Asha"}).status_code == 200
        same = c.post(url, json={"method": "second_approver", "verified_by": " ASHA "})
        other = c.post(url, json={"method": "second_approver", "verified_by": "Manager"}).json()
    assert same.status_code == 422 and "two different people" in same.json()["detail"]
    assert other["status"] == "released"


def test_a_step_up_still_needs_only_one_person() -> None:
    with _client(0.12) as c:  # caution: step-up for a normal amount
        sid = _upload(c)
        d = c.post("/v1/transactions/check", json={"session_id": sid, "amount_inr": 100}).json()
        v = c.post(
            f"/v1/transactions/{d['decision_id']}/verify",
            json={"method": "otp", "verified_by": "Asha"},
        ).json()
    assert d["action"] == "step_up" and v["status"] == "released"


# --- webhook --------------------------------------------------------------------------------


def _server(handler_cls) -> tuple[ThreadingHTTPServer, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}/hook"


SECRET = "s3cret-for-tests-0123"


def test_webhook_events_carry_the_chain_head() -> None:
    got: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            got.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(204)
            self.end_headers()

        def log_message(self, *a) -> None:
            pass

    server, url = _server(Handler)
    try:
        with _client(alert_webhook_url=url, alert_webhook_secret=SECRET) as c:
            sid = _upload(c)
            c.post("/v1/transactions/check", json={"session_id": sid, "amount_inr": 75_000})
            head = _verify(c)["head"]
            for _ in range(100):
                if len(got) >= 2:
                    break
                time.sleep(0.05)
    finally:
        server.shutdown()
    assert [e["event"] for e in got] == ["call_flagged", "transaction_held"]
    assert got[-1]["evidence_head"] == head and got[0]["evidence_head"] != head


def test_a_redirecting_webhook_is_a_failed_delivery_and_is_not_followed(caplog) -> None:
    hits: list[str] = []

    class Target(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            hits.append("GET")
            self.send_response(200)
            self.end_headers()

        do_POST = do_GET

        def log_message(self, *a) -> None:
            pass

    target, target_url = _server(Target)

    class Redirect(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.send_response(302)
            self.send_header("Location", target_url)
            self.end_headers()

        def log_message(self, *a) -> None:
            pass

    redirect, url = _server(Redirect)
    try:
        n = webhook.Notifier(url, SECRET)
        n.send("call_flagged", {"session_id": "s"})
        n.close()
    finally:
        redirect.shutdown()
        target.shutdown()
    assert hits == [] and "not delivered: HTTPError" in caplog.text


def test_the_backlog_is_capped(monkeypatch, caplog) -> None:
    release = threading.Event()

    class Slow(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            release.wait(5)
            self.send_response(204)
            self.end_headers()

        def log_message(self, *a) -> None:
            pass

    monkeypatch.setattr(webhook, "MAX_PENDING", 3)
    server, url = _server(Slow)
    try:
        n = webhook.Notifier(url, SECRET)
        for i in range(10):
            n.send("live_alert", {"i": i})
        assert n.dropped == 7 and "dropped" in caplog.text
        release.set()
        n.close()
    finally:
        server.shutdown()
