from datetime import UTC, datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from conftest import FixedDetector, make_wav
from voxguard.api.db import AlertRow, BlocklistRow, SessionRow

T0 = datetime(2026, 11, 2, 12, 0, 0, tzinfo=UTC)


def _session(sid: str, at: datetime, flagged: bool = False, caller: str | None = None):
    return SessionRow(
        session_id=sid, created_at=at, caller_id=caller, label="fake" if flagged else "real",
        fake_score=0.9 if flagged else 0.1, threshold=0.5, flagged=flagged,
        model_version="t", latency_ms=1.0,
    )  # fmt: skip


def _seed(client: TestClient, *rows) -> None:
    with client.app.state.session_factory() as db:
        for r in rows:
            db.add(r)
            db.flush()  # keep insert order explicit (alerts reference sessions)
        db.commit()


def _alert(aid: str, at: datetime, sid: str = "s"):
    return AlertRow(alert_id=aid, created_at=at, session_id=sid, fake_score=0.9, message="m")


# --- sessions --------------------------------------------------------------------------


def test_sessions_empty(client: TestClient) -> None:
    r = client.get("/v1/sessions")
    assert r.status_code == 200 and r.json() == []


def test_sessions_newest_first_with_utc_timestamps(client: TestClient) -> None:
    _seed(client, *(_session(f"s{i}", T0 + timedelta(minutes=i)) for i in (1, 3, 2)))
    body = client.get("/v1/sessions").json()
    assert [s["session_id"] for s in body] == ["s3", "s2", "s1"]
    assert body[0]["created_at"] == "2026-11-02T12:03:00Z"  # the serialized form, not the object


def test_sessions_flagged_filter(client: TestClient) -> None:
    _seed(client, _session("a", T0), _session("b", T0 + timedelta(seconds=1), True))
    assert [s["session_id"] for s in client.get("/v1/sessions?flagged_only=true").json()] == ["b"]
    assert len(client.get("/v1/sessions?flagged_only=false").json()) == 2
    assert len(client.get("/v1/sessions").json()) == 2


def test_sessions_limit(client: TestClient) -> None:
    _seed(client, *(_session(f"s{i}", T0 + timedelta(seconds=i)) for i in range(5)))
    assert [s["session_id"] for s in client.get("/v1/sessions?limit=2").json()] == ["s4", "s3"]


@pytest.mark.parametrize("limit", ["0", "-1", "501", "abc"])
def test_sessions_invalid_limit(client: TestClient, limit: str) -> None:
    assert client.get(f"/v1/sessions?limit={limit}").status_code == 422


def test_sessions_limit_bounds_accepted(client: TestClient) -> None:
    assert client.get("/v1/sessions?limit=1").status_code == 200
    assert client.get("/v1/sessions?limit=500").status_code == 200


def test_sessions_from_analyze_round_trip(client_factory) -> None:
    with client_factory(FixedDetector(0.9)) as c:
        sid = c.post(
            "/v1/analyze", files={"file": ("a.wav", make_wav(0.5))}, data={"caller_id": " bob "}
        ).json()["session_id"]
        (s,) = c.get("/v1/sessions?flagged_only=true").json()
        assert s["session_id"] == sid and s["caller_id"] == "bob" and s["flagged"] is True
        assert s["created_at"].endswith("Z")


# --- blocklist -------------------------------------------------------------------------


def test_blocklist_empty(client: TestClient) -> None:
    assert client.get("/v1/blocklist").json() == []


def test_blocklist_ordering_and_structure(client: TestClient) -> None:
    def row(cid: str, hits: int, last_min: int) -> BlocklistRow:
        return BlocklistRow(
            caller_id=cid, hit_count=hits, first_seen=T0, last_seen=T0 + timedelta(minutes=last_min)
        )

    _seed(client, row("low", 1, 50), row("old_tie", 3, 1), row("new_tie", 3, 9), row("top", 5, 0))
    body = client.get("/v1/blocklist").json()
    assert [b["caller_id"] for b in body] == ["top", "new_tie", "old_tie", "low"]
    assert set(body[0]) == {"caller_id", "hit_count", "first_seen", "last_seen"}
    assert body[0]["first_seen"] == "2026-11-02T12:00:00Z"


def test_blocklist_via_analyze(client_factory) -> None:
    with client_factory(FixedDetector(0.9)) as c:
        for _ in range(2):
            c.post("/v1/analyze", files={"file": ("a.wav", make_wav(0.5))}, data={"caller_id": "x"})
        (entry,) = c.get("/v1/blocklist").json()
        assert entry["caller_id"] == "x" and entry["hit_count"] == 2


# --- alerts ----------------------------------------------------------------------------


def _alerts_fixture(client: TestClient) -> None:
    _seed(
        client,
        _session("s", T0, True),
        *(_alert(f"a{i}", T0 + timedelta(minutes=i)) for i in (2, 1, 3)),
    )


def test_alerts_empty(client: TestClient) -> None:
    assert client.get("/v1/alerts").json() == []


def test_alerts_without_since_newest_first_with_limit(client: TestClient) -> None:
    _alerts_fixture(client)
    assert [a["alert_id"] for a in client.get("/v1/alerts").json()] == ["a3", "a2", "a1"]
    assert [a["alert_id"] for a in client.get("/v1/alerts?limit=2").json()] == ["a3", "a2"]


def test_alerts_since_strictly_greater_oldest_first(client: TestClient) -> None:
    _alerts_fixture(client)
    boundary = (T0 + timedelta(minutes=1)).isoformat()
    body = client.get("/v1/alerts", params={"since": boundary}).json()
    assert [a["alert_id"] for a in body] == ["a2", "a3"]  # a1 is exactly at since: excluded


def test_alerts_since_with_limit_returns_oldest_newer(client: TestClient) -> None:
    _alerts_fixture(client)
    body = client.get("/v1/alerts", params={"since": T0.isoformat(), "limit": 2}).json()
    assert [a["alert_id"] for a in body] == ["a1", "a2"]


def test_alerts_since_timezone_aware_is_normalized(client: TestClient) -> None:
    _alerts_fixture(client)
    # 17:31+05:30 == 12:01 UTC, the a1 boundary
    since = datetime(2026, 11, 2, 17, 31, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    body = client.get("/v1/alerts", params={"since": since.isoformat()}).json()
    assert [a["alert_id"] for a in body] == ["a2", "a3"]


def test_alerts_since_naive_is_utc(client: TestClient) -> None:
    _alerts_fixture(client)
    body = client.get("/v1/alerts", params={"since": "2026-11-02T12:01:00"}).json()
    assert [a["alert_id"] for a in body] == ["a2", "a3"]


def test_alerts_since_z_suffix(client: TestClient) -> None:
    _alerts_fixture(client)
    body = client.get("/v1/alerts", params={"since": "2026-11-02T12:02:00Z"}).json()
    assert [a["alert_id"] for a in body] == ["a3"]


@pytest.mark.parametrize("since", ["not-a-date", "2026-13-45", "0001-01-01T00:00:00+14:00"])
def test_alerts_invalid_since_is_422(client: TestClient, since: str) -> None:
    assert client.get("/v1/alerts", params={"since": since}).status_code == 422


@pytest.mark.parametrize("limit", ["0", "501", "x"])
def test_alerts_invalid_limit(client: TestClient, limit: str) -> None:
    assert client.get(f"/v1/alerts?limit={limit}").status_code == 422


def test_alerts_serialized_timestamps_are_utc(client: TestClient) -> None:
    _alerts_fixture(client)
    assert client.get("/v1/alerts?limit=1").json()[0]["created_at"] == "2026-11-02T12:03:00Z"


def test_alerts_structure_from_analyze(client_factory) -> None:
    with client_factory(FixedDetector(0.9)) as c:
        sid = c.post("/v1/analyze", files={"file": ("a.wav", make_wav(0.5))}).json()["session_id"]
        (a,) = c.get("/v1/alerts").json()
        assert set(a) == {
            "alert_id",
            "created_at",
            "session_id",
            "caller_id",
            "fake_score",
            "message",
        }
        assert a["session_id"] == sid and a["caller_id"] is None
