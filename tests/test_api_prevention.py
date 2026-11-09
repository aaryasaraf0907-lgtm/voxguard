import math
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from conftest import FixedDetector, make_wav
from voxguard.api import prevention
from voxguard.api.db import (
    AlertRow,
    BlocklistRow,
    SessionRow,
    create_tables,
    make_engine,
)
from voxguard.api.deps import Settings
from voxguard.api.main import create_app
from voxguard.inference import Prediction


def _post(client: TestClient, **form: str):
    return client.post("/v1/analyze", files={"file": ("a.wav", make_wav(0.5))}, data=form)


def _count(client: TestClient, model) -> int:
    with client.app.state.session_factory() as db:
        return db.scalar(select(func.count()).select_from(model))


def _all(client: TestClient, model) -> list:
    with client.app.state.session_factory() as db:
        return list(db.scalars(select(model)))


# --- is_flagged / decide ---------------------------------------------------------------


def test_is_flagged_below_above_equal() -> None:
    assert prevention.is_flagged(0.49, 0.5) is False
    assert prevention.is_flagged(0.51, 0.5) is True
    assert prevention.is_flagged(0.5, 0.5) is True


@pytest.mark.parametrize("score,thr", [(math.nan, 0.5), (0.5, math.nan)])
def test_is_flagged_rejects_nan(score: float, thr: float) -> None:
    with pytest.raises(ValueError):
        prevention.is_flagged(score, thr)


@pytest.mark.parametrize(
    ("raw", "expected"), [(None, None), ("", None), ("   ", None), (" +91 1 ", "+91 1")]
)
def test_normalize_caller_id(raw: str | None, expected: str | None) -> None:
    assert prevention.normalize_caller_id(raw) == expected


# --- alerts ----------------------------------------------------------------------------


def test_flagged_creates_one_alert_with_session_and_score(client_factory) -> None:
    with client_factory(FixedDetector(0.9)) as c:
        body = _post(c, caller_id="alice").json()
        (alert,) = _all(c, AlertRow)
        assert alert.session_id == body["session_id"] and alert.fake_score == 0.9
        assert alert.caller_id == "alice" and "0.9000" in alert.message
        assert body["flagged"] is True and body["label"] == "fake"


def test_not_flagged_creates_no_alert_or_blocklist(client_factory) -> None:
    with client_factory(FixedDetector(0.1)) as c:
        body = _post(c, caller_id="alice").json()
        assert body["flagged"] is False and body["label"] == "real"
        assert _count(c, SessionRow) == 1
        assert _count(c, AlertRow) == 0 and _count(c, BlocklistRow) == 0


def test_flagged_without_caller_still_alerts(client_factory) -> None:
    with client_factory(FixedDetector(0.9)) as c:
        for kw in ({}, {"caller_id": "   "}):
            assert _post(c, **kw).status_code == 200
        assert _count(c, AlertRow) == 2 and _count(c, BlocklistRow) == 0
        assert all(a.caller_id is None for a in _all(c, AlertRow))
        assert all(s.caller_id is None for s in _all(c, SessionRow))


def test_score_equal_to_threshold_is_flagged(client_factory) -> None:
    with client_factory(FixedDetector(0.5, 0.5)) as c:
        assert _post(c).json()["flagged"] is True


def test_alert_message_keeps_four_decimals(client_factory) -> None:
    with client_factory(FixedDetector(0.9998, threshold=0.5)) as c:
        _post(c)
        (alert,) = _all(c, AlertRow)
        assert alert.message == "Flagged call: fake_score 0.9998 >= threshold 0.5000"


# --- blocklist -------------------------------------------------------------------------


def test_blocklist_counts_and_timestamps(client_factory) -> None:
    with client_factory(FixedDetector(0.9)) as c:
        _post(c, caller_id="alice")
        (first,) = _all(c, BlocklistRow)
        assert first.hit_count == 1 and first.first_seen == first.last_seen
        _post(c, caller_id="alice")
        (second,) = _all(c, BlocklistRow)
        assert second.hit_count == 2
        assert second.first_seen == first.first_seen and second.last_seen > first.last_seen


def test_caller_ids_normalized_consistently(client_factory) -> None:
    with client_factory(FixedDetector(0.9)) as c:
        _post(c, caller_id="  alice ")
        _post(c, caller_id="alice")
        (entry,) = _all(c, BlocklistRow)
        assert entry.caller_id == "alice" and entry.hit_count == 2
        assert {s.caller_id for s in _all(c, SessionRow)} == {"alice"}
        assert {a.caller_id for a in _all(c, AlertRow)} == {"alice"}


def test_different_callers_separate_entries(client_factory) -> None:
    with client_factory(FixedDetector(0.9)) as c:
        _post(c, caller_id="alice")
        _post(c, caller_id="bob")
        assert {b.caller_id for b in _all(c, BlocklistRow)} == {"alice", "bob"}


# --- transaction integrity -------------------------------------------------------------


def test_alert_failure_rolls_back_session(client_factory, monkeypatch) -> None:
    with client_factory(FixedDetector(0.9)) as c:
        # a real constraint violation: the next alert id collides with an existing alert
        with c.app.state.session_factory() as db:
            db.add(
                SessionRow(
                    session_id="seed",
                    created_at=datetime.now(UTC),
                    label="fake",
                    fake_score=0.9,
                    threshold=0.5,
                    flagged=True,
                    model_version="x",
                    latency_ms=1.0,
                )
            )
            db.flush()  # no relationship(), so insert order is explicit
            db.add(
                AlertRow(
                    alert_id="dup",
                    created_at=datetime.now(UTC),
                    session_id="seed",
                    fake_score=0.9,
                    message="m",
                )
            )
            db.commit()
        monkeypatch.setattr(prevention.uuid, "uuid4", lambda: type("U", (), {"hex": "dup"})())
        r = _post(c, caller_id="alice")
        assert r.status_code == 500
        assert _count(c, SessionRow) == 1 and _count(c, AlertRow) == 1
        assert _count(c, BlocklistRow) == 0


def test_blocklist_failure_rolls_back_session_and_alert(client_factory, monkeypatch) -> None:
    def broken(db, caller_id, seen_at) -> None:
        db.execute(text("INSERT INTO no_such_table VALUES (1)"))

    with client_factory(FixedDetector(0.9)) as c:
        monkeypatch.setattr(prevention, "upsert_blocklist", broken)
        assert _post(c, caller_id="alice").status_code == 500
        assert _count(c, SessionRow) == 0 and _count(c, AlertRow) == 0
        monkeypatch.undo()
        assert _post(c, caller_id="alice").status_code == 200  # app still healthy
        assert _count(c, AlertRow) == 1


def test_detector_failure_writes_nothing(client_factory) -> None:
    class Broken(FixedDetector):
        def predict(self, waveform, sample_rate) -> Prediction:
            raise RuntimeError("boom")

    with client_factory(Broken(0.9)) as c:
        assert _post(c, caller_id="alice").status_code == 500
        assert (_count(c, SessionRow), _count(c, AlertRow), _count(c, BlocklistRow)) == (0, 0, 0)


def test_nan_score_from_detector_writes_nothing(client_factory) -> None:
    with client_factory(FixedDetector(math.nan)) as c:
        assert _post(c).status_code == 500
        assert _count(c, SessionRow) == 0


def test_foreign_key_enforced(client_factory) -> None:
    with client_factory() as c, c.app.state.session_factory() as db:
        db.add(
            AlertRow(
                alert_id="a",
                created_at=datetime.now(UTC),
                session_id="missing",
                fake_score=0.9,
                message="m",
            )
        )
        with pytest.raises(IntegrityError):
            db.commit()


def test_foreign_key_enforced_on_file_db(tmp_path) -> None:
    engine = make_engine(f"sqlite:///{tmp_path / 'fk.db'}")
    create_tables(engine)
    with sessionmaker(engine)() as db:
        db.add(
            AlertRow(
                alert_id="a",
                created_at=datetime.now(UTC),
                session_id="missing",
                fake_score=0.9,
                message="m",
            )
        )
        with pytest.raises(IntegrityError):
            db.commit()
    engine.dispose()


# --- concurrency -----------------------------------------------------------------------


def test_concurrent_flagged_sessions_same_caller(tmp_path) -> None:
    """Separate connections (file DB) each flag the same caller; the upsert must not lose hits."""
    engine = make_engine(f"sqlite:///{tmp_path / 'c.db'}")
    create_tables(engine)
    factory = sessionmaker(engine, expire_on_commit=False)
    workers, per_worker = 8, 5
    barrier = threading.Barrier(workers)

    def work(i: int) -> None:
        barrier.wait()
        for j in range(per_worker):
            with factory() as db:
                row = SessionRow(
                    session_id=f"s{i}-{j}",
                    created_at=datetime.now(UTC),
                    caller_id="alice",
                    label="fake",
                    fake_score=0.9,
                    threshold=0.5,
                    flagged=True,
                    model_version="x",
                    latency_ms=1.0,
                )
                db.add(row)
                prevention.record_flag(db, row)
                db.commit()

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(work, range(workers)))  # re-raises any worker error

    with factory() as db:
        (entry,) = db.scalars(select(BlocklistRow)).all()
        assert entry.hit_count == workers * per_worker
        assert db.scalar(select(func.count()).select_from(AlertRow)) == workers * per_worker
    engine.dispose()


def test_concurrent_bare_upsert_is_atomic(tmp_path) -> None:
    """The upsert alone, with no earlier write to serialize on: no lost or duplicate rows."""
    engine = make_engine(f"sqlite:///{tmp_path / 'u.db'}")
    create_tables(engine)
    factory = sessionmaker(engine)
    workers, per_worker = 8, 10
    barrier = threading.Barrier(workers)

    def work(_: int) -> None:
        barrier.wait()
        for _ in range(per_worker):
            with factory() as db:
                prevention.upsert_blocklist(db, "alice", datetime.now(UTC))
                db.commit()

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(work, range(workers)))

    with factory() as db:
        (entry,) = db.scalars(select(BlocklistRow)).all()
        assert entry.hit_count == workers * per_worker
    engine.dispose()


# --- threshold modes -------------------------------------------------------------------


def test_eer_mode_uses_detector_threshold(client_factory) -> None:
    with client_factory(FixedDetector(0.6, threshold=0.7)) as c:
        body = _post(c).json()
        assert body["threshold"] == 0.7 and body["flagged"] is False and body["label"] == "real"


def test_low_far_mode_uses_configured_threshold_consistently(client_factory) -> None:
    det = FixedDetector(0.3, threshold=0.5)  # the detector alone would say "real"
    with client_factory(det, threshold_mode="low_far", low_far_threshold=0.2) as c:
        body = _post(c, caller_id="alice").json()
        assert body["threshold"] == 0.2
        assert body["flagged"] is True and body["label"] == "fake"
        (row,) = _all(c, SessionRow)
        assert (row.threshold, row.flagged, row.label) == (0.2, True, "fake")
        assert _count(c, AlertRow) == 1 and _count(c, BlocklistRow) == 1


def test_low_far_mode_can_also_unflag(client_factory) -> None:
    with client_factory(
        FixedDetector(0.6, 0.5), threshold_mode="low_far", low_far_threshold=0.9
    ) as c:
        body = _post(c).json()
        assert body["flagged"] is False and body["label"] == "real"
        assert _count(c, AlertRow) == 0


def test_label_and_flagged_never_disagree(client_factory) -> None:
    for mode_kw in ({}, {"threshold_mode": "low_far", "low_far_threshold": 0.4}):
        for score in (0.0, 0.39, 0.4, 0.5, 0.99):
            with client_factory(FixedDetector(score), **mode_kw) as c:
                b = _post(c).json()
                assert b["flagged"] == (b["label"] == "fake") == (score >= b["threshold"])


def test_low_far_without_threshold_fails_startup() -> None:
    with pytest.raises(ValueError, match="VOXGUARD_LOW_FAR_THRESHOLD"):
        create_app(settings=Settings(db_url="sqlite://", threshold_mode="low_far"))


@pytest.mark.parametrize("bad", [-0.1, 1.1, math.nan, math.inf, -math.inf])
def test_low_far_rejects_invalid_values(bad: float) -> None:
    with pytest.raises(ValueError, match="VOXGUARD_LOW_FAR_THRESHOLD"):
        Settings(threshold_mode="low_far", low_far_threshold=bad)


def test_unknown_mode_rejected() -> None:
    with pytest.raises(ValueError, match="VOXGUARD_THRESHOLD_MODE"):
        Settings(threshold_mode="bogus")  # type: ignore[arg-type]


def test_env_threshold_settings(monkeypatch) -> None:
    for n in ("VOXGUARD_THRESHOLD_MODE", "VOXGUARD_LOW_FAR_THRESHOLD"):
        monkeypatch.delenv(n, raising=False)
    assert Settings.from_env().threshold_mode == "eer"
    monkeypatch.setenv("VOXGUARD_THRESHOLD_MODE", "low_far")
    with pytest.raises(ValueError, match="VOXGUARD_LOW_FAR_THRESHOLD"):
        Settings.from_env()
    monkeypatch.setenv("VOXGUARD_LOW_FAR_THRESHOLD", "abc")
    with pytest.raises(ValueError, match="VOXGUARD_LOW_FAR_THRESHOLD"):
        Settings.from_env()
    monkeypatch.setenv("VOXGUARD_LOW_FAR_THRESHOLD", "0.25")
    s = Settings.from_env()
    assert (s.threshold_mode, s.low_far_threshold) == ("low_far", 0.25)
    monkeypatch.setenv("VOXGUARD_THRESHOLD_MODE", "nope")
    with pytest.raises(ValueError, match="VOXGUARD_THRESHOLD_MODE"):
        Settings.from_env()


# --- database URL validation -----------------------------------------------------------


@pytest.mark.parametrize("url", ["sqlite://", "sqlite:///x.db", "sqlite:///:memory:"])
def test_sqlite_urls_accepted(url: str) -> None:
    Settings(db_url=url)
    make_engine(url).dispose()


@pytest.mark.parametrize(
    "url", ["postgresql://user:secret@host/db", "mysql+pymysql://u:secret@h/d", "oracle://a:b@c/d"]
)
def test_non_sqlite_urls_rejected_without_leaking_credentials(url: str) -> None:
    for build in (lambda: Settings(db_url=url), lambda: make_engine(url)):
        with pytest.raises(ValueError, match="SQLite") as exc:
            build()
        assert "secret" not in str(exc.value) and "VOXGUARD_DB_URL" in str(exc.value)


def test_garbage_db_url_rejected() -> None:
    with pytest.raises(ValueError, match="VOXGUARD_DB_URL"):
        Settings(db_url="not a url")


def test_env_non_sqlite_url_fails_fast(monkeypatch) -> None:
    monkeypatch.setenv("VOXGUARD_DB_URL", "postgresql://u:secret@h/d")
    with pytest.raises(ValueError, match="SQLite"):
        create_app()
