"""SQLite under concurrent analyses (found by scripts/load_test.py, 18 Nov 2026): WAL with a 30 s
lock wait (no "database is locked" 500s), and timestamps that follow commit order (an
``/v1/alerts?since=`` poller misses nothing)."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import text

from conftest import FixedDetector, make_wav
from voxguard.api.db import BUSY_TIMEOUT_MS, make_engine
from voxguard.api.deps import Settings
from voxguard.api.main import create_app


def test_file_database_uses_wal_and_waits_for_the_lock(tmp_path: Path) -> None:
    engine = make_engine(f"sqlite:///{tmp_path / 'v.db'}")
    with engine.connect() as conn:
        assert conn.execute(text("PRAGMA journal_mode")).scalar() == "wal"
        assert conn.execute(text("PRAGMA busy_timeout")).scalar() == BUSY_TIMEOUT_MS
        assert conn.execute(text("PRAGMA foreign_keys")).scalar() == 1
    engine.dispose()


def test_in_memory_database_still_works(tmp_path: Path) -> None:
    engine = make_engine("sqlite://")
    with engine.connect() as conn:
        assert conn.execute(text("PRAGMA journal_mode")).scalar() == "memory"
    engine.dispose()


def test_concurrent_analyses_commit_in_timestamp_order(tmp_path: Path) -> None:
    """Without the write lock before the timestamp, two threads already committed out of order
    (the load test saw it at 2 workers); a since-cursor poller then skips the late row."""
    db_file = tmp_path / "v.db"
    settings = Settings(
        db_url=f"sqlite:///{db_file}", max_running_analyses=4, max_waiting_analyses=64
    )
    app = create_app(FixedDetector(0.9), settings)
    wav = make_wav(0.05)
    with TestClient(app) as c:

        def post(i: int) -> int:
            files = {"file": ("a.wav", wav)}
            return c.post("/v1/analyze", files=files, data={"caller_id": f"c{i % 5}"}).status_code

        with ThreadPoolExecutor(16) as pool:
            statuses = list(pool.map(post, range(400)))
        assert statuses == [200] * 400
        blocklist = c.get("/v1/blocklist").json()
        assert sorted(b["hit_count"] for b in blocklist) == [80] * 5
    with sqlite3.connect(db_file) as conn:
        for table in ("sessions", "alerts"):
            stamps = [r[0] for r in conn.execute(f"SELECT created_at FROM {table} ORDER BY rowid")]
            assert len(stamps) == 400
            assert stamps == sorted(stamps), f"{table}: commit order differs from timestamp order"
