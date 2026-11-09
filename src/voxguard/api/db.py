"""SQLAlchemy 2 models and engine setup: sessions, alerts and the caller blocklist."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import DateTime, Engine, ForeignKey, TypeDecorator, create_engine, event
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import ArgumentError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.pool import StaticPool


class UTCDateTime(TypeDecorator[datetime]):
    """Stored as naive UTC (SQLite keeps no tzinfo); always returned timezone-aware UTC."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Any) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:  # naive values are taken as UTC
            return value
        return value.astimezone(UTC).replace(tzinfo=None)

    def process_result_value(self, value: datetime | None, dialect: Any) -> datetime | None:
        return None if value is None else value.replace(tzinfo=UTC)


class Base(DeclarativeBase):
    pass


class SessionRow(Base):
    __tablename__ = "sessions"

    session_id: Mapped[str] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, index=True)
    caller_id: Mapped[str | None] = mapped_column(index=True)
    label: Mapped[str]
    fake_score: Mapped[float]
    threshold: Mapped[float]
    flagged: Mapped[bool]
    codec_applied: Mapped[str | None]
    model_version: Mapped[str]
    latency_ms: Mapped[float]


class AlertRow(Base):
    __tablename__ = "alerts"

    alert_id: Mapped[str] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, index=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.session_id"), index=True)
    caller_id: Mapped[str | None]
    fake_score: Mapped[float]
    message: Mapped[str]


class BlocklistRow(Base):
    __tablename__ = "blocklist"

    caller_id: Mapped[str] = mapped_column(primary_key=True)
    hit_count: Mapped[int]
    first_seen: Mapped[datetime] = mapped_column(UTCDateTime)
    last_seen: Mapped[datetime] = mapped_column(UTCDateTime)


def require_sqlite_url(url: str) -> URL:
    """The blocklist upsert uses SQLite's ON CONFLICT, so only SQLite URLs are supported.

    Error messages name the backend only, never the URL (it may contain credentials).
    """
    try:
        parsed = make_url(url)
    except ArgumentError:
        raise ValueError(
            "VOXGUARD_DB_URL is not a valid database URL; expected sqlite:///path/to/file.db"
        ) from None
    if parsed.get_backend_name() != "sqlite":
        raise ValueError(
            f"VOXGUARD_DB_URL must be a SQLite URL (sqlite:///...), got backend "
            f"{parsed.get_backend_name()!r}: only SQLite is supported"
        )
    return parsed


def make_engine(url: str) -> Engine:
    parsed = require_sqlite_url(url)
    # FastAPI runs sync handlers on a threadpool, so SQLite connections cross threads.
    kwargs: dict = {"connect_args": {"check_same_thread": False}}
    if parsed.database in (None, "", ":memory:"):
        kwargs["poolclass"] = StaticPool  # one shared connection, else each thread sees a new DB
    engine = create_engine(url, **kwargs)

    @event.listens_for(engine, "connect")
    def _enforce_foreign_keys(dbapi_connection: Any, _record: Any) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine


def create_tables(engine: Engine) -> None:
    Base.metadata.create_all(engine)
