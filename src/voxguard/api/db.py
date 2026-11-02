"""SQLAlchemy 2 models and engine setup (sessions table only; prevention tables come later)."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Engine, create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.pool import StaticPool


class Base(DeclarativeBase):
    pass


class SessionRow(Base):
    __tablename__ = "sessions"

    session_id: Mapped[str] = mapped_column(primary_key=True)
    # SQLite does not keep tzinfo: values are written as UTC and read back naive.
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    caller_id: Mapped[str | None] = mapped_column(index=True)
    label: Mapped[str]
    fake_score: Mapped[float]
    threshold: Mapped[float]
    flagged: Mapped[bool]
    codec_applied: Mapped[str | None]
    model_version: Mapped[str]
    latency_ms: Mapped[float]


def make_engine(url: str) -> Engine:
    parsed = make_url(url)
    if parsed.get_backend_name() != "sqlite":
        return create_engine(url)
    # FastAPI runs sync handlers on a threadpool, so SQLite connections cross threads.
    kwargs: dict = {"connect_args": {"check_same_thread": False}}
    if parsed.database in (None, "", ":memory:"):
        kwargs["poolclass"] = StaticPool  # one shared connection, else each thread sees a new DB
    return create_engine(url, **kwargs)


def create_tables(engine: Engine) -> None:
    Base.metadata.create_all(engine)
