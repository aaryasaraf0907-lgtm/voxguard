"""Runtime settings and FastAPI dependencies."""

from __future__ import annotations

import math
import os
from collections.abc import Iterator
from dataclasses import dataclass

from fastapi import Request
from sqlalchemy.orm import Session

from voxguard.inference import Detector

DEFAULT_DB_URL = "sqlite:///./voxguard.db"
DEFAULT_MAX_UPLOAD_MB = 25.0
DEFAULT_MAX_AUDIO_SECONDS = 60.0


def _positive_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a positive number, got {raw!r}") from None
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number, got {raw!r}")
    return value


@dataclass(frozen=True)
class Settings:
    max_upload_bytes: int = int(DEFAULT_MAX_UPLOAD_MB * 1024 * 1024)
    max_audio_seconds: float = DEFAULT_MAX_AUDIO_SECONDS
    db_url: str = DEFAULT_DB_URL

    @classmethod
    def from_env(cls) -> Settings:
        """Read VOXGUARD_MAX_UPLOAD_MB, VOXGUARD_MAX_AUDIO_SECONDS, VOXGUARD_DB_URL.

        Unset or empty values use the defaults; invalid values raise ValueError.
        """
        upload_mb = _positive_float_env("VOXGUARD_MAX_UPLOAD_MB", DEFAULT_MAX_UPLOAD_MB)
        return cls(
            max_upload_bytes=max(1, int(upload_mb * 1024 * 1024)),
            max_audio_seconds=_positive_float_env(
                "VOXGUARD_MAX_AUDIO_SECONDS", DEFAULT_MAX_AUDIO_SECONDS
            ),
            db_url=os.environ.get("VOXGUARD_DB_URL", "").strip() or DEFAULT_DB_URL,
        )


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_detector_dep(request: Request) -> Detector:
    return request.app.state.detector


def get_db(request: Request) -> Iterator[Session]:
    """One SQLAlchemy session per request; the handler commits, this always closes."""
    with request.app.state.session_factory() as session:
        yield session
