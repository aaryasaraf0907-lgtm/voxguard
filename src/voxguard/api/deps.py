"""Runtime settings and FastAPI dependencies."""

from __future__ import annotations

import math
import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

from fastapi import Request
from sqlalchemy.orm import Session

from voxguard.api.db import require_sqlite_url
from voxguard.api.thresholds import ThresholdSet
from voxguard.data.codecs import CODECS
from voxguard.inference import Detector, threshold_from_env

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
    threshold_mode: Literal["eer", "low_far"] = "eer"
    low_far_threshold: float | None = None
    simulate_codec: str | None = None  # codec used when a request sets simulate_codec=true
    threshold_override: float | None = None  # VOXGUARD_THRESHOLD if set: beats thresholds.json
    device: str = "cpu"  # VOXGUARD_DEVICE for the real detector: cpu | cuda | auto

    def __post_init__(self) -> None:
        require_sqlite_url(self.db_url)
        if self.device not in ("cpu", "cuda", "auto"):
            raise ValueError(f"VOXGUARD_DEVICE must be cpu, cuda or auto, got {self.device!r}")
        if self.simulate_codec is not None and self.simulate_codec not in CODECS:
            raise ValueError(
                f"VOXGUARD_SIMULATE_CODEC must be one of {sorted(CODECS)}, "
                f"got {self.simulate_codec!r}"
            )
        t = self.threshold_override
        if t is not None and (not math.isfinite(t) or not 0.0 <= t <= 1.0):
            raise ValueError(f"VOXGUARD_THRESHOLD must be in [0, 1], got {t!r}")
        if self.threshold_mode not in ("eer", "low_far"):
            raise ValueError(
                f"VOXGUARD_THRESHOLD_MODE must be 'eer' or 'low_far', got {self.threshold_mode!r}"
            )
        if self.threshold_mode == "low_far":
            t = self.low_far_threshold
            if t is None or not math.isfinite(t) or not 0.0 <= t <= 1.0:
                raise ValueError(
                    "VOXGUARD_LOW_FAR_THRESHOLD must be set to a number in [0, 1] "
                    f"when VOXGUARD_THRESHOLD_MODE=low_far, got {t!r}"
                )

    @classmethod
    def from_env(cls) -> Settings:
        """Read the VOXGUARD_* limits, DB URL and threshold mode (see .env.example).

        Unset or empty values use the defaults; invalid values raise ValueError.
        """
        upload_mb = _positive_float_env("VOXGUARD_MAX_UPLOAD_MB", DEFAULT_MAX_UPLOAD_MB)
        mode = os.environ.get("VOXGUARD_THRESHOLD_MODE", "").strip().lower() or "eer"
        raw_low_far = os.environ.get("VOXGUARD_LOW_FAR_THRESHOLD", "").strip()
        try:
            low_far = float(raw_low_far) if raw_low_far else None
        except ValueError:
            raise ValueError(
                f"VOXGUARD_LOW_FAR_THRESHOLD must be a number in [0, 1], got {raw_low_far!r}"
            ) from None
        return cls(
            max_upload_bytes=max(1, int(upload_mb * 1024 * 1024)),
            max_audio_seconds=_positive_float_env(
                "VOXGUARD_MAX_AUDIO_SECONDS", DEFAULT_MAX_AUDIO_SECONDS
            ),
            db_url=os.environ.get("VOXGUARD_DB_URL", "").strip() or DEFAULT_DB_URL,
            threshold_mode=mode,  # type: ignore[arg-type]  # validated in __post_init__
            low_far_threshold=low_far,
            simulate_codec=os.environ.get("VOXGUARD_SIMULATE_CODEC", "").strip() or None,
            threshold_override=(  # unset or empty: not an override
                threshold_from_env() if os.environ.get("VOXGUARD_THRESHOLD", "").strip() else None
            ),
            device=os.environ.get("VOXGUARD_DEVICE", "").strip().lower() or "cpu",
        )


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_detector_dep(request: Request) -> Detector:
    return request.app.state.detector


def get_thresholds(request: Request) -> ThresholdSet | None:
    return request.app.state.thresholds


def get_db(request: Request) -> Iterator[Session]:
    """One SQLAlchemy session per request; the handler commits, this always closes."""
    with request.app.state.session_factory() as session:
        yield session
