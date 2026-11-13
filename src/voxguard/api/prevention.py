"""Prevention layer: the flagging rule, threshold modes, alerts and the caller blocklist."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from voxguard.api.db import AlertRow, BlocklistRow, SessionRow
from voxguard.api.deps import Settings
from voxguard.api.thresholds import ThresholdSet
from voxguard.inference import Prediction

Channel = Literal["clean", "phone"]


@dataclass(frozen=True)
class Decision:
    """The one outcome used for the response, the stored session and the alert."""

    threshold: float
    flagged: bool
    label: Literal["real", "fake"]


def is_flagged(fake_score: float, threshold: float) -> bool:
    """The prevention rule (CLAUDE.md section 7). The only place it is written."""
    if math.isnan(fake_score) or math.isnan(threshold):
        raise ValueError("fake_score and threshold must not be NaN")
    return fake_score >= threshold


def channel_for(known_clean: bool, codec_applied: str | None) -> Channel:
    """Which threshold applies (agreed with Aarya): phone-line audio and audio of unknown
    origin use the phone threshold; only audio known to be clean uses the clean one."""
    return "clean" if known_clean and codec_applied is None else "phone"


def effective_threshold(
    pred: Prediction,
    settings: Settings,
    thresholds: ThresholdSet | None = None,
    channel: Channel = "phone",
) -> float:
    """Precedence: low_far mode's explicit threshold, then an explicit VOXGUARD_THRESHOLD,
    then the checkpoint's thresholds.json (one clean and one phone value; per-codec values
    are never used here), then the detector's own threshold (legacy behaviour)."""
    if settings.threshold_mode == "low_far":
        assert settings.low_far_threshold is not None  # enforced by Settings
        return settings.low_far_threshold
    if settings.threshold_override is not None:
        return settings.threshold_override
    if thresholds is not None:
        return thresholds.clean if channel == "clean" else thresholds.phone
    return pred.threshold


def decide(
    pred: Prediction,
    settings: Settings,
    thresholds: ThresholdSet | None = None,
    channel: Channel = "phone",
) -> Decision:
    threshold = effective_threshold(pred, settings, thresholds, channel)
    flagged = is_flagged(pred.fake_score, threshold)
    return Decision(threshold=threshold, flagged=flagged, label="fake" if flagged else "real")


def normalize_caller_id(raw: str | None) -> str | None:
    """Strip surrounding whitespace; empty means missing. No other rewriting."""
    if raw is None:
        return None
    return raw.strip() or None


def upsert_blocklist(db: Session, caller_id: str, seen_at: datetime) -> None:
    """Atomic insert-or-increment, safe under concurrent requests for the same caller."""
    stmt = sqlite_insert(BlocklistRow).values(
        caller_id=caller_id, hit_count=1, first_seen=seen_at, last_seen=seen_at
    )
    db.execute(
        stmt.on_conflict_do_update(
            index_elements=[BlocklistRow.caller_id],
            set_={
                "hit_count": BlocklistRow.hit_count + 1,
                "last_seen": stmt.excluded.last_seen,
            },
        )
    )


def record_flag(db: Session, session_row: SessionRow) -> None:
    """Add the alert and blocklist update for a flagged session. The caller commits."""
    db.flush()  # the session row must exist before the alert that references it
    db.add(
        AlertRow(
            alert_id=uuid.uuid4().hex,
            created_at=session_row.created_at,
            session_id=session_row.session_id,
            caller_id=session_row.caller_id,
            fake_score=session_row.fake_score,
            message=(
                f"Flagged call: fake_score {session_row.fake_score:.4f} "
                f">= threshold {session_row.threshold:.4f}"
            ),
        )
    )
    if session_row.caller_id is not None:
        upsert_blocklist(db, session_row.caller_id, session_row.created_at)
