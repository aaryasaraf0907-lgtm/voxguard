"""Pydantic models for the REST contract (CLAUDE.md section 7)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Label = Literal["real", "fake"]


class HealthResponse(BaseModel):
    status: str
    detector: str
    model_version: str


class AnalyzeResponse(BaseModel):
    session_id: str
    label: Label
    fake_score: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    latency_ms: float = Field(ge=0.0)
    model_version: str
    flagged: bool
    codec_applied: str | None = None
    codec_ms: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Wall time (ms) of the phone-line simulation: resampling to 16 kHz plus the ffmpeg "
            "encode/decode. Null when no codec was applied. Not included in latency_ms, which "
            "is detector inference only."
        ),
    )
    timeline: list[TimelinePoint] | None = Field(
        default=None,
        description=(
            "Only with timeline=true: each detector window's score and the running average, in "
            "time order. The last running_score equals fake_score. Not stored."
        ),
    )
    first_flag_s: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Only with timeline=true: seconds of audio after which the running average first "
            "reached the threshold; null if it never did. Not stored."
        ),
    )


class TimelinePoint(BaseModel):
    """One detector window of the live-monitoring timeline (``timeline=true``)."""

    start_s: float = Field(ge=0.0)  # window start, seconds into the (16 kHz) audio
    end_s: float = Field(ge=0.0)  # the last window overlaps the one before it
    fake_score: float = Field(ge=0.0, le=1.0)  # this window alone
    running_score: float = Field(ge=0.0, le=1.0)  # mean of the windows up to this one
    flagged: bool  # running_score >= threshold (the prevention rule)


class CodecInfo(BaseModel):
    name: str
    available: bool  # this server's ffmpeg has the encoder
    bitrate: int | None  # bit/s used, as in the evaluation sets; None for fixed-rate codecs


class CodecsResponse(BaseModel):
    default: str | None  # VOXGUARD_SIMULATE_CODEC, or None when not configured
    codecs: list[CodecInfo]


class SessionRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    session_id: str
    created_at: datetime
    caller_id: str | None = None
    label: Label
    fake_score: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    flagged: bool
    codec_applied: str | None = None
    model_version: str


class BlocklistEntry(BaseModel):
    caller_id: str
    hit_count: int = Field(ge=1)
    first_seen: datetime
    last_seen: datetime


class AlertRecord(BaseModel):
    alert_id: str
    created_at: datetime
    session_id: str
    caller_id: str | None = None
    fake_score: float = Field(ge=0.0, le=1.0)
    message: str


TxAction = Literal["allow", "step_up", "hold"]
TxTier = Literal["unknown", "low", "caution", "high"]
TxMethod = Literal["callback", "otp", "second_approver"]


class TransactionCheck(BaseModel):
    """POST /v1/transactions/check: a transfer to approve during, or after, a checked call."""

    session_id: str = Field(min_length=1, max_length=64)
    amount_inr: float = Field(gt=0, le=1e9)
    beneficiary: str | None = Field(default=None, max_length=64)  # a label, e.g. "A/c ...1234"


class TransactionVerify(BaseModel):
    """POST /v1/transactions/{decision_id}/verify: a verification step someone has completed."""

    method: TxMethod
    verified_by: str = Field(min_length=1, max_length=64)
    note: str | None = Field(default=None, max_length=200)


class TransactionEvent(BaseModel):
    created_at: datetime
    kind: Literal["decided", "verified", "released"]
    method: TxMethod | None = None
    verified_by: str | None = None
    note: str | None = None


class TransactionDecision(BaseModel):
    decision_id: str
    created_at: datetime
    session_id: str
    amount_inr: float
    beneficiary: str | None = None
    high_value: bool  # at or above VOXGUARD_HIGH_VALUE_INR
    live: bool  # checked while the call was still going on
    tier: TxTier
    action: TxAction
    status: Literal["allowed", "pending", "held", "released"]
    score: float | None = None  # the call's score (running average if live); None before any
    threshold: float
    windows: int | None = None  # scored windows of a live call; None for a finished session
    reasons: list[str]  # codes: risk_<tier>, not_enough_audio, caller_flagged_before, high_value
    required_steps: list[TxMethod]  # step_up: any one; hold: all
    completed_steps: list[TxMethod] = []
    events: list[TransactionEvent] = []


class CallWindow(BaseModel):
    """One scored window of a live call (feature-level log; no audio)."""

    model_config = ConfigDict(from_attributes=True)

    t_s: float
    window_score: float
    running_score: float
