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
