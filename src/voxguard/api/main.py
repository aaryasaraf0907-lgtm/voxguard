"""FastAPI app. Run: `uvicorn voxguard.api.main:create_app --factory`."""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from voxguard.api import codec_sim, prevention
from voxguard.api.audio import decode_audio
from voxguard.api.db import AlertRow, BlocklistRow, SessionRow, create_tables, make_engine
from voxguard.api.deps import Settings, get_db, get_detector_dep, get_settings
from voxguard.api.detector_loader import load_detector
from voxguard.api.limits import BodySizeLimitMiddleware
from voxguard.inference import Detector
from voxguard.schemas import (
    AlertRecord,
    AnalyzeResponse,
    BlocklistEntry,
    HealthResponse,
    SessionRecord,
)

logger = logging.getLogger(__name__)

ANALYZE_PATH = "/v1/analyze"
MULTIPART_OVERHEAD_BYTES = 64 * 1024  # boundaries and form fields around the file


def create_app(detector: Detector | None = None, settings: Settings | None = None) -> FastAPI:
    """Build the app. Invalid VOXGUARD_* settings raise ValueError here (fail fast)."""
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = make_engine(settings.db_url)
        create_tables(engine)
        app.state.settings = settings
        app.state.detector = detector or load_detector(settings.device, settings.threshold_mode)
        app.state.session_factory = sessionmaker(engine, expire_on_commit=False)
        yield
        engine.dispose()

    app = FastAPI(title="VoxGuard", lifespan=lifespan)
    app.add_middleware(
        BodySizeLimitMiddleware,
        max_body_bytes=settings.max_upload_bytes + MULTIPART_OVERHEAD_BYTES,
        path=ANALYZE_PATH,
    )

    @app.get("/health", response_model=HealthResponse)
    def health(det: Annotated[Detector, Depends(get_detector_dep)]) -> HealthResponse:
        kind = type(det).__name__.removesuffix("Detector").lower()
        return HealthResponse(status="ok", detector=kind, model_version=det.model_version)

    @app.post(ANALYZE_PATH, response_model=AnalyzeResponse)
    def analyze(
        file: Annotated[UploadFile, File()],
        det: Annotated[Detector, Depends(get_detector_dep)],
        db: Annotated[Session, Depends(get_db)],
        cfg: Annotated[Settings, Depends(get_settings)],
        caller_id: Annotated[str | None, Form(max_length=128)] = None,
        simulate_codec: Annotated[bool, Form()] = False,
    ) -> AnalyzeResponse:
        if simulate_codec:
            if cfg.simulate_codec is None:
                raise HTTPException(
                    501, "Codec simulation is not configured (set VOXGUARD_SIMULATE_CODEC)"
                )
            try:
                codec_sim.ensure_available(cfg.simulate_codec)
            except codec_sim.CodecUnavailableError as exc:
                logger.error("codec simulation unavailable: %s", exc)
                raise HTTPException(503, "Codec simulation is unavailable on this server") from None

        # sync handler: read the underlying file object, bounded by the upload limit
        data = file.file.read(cfg.max_upload_bytes + 1)
        if not data:
            raise HTTPException(400, "Empty upload")
        if len(data) > cfg.max_upload_bytes:
            raise HTTPException(413, "Upload too large")

        waveform, sample_rate = decode_audio(data, cfg.max_audio_seconds)

        codec_applied: str | None = None
        if simulate_codec:  # before inference; on failure nothing is stored or scored
            try:
                waveform, sample_rate = codec_sim.simulate_phone_line(
                    waveform, sample_rate, cfg.simulate_codec
                )
            except codec_sim.CodecUnavailableError:
                raise HTTPException(503, "Codec simulation is unavailable on this server") from None
            except codec_sim.CodecSimulationError:
                raise HTTPException(500, "Codec simulation failed") from None
            codec_applied = cfg.simulate_codec

        try:
            pred = det.predict(waveform, sample_rate)
            decision = prevention.decide(pred, cfg)
        except Exception:
            logger.exception("Detector failed")
            raise HTTPException(500, "Detector failed") from None

        row = SessionRow(
            session_id=uuid.uuid4().hex,
            created_at=datetime.now(UTC),
            caller_id=prevention.normalize_caller_id(caller_id),
            label=decision.label,
            fake_score=pred.fake_score,
            threshold=decision.threshold,
            flagged=decision.flagged,
            codec_applied=codec_applied,
            model_version=pred.model_version,
            latency_ms=pred.latency_ms,
        )
        try:  # session, alert and blocklist update commit together or not at all
            db.add(row)
            if row.flagged:
                prevention.record_flag(db, row)
            db.commit()
        except SQLAlchemyError:
            db.rollback()
            logger.exception("Could not store session")
            raise HTTPException(500, "Could not store session") from None

        return AnalyzeResponse(
            session_id=row.session_id,
            label=decision.label,
            fake_score=pred.fake_score,
            threshold=decision.threshold,
            latency_ms=pred.latency_ms,
            model_version=pred.model_version,
            flagged=decision.flagged,
            codec_applied=codec_applied,
        )

    @app.get("/v1/sessions", response_model=list[SessionRecord])
    def list_sessions(
        db: Annotated[Session, Depends(get_db)],
        flagged_only: bool = False,
        limit: Annotated[int, Query(ge=1, le=500)] = 50,
    ) -> list[SessionRecord]:
        stmt = select(SessionRow).order_by(
            SessionRow.created_at.desc(), SessionRow.session_id.desc()
        )
        if flagged_only:
            stmt = stmt.where(SessionRow.flagged.is_(True))
        rows = db.scalars(stmt.limit(limit))
        return [SessionRecord.model_validate(r) for r in rows]

    @app.get("/v1/blocklist", response_model=list[BlocklistEntry])
    def list_blocklist(db: Annotated[Session, Depends(get_db)]) -> list[BlocklistEntry]:
        stmt = select(BlocklistRow).order_by(
            BlocklistRow.hit_count.desc(), BlocklistRow.last_seen.desc(), BlocklistRow.caller_id
        )
        return [
            BlocklistEntry(
                caller_id=r.caller_id,
                hit_count=r.hit_count,
                first_seen=r.first_seen,
                last_seen=r.last_seen,
            )
            for r in db.scalars(stmt)
        ]

    @app.get("/v1/alerts", response_model=list[AlertRecord])
    def list_alerts(
        db: Annotated[Session, Depends(get_db)],
        since: datetime | None = None,
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
    ) -> list[AlertRecord]:
        if since is None:  # latest alerts, newest first
            stmt = select(AlertRow).order_by(AlertRow.created_at.desc(), AlertRow.alert_id.desc())
        else:  # polling cursor: strictly newer, oldest first so clients advance without gaps
            try:
                since_utc = (
                    since.replace(tzinfo=UTC) if since.tzinfo is None else since.astimezone(UTC)
                )
            except OverflowError:
                raise HTTPException(422, "since is out of range") from None
            stmt = (
                select(AlertRow)
                .where(AlertRow.created_at > since_utc)
                .order_by(AlertRow.created_at, AlertRow.alert_id)
            )
        return [
            AlertRecord(
                alert_id=r.alert_id,
                created_at=r.created_at,
                session_id=r.session_id,
                caller_id=r.caller_id,
                fake_score=r.fake_score,
                message=r.message,
            )
            for r in db.scalars(stmt.limit(limit))
        ]

    return app
