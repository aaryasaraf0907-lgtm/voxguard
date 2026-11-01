"""FastAPI app. Run: `uvicorn voxguard.api.main:create_app --factory`."""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from voxguard.api.audio import decode_audio
from voxguard.api.db import SessionRow, create_tables, make_engine
from voxguard.api.deps import Settings, get_db, get_detector_dep, get_settings
from voxguard.api.limits import BodySizeLimitMiddleware
from voxguard.inference import Detector, get_detector
from voxguard.schemas import AnalyzeResponse, HealthResponse

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
        app.state.detector = detector or get_detector()
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
            raise HTTPException(501, "Codec simulation is not implemented yet")

        # sync handler: read the underlying file object, bounded by the upload limit
        data = file.file.read(cfg.max_upload_bytes + 1)
        if not data:
            raise HTTPException(400, "Empty upload")
        if len(data) > cfg.max_upload_bytes:
            raise HTTPException(413, "Upload too large")

        waveform, sample_rate = decode_audio(data, cfg.max_audio_seconds)

        try:
            pred = det.predict(waveform, sample_rate)
        except Exception:
            logger.exception("Detector failed")
            raise HTTPException(500, "Detector failed") from None

        row = SessionRow(
            session_id=uuid.uuid4().hex,
            created_at=datetime.now(UTC),
            caller_id=caller_id or None,
            label=pred.label,
            fake_score=pred.fake_score,
            threshold=pred.threshold,
            flagged=pred.fake_score >= pred.threshold,
            codec_applied=None,
            model_version=pred.model_version,
            latency_ms=pred.latency_ms,
        )
        try:
            db.add(row)
            db.commit()
        except SQLAlchemyError:
            db.rollback()
            logger.exception("Could not store session")
            raise HTTPException(500, "Could not store session") from None

        return AnalyzeResponse(
            session_id=row.session_id,
            label=pred.label,
            fake_score=pred.fake_score,
            threshold=pred.threshold,
            latency_ms=pred.latency_ms,
            model_version=pred.model_version,
            flagged=row.flagged,
            codec_applied=row.codec_applied,
        )

    return app
