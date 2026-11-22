"""FastAPI app. Run: `uvicorn voxguard.api.main:create_app --factory`."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from voxguard.api import codec_sim, evidence, prevention
from voxguard.api.audio import decode_audio
from voxguard.api.db import (
    AlertRow,
    BlocklistRow,
    SessionRow,
    create_tables,
    make_engine,
)
from voxguard.api.deps import (
    Settings,
    get_db,
    get_detector_dep,
    get_gate,
    get_language_checker,
    get_settings,
    get_thresholds,
)
from voxguard.api.detector_loader import (
    load_detector,
    load_model_thresholds,
    warn_if_default_threshold,
)
from voxguard.api.gate import RETRY_AFTER_S, AnalysisGate, Busy
from voxguard.api.limits import BodySizeLimitMiddleware
from voxguard.api.monitor_page import render as render_monitor
from voxguard.api.stream import handle_stream
from voxguard.api.thresholds import ThresholdSet
from voxguard.api.timeline import WindowScore, build_timeline, score_windows
from voxguard.api.transactions import router as transactions_router
from voxguard.api.webhook import Notifier
from voxguard.data.codecs import CODECS
from voxguard.inference import Detector, Prediction
from voxguard.schemas import (
    AlertRecord,
    AnalyzeResponse,
    BlocklistEntry,
    CodecInfo,
    CodecsResponse,
    HealthResponse,
    SessionRecord,
)

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(levelname)s %(name)s: %(message)s"
ANALYZE_PATH = "/v1/analyze"
MULTIPART_OVERHEAD_BYTES = 64 * 1024  # boundaries and form fields around the file


def _codec_unavailable(codec: str) -> str:
    return f"Codec simulation for {codec} is unavailable on this server."


def check_language(checker: object | None, waveform, sample_rate: int):  # noqa: ANN001
    """The language check on the audio as received; None when off, too short or failed (the
    normal verdict then applies: a failed check must never block an analysis)."""
    if checker is None:
        return None
    try:
        return checker.check(waveform, sample_rate)  # type: ignore[attr-defined]
    except Exception:
        logger.exception("language check failed; giving the normal verdict")
        return None


def _score(
    data: bytes,
    chosen: str | None,
    det: Detector,
    cfg: Settings,
    with_timeline: bool = False,
    checker: object | None = None,
) -> tuple[Prediction, str | None, float | None, list[WindowScore] | None, object | None]:
    """Decode, check the language, simulate the phone line if ``chosen``, run the detector.

    Returns (prediction, codec applied, codec wall time in ms, per-window scores if
    ``with_timeline``, language result or None). Raises HTTPException; on any failure nothing
    has been scored or stored."""
    waveform, sample_rate = decode_audio(data, cfg.max_audio_seconds, cfg.max_upload_bytes)
    language = check_language(checker, waveform, sample_rate)  # the original, not the codec
    codec_ms: float | None = None
    if chosen is not None:  # before inference; never falls back to the original audio
        t0 = time.perf_counter()
        try:
            waveform, sample_rate = codec_sim.simulate_phone_line(waveform, sample_rate, chosen)
        except codec_sim.CodecUnavailableError:
            raise HTTPException(503, _codec_unavailable(chosen)) from None
        except codec_sim.CodecInputTooShortError:
            ms = codec_sim.MIN_CODEC_SAMPLES * 1000 // codec_sim.SAMPLE_RATE
            raise HTTPException(
                422, f"Audio is too short for phone-line simulation (at least {ms} ms)"
            ) from None
        except codec_sim.CodecSimulationError:
            raise HTTPException(500, "Codec simulation failed") from None
        codec_ms = (time.perf_counter() - t0) * 1000.0
    windows: list[WindowScore] | None = None
    try:
        if with_timeline:  # same windows and verdict, each window scored on its own
            pred, windows = score_windows(det, waveform, sample_rate)
        else:
            pred = det.predict(waveform, sample_rate)
    except Exception:
        logger.exception("Detector failed")
        raise HTTPException(500, "Detector failed") from None
    return pred, chosen, codec_ms, windows, language


def language_fields(language: object | None) -> dict[str, object]:
    if language is None:
        return {}
    from voxguard.language import language_name

    return {
        "language": language.code,  # type: ignore[attr-defined]
        "language_name": language_name(language.code),  # type: ignore[attr-defined]
        "language_supported": language.supported,  # type: ignore[attr-defined]
    }


def _timeline_fields(windows: list[WindowScore], threshold: float) -> dict[str, object]:
    points, first_flag_s = build_timeline(windows, threshold)
    return {"timeline": points, "first_flag_s": first_flag_s}


def configure_logging() -> None:
    """Show LineTrue's own INFO lines (model loaded, warm-up, thresholds) when served.

    uvicorn configures only its own loggers, so without this the ``voxguard`` INFO records are
    dropped and only warnings reach the console. Nothing changes when logging is already
    configured (a root handler exists, e.g. under pytest or in an embedding application)."""
    package = logging.getLogger("voxguard")
    if logging.getLogger().handlers or package.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    package.addHandler(handler)
    package.setLevel(logging.INFO)


def create_app(
    detector: Detector | None = None,
    settings: Settings | None = None,
    thresholds: ThresholdSet | None = None,
    language_checker: object | None = None,
) -> FastAPI:
    """Build the app. Invalid VOXGUARD_* settings raise ValueError here (fail fast).

    Without ``detector`` the configured one is loaded at startup, with its thresholds.json
    (real detector only). ``thresholds`` is given directly in tests."""
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging()
        engine = make_engine(settings.db_url)
        create_tables(engine)
        app.state.settings = settings
        app.state.gate = AnalysisGate(settings.max_running_analyses, settings.max_waiting_analyses)
        app.state.live_streams = 0  # open /v1/stream connections (event loop only)
        app.state.live_calls = {}  # session_id -> stream.LiveCall, while the call is going on
        app.state.notifier = Notifier(settings.alert_webhook_url, settings.alert_webhook_secret)
        det = detector or load_detector(settings.device)
        if detector is None and thresholds is None:
            found = load_model_thresholds(det)
            warn_if_default_threshold(
                det, found, settings.threshold_mode, settings.threshold_override
            )
        else:
            found = thresholds
        app.state.detector = det
        app.state.thresholds = found
        checker = language_checker
        if checker is None and settings.language_check:  # fail fast, like the detector
            from voxguard.language import WhisperLanguageChecker

            checker = WhisperLanguageChecker(settings.language_model)
            logger.info("language check on: %s", settings.language_model)
        app.state.language_checker = checker
        app.state.session_factory = sessionmaker(engine, expire_on_commit=False)
        try:
            yield
        finally:  # also when shutdown is interrupted
            app.state.notifier.close()
            app.state.gate.close()
            engine.dispose()

    app = FastAPI(title="LineTrue", lifespan=lifespan)
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
        request: Request,
        file: Annotated[UploadFile, File()],
        det: Annotated[Detector, Depends(get_detector_dep)],
        db: Annotated[Session, Depends(get_db)],
        cfg: Annotated[Settings, Depends(get_settings)],
        model_thresholds: Annotated[ThresholdSet | None, Depends(get_thresholds)],
        gate: Annotated[AnalysisGate, Depends(get_gate)],
        checker: Annotated[object | None, Depends(get_language_checker)],
        caller_id: Annotated[str | None, Form(max_length=128)] = None,
        simulate_codec: Annotated[bool, Form()] = False,
        codec: Annotated[str | None, Form(max_length=32)] = None,
        timeline: Annotated[bool, Form()] = False,
    ) -> AnalyzeResponse:
        # A `codec` field selects the codec and implies simulation; otherwise simulate_codec=true
        # uses the server default (VOXGUARD_SIMULATE_CODEC). Never substituted with another codec.
        chosen: str | None = None
        if codec is not None:
            if codec not in CODECS:
                raise HTTPException(422, f"Unknown codec {codec!r}; known: {sorted(CODECS)}")
            chosen = codec
        elif simulate_codec:
            if cfg.simulate_codec is None:
                raise HTTPException(
                    501, "Codec simulation is not configured (set VOXGUARD_SIMULATE_CODEC)"
                )
            chosen = cfg.simulate_codec
        if chosen is not None:
            try:
                codec_sim.ensure_available(chosen)
            except codec_sim.CodecUnavailableError as exc:
                logger.error("codec simulation unavailable: %s", exc)
                raise HTTPException(503, _codec_unavailable(chosen)) from None

        # sync handler: read the underlying file object, bounded by the upload limit
        data = file.file.read(cfg.max_upload_bytes + 1)
        if not data:
            raise HTTPException(400, "Empty upload")
        if len(data) > cfg.max_upload_bytes:
            raise HTTPException(413, "Upload too large")

        try:
            # decode, simulation and inference hold the memory: bounded, on dedicated threads
            pred, codec_applied, codec_ms, windows, language = gate.run(
                _score, data, chosen, det, cfg, timeline, checker
            )
        except Busy:
            logger.warning("analysis refused: server busy (%d running)", gate.running)
            raise HTTPException(
                503,
                "Server busy: too many analyses at once. Try again in a few seconds.",
                headers={"Retry-After": str(RETRY_AFTER_S)},
            ) from None
        try:
            # no request field says audio is known to be clean: uploads are of unknown origin
            channel = prevention.channel_for(known_clean=False, codec_applied=codec_applied)
            decision = prevention.decide(pred, cfg, model_thresholds, channel)
        except Exception:
            logger.exception("Detector failed")
            raise HTTPException(500, "Detector failed") from None
        if language is not None and not language.supported:  # type: ignore[attr-defined]
            decision = prevention.unsupported_language(decision)

        try:  # session, alert and blocklist update commit together or not at all
            row = prevention.store_session(
                db, uuid.uuid4().hex, caller_id, pred, decision, codec_applied, language=language
            )
        except SQLAlchemyError:
            db.rollback()
            logger.exception("Could not store session")
            raise HTTPException(500, "Could not store session") from None
        if row.flagged:
            request.app.state.notifier.send(
                "call_flagged",
                {
                    "session_id": row.session_id,
                    "caller_id": row.caller_id,
                    "source": "upload",
                    "fake_score": row.fake_score,
                    "threshold": row.threshold,
                    "model_version": row.model_version,
                    "evidence_head": evidence.head(db),
                },
            )

        return AnalyzeResponse(
            session_id=row.session_id,
            label=decision.label,
            fake_score=pred.fake_score,
            threshold=decision.threshold,
            latency_ms=pred.latency_ms,
            model_version=pred.model_version,
            flagged=decision.flagged,
            codec_applied=codec_applied,
            codec_ms=codec_ms,
            **(_timeline_fields(windows, decision.threshold) if windows is not None else {}),
            **language_fields(language),
        )

    # live call monitor (api/stream.py); the /monitor page is its browser client
    app.add_api_websocket_route("/v1/stream", handle_stream)
    app.include_router(transactions_router)  # transfer checks, call timelines, incident reports

    @app.get("/monitor", response_class=HTMLResponse, include_in_schema=False)
    def monitor() -> HTMLResponse:
        """The live call monitor page (browser client of /v1/stream)."""
        return HTMLResponse(render_monitor(), headers={"Cache-Control": "no-store"})

    @app.get("/v1/codecs", response_model=CodecsResponse)
    def list_codecs(cfg: Annotated[Settings, Depends(get_settings)]) -> CodecsResponse:
        """Every registered codec, whether this server's ffmpeg can run it, and its bitrate."""
        available = codec_sim.available_codecs() if codec_sim.ffmpeg_path() else {}
        return CodecsResponse(
            default=cfg.simulate_codec,
            codecs=[
                CodecInfo(
                    name=name,
                    available=bool(available.get(name)),
                    bitrate=codec_sim.eval_bitrate(name),
                )
                for name in CODECS
            ],
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
