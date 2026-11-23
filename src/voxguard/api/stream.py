"""Live call monitor: ``WS /v1/stream`` scores a call while it is still going on.

The client (the ``/monitor`` page or ``scripts/stream_call.py``) sends raw mono float32 audio
(little-endian) in small binary frames at its own sample rate, given as ``?sample_rate=``. Once
the detector's window (64,600 samples at 16 kHz, about 4 s) has arrived, the latest window is
scored every HOP_S seconds of audio, exactly as ``POST /v1/analyze`` would score that 4 s clip:
phone-line codec first if ``?codec=`` is given, then ``Detector.predict`` (which resamples to
16 kHz itself). Each score is pushed back at once with the running average of the window scores,
and an ``alert`` is sent the first time that average reaches ALERT_FACTOR x the threshold
(``api/risk.py``); the stored verdict still uses the threshold itself (the prevention rule).

When the client sends ``{"type": "end"}`` or disconnects, one session is stored through the usual
prevention layer (alert and blocklist when flagged). Its score is the running average of the live
windows: the stream's verdict. Only the latest window of audio is kept, in memory; nothing is
written to disk (CLAUDE.md section 15: no recordings retained).

Messages from the server (JSON text frames): ``ready``, ``score``, ``alert``, ``busy`` (a window
was skipped because every analysis slot stayed taken for one hop), ``limit``, ``final``,
``error``. Limits: ``VOXGUARD_MAX_STREAMS`` streams at once, ``VOXGUARD_MAX_STREAM_SECONDS`` of
audio per stream, frames up to MAX_FRAME_BYTES, and IDLE_TIMEOUT_S without a frame.

What it is not: it listens to a microphone or a file, not a phone line; the codec runs per window,
not across the whole call; the first score needs one window of audio and an alert needs two
(MIN_ALERT_WINDOWS).
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
import uuid
from dataclasses import dataclass, field

import numpy as np
from anyio import to_thread
from fastapi import WebSocket, WebSocketDisconnect
from sqlalchemy.exc import SQLAlchemyError

from voxguard.api import codec_sim, evidence, prevention
from voxguard.api.deps import Settings
from voxguard.api.gate import AnalysisGate, Busy
from voxguard.api.risk import ALERT_FACTOR, CAUTION_FACTOR, MIN_ALERT_WINDOWS
from voxguard.data.codecs import CODECS
from voxguard.inference import NUM_SAMPLES, TARGET_SR, Detector, Prediction

logger = logging.getLogger(__name__)

HOP_S = 2.0  # seconds of audio between scores, once the first window has arrived
MIN_RATE, MAX_RATE = 8_000, 48_000
MAX_FRAME_BYTES = 1024 * 1024  # about 5 s at 48 kHz; clients send about 0.25 s per frame
IDLE_TIMEOUT_S = 30.0
MIN_FINAL_S = 1.0  # a call shorter than one window is scored once at the end if this long
WINDOW_S = NUM_SAMPLES / TARGET_SR

# WebSocket close codes (RFC 6455)
CLOSE_NORMAL, CLOSE_POLICY, CLOSE_TRY_AGAIN, CLOSE_ERROR = 1000, 1008, 1013, 1011


class StreamError(Exception):
    """A client error: reported in an ``error`` message, then the stream closes."""


@dataclass
class WindowCutter:
    """Keeps only the latest window of audio and says when the next one is due."""

    sample_rate: int
    window_n: int = 0
    hop_n: int = 0
    buffer: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    received: int = 0
    next_end: int = 0

    def __post_init__(self) -> None:
        self.window_n = math.ceil(NUM_SAMPLES * self.sample_rate / TARGET_SR)
        self.hop_n = round(HOP_S * self.sample_rate)
        self.next_end = self.window_n

    def feed(self, samples: np.ndarray) -> tuple[np.ndarray, float, int] | None:
        """Add samples. Returns (latest window, its end in seconds, windows skipped because
        several came due at once) when a window is due, else None. Only the newest due window
        is returned: if scoring falls behind, the monitor stays current instead of lagging."""
        self.received += len(samples)
        self.buffer = np.concatenate([self.buffer, samples])[-self.window_n :]
        if self.received < self.next_end:
            return None
        due = 1 + (self.received - self.next_end) // self.hop_n
        self.next_end += due * self.hop_n
        end = self.received  # the newest audio: the window ends now
        return self.buffer.copy(), end / self.sample_rate, due - 1

    @property
    def seconds(self) -> float:
        return self.received / self.sample_rate


@dataclass
class LiveCall:
    threshold: float
    session_id: str = ""
    caller_id: str | None = None
    codec: str | None = None
    scores: list[float] = field(default_factory=list)
    points: list[tuple[float, float, float]] = field(default_factory=list)  # t_s, window, running
    latencies: list[float] = field(default_factory=list)
    model_version: str = ""
    first_flag_s: float | None = None
    skipped: int = 0

    @property
    def running(self) -> float:
        return sum(self.scores) / len(self.scores)

    def add(self, score: float, latency_ms: float, model_version: str, t_s: float) -> bool:
        """Record a window score. True if this one first brought the running average to the
        alert level, ALERT_FACTOR x the threshold (the moment to alert)."""
        self.scores.append(score)
        self.latencies.append(latency_ms)
        self.points.append((t_s, score, self.running))
        self.model_version = model_version
        if (
            self.first_flag_s is None
            and len(self.scores) >= MIN_ALERT_WINDOWS
            and prevention.is_flagged(self.running, ALERT_FACTOR * self.threshold)
        ):
            self.first_flag_s = t_s
            return True
        return False


def origin_allowed(
    origin: str | None, host: str | None, secure: bool, extra: tuple[str, ...]
) -> bool:
    """Browsers send Origin on a WebSocket handshake; other clients (stream_call.py) do not. A
    page on another site must not be able to stream to, and read scores from, a local API."""
    if origin is None:
        return True
    origin = origin.rstrip("/")
    own = f"{'https' if secure else 'http'}://{host}" if host else None
    return origin == own or origin in extra


def _score_window(
    det: Detector, window: np.ndarray, sample_rate: int, codec: str | None
) -> tuple[Prediction, float | None]:
    """As POST /v1/analyze scores a clip: codec first if chosen, then the detector."""
    codec_ms = None
    if codec is not None:
        t0 = time.perf_counter()
        window, sample_rate = codec_sim.simulate_phone_line(window, sample_rate, codec)
        codec_ms = (time.perf_counter() - t0) * 1000.0
    return det.predict(window, sample_rate), codec_ms


def _params(ws: WebSocket, cfg: Settings) -> tuple[int, str | None, str | None]:
    q = ws.query_params
    try:
        rate = int(q.get("sample_rate", ""))
    except ValueError:
        raise StreamError("sample_rate must be a whole number of Hz") from None
    if not MIN_RATE <= rate <= MAX_RATE:
        raise StreamError(f"sample_rate must be between {MIN_RATE} and {MAX_RATE} Hz")
    caller = prevention.normalize_caller_id(q.get("caller_id"))  # as the blocklist stores it
    if caller is not None and len(caller) > 128:
        raise StreamError("caller_id is longer than 128 characters")
    codec = q.get("codec") or None
    if codec is not None:
        if codec not in CODECS:
            raise StreamError(f"Unknown codec {codec!r}; known: {sorted(CODECS)}")
        try:
            codec_sim.ensure_available(codec)
        except codec_sim.CodecUnavailableError:
            raise StreamError(
                f"Codec simulation for {codec} is unavailable on this server."
            ) from None
    return rate, caller, codec


def _samples(message: dict) -> np.ndarray | None:
    """Audio from a binary frame; None for {"type": "end"}. Raises StreamError otherwise."""
    data = message.get("bytes")
    if data is not None:
        if len(data) > MAX_FRAME_BYTES:
            raise StreamError(f"audio frames must be at most {MAX_FRAME_BYTES} bytes")
        if len(data) % 4:
            raise StreamError("audio frames must hold whole float32 samples")
        x = np.frombuffer(data, dtype="<f4").astype(np.float32)
        if not np.isfinite(x).all():
            raise StreamError("audio contains NaN or infinite samples")
        return x
    try:
        kind = json.loads(message.get("text") or "").get("type")
    except (ValueError, AttributeError):
        kind = None
    if kind == "end":
        return None
    raise StreamError('send binary float32 audio frames, then {"type": "end"}')


async def handle_stream(ws: WebSocket) -> None:
    state = ws.app.state
    cfg: Settings = state.settings
    if not origin_allowed(
        ws.headers.get("origin"), ws.headers.get("host"), ws.url.scheme == "wss", cfg.stream_origins
    ):
        await ws.close(code=CLOSE_POLICY)  # before accept: the handshake is refused (403)
        return
    await ws.accept()
    if state.live_streams >= cfg.max_streams:
        await ws.send_json({"type": "error", "detail": "Server busy: too many live calls at once."})
        await ws.close(code=CLOSE_TRY_AGAIN)
        return
    state.live_streams += 1
    try:
        await _run(ws, cfg)
    finally:
        state.live_streams -= 1


async def _run(ws: WebSocket, cfg: Settings) -> None:
    state = ws.app.state
    det: Detector = state.detector
    gate: AnalysisGate = state.gate
    try:
        rate, caller, codec = _params(ws, cfg)
    except StreamError as exc:
        await ws.send_json({"type": "error", "detail": str(exc)})
        await ws.close(code=CLOSE_POLICY)
        return
    channel = prevention.channel_for(known_clean=False, codec_applied=codec)
    probe = Prediction(0.0, "real", det.threshold, 0.0, det.model_version)
    threshold = prevention.effective_threshold(probe, cfg, state.thresholds, channel)
    session_id = uuid.uuid4().hex
    cutter = WindowCutter(rate)
    call = LiveCall(threshold, session_id=session_id, caller_id=caller, codec=codec)
    state.live_calls[session_id] = call  # read by POST /v1/transactions/check during the call
    try:
        await _serve(ws, cfg, det, gate, codec, cutter, call, caller, session_id)
    finally:
        state.live_calls.pop(session_id, None)


async def _serve(
    ws: WebSocket,
    cfg: Settings,
    det: Detector,
    gate: AnalysisGate,
    codec: str | None,
    cutter: WindowCutter,
    call: LiveCall,
    caller: str | None,
    session_id: str,
) -> None:
    state = ws.app.state
    await ws.send_json(
        {
            "type": "ready",
            "session_id": session_id,
            "threshold": call.threshold,
            "window_s": WINDOW_S,
            "hop_s": HOP_S,
            "codec": codec,
            "model_version": det.model_version,
            "min_alert_windows": MIN_ALERT_WINDOWS,
            "caution_factor": CAUTION_FACTOR,
            "alert_factor": ALERT_FACTOR,
        }
    )
    connected = True
    try:
        while True:
            try:
                message = await asyncio.wait_for(ws.receive(), IDLE_TIMEOUT_S)
            except TimeoutError:
                raise StreamError(f"no audio for {IDLE_TIMEOUT_S:g} s") from None
            if message["type"] == "websocket.disconnect":
                connected = False
                break
            samples = _samples(message)
            if samples is None:
                break
            due = cutter.feed(samples)
            if due is not None:
                await _score_due(ws, det, gate, codec, cutter, call, due)
            if cutter.seconds >= cfg.max_stream_seconds:
                await ws.send_json({"type": "limit", "seconds": cfg.max_stream_seconds})
                break
    except StreamError as exc:
        await ws.send_json({"type": "error", "detail": str(exc)})
        await ws.close(code=CLOSE_POLICY)
        return
    except WebSocketDisconnect:
        connected = False
    except Exception:
        logger.exception("live stream failed")
        if connected:
            await ws.send_json({"type": "error", "detail": "Live analysis failed"})
            await ws.close(code=CLOSE_ERROR)
        return
    final = await _finish(state, codec, cutter, call, caller, session_id)
    if connected:
        await ws.send_json(final)
        await ws.close(code=CLOSE_NORMAL)


async def _score_due(
    ws: WebSocket,
    det: Detector,
    gate: AnalysisGate,
    codec: str | None,
    cutter: WindowCutter,
    call: LiveCall,
    due: tuple[np.ndarray, float, int],
) -> None:
    window, t_s, behind = due
    call.skipped += behind
    try:
        pred, codec_ms = await to_thread.run_sync(
            lambda: gate.run(_score_window, det, window, cutter.sample_rate, codec, wait_s=HOP_S)
        )
    except Busy:
        call.skipped += 1
        await ws.send_json({"type": "busy", "t_s": t_s})
        return
    except (codec_sim.CodecUnavailableError, codec_sim.CodecSimulationError):
        raise StreamError("Codec simulation failed") from None
    alert = call.add(pred.fake_score, pred.latency_ms, pred.model_version, t_s)
    await ws.send_json(
        {
            "type": "score",
            "t_s": t_s,
            "window_score": pred.fake_score,
            "running_score": call.running,
            "flagged": prevention.is_flagged(call.running, call.threshold),
            "latency_ms": pred.latency_ms,
            "codec_ms": codec_ms,
            "windows": len(call.scores),
        }
    )
    if alert:
        # The alert fires at ALERT_FACTOR x the threshold, which is at least the caution/high
        # boundary, so an alert is always "high" (the field stays for clients and webhooks).
        level = "high"
        await ws.send_json(
            {
                "type": "alert",
                "t_s": t_s,
                "running_score": call.running,
                "threshold": call.threshold,
                "level": level,
            }
        )
        ws.app.state.notifier.send(
            "live_alert",
            {
                "session_id": call.session_id,
                "caller_id": call.caller_id,
                "t_s": t_s,
                "running_score": call.running,
                "threshold": call.threshold,
                "level": level,
            },
        )


async def _finish(
    state: object,
    codec: str | None,
    cutter: WindowCutter,
    call: LiveCall,
    caller: str | None,
    session_id: str,
) -> dict:
    """Store the call's session (if anything was scored) and build the ``final`` message."""
    det: Detector = state.detector  # type: ignore[attr-defined]
    gate: AnalysisGate = state.gate  # type: ignore[attr-defined]
    cfg: Settings = state.settings  # type: ignore[attr-defined]
    if not call.scores and cutter.seconds >= MIN_FINAL_S:  # shorter than one window: score it
        try:
            pred, _ = await to_thread.run_sync(
                lambda: gate.run(_score_window, det, cutter.buffer, cutter.sample_rate, codec)
            )
            call.add(pred.fake_score, pred.latency_ms, pred.model_version, cutter.seconds)
        except Exception:  # busy, codec or detector failure: still send `final` (no verdict)
            logger.exception("live stream: the short call could not be scored")
    base = {
        "type": "final",
        "session_id": session_id,
        "duration_s": cutter.seconds,
        "windows": len(call.scores),
        "skipped": call.skipped,
        "codec_applied": codec,
        "first_flag_s": call.first_flag_s,
    }
    if not call.scores:
        return {**base, "stored": False}
    clip = Prediction(
        fake_score=min(1.0, max(0.0, call.running)),
        label="real",
        threshold=det.threshold,
        latency_ms=sum(call.latencies) / len(call.latencies),
        model_version=call.model_version,
    )
    channel = prevention.channel_for(known_clean=False, codec_applied=codec)
    decision = prevention.decide(clip, cfg, state.thresholds, channel)  # type: ignore[attr-defined]

    def store() -> str | None:
        """The evidence chain head after storing, or None if the call could not be stored."""
        with state.session_factory() as db:  # type: ignore[attr-defined]
            try:
                prevention.store_session(
                    db, session_id, caller, clip, decision, codec, windows=call.points
                )
            except SQLAlchemyError:
                logger.exception("live stream: could not store session")
                return None
            return evidence.head(db)

    head = await to_thread.run_sync(store)
    stored = head is not None
    if stored and decision.flagged:
        state.notifier.send(  # type: ignore[attr-defined]
            "call_flagged",
            {
                "session_id": session_id,
                "caller_id": caller,
                "source": "live",
                "fake_score": clip.fake_score,
                "threshold": decision.threshold,
                "model_version": clip.model_version,
                "evidence_head": head,
            },
        )
    return {
        **base,
        "stored": stored,
        "label": decision.label,
        "fake_score": clip.fake_score,
        "threshold": decision.threshold,
        "flagged": decision.flagged,
        "latency_ms": clip.latency_ms,
        "model_version": clip.model_version,
    }
