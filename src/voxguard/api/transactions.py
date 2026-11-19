"""Transfer checks against a call's risk, the call's feature-level timeline and its incident
report (api/policy.py has the rules; api/report_page.py renders the report).

- ``POST /v1/transactions/check``: decide a transfer from the call's risk. A call still going on
  is read from the live monitor (its running average so far); a finished one from its stored
  session. Each decision is stored with an audit trail.
- ``POST /v1/transactions/{decision_id}/verify``: record a completed verification step; the
  transfer is released once the policy's steps are done.
- ``GET /v1/transactions``: decisions with their audit trail, newest first.
- ``GET /v1/sessions/{session_id}/timeline``: a live call's per-window scores (no audio).
- ``GET /v1/sessions/{session_id}/report``: a printable incident report (no audio).
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from voxguard.api import policy
from voxguard.api.db import (
    BlocklistRow,
    CallWindowRow,
    SessionRow,
    TransactionEventRow,
    TransactionRow,
    begin_write,
)
from voxguard.api.deps import Settings, get_db, get_settings
from voxguard.api.report_page import render_report
from voxguard.schemas import (
    CallWindow,
    TransactionCheck,
    TransactionDecision,
    TransactionEvent,
    TransactionVerify,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _flagged_before(db: Session, caller_id: str | None, this_session_flagged: bool) -> bool:
    """True if the caller was flagged on an earlier call (the blocklist counts flagged calls; a
    stored flagged session has already added one)."""
    if caller_id is None:
        return False
    hits = db.scalar(select(BlocklistRow.hit_count).where(BlocklistRow.caller_id == caller_id))
    return (hits or 0) - (1 if this_session_flagged else 0) > 0


def _call_risk(
    request: Request, db: Session, session_id: str
) -> tuple[float | None, float, int | None, str | None, bool, bool]:
    """(score, threshold, live windows, caller_id, live, this session flagged)."""
    call = request.app.state.live_calls.get(session_id)
    if call is not None:
        windows = len(call.scores)
        return (call.running if windows else None, call.threshold, windows, call.caller_id, True,
                False)  # fmt: skip
    row = db.get(SessionRow, session_id)
    if row is None:
        raise HTTPException(404, "Unknown session_id: no live call or stored session")
    return row.fake_score, row.threshold, None, row.caller_id, False, row.flagged


def _decision(db: Session, row: TransactionRow) -> TransactionDecision:
    events = db.scalars(
        select(TransactionEventRow)
        .where(TransactionEventRow.decision_id == row.decision_id)
        .order_by(TransactionEventRow.id)
    ).all()
    completed = sorted({e.method for e in events if e.kind == "verified" and e.method})
    return TransactionDecision(
        decision_id=row.decision_id, created_at=row.created_at, session_id=row.session_id,
        amount_inr=row.amount_inr, beneficiary=row.beneficiary, high_value=row.high_value,
        live=row.live, tier=row.tier, action=row.action, status=row.status, score=row.score,
        threshold=row.threshold, windows=row.windows, reasons=json.loads(row.reasons),
        required_steps=json.loads(row.required_steps), completed_steps=completed,
        events=[TransactionEvent(created_at=e.created_at, kind=e.kind, method=e.method,
                                 verified_by=e.verified_by, note=e.note) for e in events],
    )  # fmt: skip


def _commit(db: Session) -> None:
    try:
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        logger.exception("Could not store the transaction decision")
        raise HTTPException(500, "Could not store the transaction decision") from None


@router.post("/v1/transactions/check", response_model=TransactionDecision)
def check_transaction(
    body: TransactionCheck,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    cfg: Annotated[Settings, Depends(get_settings)],
) -> TransactionDecision:
    score, threshold, windows, caller, live, flagged = _call_risk(request, db, body.session_id)
    base = policy.tier_for(score if score is not None else 0.0, threshold, windows)
    before = _flagged_before(db, caller, flagged)
    tier = policy.escalate(base, before)
    action, high_value = policy.decide(tier, body.amount_inr, cfg.high_value_inr,
                                       dict(cfg.transaction_policy))  # fmt: skip
    reasons = [f"risk_{base}"] + (["not_enough_audio"] if base == "unknown" else [])
    reasons += (["caller_flagged_before"] if before else []) + (
        ["high_value"] if high_value else []
    )
    now = datetime.now(UTC)
    row = TransactionRow(
        decision_id=uuid.uuid4().hex, created_at=now, session_id=body.session_id,
        amount_inr=body.amount_inr, beneficiary=(body.beneficiary or "").strip() or None,
        high_value=high_value, live=live, tier=tier, action=action,
        status=policy.initial_status(action), score=score, threshold=threshold, windows=windows,
        reasons=json.dumps(reasons), required_steps=json.dumps(policy.required_steps(action)),
    )  # fmt: skip
    db.add(row)
    db.flush()
    db.add(TransactionEventRow(decision_id=row.decision_id, created_at=now, kind="decided"))
    _commit(db)
    if action != "allow":
        request.app.state.notifier.send(
            "transaction_held" if action == "hold" else "transaction_step_up",
            {"decision_id": row.decision_id, "session_id": row.session_id, "caller_id": caller,
             "amount_inr": row.amount_inr, "tier": tier, "action": action, "live": live},
        )  # fmt: skip
    return _decision(db, row)


@router.post("/v1/transactions/{decision_id}/verify", response_model=TransactionDecision)
def verify_transaction(
    decision_id: str,
    body: TransactionVerify,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
) -> TransactionDecision:
    begin_write(db)  # one verification at a time: the release decision sees every step
    row = db.get(TransactionRow, decision_id)
    if row is None:
        raise HTTPException(404, "Unknown decision_id")
    if row.status in ("allowed", "released"):
        raise HTTPException(409, f"Nothing to verify: the transfer is already {row.status}")
    if body.method not in json.loads(row.required_steps):
        raise HTTPException(422, f"{body.method} is not one of this decision's steps")
    now = datetime.now(UTC)
    db.add(TransactionEventRow(decision_id=decision_id, created_at=now, kind="verified",
                               method=body.method, verified_by=body.verified_by.strip(),
                               note=(body.note or "").strip() or None))  # fmt: skip
    db.flush()
    done = set(
        db.scalars(
            select(TransactionEventRow.method).where(
                TransactionEventRow.decision_id == decision_id,
                TransactionEventRow.kind == "verified",
            )
        )
    )
    if policy.released(row.action, done):  # type: ignore[arg-type]
        row.status = "released"
        db.add(TransactionEventRow(decision_id=decision_id, created_at=now, kind="released"))
    _commit(db)
    if row.status == "released":
        request.app.state.notifier.send(
            "transaction_released",
            {"decision_id": decision_id, "session_id": row.session_id, "methods": sorted(done)},
        )
    return _decision(db, row)


@router.get("/v1/transactions", response_model=list[TransactionDecision])
def list_transactions(
    db: Annotated[Session, Depends(get_db)],
    session_id: str | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
) -> list[TransactionDecision]:
    stmt = select(TransactionRow).order_by(
        TransactionRow.created_at.desc(), TransactionRow.decision_id.desc()
    )
    if session_id is not None:
        stmt = stmt.where(TransactionRow.session_id == session_id)
    return [_decision(db, r) for r in db.scalars(stmt.limit(limit))]


def _windows(db: Session, session_id: str) -> list[CallWindow]:
    rows = db.scalars(
        select(CallWindowRow)
        .where(CallWindowRow.session_id == session_id)
        .order_by(CallWindowRow.t_s)
    )
    return [CallWindow.model_validate(r) for r in rows]


@router.get("/v1/sessions/{session_id}/timeline", response_model=list[CallWindow])
def session_timeline(session_id: str, db: Annotated[Session, Depends(get_db)]) -> list[CallWindow]:
    """A live call's per-window scores; empty for an uploaded recording (not stored)."""
    if db.get(SessionRow, session_id) is None:
        raise HTTPException(404, "Unknown session_id")
    return _windows(db, session_id)


@router.get("/v1/sessions/{session_id}/report", response_class=HTMLResponse)
def session_report(
    session_id: str,
    db: Annotated[Session, Depends(get_db)],
    lang: Literal["en", "hi"] = "en",
) -> HTMLResponse:
    row = db.get(SessionRow, session_id)
    if row is None:
        raise HTTPException(404, "Unknown session_id")
    stmt = select(TransactionRow).where(TransactionRow.session_id == session_id)
    decisions = [_decision(db, r) for r in db.scalars(stmt.order_by(TransactionRow.created_at))]
    html = render_report(row, _windows(db, session_id), decisions, lang)
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})
