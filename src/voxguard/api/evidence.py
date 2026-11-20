"""Tamper-evident evidence trail: an append-only, hash-chained log of what LineTrue recorded.

Every stored session (upload or live call), a live call's per-window scores, every transfer
decision and every verification step adds one entry to ``evidence_log``, in the same database
transaction as the record itself. Each entry holds:

- ``record_digest``: SHA-256 of the record's fields, canonically serialised (sorted keys, fixed
  number formats);
- ``prev_hash``: the previous entry's hash (64 zeros for the first);
- ``hash``: SHA-256 of ``prev_hash | seq | kind | ref | created_at | record_digest``.

Writers hold SQLite's write lock (``begin_write``) while appending, so entries are strictly
ordered. ``verify`` walks the chain and recomputes everything from the records as they are now:
an edited record, a deleted one, an edited or removed log entry, or a re-ordered chain is
reported. Alone, a chain shows that the log is internally consistent; someone able to rewrite the
whole database could rebuild it. So the chain head is also sent out with every webhook event and
printed on each incident report: a copy held elsewhere anchors it.

What it is not: no audio (there is none to keep), no signatures from a trusted third party, and
no claim of legal admissibility. It makes tampering detectable, it does not prevent it.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from voxguard.api.db import (
    CallWindowRow,
    EvidenceRow,
    SessionRow,
    TransactionEventRow,
    TransactionRow,
)

GENESIS = "0" * 64


def _canon(value: Any) -> Any:
    if isinstance(value, float):
        return format(value, ".17g")  # exact round trip, independent of repr changes
    if isinstance(value, datetime):
        aware = value if value.tzinfo else value.replace(tzinfo=UTC)
        return aware.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    if isinstance(value, list | tuple):
        return [_canon(v) for v in value]
    if isinstance(value, dict):
        return {k: _canon(v) for k, v in value.items()}
    return value


def digest(payload: dict[str, Any]) -> str:
    body = json.dumps(_canon(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(body.encode()).hexdigest()


def entry_hash(prev: str, seq: int, kind: str, ref: str, created_at: datetime, rec: str) -> str:
    line = "|".join([prev, str(seq), kind, ref, _canon(created_at), rec])
    return hashlib.sha256(line.encode()).hexdigest()


# --- what each kind of record contributes -------------------------------------------------------


def session_payload(row: SessionRow) -> dict[str, Any]:
    return {
        "session_id": row.session_id, "created_at": row.created_at, "caller_id": row.caller_id,
        "label": row.label, "fake_score": row.fake_score, "threshold": row.threshold,
        "flagged": row.flagged, "codec_applied": row.codec_applied,
        "model_version": row.model_version, "latency_ms": row.latency_ms,
    }  # fmt: skip


def windows_payload(rows: list[CallWindowRow]) -> dict[str, Any]:
    return {"windows": [[r.t_s, r.window_score, r.running_score] for r in rows]}


def decision_payload(row: TransactionRow) -> dict[str, Any]:
    return {
        "decision_id": row.decision_id, "created_at": row.created_at,
        "session_id": row.session_id, "amount_inr": row.amount_inr,
        "beneficiary": row.beneficiary, "high_value": row.high_value, "live": row.live,
        "tier": row.tier, "action": row.action, "score": row.score, "threshold": row.threshold,
        "windows": row.windows, "reasons": row.reasons, "required_steps": row.required_steps,
    }  # fmt: skip


def event_payload(row: TransactionEventRow) -> dict[str, Any]:
    return {
        "event_id": row.id, "decision_id": row.decision_id, "created_at": row.created_at,
        "kind": row.kind, "method": row.method, "verified_by": row.verified_by, "note": row.note,
    }  # fmt: skip


# --- writing ------------------------------------------------------------------------------------


def head(db: Session) -> str:
    return db.scalar(select(EvidenceRow.hash).order_by(EvidenceRow.seq.desc()).limit(1)) or GENESIS


def append(
    db: Session, kind: str, ref: str, session_id: str, payload: dict[str, Any]
) -> EvidenceRow:
    """Add one entry. The caller holds the write lock (``begin_write``) and commits."""
    last = db.scalars(select(EvidenceRow).order_by(EvidenceRow.seq.desc()).limit(1)).first()
    seq = (last.seq + 1) if last else 1
    prev = last.hash if last else GENESIS
    now = datetime.now(UTC)
    rec = digest(payload)
    row = EvidenceRow(seq=seq, created_at=now, kind=kind, ref=ref, session_id=session_id,
                      record_digest=rec, prev_hash=prev,
                      hash=entry_hash(prev, seq, kind, ref, now, rec))  # fmt: skip
    db.add(row)
    db.flush()
    return row


# --- checking -----------------------------------------------------------------------------------


def _current_payload(db: Session, e: EvidenceRow) -> dict[str, Any] | None:
    """The record an entry points at, as it is in the database now; None if it is gone."""
    if e.kind == "session":
        row = db.get(SessionRow, e.ref)
        return session_payload(row) if row else None
    if e.kind == "windows":
        rows = list(db.scalars(select(CallWindowRow).where(CallWindowRow.session_id == e.ref)
                               .order_by(CallWindowRow.t_s)))  # fmt: skip
        return windows_payload(rows) if rows else None
    if e.kind == "decision":
        row = db.get(TransactionRow, e.ref)
        return decision_payload(row) if row else None
    if e.kind == "event":
        row = db.get(TransactionEventRow, int(e.ref))
        return event_payload(row) if row else None
    return None


def verify(db: Session, session_id: str | None = None) -> dict[str, Any]:
    """Walk the whole chain. Returns ``ok``, the number of entries, the head, the problems found
    (``seq``, ``kind``, ``ref``, ``problem``), and the entries for ``session_id`` if given."""
    entries = list(db.scalars(select(EvidenceRow).order_by(EvidenceRow.seq)))
    problems: list[dict[str, Any]] = []
    prev, expected_seq = GENESIS, 1
    for e in entries:
        where = {"seq": e.seq, "kind": e.kind, "ref": e.ref}
        if e.seq != expected_seq:
            problems.append({**where, "problem": f"entries missing before seq {e.seq}"})
        if e.prev_hash != prev:
            problems.append({**where, "problem": "does not link to the previous entry"})
        if entry_hash(e.prev_hash, e.seq, e.kind, e.ref, e.created_at, e.record_digest) != e.hash:
            problems.append({**where, "problem": "entry was altered"})
        current = _current_payload(db, e)
        if current is None:
            problems.append({**where, "problem": "record was deleted"})
        elif digest(current) != e.record_digest:
            problems.append({**where, "problem": "record was altered"})
        prev, expected_seq = e.hash, e.seq + 1
    out: dict[str, Any] = {
        "ok": not problems, "entries": len(entries), "head": prev if entries else GENESIS,
        "first_entry_at": entries[0].created_at if entries else None, "problems": problems,
    }  # fmt: skip
    if session_id is not None:
        out["session_entries"] = [
            {
                "seq": e.seq,
                "kind": e.kind,
                "ref": e.ref,
                "created_at": e.created_at,
                "hash": e.hash,
            }  # fmt: skip
            for e in entries
            if e.session_id == session_id
        ]
    return out
