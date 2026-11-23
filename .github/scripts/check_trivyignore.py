"""Validate .trivyignore so no image finding is suppressed silently or forever. Stdlib only.

Each accepted finding is one line ``<ID> exp:<YYYY-MM-DD>`` (Trivy's own expiry syntax: an
expired entry is reported again by Trivy), directly under a ``# reason: <why>`` line. Other
comments never count as a reason. A line without an expiry, without a reason directly above,
with a date that has passed or with an unexpected ID stops the scan. Prints the accepted
entries for the job summary.

Usage: check_trivyignore.py [PATH]   (default: .trivyignore at the repository root)
"""

from __future__ import annotations

import datetime as dt
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REASON_RE = re.compile(r"^#\s*reason:\s*(?P<reason>\S.*)$")
ENTRY_RE = re.compile(
    r"^(?P<id>(CVE-\d{4}-\d+|GHSA(-[23456789cfghjmpqrvwx]{4}){3}))\s+exp:(?P<date>\S+)$"
)


class TrivyIgnoreError(ValueError):
    pass


def entries(path: Path, today: dt.date | None = None) -> list[tuple[str, str, str]]:
    today = today or dt.date.today()
    found: list[tuple[str, str, str]] = []
    reason: str | None = None
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            reason = None  # a reason applies only to the entry directly below it
            continue
        if line.startswith("#"):
            m = REASON_RE.match(line)  # other comments (the header) never count as a reason
            reason = m["reason"] if m else None
            continue
        m = ENTRY_RE.match(line)
        if m is None:
            raise TrivyIgnoreError(
                f"{path.name}:{number}: expected '<CVE/GHSA id> exp:<YYYY-MM-DD>'"
            )
        try:
            expiry = dt.date.fromisoformat(m["date"])
        except ValueError as exc:
            raise TrivyIgnoreError(f"{path.name}:{number}: bad expiry date {m['date']!r}") from exc
        if expiry < today:
            raise TrivyIgnoreError(f"{path.name}:{number}: the exception for {m['id']} expired")
        if reason is None:
            raise TrivyIgnoreError(
                f"{path.name}:{number}: {m['id']} needs a '# reason: <why>' line directly above"
            )
        found.append((m["id"], m["date"], reason))
        reason = None
    return found


def main(argv: list[str]) -> int:
    path = Path(argv[0]) if argv else ROOT / ".trivyignore"
    try:
        accepted = entries(path)
    except TrivyIgnoreError as exc:
        print(f"trivyignore: {exc}", file=sys.stderr)
        return 1
    if accepted:
        print("Accepted image findings in force (excluded from the gate):")
        for vuln_id, date, reason in accepted:
            print(f"- {vuln_id} (expires {date}): {reason}")
    else:
        print("No accepted image findings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
