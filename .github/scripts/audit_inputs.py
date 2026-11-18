"""Inputs for the dependency audit (.github/scripts/audit.sh). Standard library only.

1. Every package in uv.lock (all extras, groups and platforms; transitive ones included) as an
   exact ``name==version`` pin, with environment markers dropped, so a package that only applies
   to another platform or Python version (Windows, 3.13+) is still audited. Two locked versions
   of one package go into different files, since one requirements file can hold one pin per
   name. Anything not from a package index (git, path, URL) cannot be audited by version and
   stops the audit.
2. The accepted exceptions in .github/pip-audit-ignore.txt, validated: each line is
   ``<ID> review:<YYYY-MM-DD> <reason>``, the review date must not have passed, and the reason
   must be given. A malformed or expired entry stops the audit, so no exception is applied
   silently or forever.

Usage: audit_inputs.py OUT_DIR  ->  writes OUT_DIR/req-N.txt and OUT_DIR/ignore-args.txt, and
prints a summary (packages audited, exceptions in force).
"""

from __future__ import annotations

import datetime as dt
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "uv.lock"
IGNORE = ROOT / ".github" / "pip-audit-ignore.txt"
PROJECT = "voxguard"
ID_RE = re.compile(r"^(PYSEC-\d{4}-\d+|GHSA(-[23456789cfghjmpqrvwx]{4}){3}|CVE-\d{4}-\d+)$")
LINE_RE = re.compile(r"^(?P<id>\S+)\s+review:(?P<date>\d{4}-\d{2}-\d{2})\s+(?P<reason>\S.*)$")


class AuditInputError(ValueError):
    pass


def locked_pins(lock: Path = LOCK) -> list[tuple[str, str]]:
    data = tomllib.loads(lock.read_text(encoding="utf-8"))
    pins, unauditable = [], []
    for pkg in data.get("package", []):
        if pkg["name"] == PROJECT:
            continue
        if "registry" not in pkg.get("source", {}):
            unauditable.append(f"{pkg['name']} ({', '.join(pkg.get('source', {})) or 'no source'})")
            continue
        pins.append((pkg["name"], pkg["version"]))
    if unauditable:
        raise AuditInputError(
            "not from a package index, cannot be audited: " + "; ".join(unauditable)
        )
    if not pins:
        raise AuditInputError(f"no packages found in {lock}")
    return sorted(pins)


def split_unique(pins: list[tuple[str, str]]) -> list[list[str]]:
    """Requirement files with at most one pin per package name."""
    files: list[dict[str, str]] = []
    for name, version in pins:
        target = next((f for f in files if name not in f), None)
        if target is None:
            target = {}
            files.append(target)
        target[name] = version
    return [[f"{n}=={v}" for n, v in sorted(f.items())] for f in files]


def exceptions(path: Path = IGNORE, today: dt.date | None = None) -> list[tuple[str, str, str]]:
    today = today or dt.date.today()
    found = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = LINE_RE.match(line)
        if m is None:
            raise AuditInputError(
                f"{path.name}:{number}: expected '<ID> review:<YYYY-MM-DD> <reason>', got {line!r}"
            )
        vuln_id, date, reason = m["id"], m["date"], m["reason"]
        if not ID_RE.match(vuln_id):
            raise AuditInputError(f"{path.name}:{number}: not a PYSEC, GHSA or CVE id: {vuln_id!r}")
        try:
            review = dt.date.fromisoformat(date)
        except ValueError as exc:
            raise AuditInputError(f"{path.name}:{number}: bad review date {date!r}") from exc
        if review < today:
            raise AuditInputError(
                f"{path.name}:{number}: the exception for {vuln_id} expired on {date}; "
                "re-check it and fix, or renew it with a new date and reason"
            )
        found.append((vuln_id, date, reason))
    return found


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    out = Path(argv[0])
    try:
        pins = locked_pins()
        accepted = exceptions()
    except AuditInputError as exc:
        print(f"audit inputs: {exc}", file=sys.stderr)
        return 1
    files = split_unique(pins)
    for i, lines in enumerate(files):
        (out / f"req-{i}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    args = [a for vuln_id, _, _ in accepted for a in ("--ignore-vuln", vuln_id)]
    (out / "ignore-args.txt").write_text("\n".join(args) + ("\n" if args else ""), encoding="utf-8")
    audited = sum(len(f) for f in files)
    assert audited == len(pins)  # every locked pin is in exactly one file
    print(f"Auditing all {audited} locked packages ({len({n for n, _ in pins})} names) "
          f"in {len(files)} requirement file(s).")  # fmt: skip
    if accepted:
        print("Accepted exceptions in force:")
        for vuln_id, date, reason in accepted:
            print(f"- {vuln_id} (review by {date}): {reason}")
    else:
        print("No accepted exceptions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
