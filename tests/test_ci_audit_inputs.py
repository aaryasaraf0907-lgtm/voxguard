"""The dependency audit's inputs (.github/scripts/audit_inputs.py): every locked package is
audited, and an accepted exception is never applied silently or after its review date."""

from __future__ import annotations

import datetime as dt
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "audit_inputs", ROOT / ".github" / "scripts" / "audit_inputs.py"
)
assert _spec is not None and _spec.loader is not None
audit_inputs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit_inputs)

TODAY = dt.date(2026, 11, 18)


def write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_every_package_in_the_real_lock_is_audited() -> None:
    import tomllib

    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    expected = {(p["name"], p["version"]) for p in lock["package"] if p["name"] != "voxguard"}
    pins = audit_inputs.locked_pins()
    assert set(pins) == expected
    files = audit_inputs.split_unique(pins)
    assert sorted(line for f in files for line in f) == sorted(f"{n}=={v}" for n, v in expected)


def test_two_versions_of_one_package_go_to_different_files() -> None:
    files = audit_inputs.split_unique([("a", "1"), ("a", "2"), ("b", "1")])
    assert files == [["a==1", "b==1"], ["a==2"]]


def test_platform_only_packages_are_kept(tmp_path: Path) -> None:
    lock = write(tmp_path, "uv.lock", """
[[package]]
name = "colorama"
version = "0.4.6"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "voxguard"
version = "0.1.0"
source = { editable = "." }
""")  # fmt: skip
    assert audit_inputs.locked_pins(lock) == [("colorama", "0.4.6")]


def test_a_package_not_from_an_index_stops_the_audit(tmp_path: Path) -> None:
    lock = write(tmp_path, "uv.lock", """
[[package]]
name = "thing"
version = "1.0"
source = { git = "https://example.invalid/thing.git" }
""")  # fmt: skip
    with pytest.raises(audit_inputs.AuditInputError, match="cannot be audited"):
        audit_inputs.locked_pins(lock)


def test_a_valid_exception_is_accepted_and_listed(tmp_path: Path) -> None:
    path = write(tmp_path, "ignore.txt", "# comment\n\nPYSEC-2026-1 review:2026-12-01 no fix yet\n")
    assert audit_inputs.exceptions(path, TODAY) == [("PYSEC-2026-1", "2026-12-01", "no fix yet")]


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("PYSEC-2026-1", "expected"),  # no review date or reason
        ("PYSEC-2026-1 review:2026-12-01", "expected"),  # no reason
        ("PYSEC-2026-1 no fix yet", "expected"),  # no review date
        ("PYSEC-2026-1 review:2026-11-17 no fix yet", "expired"),
        ("PYSEC-2026-1 review:2026-13-01 no fix yet", "bad review date"),
        ("pysec-2026-1 review:2026-12-01 no fix yet", "not a PYSEC"),
        ("--ignore-vuln review:2026-12-01 injected", "not a PYSEC"),
    ],
)
def test_a_bad_or_expired_exception_stops_the_audit(
    tmp_path: Path, line: str, message: str
) -> None:
    path = write(tmp_path, "ignore.txt", line + "\n")
    with pytest.raises(audit_inputs.AuditInputError, match=message):
        audit_inputs.exceptions(path, TODAY)


def test_the_committed_exception_file_is_valid() -> None:
    audit_inputs.exceptions()


_tspec = importlib.util.spec_from_file_location(
    "check_trivyignore", ROOT / ".github" / "scripts" / "check_trivyignore.py"
)
assert _tspec is not None and _tspec.loader is not None
check_trivyignore = importlib.util.module_from_spec(_tspec)
_tspec.loader.exec_module(check_trivyignore)


def test_a_trivy_exception_needs_a_reason_directly_above_and_an_expiry(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        ".trivyignore",
        "# header\n\n# reason: base image, no fix yet\nCVE-2026-1 exp:2026-12-01\n",
    )
    assert check_trivyignore.entries(path, TODAY) == [
        ("CVE-2026-1", "2026-12-01", "base image, no fix yet")
    ]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("CVE-2026-1 exp:2026-12-01\n", "needs a '# reason"),
        ("# just a comment\nCVE-2026-1 exp:2026-12-01\n", "needs a '# reason"),  # not a reason
        ("# reason: x\n\nCVE-2026-1 exp:2026-12-01\n", "needs a '# reason"),  # not directly above
        ("# reason: x\nCVE-2026-1\n", "expected"),  # no expiry: would apply forever
        ("# reason: x\nCVE-2026-1 exp:2026-11-17\n", "expired"),
        ("# reason: x\nCVE-2026-1 exp:2026-02-30\n", "bad expiry"),
        (
            "# reason: x\nCVE-2026-1 exp:2026-12-01\nCVE-2026-2 exp:2026-12-01\n",
            "needs a '# reason",
        ),
        ("# reason: x\n*\n", "expected"),
    ],
)
def test_a_bad_trivy_exception_stops_the_scan(tmp_path: Path, text: str, message: str) -> None:
    path = write(tmp_path, ".trivyignore", text)
    with pytest.raises(check_trivyignore.TrivyIgnoreError, match=message):
        check_trivyignore.entries(path, TODAY)


def test_the_committed_trivyignore_is_valid() -> None:
    check_trivyignore.entries(ROOT / ".trivyignore")
