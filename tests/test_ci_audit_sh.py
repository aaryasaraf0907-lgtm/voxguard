""".github/scripts/audit.sh with a stub `uvx` (no network): the accepted exceptions reach
pip-audit as --ignore-vuln arguments, none when the list is empty, and a failing audit fails the
script. It must also run on macOS's bash 3.2 (`make audit`), so no bash-4-only builtins."""

from __future__ import annotations

import datetime as dt
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".github" / "scripts"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _tree(tmp_path: Path, ignore: str, uvx_exit: int = 0) -> tuple[Path, Path, dict[str, str]]:
    """A copy of the scripts, the real uv.lock and the given ignore file, plus a stub uvx that
    records its arguments (one call per line, NUL between arguments)."""
    scripts = tmp_path / "repo" / ".github" / "scripts"
    scripts.mkdir(parents=True)
    for name in ("audit.sh", "audit_inputs.py"):
        shutil.copy2(SCRIPTS / name, scripts / name)
    shutil.copy2(ROOT / "uv.lock", tmp_path / "repo" / "uv.lock")
    (tmp_path / "repo" / ".github" / "pip-audit-ignore.txt").write_text(ignore)
    bin_dir, calls = tmp_path / "bin", tmp_path / "calls"
    bin_dir.mkdir()
    uvx = bin_dir / "uvx"
    uvx.write_text(
        f'#!/usr/bin/env bash\nprintf "%s\\0" "$@" >> {calls}\necho >> {calls}\nexit {uvx_exit}\n'
    )
    uvx.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
    env.pop("GITHUB_STEP_SUMMARY", None)
    return scripts / "audit.sh", calls, env


def _calls(calls: Path) -> list[list[str]]:
    return [line.strip("\0").split("\0") for line in calls.read_text().splitlines() if line]


def test_empty_exception_list_passes_no_ignore_arguments(tmp_path: Path) -> None:
    script, calls, env = _tree(tmp_path, "# no exceptions\n")
    proc = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    runs = _calls(calls)
    assert runs and all("--ignore-vuln" not in args for args in runs)
    assert all(args[0].startswith("pip-audit==") and "--no-deps" in args for args in runs)


def test_accepted_exceptions_reach_every_pip_audit_run(tmp_path: Path) -> None:
    review = (dt.date.today() + dt.timedelta(days=30)).isoformat()
    ignore = (
        f"PYSEC-2099-1 review:{review} no fixed version yet\n"
        f"GHSA-2c3f-4g5h-6j7m review:{review} only in the training extra\n"
    )
    script, calls, env = _tree(tmp_path, ignore)
    proc = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    for args in _calls(calls):
        pairs = [args[i + 1] for i, a in enumerate(args) if a == "--ignore-vuln"]
        assert pairs == ["PYSEC-2099-1", "GHSA-2c3f-4g5h-6j7m"]


def test_a_failing_audit_fails_the_script(tmp_path: Path) -> None:
    script, _, env = _tree(tmp_path, "", uvx_exit=1)
    assert subprocess.run(["bash", str(script)], env=env, capture_output=True).returncode == 1


def test_no_bash4_only_builtins() -> None:
    """macOS ships bash 3.2: no mapfile/readarray, associative arrays or case-changing expansion,
    and arrays are expanded with the ${a[@]+"${a[@]}"} guard (empty arrays are unbound there)."""
    code = "\n".join(
        line
        for line in (SCRIPTS / "audit.sh").read_text().splitlines()
        if not line.lstrip().startswith("#")
    )
    assert not re.search(r"\b(mapfile|readarray)\b|declare -A|\$\{\w+(,,|\^\^)", code)
    for use in re.findall(r'\S*"\$\{(\w+)\[@\]\}"\S*', code):
        assert f'${{{use}[@]+"${{{use}[@]}}"}}' in code
