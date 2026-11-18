"""CI's ffmpeg install (.github/scripts/install_ffmpeg.sh): one apt mirror being down must not
fail the job. Run with stub commands (no sudo, no network)."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".github" / "scripts" / "install_ffmpeg.sh"
PRIMARY = "https://mirrors.edge.kernel.org/ubuntu/"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def run(tmp_path: Path, mirrors: str | None, fail_times: int) -> tuple[int, list[str], str]:
    mirrors_file = tmp_path / "apt-mirrors.txt"
    if mirrors is not None:
        mirrors_file.write_text(mirrors)
    calls = tmp_path / "calls"
    counter = tmp_path / "failures-left"
    counter.write_text(str(fail_times))
    apt = tmp_path / "apt-get"
    apt.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "$*" >> {calls}\n'
        f"left=$(cat {counter})\n"
        'if [[ "$*" == *update* && $left -gt 0 ]]; then\n'
        f"  echo $((left - 1)) > {counter}; exit 100\n"
        "fi\n"
    )
    apt.chmod(0o755)
    env = {**os.environ, "MIRRORS_FILE": str(mirrors_file), "SUDO": "", "APT_GET": str(apt),
           "SLEEP": "true"}  # fmt: skip
    proc = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True,
                          timeout=60)  # fmt: skip
    lines = calls.read_text().splitlines() if calls.exists() else []
    return proc.returncode, lines, mirrors_file.read_text()


def test_https_mirror_first_and_runner_mirrors_kept_as_fallbacks(tmp_path: Path) -> None:
    runner = "http://azure.archive.ubuntu.com/ubuntu/\thttp://archive.ubuntu.com/ubuntu/\n"
    code, calls, mirrors = run(tmp_path, runner, fail_times=0)
    assert code == 0
    assert mirrors.splitlines()[0] == PRIMARY
    assert mirrors.splitlines()[1:] == runner.splitlines()
    assert any("install -y --no-install-recommends ffmpeg" in c for c in calls)
    assert all("Acquire::ForceIPv4=true" in c for c in calls)


def test_no_duplicate_when_the_https_mirror_is_already_listed(tmp_path: Path) -> None:
    code, _, mirrors = run(tmp_path, f"{PRIMARY}\nhttp://azure.archive.ubuntu.com/ubuntu/\n", 0)
    assert code == 0
    assert mirrors.splitlines().count(PRIMARY) == 1


def test_works_without_an_existing_mirror_file(tmp_path: Path) -> None:
    code, _, mirrors = run(tmp_path, None, fail_times=0)
    assert code == 0
    assert mirrors.splitlines() == [PRIMARY]


def test_a_failed_attempt_is_retried(tmp_path: Path) -> None:
    code, calls, _ = run(tmp_path, "", fail_times=2)
    assert code == 0
    assert sum("update" in c for c in calls) == 3


def test_three_failed_attempts_fail_the_step(tmp_path: Path) -> None:
    code, calls, _ = run(tmp_path, "", fail_times=5)
    assert code == 1
    assert sum("update" in c for c in calls) == 3
    assert not any("install" in c for c in calls)
