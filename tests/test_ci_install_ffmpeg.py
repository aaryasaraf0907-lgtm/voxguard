"""CI's ffmpeg install (.github/scripts/install_ffmpeg.sh): an apt mirror being down or slow must
not fail or stall the job. Run with stub commands (no sudo, no network)."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".github" / "scripts" / "install_ffmpeg.sh"
ARCHIVE = "https://archive.ubuntu.com/ubuntu/"
KERNEL = "https://mirrors.edge.kernel.org/ubuntu/"
AZURE = "http://azure.archive.ubuntu.com/ubuntu/"
SECURITY = "https://security.ubuntu.com/ubuntu/"
RUNNER = f"{AZURE}\tpriority:1\n{ARCHIVE}\tpriority:2\n{SECURITY}\tpriority:3\n"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def run(
    tmp_path: Path, mirrors: str | None, speeds: dict[str, str], apt_failures: int = 0
) -> tuple[int, list[str], list[str], list[str]]:
    """speeds: mirror URL -> probe speed in bytes/s, or "fail"; unlisted mirrors fail."""
    mirrors_file = tmp_path / "apt-mirrors.txt"
    if mirrors is not None:
        mirrors_file.write_text(mirrors)
    calls, probes, counter = tmp_path / "calls", tmp_path / "probes", tmp_path / "left"
    counter.write_text(str(apt_failures))
    apt = tmp_path / "apt-get"
    apt.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "$*" >> {calls}\n'
        f"left=$(cat {counter})\n"
        'if [[ "$*" == *update* && $left -gt 0 ]]; then\n'
        f"  echo $((left - 1)) > {counter}; exit 100\n"
        "fi\n"
    )
    cases = "\n".join(
        f'  "{url}dists/test/InRelease") '
        + ("exit 28 ;;" if speed == "fail" else f"echo -n {speed} ;;")
        for url, speed in speeds.items()
    )
    curl = tmp_path / "curl"
    curl.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "${{@: -1}}" >> {probes}\n'
        f'case "${{@: -1}}" in\n{cases}\n  *) exit 7 ;;\nesac\n'
    )
    for stub in (apt, curl):
        stub.chmod(0o755)
    env = {**os.environ, "MIRRORS_FILE": str(mirrors_file), "SUDO": "", "APT_GET": str(apt),
           "SLEEP": "true", "CURL": str(curl), "CODENAME": "test"}  # fmt: skip
    proc = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True,
                          timeout=60)  # fmt: skip

    def read(p: Path) -> list[str]:
        return p.read_text().splitlines() if p.exists() else []

    lines = mirrors_file.read_text().splitlines()
    # every line is "<url>\tpriority:<rank>", ranks 1..N in file order (apt would otherwise pick
    # among untagged mirrors at random)
    assert [line.split("\t")[1] for line in lines] == [
        f"priority:{i}" for i in range(1, len(lines) + 1)
    ]
    return proc.returncode, read(calls), [line.split("\t")[0] for line in lines], read(probes)


def test_fastest_mirror_first_and_every_mirror_kept(tmp_path: Path) -> None:
    speeds = {KERNEL: "100000", ARCHIVE: "11000000", AZURE: "5000000", SECURITY: "fail"}
    code, calls, mirrors, probes = run(tmp_path, RUNNER, speeds)
    assert code == 0
    assert mirrors == [ARCHIVE, AZURE, KERNEL, SECURITY]  # runner's tags replaced by the ranking
    assert len(probes) == 4  # archive.ubuntu.com is in both lists but probed once
    assert any("install -y --no-install-recommends ffmpeg" in c for c in calls)
    assert all("Acquire::ForceIPv4=true" in c for c in calls)


def test_an_unreachable_mirror_goes_last_but_stays_as_a_fallback(tmp_path: Path) -> None:
    speeds = {KERNEL: "fail", ARCHIVE: "200000", AZURE: "fail", SECURITY: "fail"}
    code, _, mirrors, _ = run(tmp_path, RUNNER, speeds)
    assert code == 0
    assert mirrors[0] == ARCHIVE
    assert set(mirrors) == {ARCHIVE, KERNEL, AZURE, SECURITY}


def test_works_without_an_existing_mirror_file(tmp_path: Path) -> None:
    code, _, mirrors, _ = run(tmp_path, None, {KERNEL: "300", ARCHIVE: "200"})
    assert code == 0
    assert mirrors == [KERNEL, ARCHIVE]


def test_all_probes_failing_keeps_the_list_and_still_tries_apt(tmp_path: Path) -> None:
    code, calls, mirrors, _ = run(tmp_path, RUNNER, {})
    assert code == 0
    assert set(mirrors) == {ARCHIVE, KERNEL, AZURE, SECURITY}
    assert any("update" in c for c in calls)


def test_a_failed_attempt_is_retried(tmp_path: Path) -> None:
    code, calls, _, _ = run(tmp_path, "", {ARCHIVE: "1"}, apt_failures=2)
    assert code == 0
    assert sum("update" in c for c in calls) == 3


def test_three_failed_attempts_fail_the_step(tmp_path: Path) -> None:
    code, calls, _, _ = run(tmp_path, "", {ARCHIVE: "1"}, apt_failures=5)
    assert code == 1
    assert sum("update" in c for c in calls) == 3
    assert not any("install" in c for c in calls)
