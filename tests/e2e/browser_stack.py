"""Browser end-to-end fixtures: the real API (stub detector) and the Streamlit UI as processes.

A helper module imported by test_browser.py, deliberately not a ``conftest.py``: the other tests
import helpers with ``from conftest import ...``, which a second conftest module would shadow.

Opt-in (``VOXGUARD_E2E_TESTS=1``, ``make e2e``; the skip is in the test module): needs the
``e2e`` dependency group, Chromium (``uv run playwright install chromium``) and ffmpeg.
Generated audio only; nothing leaves the temporary directory.
"""

from __future__ import annotations

import atexit
import contextlib
import io
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voxguard.api.audio import decode_audio
from voxguard.inference import StubDetector

REPO = Path(__file__).resolve().parents[2]
SR = 16_000
START_TIMEOUT_S = 60.0


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wav_bytes(seconds: float, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal(int(seconds * SR)) * 0.1).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, x, SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def stub_score(data: bytes) -> float:
    """The score the API's stub detector gives these bytes (same decoder, same hash)."""
    waveform, sr = decode_audio(data, 60.0, 25 * 1024 * 1024)
    return StubDetector(threshold=0.5).predict(waveform, sr).fake_score


@dataclass(frozen=True)
class Clips:
    near: Path  # flagged, inside the caution band
    clear: Path  # flagged, at least CAUTION_FACTOR x the threshold: no caution
    below: Path  # not flagged
    m4a: Path  # AAC in MP4 with the index at the end, as recorders write it
    too_long: Path  # over the 60 s limit
    corrupt: Path  # not audio
    threshold: float


def make_clips(tmp: Path) -> Clips:
    """Choose seeds so each clip lands where the checks need it, with the threshold set from
    the near clip's stub score (score / 1.2): deterministic, no real detector involved."""
    from voxguard.ui.app import CAUTION_FACTOR  # streamlit: imported only when the tests run

    scored = [(stub_score(wav_bytes(3.0, seed)), seed) for seed in range(200)]
    near_score, near_seed = next((s, k) for s, k in scored if 0.2 <= s <= 0.6)
    threshold = near_score / 1.2
    clear_seed = next(k for s, k in scored if s >= CAUTION_FACTOR * threshold * 1.05)
    below_seed = next(k for s, k in scored if s < threshold * 0.95)
    paths = {}
    for name, seed in (("near", near_seed), ("clear", clear_seed), ("below", below_seed)):
        paths[name] = tmp / f"{name}.wav"
        paths[name].write_bytes(wav_bytes(3.0, seed))
    paths["too_long"] = tmp / "too_long.wav"
    paths["too_long"].write_bytes(wav_bytes(61.0, 1))
    paths["corrupt"] = tmp / "corrupt.wav"
    paths["corrupt"].write_bytes(b"RIFF\x00\x00\x00\x00not really audio" * 8)
    paths["m4a"] = tmp / "tone.m4a"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i",
         "sine=frequency=440:duration=3:sample_rate=16000", "-c:a", "aac", "-b:a", "64k",
         str(paths["m4a"])],
        check=True, timeout=60,
    )  # fmt: skip
    return Clips(threshold=threshold, **paths)


def log_tail(path: Path, lines: int = 40) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "(no log)"
    return "\n".join(text[-lines:]) or "(empty log)"


def spawn(cmd: list[str], env: dict[str, str], log: Path) -> subprocess.Popen[bytes]:
    """Start a server in its own process group (so stop() also reaches any children), with
    stdout and stderr in a log file for diagnostics."""
    with log.open("ab") as out:
        return subprocess.Popen(
            cmd, cwd=REPO, env=env, stdin=subprocess.DEVNULL, stdout=out,
            stderr=subprocess.STDOUT, start_new_session=True,
        )  # fmt: skip


def stop(proc: subprocess.Popen[bytes] | None) -> None:
    """Terminate the process group, then kill it if it is still there after 10 s; always reaped."""
    if proc is None or proc.poll() is not None:
        return
    for sig, wait_s in ((signal.SIGTERM, 10), (signal.SIGKILL, 10)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            break
        try:
            proc.wait(timeout=wait_s)
            break
        except subprocess.TimeoutExpired:
            continue


def wait_ready(
    url: str, proc: subprocess.Popen[bytes], log: Path, check: Callable[[bytes], bool]
) -> None:
    """Poll ``url`` until ``check(body)`` holds. Stops the process and fails with its log tail
    if it exits, answers wrongly (another server on the port) or is not ready in time."""
    deadline = time.monotonic() + START_TIMEOUT_S
    problem = f"not ready within {START_TIMEOUT_S:.0f} s"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            problem = f"exited with code {proc.returncode}"
            break
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                body = resp.read()
            if check(body):
                return
            problem = f"answered, but not as the server this test started: {body[:200]!r}"
        except OSError:
            pass
        time.sleep(0.3)
    stop(proc)
    raise RuntimeError(f"{url}: {problem}\n--- {log.name} (last lines) ---\n{log_tail(log)}")


@dataclass
class Stack:
    api_url: str
    ui_url: str
    clips: Clips
    api_env: dict[str, str]
    api_port: int
    logs: Path
    api: subprocess.Popen[bytes] | None = None

    def start_api(self) -> None:
        log = self.logs / "api.log"
        self.api = spawn(
            [sys.executable, "-m", "uvicorn", "voxguard.api.main:create_app", "--factory",
             "--host", "127.0.0.1", "--port", str(self.api_port)],
            self.api_env, log,
        )  # fmt: skip

        def is_ours(body: bytes) -> bool:
            health = json.loads(body)
            return health.get("detector") == "stub" and health.get("status") == "ok"

        wait_ready(f"{self.api_url}/health", self.api, log, is_ours)

    def stop_api(self) -> None:
        stop(self.api)


def artifacts_dir(tmp: Path) -> Path:
    """Where failure screenshots and server logs go: VOXGUARD_E2E_ARTIFACTS (CI uploads it on
    failure) or the session's temporary directory."""
    path = Path(os.environ.get("VOXGUARD_E2E_ARTIFACTS") or tmp / "artifacts")
    path.mkdir(parents=True, exist_ok=True)
    return path


E2E_POLICY = {
    "unknown": ["step_up", "step_up"],
    "low": ["allow", "step_up"],
    "caution": ["step_up", "hold"],
    "high": ["hold", "hold"],
}


@pytest.fixture(scope="session")
def stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Stack]:
    tmp = tmp_path_factory.mktemp("e2e")
    logs = artifacts_dir(tmp)
    clips = make_clips(tmp)
    api_port, ui_port = free_port(), free_port()
    base = {k: v for k, v in os.environ.items() if not k.startswith("VOXGUARD_")}
    api_env = {
        **base,
        "VOXGUARD_DETECTOR": "stub",
        "VOXGUARD_DB_URL": f"sqlite:///{tmp / 'e2e.db'}",
        "VOXGUARD_SIMULATE_CODEC": "g711_mulaw",
        "VOXGUARD_THRESHOLD": repr(clips.threshold),
        # a high-value transfer needs at least one verification step at every risk tier, so the
        # browser test of the transfer panel always exercises the verification buttons
        "VOXGUARD_TRANSACTION_POLICY": json.dumps(E2E_POLICY),
    }
    s = Stack(
        api_url=f"http://127.0.0.1:{api_port}",
        ui_url=f"http://127.0.0.1:{ui_port}",
        clips=clips,
        api_env=api_env,
        api_port=api_port,
        logs=logs,
    )
    ui: subprocess.Popen[bytes] | None = None
    try:
        s.start_api()
        ui_log = logs / "ui.log"
        ui = spawn(
            [sys.executable, "-m", "streamlit", "run", "src/voxguard/ui/app.py",
             "--server.port", str(ui_port), "--server.address", "127.0.0.1",
             "--server.headless", "true", "--browser.gatherUsageStats", "false"],
            {**base, "VOXGUARD_API_URL": s.api_url}, ui_log,
        )  # fmt: skip
        wait_ready(f"{s.ui_url}/_stcore/health", ui, ui_log, lambda body: body.strip() == b"ok")
        yield s
    finally:
        stop(ui)
        s.stop_api()


_PLAYWRIGHT: list[object] = []  # one sync Playwright per process, shared by every test module


def playwright() -> object:
    """The process's Playwright instance, started on first use and stopped at exit. A session
    fixture imported into two test modules runs once per module, and one process cannot run two
    sync Playwright instances, so the instance itself must be shared."""
    if not _PLAYWRIGHT:
        from playwright.sync_api import sync_playwright  # e2e group only; tests skip without it

        _PLAYWRIGHT.append(sync_playwright().start())
        atexit.register(_PLAYWRIGHT[0].stop)  # type: ignore[attr-defined]
    return _PLAYWRIGHT[0]


@pytest.fixture(scope="session")
def browser() -> Iterator[object]:
    b = playwright().chromium.launch()  # type: ignore[attr-defined]
    try:
        yield b
    finally:
        b.close()


@pytest.fixture
def page(browser: object, stack: Stack, request: pytest.FixtureRequest) -> Iterator[object]:
    context = browser.new_context(viewport={"width": 1400, "height": 1100})  # type: ignore[attr-defined]
    pg = context.new_page()
    pg.set_default_timeout(30_000)
    failed_before = request.session.testsfailed
    try:
        yield pg
    finally:
        # The call phase is reported before teardown, so a failure of this test is counted here.
        if request.session.testsfailed > failed_before:  # keep evidence of what the page showed
            name = re.sub(r"[^A-Za-z0-9_.-]+", "_", request.node.name)
            # a crashed page must not hide the test's own failure
            with contextlib.suppress(Exception):
                pg.screenshot(path=str(stack.logs / f"{name}.png"), full_page=True)
        context.close()
