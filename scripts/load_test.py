"""Load test for the VoxGuard API: closed-loop concurrency levels through the public API.

    python scripts/load_test.py --api-url http://127.0.0.1:18300 --mix A --levels 1,2,4,8 \\
        --out ~/private/loadtest-1 [--container voxguard-load-api-1] [--poll]

N workers each send the next POST /v1/analyze as soon as the last one answers. Per level: two
warm-up analyses, then at least --duration seconds and --min-requests requests. Audio is
synthetic (seeded tone + noise), generated in memory and never written to disk.

Mixes:
  A  4 s, 16 kHz WAV, no codec (the CLAUDE.md section 10 clip)
  B  A with codec=g711_mulaw
  C  2-60 s, 16 or 44.1 kHz, half with a codec, about 5% M4A (when ffmpeg can make one)
  D  limits: 23 MB uploads that are valid (200), over the upload limit (413), over 60 s (422)

Checks per level (exact, from responses and the API's own lists): every status is the expected
one, no 5xx, no client time-out; each caller's blocklist hit_count equals its flagged responses
(callers are unique to the run and level); with --container, the container did not restart and
stayed healthy, and no ffmpeg process or decode directory is left over. With --poll, a background
reader polls /v1/alerts?since= like a client, and the alerts it saw are compared with the flagged
responses.

Safety: the API URL must be on this machine and not a demo port (8000/8501): run it against a
scratch API or a scratch compose project, never the demo database. The output directory must be
new and outside the git tree; it holds JSON and Markdown only (no audio).
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import shutil
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_audio_sanity import (  # noqa: E402  (scripts/ is not a package)
    UsageError,
    _multipart,
    check_api_url,
    inside_git_tree,
)

logger = logging.getLogger("load_test")

CODEC = "g711_mulaw"
CLIENT_TIMEOUT_S = 60.0  # the UI's analyze read time-out
LIST_TIMEOUT_S = 10.0  # the UI's read time-out for lists
WARMUP = 2
MAX_NO_ANSWER_IN_A_ROW = 10  # the server is gone (e.g. OOM-killed): stop the level
RECOVERY_WAIT_S = 180.0  # after a level, how long to wait for /health before the checks
SAMPLE_EVERY_S = 2.0
MIXES = ("A", "B", "C", "D")
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


# --- audio --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Clip:
    kind: str  # e.g. "wav 4s 16000Hz", used to group results
    data: bytes
    filename: str
    codec: str | None
    expect: int  # expected HTTP status


def synth(seconds: float, sr: int, channels: int = 1, seed: int = 0) -> np.ndarray:
    """Seeded speech-band tone plus noise, float32, shape (samples, channels)."""
    rng = np.random.default_rng(seed)
    n = int(round(seconds * sr))
    t = np.arange(n, dtype=np.float64) / sr
    f0 = 120 + 40 * rng.random()
    tone = 0.3 * np.sin(2 * np.pi * f0 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
    x = tone[:, None] + 0.05 * rng.standard_normal((n, channels))
    return x.astype(np.float32)


def wav_bytes(seconds: float, sr: int, channels: int = 1, seed: int = 0) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, synth(seconds, sr, channels, seed), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def m4a_bytes(seconds: float, seed: int = 0) -> bytes | None:
    """AAC in MP4 via ffmpeg (pipes only, nothing on disk); None if ffmpeg cannot make one."""
    exe = shutil.which("ffmpeg")
    if exe is None:
        return None
    cmd = [
        exe,
        "-v",
        "error",
        "-nostdin",
        "-f",
        "f32le",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-i",
        "-",
        "-c:a",
        "aac",
        "-b:a",
        "64k",
        "-movflags",
        "frag_keyframe+empty_moov",
        "-f",
        "mp4",
        "-",
    ]
    try:
        out = subprocess.run(
            cmd,
            input=synth(seconds, 16_000, seed=seed).tobytes(),
            capture_output=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout if out.returncode == 0 and out.stdout else None


def build_clips(mix: str, seed: int, max_upload_mb: float = 25.0) -> list[Clip]:
    """The clips a mix draws from, in a fixed order (workers cycle through them)."""
    if mix == "A":
        return [Clip("wav 4s 16000Hz", wav_bytes(4, 16_000, seed=seed), "a.wav", None, 200)]
    if mix == "B":
        return [
            Clip(f"wav 4s 16000Hz {CODEC}", wav_bytes(4, 16_000, seed=seed), "b.wav", CODEC, 200)
        ]
    if mix == "C":
        clips = []
        for i, (secs, sr) in enumerate(
            (s, r) for s in (2, 4, 10, 30, 60) for r in (16_000, 44_100)
        ):
            data = wav_bytes(secs, sr, seed=seed + i)
            for codec in (None, CODEC):
                kind = f"wav {secs}s {sr}Hz" + (f" {codec}" if codec else "")
                clips.append(Clip(kind, data, "c.wav", codec, 200))
        m4a = m4a_bytes(4, seed)
        if m4a is not None:  # one in 21: about 5%
            clips.append(Clip("m4a 4s", m4a, "c.m4a", None, 200))
        return clips
    if mix == "D":
        big = wav_bytes(30, 192_000, channels=2, seed=seed)  # 23 MB, valid
        limit = int(max_upload_mb * 1024 * 1024)
        over = big + b"\0" * (limit + 1 - len(big))  # one byte over the upload limit
        return [
            Clip("wav 30s 192kHz stereo 23MB", big, "d.wav", None, 200),
            Clip("over upload limit", over, "d.wav", None, 413),
            Clip(
                "wav 61s 16000Hz (too long)", wav_bytes(61, 16_000, seed=seed), "d.wav", None, 422
            ),
        ]
    raise UsageError(f"--mix must be one of {MIXES}")


# --- HTTP ---------------------------------------------------------------------------------------


@dataclass
class Result:
    level: int
    kind: str
    caller: str
    expect: int
    status: int | None = None  # None: no HTTP answer (time-out or connection error)
    error: str | None = None
    client_ms: float = 0.0
    latency_ms: float | None = None
    codec_ms: float | None = None
    flagged: bool | None = None
    busy: bool = False  # 503 with Retry-After: the API's analysis queue was full (by design)


def get_json(base: str, path: str, timeout: float = LIST_TIMEOUT_S) -> Any:
    with _OPENER.open(f"{base}{path}", timeout=timeout) as resp:
        return json.loads(resp.read())


def analyze(base: str, clip: Clip, caller: str, level: int) -> Result:
    form = {"caller_id": caller}
    if clip.codec:
        form["codec"] = clip.codec
    body, boundary = _multipart(clip.filename, clip.data, form)
    req = urllib.request.Request(
        f"{base}/v1/analyze",
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    r = Result(level, clip.kind, caller, clip.expect)
    t0 = time.perf_counter()
    try:
        with _OPENER.open(req, timeout=CLIENT_TIMEOUT_S) as resp:
            r.status, payload = resp.status, json.loads(resp.read())
        r.latency_ms, r.codec_ms = payload["latency_ms"], payload.get("codec_ms")
        r.flagged = payload["flagged"]
    except urllib.error.HTTPError as exc:
        r.status = exc.code
        r.busy = exc.code == 503 and exc.headers.get("Retry-After") is not None
        exc.close()
    except (TimeoutError, urllib.error.URLError, OSError) as exc:
        r.error = type(getattr(exc, "reason", exc)).__name__
    r.client_ms = (time.perf_counter() - t0) * 1000.0
    return r


# --- background observers -----------------------------------------------------------------------


class AlertPoller(threading.Thread):
    """Polls like a client: /v1/alerts?since=<cursor> every second, plus the two lists the UI
    shows, timing each. Collects this run's alert ids (by caller prefix)."""

    def __init__(self, base: str, prefix: str, since: str) -> None:
        super().__init__(daemon=True)
        self.base, self.prefix, self.cursor = base, prefix, since
        self.seen: set[str] = set()
        self.list_ms: dict[str, list[float]] = {"sessions": [], "blocklist": [], "alerts": []}
        self.errors = 0
        self.stop = threading.Event()

    def poll_once(self) -> None:
        for name, path in (("sessions", "/v1/sessions?limit=50"), ("blocklist", "/v1/blocklist")):
            t0 = time.perf_counter()
            try:
                get_json(self.base, path)
                self.list_ms[name].append((time.perf_counter() - t0) * 1000.0)
            except (OSError, ValueError):
                self.errors += 1
        while True:  # page until a short page: the cursor only moves forward
            t0 = time.perf_counter()
            try:
                page = get_json(self.base, f"/v1/alerts?limit=500&since={self.cursor}")
            except (OSError, ValueError):
                self.errors += 1
                return
            self.list_ms["alerts"].append((time.perf_counter() - t0) * 1000.0)
            for a in page:
                if (a.get("caller_id") or "").startswith(self.prefix):
                    self.seen.add(a["alert_id"])
            if page:
                self.cursor = urllib.request.quote(page[-1]["created_at"])
            if len(page) < 500:
                return

    def run(self) -> None:
        while not self.stop.wait(1.0):
            self.poll_once()


class ContainerSampler(threading.Thread):
    """`docker stats` every SAMPLE_EVERY_S: memory (MiB) and CPU (%) of one container."""

    def __init__(self, container: str) -> None:
        super().__init__(daemon=True)
        self.container = container
        self.samples: list[tuple[float, float]] = []
        self.stop = threading.Event()

    def run(self) -> None:
        while not self.stop.is_set():
            out = docker(
                "stats", "--no-stream", "--format", "{{.MemUsage}}|{{.CPUPerc}}", self.container
            )
            parsed = parse_stats(out)
            if parsed is not None:
                self.samples.append(parsed)
            self.stop.wait(SAMPLE_EVERY_S)


def docker(*args: str) -> str:
    try:
        out = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


_UNITS = {
    "B": 1 / 2**20,
    "KiB": 1 / 1024,
    "MiB": 1.0,
    "GiB": 1024.0,
    "kB": 1e3 / 2**20,
    "MB": 1e6 / 2**20,
    "GB": 1e9 / 2**20,
}


def parse_stats(line: str) -> tuple[float, float] | None:
    """'1.2GiB / 4GiB|350.5%' -> (1228.8, 350.5); None if unreadable."""
    try:
        mem, cpu = line.split("|")
        used = mem.split("/")[0].strip()
        unit = next(u for u in sorted(_UNITS, key=len, reverse=True) if used.endswith(u))
        return float(used[: -len(unit)]) * _UNITS[unit], float(cpu.strip().rstrip("%"))
    except (ValueError, StopIteration):
        return None


def container_state(container: str) -> dict[str, Any]:
    raw = docker(
        "inspect",
        "--format",
        "{{.RestartCount}}|{{.State.Health.Status}}|{{.State.StartedAt}}",
        container,
    )
    parts = raw.split("|") if raw else ["", "", ""]
    return {"restarts": parts[0], "health": parts[1], "started_at": parts[2]}


def container_leftovers(container: str) -> dict[str, str]:
    script = (
        "cat /proc/[0-9]*/comm 2>/dev/null | grep -c -E '^ff(mpeg|probe)$'; "
        "ls -d /tmp/voxguard-decode-* 2>/dev/null | wc -l"
    )
    out = docker("exec", container, "sh", "-c", script).split()
    return {
        "ffmpeg_processes": out[0] if out else "?",
        "decode_dirs": out[1] if len(out) > 1 else "?",
    }


# --- one level ----------------------------------------------------------------------------------


def run_level(
    base: str,
    clips: list[Clip],
    level: int,
    prefix: str,
    duration: float,
    min_requests: int,
    callers: int,
    send: Callable[[str, Clip, str, int], Result] = analyze,
) -> tuple[list[Result], float, bool]:
    """Closed loop with ``level`` workers. Returns (results, wall seconds, stopped early).

    The level stops early once MAX_NO_ANSWER_IN_A_ROW requests in a row got no HTTP answer:
    without that, a dead server turns the loop into thousands of instant connection errors."""
    results: list[Result] = []
    lock = threading.Lock()
    counter = iter(range(10**9))
    t_end = time.monotonic() + duration
    no_answer_run = [0]
    stopped = threading.Event()

    def worker(w: int) -> None:
        while not stopped.is_set():
            with lock:
                i = next(counter)
                if time.monotonic() >= t_end and i >= min_requests:
                    return
            clip = clips[i % len(clips)]
            r = send(base, clip, f"{prefix}-L{level}-c{i % callers}", level)
            with lock:
                results.append(r)
                no_answer_run[0] = no_answer_run[0] + 1 if r.status is None else 0
                if no_answer_run[0] >= MAX_NO_ANSWER_IN_A_ROW:
                    stopped.set()

    t0 = time.monotonic()
    threads = [threading.Thread(target=worker, args=(w,), daemon=True) for w in range(level)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results, time.monotonic() - t0, stopped.is_set()


def pct(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile (q in 0-100); None for no values."""
    if not values:
        return None
    s = sorted(values)
    return s[max(0, math.ceil(q / 100 * len(s)) - 1)]


def blocklist_check(results: list[Result], blocklist: list[dict[str, Any]], prefix: str) -> dict:
    """Each caller's hit_count must equal its flagged responses. Inexact if any request got no
    answer (it may or may not have been stored)."""
    expected = Counter(r.caller for r in results if r.flagged)
    actual = {
        b["caller_id"]: b["hit_count"] for b in blocklist if b["caller_id"].startswith(prefix)
    }
    mismatches = sorted(
        c for c in set(expected) | set(actual) if expected.get(c, 0) != actual.get(c, 0)
    )
    unanswered = sum(r.status is None for r in results)
    return {"exact": unanswered == 0, "mismatches": mismatches, "flagged": sum(expected.values())}


def summarize(results: list[Result], wall_s: float) -> dict[str, Any]:
    ok = [r for r in results if r.status == 200]
    lat = [r.latency_ms for r in ok if r.latency_ms is not None]
    codec = [r.codec_ms for r in ok if r.codec_ms is not None]
    client = [r.client_ms for r in results if r.status is not None]
    return {
        "requests": len(results),
        "wall_s": round(wall_s, 1),
        "rps": round(len(results) / wall_s, 2) if wall_s else None,
        "status": dict(Counter(str(r.status) if r.status else r.error for r in results)),
        "busy": sum(r.busy for r in results),
        "unexpected_status": sum(r.status != r.expect and not r.busy for r in results),
        "server_errors": sum((r.status or 0) >= 500 and not r.busy for r in results),
        "no_answer": sum(r.status is None for r in results),
        **{f"client_ms_p{q}": pct(client, q) for q in (50, 95, 99)},
        **{f"latency_ms_p{q}": pct(lat, q) for q in (50, 95, 99)},
        "codec_ms_p95": pct(codec, 95),
        "by_kind": {
            k: {
                "n": n,
                "latency_ms_p95": pct(
                    [r.latency_ms for r in ok if r.kind == k and r.latency_ms is not None], 95
                ),
            }
            for k, n in Counter(r.kind for r in results).items()
        },
    }


# --- report -------------------------------------------------------------------------------------


def _f(v: Any) -> str:
    if v is None:
        return "-"
    return f"{v:.0f}" if isinstance(v, float) else str(v)


def render_markdown(report: dict[str, Any]) -> str:
    run = report["run"]
    lines = [
        f"# Load test {run['started']} (mix {run['mix']})",
        "",
        f"API {run['api_url']}, detector `{run['health'].get('detector')}`, model "
        f"`{run['health'].get('model_version')}`, commit `{run['commit']}`. Closed loop, "
        f"{WARMUP} warm-up analyses, then at least {run['duration_s']:g} s and "
        f"{run['min_requests']} requests per level. Synthetic audio (seed {run['seed']}).",
        "",
        "| Level | Requests | Req/s | Statuses | Client p50 / p95 / p99 ms "
        "| Detector p50 / p95 / p99 ms | Codec p95 ms | Peak mem MiB | Checks |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for lv in report["levels"]:
        s = lv["summary"]
        statuses = ", ".join(f"{k}: {v}" for k, v in sorted(s["status"].items()))
        checks = "pass" if lv["passed"] else "FAIL: " + "; ".join(lv["failures"])
        lines.append(
            f"| {lv['level']} | {s['requests']} | {_f(s['rps'])} | {statuses} | "
            f"{_f(s['client_ms_p50'])} / {_f(s['client_ms_p95'])} / {_f(s['client_ms_p99'])} | "
            f"{_f(s['latency_ms_p50'])} / {_f(s['latency_ms_p95'])} / {_f(s['latency_ms_p99'])} | "
            f"{_f(s['codec_ms_p95'])} | {_f(lv.get('peak_mem_mib'))} | {checks} |"
        )
    if report.get("goals"):
        lines += ["", *(f"- {g}" for g in report["goals"])]
    return "\n".join(lines) + "\n"


def goals(mix: str, levels: list[dict[str, Any]]) -> list[str]:
    """CLAUDE.md section 10's CPU goal and the demo criterion, where this run measured them."""
    out = []
    by = {lv["level"]: lv["summary"] for lv in levels}
    if mix == "A" and 1 in by and by[1]["latency_ms_p95"] is not None:
        p95 = by[1]["latency_ms_p95"]
        out.append(
            f"Section 10 goal (4 s clip, one at a time, detector p95 < 2000 ms): "
            f"{p95:.0f} ms, {'met' if p95 < 2000 else 'not met'}."
        )
    if mix in ("A", "B") and 2 in by and by[2]["client_ms_p95"] is not None:
        p95, bad = by[2]["client_ms_p95"], by[2]["server_errors"] + by[2]["no_answer"]
        met = p95 < 5000 and bad == 0
        out.append(
            f"Demo criterion (2 users at once, client p95 < 5000 ms, no 5xx or time-out): "
            f"{p95:.0f} ms, {bad} failures, {'met' if met else 'not met'}."
        )
    return out


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    commit = out.stdout.strip() or "unknown"
    try:
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        dirty = ""
    return f"{commit}+uncommitted changes" if dirty else commit


def wait_for_api(base: str, timeout: float = RECOVERY_WAIT_S) -> bool:
    """True once /health answers (the container may be restarting after an OOM kill)."""
    t_end = time.monotonic() + timeout
    while time.monotonic() < t_end:
        try:
            get_json(base, "/health")
            return True
        except (OSError, ValueError):
            time.sleep(2)
    return False


def check_out_dir(raw: str) -> Path:
    out = Path(raw).expanduser().resolve()
    if inside_git_tree(out):
        raise UsageError("--out must be outside the git tree (reports stay private)")
    if out.exists() and any(out.iterdir()):
        raise UsageError("--out must be a new or empty directory")
    return out


def parse_levels(raw: str) -> list[int]:
    try:
        levels = [int(x) for x in raw.split(",") if x.strip()]
    except ValueError:
        raise UsageError("--levels is a comma-separated list of positive integers") from None
    if not levels or any(n < 1 or n > 256 for n in levels):
        raise UsageError("--levels must be between 1 and 256")
    return levels


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--api-url", required=True)
    p.add_argument("--mix", choices=MIXES, default="A")
    p.add_argument("--levels", default="1,2,4,8")
    p.add_argument("--duration", type=float, default=60.0, help="seconds per level (minimum)")
    p.add_argument("--min-requests", type=int, default=30)
    p.add_argument("--callers", type=int, default=10, help="caller ids per level")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-upload-mb", type=float, default=25.0, help="the API's upload limit")
    p.add_argument("--container", help="API container name, for docker stats and health")
    p.add_argument("--poll", action="store_true", help="background alert poller, like a client")
    p.add_argument("--report-only", action="store_true", help="exit 0 even if checks fail")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    try:
        base = check_api_url(args.api_url, allow_remote=False)
        levels = parse_levels(args.levels)
        out = check_out_dir(args.out)
        if args.callers < 1 or args.min_requests < 1 or args.duration < 0:
            raise UsageError("--callers and --min-requests must be >= 1, --duration >= 0")
    except UsageError as exc:
        p.error(str(exc))
    health = get_json(base, "/health")
    clips = build_clips(args.mix, args.seed, args.max_upload_mb)
    prefix = f"lt{uuid.uuid4().hex[:8]}"
    started = datetime.now(UTC)
    logger.info(
        "run %s: mix %s, %d clip kinds, detector %s",
        prefix,
        args.mix,
        len(clips),
        health.get("detector"),
    )
    for i in range(WARMUP):  # documented: the first analyses after start or idle are slow
        r = analyze(base, clips[0], f"{prefix}-warmup", 0)
        logger.info("warm-up %d: HTTP %s, %.0f ms", i + 1, r.status, r.client_ms)

    report_levels = []
    for level in levels:
        before = container_state(args.container) if args.container else None
        sampler = ContainerSampler(args.container) if args.container else None
        poller = (
            AlertPoller(
                base, f"{prefix}-L{level}-", urllib.request.quote(datetime.now(UTC).isoformat())
            )
            if args.poll
            else None
        )
        for t in (sampler, poller):
            if t is not None:
                t.start()
        results, wall, stopped_early = run_level(
            base, clips, level, prefix, args.duration, args.min_requests, args.callers
        )
        for t in (sampler, poller):
            if t is not None:
                t.stop.set()
                t.join()
        summary = summarize(results, wall)
        failures = []
        if summary["unexpected_status"]:
            failures.append(f"{summary['unexpected_status']} unexpected status")
        if summary["server_errors"] or summary["no_answer"]:
            failures.append(f"{summary['server_errors']} 5xx, {summary['no_answer']} no answer")
        if stopped_early:
            failures.append(f"server stopped answering; level stopped after {len(results)}")
        entry: dict[str, Any] = {"level": level, "summary": summary}
        reachable = wait_for_api(base)
        if not reachable:
            failures.append(f"API not answering {RECOVERY_WAIT_S:g} s after the level")
        else:
            bl = blocklist_check(results, get_json(base, "/v1/blocklist"), f"{prefix}-L{level}-")
            entry["blocklist"] = bl
            # a request without an answer may or may not have been stored: no exact check then
            if bl["exact"] and bl["mismatches"]:
                failures.append(f"blocklist off for {len(bl['mismatches'])} callers")
        if poller is not None and reachable:
            poller.poll_once()  # final drain after the last commit
            entry["alerts"] = {
                "seen_by_poller": len(poller.seen),
                "flagged": bl["flagged"],
                "poll_errors": poller.errors,
                "list_ms_p95": {k: pct(v, 95) for k, v in poller.list_ms.items()},
            }
            if bl["exact"] and len(poller.seen) != bl["flagged"]:
                failures.append(f"poller saw {len(poller.seen)} of {bl['flagged']} alerts")
        if args.container:
            after = container_state(args.container)
            entry["container"] = {
                "before": before,
                "after": after,
                "leftovers": container_leftovers(args.container),
            }
            if before and (after["restarts"], after["started_at"]) != (
                before["restarts"],
                before["started_at"],
            ):
                failures.append("container restarted")
            if after["health"] != "healthy":
                failures.append(f"container {after['health'] or 'unknown'}")
            left = entry["container"]["leftovers"]
            if left != {"ffmpeg_processes": "0", "decode_dirs": "0"}:
                failures.append(f"leftovers {left}")
            mem = [m for m, _ in sampler.samples] if sampler else []
            entry["peak_mem_mib"] = max(mem) if mem else None
            entry["mem_mib_median"] = statistics.median(mem) if mem else None
        entry["failures"], entry["passed"] = failures, not failures
        entry["results"] = [asdict(r) for r in results]
        report_levels.append(entry)
        logger.info(
            "level %d: %d requests, %.2f req/s, client p95 %s ms, %s",
            level,
            summary["requests"],
            summary["rps"] or 0,
            _f(summary["client_ms_p95"]),
            "pass" if not failures else "FAIL " + "; ".join(failures),
        )

    report = {
        "run": {
            "started": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "api_url": base,
            "mix": args.mix,
            "levels": levels,
            "duration_s": args.duration,
            "min_requests": args.min_requests,
            "seed": args.seed,
            "prefix": prefix,
            "health": health,
            "commit": _git_commit(),
            "container": args.container,
        },
        "levels": report_levels,
        "goals": goals(args.mix, report_levels),
    }
    out.mkdir(parents=True, exist_ok=True)
    stem = f"load-{args.mix}-{started.strftime('%Y%m%dT%H%M%S')}"
    (out / f"{stem}.json").write_text(json.dumps(report, indent=1))
    md = render_markdown(report)
    (out / f"{stem}.md").write_text(md)
    print(md)
    ok = all(lv["passed"] for lv in report_levels)
    return 0 if ok or args.report_only else 1


if __name__ == "__main__":
    sys.exit(main())
