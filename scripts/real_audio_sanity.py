"""Real-audio sanity check: score consented, labelled recordings through the public API.

Diagnostic only. A handful of clips supports no accuracy, FAR, FRR or EER claim; the report says
so. Two subcommands (see docs/real-audio-testing.md):

    serve  Start a local API on a fresh scratch database, never the demo one. Fails closed.
           make sanity-api SANITY_DB=/private/dir/sanity-2026-11-17.db
    run    Validate the labels CSV, consent, paths and files, then score each file through
           GET /health and POST /v1/analyze only, and write a JSON + Markdown report.
           make sanity SANITY_URL=http://127.0.0.1:8100 LABELS=… AUDIO=… OUT=…

Boundaries (agreed with Aarya, 16 Nov): public API only, no detector internals, no per-window
output, ground truth only from the CSV. Uploads are never retried: after a timeout the session
may or may not exist, so the outcome is recorded as unknown and the run stops. Reports hold file
names relative to the audio folder and SHA-256 hashes, never audio, absolute paths or tokens.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import ipaddress
import json
import logging
import math
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from voxguard.schemas import AnalyzeResponse, HealthResponse

logger = logging.getLogger("voxguard.sanity")

# The UI's near-threshold band (src/voxguard/ui/app.py); a test keeps the two equal.
CAUTION_FACTOR = 1.5
DEMO_PORTS = frozenset({8000, 8501})  # compose API and UI defaults
DEFAULT_SERVE_PORT = 8100
# Excluded unless named with --allow-file (the user's decision, 15-17 Nov).
DEFAULT_EXCLUDED = frozenset({"voxguard_test.wav"})
DEMO_DB_NAMES = frozenset({"voxguard.db"})  # the API's default and the compose database
REQUIRED_COLUMNS = ("file", "ground_truth", "source", "consent")
GROUND_TRUTHS = frozenset({"genuine", "spoof"})
MAX_FILE_BYTES = 100 * 1024 * 1024  # read into memory; the API's own 25 MB limit still applies
CODEC_NAME = re.compile(r"[a-z0-9_]{1,32}")
HEALTH_TIMEOUT_S = 10.0
ANALYZE_TIMEOUT_S = 300.0
MAX_RESPONSE_BYTES = 1024 * 1024
CAVEATS = (
    "Diagnostic only: a small, hand-picked set. It supports no accuracy, FAR, FRR or EER claim.",
    "Ground truth comes only from the labels CSV, never from the API's predictions.",
    "Model identity is what the public API reports (/health and each response). The checkpoint "
    "SHA-256 and Hugging Face revision cannot be checked through it.",
    "The API stores a session for every analysed upload (and an alert when flagged). Database "
    "isolation is enforced by the serve launcher and the run guards, not reported by the API.",
)


class UsageError(Exception):
    """Unsafe or invalid input. Nothing has been sent when this is raised."""


# --- shared path checks ------------------------------------------------------------------------


def inside_git_tree(path: Path) -> bool:
    """True if ``path`` (resolved) is inside any git working tree, including other worktrees."""
    p = path.resolve()
    return any((parent / ".git").exists() for parent in (p, *p.parents))


def _overlap(a: Path, b: Path) -> bool:
    a, b = a.resolve(), b.resolve()
    return a == b or a in b.parents or b in a.parents


# --- serve: a local API on a scratch database ---------------------------------------------------


def check_scratch_db(raw: str) -> Path:
    """The scratch database path, or UsageError. Fails closed on anything doubtful."""
    given = Path(raw).expanduser()
    if os.path.lexists(given):
        raise UsageError("the scratch database already exists; give a new file name")
    path = given.resolve()
    if os.path.lexists(path):
        raise UsageError("the scratch database already exists; give a new file name")
    for suffix in ("-journal", "-wal", "-shm"):
        if os.path.lexists(str(path) + suffix):
            raise UsageError("SQLite side files for this database already exist")
    if path.name in DEMO_DB_NAMES:
        raise UsageError(f"{path.name!r} is the demo/default database name; use another name")
    if path.parts[:2] == ("/", "data") or ("docker" in path.parts and "volumes" in path.parts):
        raise UsageError("the scratch database may not live in /data or a Docker volume")
    if not path.parent.is_dir():
        raise UsageError("the scratch database's folder does not exist")
    if inside_git_tree(path.parent):
        raise UsageError("the scratch database may not be inside a git working tree")
    return path


def check_bind(host: str, port: int) -> None:
    if port in DEMO_PORTS:
        raise UsageError(f"port {port} is a demo port; use another (default {DEFAULT_SERVE_PORT})")
    if not 1024 <= port <= 65535:
        raise UsageError("port must be between 1024 and 65535")
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host == "localhost"
    if not loopback:
        raise UsageError("serve binds to a loopback address only (127.0.0.1, ::1 or localhost)")


def serve_command(args: argparse.Namespace) -> tuple[list[str], dict[str, str], Path]:
    """The uvicorn command and environment for a scratch API, after every safety check."""
    db = check_scratch_db(args.db)
    check_bind(args.host, args.port)
    env = dict(os.environ)
    env["VOXGUARD_DB_URL"] = f"sqlite:///{db}"  # absolute path: four slashes in total
    env["VOXGUARD_DETECTOR"] = args.detector
    if args.detector == "real":
        repo = args.model_repo or env.get("VOXGUARD_MODEL_REPO", "").strip()
        if not repo:
            raise UsageError("the real detector needs --model-repo or VOXGUARD_MODEL_REPO")
        env["VOXGUARD_MODEL_REPO"] = repo
    if not args.online:
        env["HF_HUB_OFFLINE"] = "1"  # load from the local cache; never download here
    for name in ("VOXGUARD_THRESHOLD", "VOXGUARD_THRESHOLD_MODE", "VOXGUARD_LOW_FAR_THRESHOLD"):
        if env.get(name, "").strip():
            logger.warning("%s is set in this shell and will apply to the scratch API", name)
    cmd = [sys.executable, "-m", "uvicorn", "voxguard.api.main:create_app", "--factory"]
    cmd += ["--host", args.host, "--port", str(args.port)]
    return cmd, env, db


def cmd_serve(args: argparse.Namespace) -> int:
    cmd, env, db = serve_command(args)
    logger.info("scratch database: %s (new; delete it when done)", db)
    logger.info("scratch API on http://%s:%d (Ctrl-C to stop)", args.host, args.port)
    # Replace this process with uvicorn, so stopping it can never leave an orphaned server.
    os.execve(cmd[0], cmd, env)
    return 0  # not reached; execve only returns by raising


# --- run: validate, then score through the public API -------------------------------------------


@dataclass(frozen=True)
class Entry:
    file: str  # relative to the audio folder, as written in the CSV
    path: Path
    ground_truth: str
    source: str
    notes: str
    sha256: str
    size: int


@dataclass
class Result:
    file: str
    ground_truth: str  # from the CSV only
    source: str
    notes: str
    sha256: str
    size: int
    codec_requested: str | None
    outcome: str  # ok | http_error | invalid_response | unknown | not_sent
    http_status: int | None = None
    detail: str | None = None
    predicted_label: str | None = None  # the API's label, kept apart from ground_truth
    fake_score: float | None = None
    threshold: float | None = None
    ratio: float | None = None  # fake_score / threshold; None when unavailable
    near_threshold: bool | None = None
    flagged: bool | None = None
    latency_ms: float | None = None
    model_version: str | None = None
    session_id: str | None = None
    codec_applied: str | None = None
    codec_ms: float | None = None
    response: dict[str, Any] | None = field(default=None)  # the JSON body as returned


def check_api_url(url: str, allow_remote: bool) -> str:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise UsageError("--api-url must be an http(s) URL with a host")
    if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise UsageError("--api-url is the API's base URL only (no path or query)")
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError:
        raise UsageError("--api-url has an invalid port") from None
    if port in DEMO_PORTS:
        raise UsageError(f"port {port} is the demo API/UI; start a scratch API with `serve`")
    host = parsed.hostname
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host == "localhost"
    if not loopback and not allow_remote:
        raise UsageError("--api-url is not on this machine; pass --allow-remote if intended")
    return url.rstrip("/")


def check_dirs(audio_dir: Path, out_dir: Path) -> None:
    if not audio_dir.is_dir():
        raise UsageError("--audio-dir does not exist or is not a folder")
    if not out_dir.is_dir():
        raise UsageError("--out does not exist or is not a folder")
    for name, d in (("--audio-dir", audio_dir), ("--out", out_dir)):
        if inside_git_tree(d):
            raise UsageError(f"{name} may not be inside a git working tree (keep data private)")
    if _overlap(audio_dir, out_dir):
        raise UsageError("--audio-dir and --out must not overlap")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_labels(labels: Path, audio_dir: Path, allowed: frozenset[str]) -> list[Entry]:
    """Every row checked (columns, consent, label, path, file) before anything is sent."""
    if not labels.is_file():
        raise UsageError("--labels does not exist")
    with labels.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        columns = [c.strip() for c in reader.fieldnames or []]
        missing = [c for c in REQUIRED_COLUMNS if c not in columns]
        if missing:
            raise UsageError(f"labels CSV lacks required columns: {missing}")
        rows = [{(k or "").strip(): (v or "").strip() for k, v in row.items()} for row in reader]
    if not rows:
        raise UsageError("labels CSV has no rows")
    root = audio_dir.resolve()
    entries: list[Entry] = []
    seen: set[str] = set()
    for n, row in enumerate(rows, start=2):  # line numbers as in a spreadsheet
        name, truth = row["file"], row["ground_truth"].lower()
        if row["consent"].lower() != "yes":
            raise UsageError(f"line {n}: consent is not 'yes'; nothing was sent")
        if truth not in GROUND_TRUTHS:
            raise UsageError(f"line {n}: ground_truth must be 'genuine' or 'spoof'")
        if not row["source"]:
            raise UsageError(f"line {n}: source is empty")
        rel = Path(name)
        if not name or rel.is_absolute() or ".." in rel.parts:
            raise UsageError(f"line {n}: file must be a path inside --audio-dir")
        path = (audio_dir / rel).resolve()
        for excluded in {rel.name, path.name} & DEFAULT_EXCLUDED - allowed:
            raise UsageError(f"line {n}: {excluded} is excluded by default (--allow-file)")
        if root not in path.parents:
            raise UsageError(f"line {n}: file resolves outside --audio-dir")
        if not path.is_file():
            raise UsageError(f"line {n}: file not found")
        if path.stat().st_size > MAX_FILE_BYTES:
            raise UsageError(f"line {n}: file is larger than {MAX_FILE_BYTES >> 20} MB")
        if path in seen_paths(entries) or rel.as_posix() in seen:  # also catches "./a" vs "a"
            raise UsageError(f"line {n}: file is listed twice")
        seen.add(rel.as_posix())
        entries.append(
            Entry(
                rel.as_posix(),
                path,
                truth,
                row["source"],
                row.get("notes", ""),
                _sha256(path),
                path.stat().st_size,
            )  # fmt: skip
        )
    return entries


def seen_paths(entries: list[Entry]) -> set[Path]:
    return {e.path for e in entries}


def unlabelled_files(audio_dir: Path, entries: list[Entry]) -> list[str]:
    labelled = seen_paths(entries)
    root = audio_dir.resolve()
    return sorted(
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p.resolve() not in labelled
    )


# No proxies: the API is local (or explicitly allowed) and private audio must not take a detour.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _read_capped(resp: Any) -> bytes:
    data = resp.read(MAX_RESPONSE_BYTES + 1)
    if len(data) > MAX_RESPONSE_BYTES:
        raise ValueError("response too large")
    return data


def get_health(base: str) -> HealthResponse:
    """GET /health. Raises RuntimeError on any failure; nothing has been uploaded yet."""
    try:
        with _OPENER.open(f"{base}/health", timeout=HEALTH_TIMEOUT_S) as resp:
            return HealthResponse.model_validate_json(_read_capped(resp))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, ValidationError) as exc:
        raise RuntimeError(f"/health failed ({type(exc).__name__}); is the API running?") from None


def _multipart(filename: str, data: bytes, form: dict[str, str]) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
        for k, v in form.items()
    ]
    safe = filename.replace('"', "").replace("\r", "").replace("\n", "")
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{safe}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n".encode()
        + data
        + b"\r\n"
    )
    return b"".join(parts) + f"--{boundary}--\r\n".encode(), boundary


def _detail(body: bytes) -> str | None:
    try:
        detail = json.loads(body).get("detail")
    except (ValueError, AttributeError):
        return None
    return None if detail is None else str(detail)[:300]


def ratio_of(score: float | None, threshold: float | None) -> float | None:
    """score / threshold, or None when either is missing, non-finite, or the threshold <= 0."""
    if score is None or threshold is None:
        return None
    if not (math.isfinite(score) and math.isfinite(threshold)) or threshold <= 0:
        return None
    return score / threshold


class TransportError(Exception):
    """The upload's fate is unknown or it never reached the server; the run stops."""

    def __init__(self, outcome: str, detail: str):
        super().__init__(detail)
        self.outcome, self.detail = outcome, detail


def analyze(base: str, entry: Entry, codec: str | None, timeout: float) -> Result:
    """POST one file, once. Never retried: a retry could store a second session."""
    result = Result(entry.file, entry.ground_truth, entry.source, entry.notes, entry.sha256,
                    entry.size, codec, "unknown")  # fmt: skip
    form = {"codec": codec} if codec else {}
    body, boundary = _multipart(entry.path.name, entry.path.read_bytes(), form)
    req = urllib.request.Request(
        f"{base}/v1/analyze", data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )  # fmt: skip
    try:
        with _OPENER.open(req, timeout=timeout) as resp:
            result.http_status = resp.status
            raw = _read_capped(resp)
    except urllib.error.HTTPError as exc:
        result.outcome, result.http_status = "http_error", exc.code
        try:
            result.detail = _detail(exc.read(64 * 1024))
        except OSError:
            result.detail = None
        return result
    except TimeoutError:
        raise TransportError("unknown", "timed out: the session may or may not exist") from None
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, ConnectionRefusedError):
            raise TransportError("not_sent", "connection refused: nothing was stored") from None
        if isinstance(exc.reason, TimeoutError):
            raise TransportError("unknown", "timed out: the session may or may not exist") from None
        raise TransportError("unknown", f"connection error ({type(exc.reason).__name__})") from None
    except (OSError, ValueError) as exc:
        raise TransportError("unknown", f"transport error ({type(exc).__name__})") from None
    try:
        payload = json.loads(raw)
        parsed = AnalyzeResponse.model_validate(payload)
    except (ValueError, ValidationError):
        result.outcome = "invalid_response"  # a session was probably stored
        result.detail = "response was not a valid AnalyzeResponse"
        return result
    result.outcome, result.response = "ok", payload
    result.predicted_label, result.fake_score = parsed.label, parsed.fake_score
    result.threshold, result.flagged = parsed.threshold, parsed.flagged
    result.latency_ms, result.model_version = parsed.latency_ms, parsed.model_version
    result.session_id, result.codec_applied = parsed.session_id, parsed.codec_applied
    result.codec_ms = parsed.codec_ms
    result.ratio = ratio_of(parsed.fake_score, parsed.threshold)
    if result.ratio is not None:  # exactly the UI's rule (src/voxguard/ui/app.py)
        t, score = parsed.threshold, parsed.fake_score
        result.near_threshold = t <= score < t * CAUTION_FACTOR
    return result


def _git_commit() -> str:
    try:
        out = subprocess.run(  # noqa: S603
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent,
            capture_output=True, text=True, timeout=10, check=False,
        )  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def summarize(results: list[Result]) -> list[dict[str, Any]]:
    """Counts per condition (codec) and ground truth. Counts only, never rates."""
    rows = []
    for codec in sorted({r.codec_requested or "" for r in results}):
        for truth in ("genuine", "spoof"):
            group = [
                r for r in results
                if (r.codec_requested or "") == codec and r.ground_truth == truth
            ]  # fmt: skip
            if not group:
                continue
            ok = [r for r in group if r.outcome == "ok"]
            rows.append({
                "condition": codec or "as uploaded", "ground_truth": truth, "listed": len(group),
                "scored": len(ok), "flagged": sum(bool(r.flagged) for r in ok),
                "not_flagged": sum(not r.flagged for r in ok),
                "near_threshold": sum(bool(r.near_threshold) for r in ok),
                "not_scored": len(group) - len(ok),
            })  # fmt: skip
    return rows


def _cell(value: object) -> str:
    """Text for one Markdown table cell: no pipes or line breaks from labels or server text."""
    return " ".join(str(value).split()).replace("|", "\\|")


def _fmt(value: float | None, digits: int = 6) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_markdown(report: dict[str, Any]) -> str:
    meta, health = report["run"], report["run"]["health"]
    lines = [
        "# Real-audio sanity check (diagnostic only)", "",
        f"- Started: {meta['started_utc']} UTC; finished: {meta['finished_utc']} UTC",
        f"- API: {meta['api_url']} (detector `{health['detector']}`, "
        f"model `{health['model_version']}`)",
        f"- Tool commit: `{meta['tool_commit']}`; labels SHA-256: `{meta['labels_sha256']}`",
        f"- Conditions: {', '.join(meta['conditions'])}",
        f"- Stopped early: {meta['stopped_early'] or 'no'}", "",
        "## Caveats", "", *[f"- {c}" for c in CAVEATS], "",
        "## Counts (not rates)", "",
        "| Condition | Ground truth | Listed | Scored | Flagged | Not flagged | Near threshold "
        "| Not scored |",
        "|---|---|---|---|---|---|---|---|",
    ]  # fmt: skip
    for s in report["summary"]:
        lines.append(
            f"| {s['condition']} | {s['ground_truth']} | {s['listed']} | {s['scored']} | "
            f"{s['flagged']} | {s['not_flagged']} | {s['near_threshold']} | {s['not_scored']} |"
        )
    lines += [
        "", "## Per file", "",
        "| File | Ground truth (CSV) | Source | Condition | Outcome | Score | Threshold | "
        "Score / threshold | Predicted label (API) | Flagged | Near threshold |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]  # fmt: skip
    for r in report["results"]:
        near = "n/a" if r["near_threshold"] is None else ("yes" if r["near_threshold"] else "no")
        flagged = "n/a" if r["flagged"] is None else ("yes" if r["flagged"] else "no")
        status = r["outcome"] + (f" ({r['http_status']})" if r["http_status"] else "")
        lines.append(
            f"| {_cell(r['file'])} | {r['ground_truth']} | {_cell(r['source'])} | "
            f"{r['codec_requested'] or 'as uploaded'} | {status} | {_fmt(r['fake_score'])} | "
            f"{_fmt(r['threshold'])} | {_fmt(r['ratio'], 3)} | {r['predicted_label'] or 'n/a'} | "
            f"{flagged} | {near} |"
        )
    problems = [r for r in report["results"] if r["outcome"] != "ok"]
    if problems or report["model_version_mismatches"]:
        lines += ["", "## Problems", ""]
        lines += [f"- {_cell(r['file'])} ({r['codec_requested'] or 'as uploaded'}): "
                  f"{r['outcome']}{' ' + str(r['http_status']) if r['http_status'] else ''}"
                  f"{': ' + _cell(r['detail']) if r['detail'] else ''}"
                  for r in problems]  # fmt: skip
        lines += [
            f"- model_version differs from /health: {m}" for m in report["model_version_mismatches"]
        ]
    if report["unlabelled_files"]:
        lines += ["", "## Unlabelled files (not sent)", ""]
        lines += [f"- {_cell(name)}" for name in report["unlabelled_files"]]
    return "\n".join(lines) + "\n"


def write_reports(out_dir: Path, report: dict[str, Any]) -> tuple[Path, Path]:
    stamp = report["run"]["started_utc"].replace(":", "").replace("-", "")
    paths = (out_dir / f"sanity-{stamp}.json", out_dir / f"sanity-{stamp}.md")
    with _private_new_file(paths[0]) as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    with _private_new_file(paths[1]) as f:
        f.write(render_markdown(report))
    return paths


def _private_new_file(path: Path) -> Any:
    """Open a new file for writing, owner-only (0600) whatever the umask; never overwrite one."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.fchmod(fd, 0o600)  # exactly 0600, even under an unusual umask
    return os.fdopen(fd, "w", encoding="utf-8")


def cmd_run(args: argparse.Namespace) -> int:
    # 1. Everything is validated before the first request (including GET /health).
    base = check_api_url(args.api_url, args.allow_remote)
    audio_dir, out_dir, labels = Path(args.audio_dir), Path(args.out), Path(args.labels)
    check_dirs(audio_dir, out_dir)
    if inside_git_tree(labels):
        raise UsageError("--labels may not be inside a git working tree (keep it private)")
    entries = load_labels(labels, audio_dir, frozenset(args.allow_file))
    unlabelled = unlabelled_files(audio_dir, entries)
    for codec in args.codec:  # sent as a form field: names only, no control characters
        if not CODEC_NAME.fullmatch(codec):
            raise UsageError("--codec must be a codec name such as g711_mulaw")
    conditions: list[str | None] = [None, *dict.fromkeys(args.codec)]
    started = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S")

    # 2. Identity.
    health = get_health(base)
    if health.detector == "stub" and not args.allow_stub:
        raise RuntimeError("the API runs the stub detector (placeholder scores); see --allow-stub")
    if args.expect_model_version and health.model_version != args.expect_model_version:
        raise RuntimeError("model_version from /health differs from --expect-model-version")

    # 3. One request per file and condition; stop at the first transport failure.
    results: list[Result] = []
    stopped: str | None = None
    for entry in entries:
        for codec in conditions:
            if stopped:
                results.append(Result(entry.file, entry.ground_truth, entry.source, entry.notes,
                                      entry.sha256, entry.size, codec, "not_sent"))  # fmt: skip
                continue
            try:
                results.append(analyze(base, entry, codec, args.timeout))
            except TransportError as exc:
                results.append(Result(entry.file, entry.ground_truth, entry.source, entry.notes,
                                      entry.sha256, entry.size, codec, exc.outcome,
                                      detail=exc.detail))  # fmt: skip
                stopped = exc.detail
                logger.error("stopping: %s (no retry)", exc.detail)
    expected = health.model_version
    mismatches = sorted(
        {r.model_version for r in results if r.model_version and r.model_version != expected}
    )
    report = {
        "run": {
            "started_utc": started,
            "finished_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S"),
            "api_url": base, "health": health.model_dump(), "tool_commit": _git_commit(),
            "labels_file": labels.name, "labels_sha256": _sha256(labels),
            "conditions": [c or "as uploaded" for c in conditions],
            "stopped_early": stopped, "caution_factor": CAUTION_FACTOR, "caveats": list(CAVEATS),
        },
        "summary": summarize(results),
        "results": [asdict(r) for r in results],
        "model_version_mismatches": mismatches,
        "unlabelled_files": unlabelled,
    }  # fmt: skip
    json_path, md_path = write_reports(out_dir, report)
    logger.info("wrote %s and %s", json_path.name, md_path.name)
    return 1 if stopped or mismatches else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("serve", help="start a local API on a fresh scratch database")
    s.add_argument("--db", required=True, help="new SQLite file outside any git tree")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=DEFAULT_SERVE_PORT)
    s.add_argument("--detector", choices=["real", "stub"], default="real")
    s.add_argument("--model-repo", default="", help="default: VOXGUARD_MODEL_REPO")
    s.add_argument("--online", action="store_true", help="allow Hugging Face downloads")

    r = sub.add_parser("run", help="score labelled, consented files through the public API")
    r.add_argument("--api-url", required=True, help="the scratch API from `serve`")
    r.add_argument("--labels", required=True, help="CSV: file,ground_truth,source,consent[,notes]")
    r.add_argument("--audio-dir", required=True)
    r.add_argument("--out", required=True, help="existing private folder for the reports")
    r.add_argument("--codec", action="append", default=[], help="also score with this codec")
    r.add_argument("--allow-stub", action="store_true", help="accept the stub detector (tests)")
    r.add_argument("--expect-model-version", default="")
    r.add_argument("--allow-remote", action="store_true", help="allow a non-loopback API URL")
    r.add_argument("--allow-file", action="append", default=[], help="include an excluded name")
    r.add_argument("--timeout", type=float, default=ANALYZE_TIMEOUT_S, help="per upload, seconds")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    try:
        return cmd_serve(args) if args.command == "serve" else cmd_run(args)
    except UsageError as exc:
        logger.error("%s", exc)
        return 2
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
