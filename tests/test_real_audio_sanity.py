"""scripts/real_audio_sanity.py: validation before any request, the scratch-API launcher's
safety checks, public-API-only scoring, no retries, and diagnostic-only reports.

Audio is generated here; nothing real is used. HTTP behaviour runs against a local test server
(and, end to end, the real FastAPI app with the stub detector) on 127.0.0.1.
"""

import argparse
import csv
import importlib.util
import io
import json
import os
import socket
import sys
import threading
import time
import urllib.error
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "real_audio_sanity", ROOT / "scripts/real_audio_sanity.py"
)
assert SPEC and SPEC.loader
sanity = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sanity  # dataclasses look the module up while it loads
SPEC.loader.exec_module(sanity)

HEALTH = {"status": "ok", "detector": "real", "model_version": "ssl:repo:epoch11"}


def _wav(seconds: float = 0.5, seed: int = 0) -> bytes:
    x = (np.random.default_rng(seed).standard_normal(int(seconds * 16_000)) * 0.1).astype("f4")
    buf = io.BytesIO()
    sf.write(buf, x, 16_000, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _response(score: float = 0.02, threshold: float = 0.0127, **kw) -> dict:
    return {
        "session_id": "s", "label": "fake" if score >= threshold else "real",
        "fake_score": score, "threshold": threshold, "latency_ms": 5.0,
        "model_version": HEALTH["model_version"], "flagged": score >= threshold,
        "codec_applied": None, "codec_ms": None, **kw,
    }  # fmt: skip


class FakeApi:
    """A local HTTP server standing in for the API. ``analyze`` is a list of behaviours, used in
    order: a dict (200 JSON), (status, dict) for an error, b"raw" for a raw 200 body, or a float
    to sleep before answering."""

    def __init__(self) -> None:
        self.health: dict = dict(HEALTH)
        self.analyze: list = []
        self.requests: list[tuple[str, str, bytes]] = []
        api = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # quiet
                pass

            def _send(self, status: int, body: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                api.requests.append(("GET", self.path, b""))
                self._send(200, json.dumps(api.health).encode())

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                api.requests.append(("POST", self.path, body))
                behaviour = api.analyze.pop(0) if api.analyze else _response()
                if isinstance(behaviour, float):
                    time.sleep(behaviour)
                    behaviour = _response()
                if isinstance(behaviour, bytes):
                    self._send(200, behaviour)
                elif isinstance(behaviour, tuple):
                    self._send(behaviour[0], json.dumps(behaviour[1]).encode())
                else:
                    self._send(200, json.dumps(behaviour).encode())

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def posts(self) -> list[bytes]:
        return [body for method, _, body in self.requests if method == "POST"]


@pytest.fixture
def api() -> Iterator[FakeApi]:
    server = FakeApi()
    yield server
    server.server.shutdown()


@pytest.fixture
def tree(tmp_path: Path) -> dict[str, Path]:
    audio, out = tmp_path / "audio", tmp_path / "out"
    audio.mkdir()
    out.mkdir()
    (audio / "a.wav").write_bytes(_wav(seed=1))
    (audio / "b.wav").write_bytes(_wav(seed=2))
    labels = tmp_path / "labels.csv"
    _labels(labels, [
        {"file": "a.wav", "ground_truth": "genuine", "source": "speaker A", "consent": "yes"},
        {"file": "b.wav", "ground_truth": "spoof", "source": "TTS", "consent": "yes"},
    ])  # fmt: skip
    return {"audio": audio, "out": out, "labels": labels, "root": tmp_path}


def _labels(path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
    columns = columns or ["file", "ground_truth", "source", "consent", "notes"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _run(api: FakeApi, tree: dict[str, Path], *extra: str, url: str | None = None) -> int:
    return sanity.main([
        "run", "--api-url", url or api.url, "--labels", str(tree["labels"]),
        "--audio-dir", str(tree["audio"]), "--out", str(tree["out"]), *extra,
    ])  # fmt: skip


def _report(tree: dict[str, Path]) -> dict:
    (path,) = tree["out"].glob("*.json")
    return json.loads(path.read_text())


# --- serve: the scratch-API launcher fails closed --------------------------------------------


def _serve_args(db: str, **kw) -> argparse.Namespace:
    base = {"db": db, "host": "127.0.0.1", "port": 8100, "detector": "real",
            "model_repo": "voxguard-team/x", "online": False}  # fmt: skip
    return argparse.Namespace(**{**base, **kw})


def test_serve_builds_an_isolated_scratch_api(tmp_path: Path) -> None:
    cmd, env, db = sanity.serve_command(_serve_args(str(tmp_path / "sanity-1.db")))
    assert db == (tmp_path / "sanity-1.db").resolve() and db.is_absolute()
    assert env["VOXGUARD_DB_URL"] == f"sqlite:///{db}"  # sqlite:////abs/path
    assert env["VOXGUARD_DETECTOR"] == "real" and env["VOXGUARD_MODEL_REPO"] == "voxguard-team/x"
    assert env["HF_HUB_OFFLINE"] == "1"  # no downloads from the launcher
    assert cmd[1:5] == ["-m", "uvicorn", "voxguard.api.main:create_app", "--factory"]
    assert cmd[-4:] == ["--host", "127.0.0.1", "--port", "8100"]
    assert not db.exists()  # the checks never create it


def _git_dir(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    return repo


@pytest.mark.parametrize(
    "case", ["exists", "dangling_symlink", "side_file", "demo_name", "data_dir",
             "docker_volume", "no_parent", "in_git"],
)  # fmt: skip
def test_serve_refuses_unsafe_databases(tmp_path: Path, case: str) -> None:
    db = tmp_path / "sanity.db"
    if case == "exists":
        db.write_bytes(b"")
    elif case == "dangling_symlink":
        db.symlink_to(tmp_path / "nowhere.db")
    elif case == "side_file":
        (tmp_path / "sanity.db-wal").write_bytes(b"")
    elif case == "demo_name":
        db = tmp_path / "voxguard.db"
    elif case == "data_dir":
        db = Path("/data/sanity.db")
    elif case == "docker_volume":
        db = Path("/var/lib/docker/volumes/voxguard_voxguard-data/_data/sanity.db")
    elif case == "no_parent":
        db = tmp_path / "missing" / "sanity.db"
    elif case == "in_git":
        db = _git_dir(tmp_path) / "sanity.db"
    with pytest.raises(sanity.UsageError):
        sanity.serve_command(_serve_args(str(db)))


@pytest.mark.parametrize(
    ("host", "port"), [("127.0.0.1", 8000), ("127.0.0.1", 8501), ("127.0.0.1", 80),
                       ("0.0.0.0", 8100), ("192.168.1.5", 8100)],
)  # fmt: skip
def test_serve_refuses_demo_ports_and_public_binds(tmp_path: Path, host: str, port: int) -> None:
    with pytest.raises(sanity.UsageError):
        sanity.serve_command(_serve_args(str(tmp_path / "s.db"), host=host, port=port))


def test_serve_real_detector_needs_a_repo(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("VOXGUARD_MODEL_REPO", raising=False)
    with pytest.raises(sanity.UsageError):
        sanity.serve_command(_serve_args(str(tmp_path / "s.db"), model_repo=""))


def test_serve_main_launches_only_when_safe(tmp_path: Path, monkeypatch) -> None:
    launched: list[tuple[str, list[str], dict]] = []

    def fake_execve(path, argv, env):  # serve replaces itself with uvicorn
        launched.append((path, argv, env))
        return None

    monkeypatch.setattr(sanity.os, "execve", fake_execve)
    (tmp_path / "taken.db").write_bytes(b"")
    assert sanity.main(["serve", "--db", str(tmp_path / "taken.db"), "--detector", "stub"]) == 2
    assert launched == []  # fails closed: nothing started
    assert sanity.main(["serve", "--db", str(tmp_path / "new.db"), "--detector", "stub"]) == 0
    ((path, argv, env),) = launched
    assert path == argv[0] and "uvicorn" in argv
    assert env["VOXGUARD_DB_URL"] == f"sqlite:///{(tmp_path / 'new.db').resolve()}"


# --- run: everything is validated before the first request --------------------------------------


@pytest.mark.parametrize(
    "url", ["http://127.0.0.1:8000", "http://localhost:8501", "ftp://127.0.0.1:8100",
            "http://127.0.0.1:8100/v1", "http://example.com:8100", "http://10.0.0.5:8100"],
)  # fmt: skip
def test_run_refuses_unsafe_urls_before_any_request(api, tree, url: str) -> None:
    assert _run(api, tree, url=url) == 2
    assert api.requests == []


def test_run_allows_a_remote_url_only_when_asked() -> None:
    with pytest.raises(sanity.UsageError):
        sanity.check_api_url("http://10.0.0.5:8100", allow_remote=False)
    assert (
        sanity.check_api_url("http://10.0.0.5:8100/", allow_remote=True) == "http://10.0.0.5:8100"
    )


@pytest.mark.parametrize("which", ["audio", "out", "labels"])
def test_run_refuses_paths_inside_git(api, tree, which: str) -> None:
    repo = _git_dir(tree["root"])
    moved = repo / tree[which].name
    tree[which].rename(moved)
    tree[which] = moved
    assert _run(api, tree) == 2
    assert api.requests == []


def test_run_refuses_overlapping_folders(api, tree) -> None:
    nested = tree["audio"] / "reports"
    nested.mkdir()
    tree["out"] = nested
    assert _run(api, tree) == 2
    assert api.requests == []


ROW = {"file": "a.wav", "ground_truth": "genuine", "source": "speaker A", "consent": "yes"}


@pytest.mark.parametrize(
    "rows",
    [
        [{**ROW, "consent": "no"}],
        [{**ROW, "consent": ""}],
        [ROW, {**ROW, "file": "b.wav", "consent": "maybe"}],  # one bad row aborts everything
        [{**ROW, "ground_truth": "real"}],  # not a ground-truth value
        [{**ROW, "source": ""}],
        [{**ROW, "file": ""}],
        [{**ROW, "file": "/etc/passwd"}],
        [{**ROW, "file": "../labels.csv"}],
        [{**ROW, "file": "missing.wav"}],
        [ROW, {**ROW, "file": "./a.wav"}],  # the same file twice
        [],
    ],
)
def test_run_validates_every_row_before_any_request(api, tree, rows: list[dict]) -> None:
    _labels(tree["labels"], rows)
    assert _run(api, tree) == 2
    assert api.requests == []
    assert list(tree["out"].iterdir()) == []


def test_run_refuses_missing_columns(api, tree) -> None:
    _labels(tree["labels"], [ROW], columns=["file", "ground_truth", "source"])
    assert _run(api, tree) == 2 and api.requests == []


def test_run_refuses_a_symlink_out_of_the_audio_folder(api, tree) -> None:
    outside = tree["root"] / "outside.wav"
    outside.write_bytes(_wav())
    (tree["audio"] / "link.wav").symlink_to(outside)
    _labels(tree["labels"], [{**ROW, "file": "link.wav"}])
    assert _run(api, tree) == 2 and api.requests == []


def test_excluded_file_needs_allow_file(api, tree) -> None:
    (tree["audio"] / "voxguard_test.wav").write_bytes(_wav())
    _labels(tree["labels"], [{**ROW, "file": "voxguard_test.wav"}])
    assert _run(api, tree, "--allow-stub") == 2 and api.requests == []
    assert _run(api, tree, "--allow-file", "voxguard_test.wav") == 0
    assert len(api.posts()) == 1


def test_excluded_file_behind_a_symlink_is_still_excluded(api, tree) -> None:
    (tree["audio"] / "voxguard_test.wav").write_bytes(_wav())
    (tree["audio"] / "innocent.wav").symlink_to(tree["audio"] / "voxguard_test.wav")
    _labels(tree["labels"], [{**ROW, "file": "innocent.wav"}])
    assert _run(api, tree) == 2 and api.requests == []


# --- run: identity, results as returned, no retries ---------------------------------------------


def test_run_scores_each_file_once_and_keeps_ground_truth_separate(api, tree) -> None:
    api.analyze = [_response(0.02), _response(0.005)]  # genuine flagged, spoof missed
    assert _run(api, tree) == 0
    assert [m for m, *_ in api.requests] == ["GET", "POST", "POST"]  # /health first
    report = _report(tree)
    a, b = report["results"]
    assert (a["file"], a["ground_truth"], a["predicted_label"]) == ("a.wav", "genuine", "fake")
    assert (b["file"], b["ground_truth"], b["predicted_label"]) == ("b.wav", "spoof", "real")
    assert a["response"] == _response(0.02)  # recorded exactly as returned
    assert a["ratio"] == pytest.approx(0.02 / 0.0127) and a["near_threshold"] is False
    assert b["near_threshold"] is False  # under the threshold: never a caution
    counts = {(s["ground_truth"]): s for s in report["summary"]}
    assert counts["genuine"]["flagged"] == 1 and counts["spoof"]["not_flagged"] == 1
    assert report["run"]["health"] == HEALTH and report["run"]["labels_file"] == "labels.csv"


def test_near_threshold_matches_the_ui_band(api, tree) -> None:
    api.analyze = [_response(0.0127), _response(0.0127 * 1.5)]
    assert _run(api, tree) == 0
    a, b = _report(tree)["results"]
    assert a["near_threshold"] is True and b["near_threshold"] is False


def test_zero_threshold_reports_the_ratio_as_unavailable(api, tree) -> None:
    api.analyze = [_response(0.3, threshold=0.0), _response(0.3, threshold=0.0)]
    assert _run(api, tree) == 0
    report = _report(tree)
    assert all(r["ratio"] is None and r["near_threshold"] is None for r in report["results"])
    assert "| n/a |" in next(tree["out"].glob("*.md")).read_text()


@pytest.mark.parametrize(
    ("score", "threshold", "expected"),
    [(0.6, 0.5, 1.2), (0.5, 0.0, None), (None, 0.5, None), (0.5, None, None),
     (float("nan"), 0.5, None), (0.5, float("inf"), None), (0.5, -0.1, None)],
)  # fmt: skip
def test_ratio_of(score, threshold, expected) -> None:
    got = sanity.ratio_of(score, threshold)
    assert got == (pytest.approx(expected) if expected is not None else None)


def test_stub_detector_is_refused_unless_allowed(api, tree) -> None:
    api.health = {**HEALTH, "detector": "stub"}
    assert _run(api, tree) == 1
    assert api.posts() == []  # only /health was asked
    assert _run(api, tree, "--allow-stub") == 0


def test_expected_model_version_mismatch_aborts(api, tree) -> None:
    assert _run(api, tree, "--expect-model-version", "other:epoch1") == 1
    assert api.posts() == []


def test_response_model_version_mismatch_is_reported(api, tree) -> None:
    api.analyze = [_response(model_version="other:epoch2"), _response()]
    assert _run(api, tree) == 1
    report = _report(tree)
    assert report["model_version_mismatches"] == ["other:epoch2"]
    assert "model_version differs" in next(tree["out"].glob("*.md")).read_text()


@pytest.mark.parametrize("status", [413, 415, 422, 500, 503])
def test_http_errors_are_recorded_and_the_run_continues(api, tree, status: int) -> None:
    api.analyze = [(status, {"detail": "Audio is longer than 60 s"}), _response()]
    assert _run(api, tree) == 0
    a, b = _report(tree)["results"]
    assert (a["outcome"], a["http_status"], a["detail"]) == (
        "http_error",
        status,
        "Audio is longer than 60 s",
    )
    assert a["fake_score"] is None and b["outcome"] == "ok"
    assert len(api.posts()) == 2  # nothing retried


@pytest.mark.parametrize(
    "body", [b"not json", b'{"label": "fake"}', json.dumps(_response(threshold=2.0)).encode()]
)
def test_malformed_responses_are_recorded_not_trusted(api, tree, body: bytes) -> None:
    api.analyze = [body, _response()]
    assert _run(api, tree) == 0
    a = _report(tree)["results"][0]
    assert a["outcome"] == "invalid_response" and a["fake_score"] is None
    assert len(api.posts()) == 2


def test_timeout_is_unknown_never_retried_and_stops_the_run(api, tree) -> None:
    api.analyze = [3.0]  # the first upload takes longer than --timeout
    assert _run(api, tree, "--timeout", "0.5") == 1
    a, b = _report(tree)["results"]
    assert a["outcome"] == "unknown" and "may or may not exist" in a["detail"]
    assert b["outcome"] == "not_sent"
    time.sleep(3)  # let the slow request finish, then count what reached the server
    assert len(api.posts()) == 1  # exactly one attempt: no retry, nothing after the stop


def test_connection_refused_is_not_sent_and_stops_the_run(api, tree, monkeypatch) -> None:
    real_open = sanity._OPENER.open

    def opener(req, timeout):
        if getattr(req, "method", "GET") == "POST":
            raise urllib.error.URLError(ConnectionRefusedError())
        return real_open(req, timeout=timeout)

    monkeypatch.setattr(sanity._OPENER, "open", opener)
    assert _run(api, tree) == 1
    a, b = _report(tree)["results"]
    assert a["outcome"] == "not_sent" and "nothing was stored" in a["detail"]
    assert b["outcome"] == "not_sent"


def test_health_failure_aborts_before_any_upload(tree) -> None:
    with socket.socket() as s:  # a port with nothing listening
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    assert sanity.main([
        "run", "--api-url", f"http://127.0.0.1:{port}", "--labels", str(tree["labels"]),
        "--audio-dir", str(tree["audio"]), "--out", str(tree["out"]),
    ]) == 1  # fmt: skip
    assert list(tree["out"].iterdir()) == []


def test_codec_option_sends_the_field_and_records_it(api, tree) -> None:
    api.analyze = [_response(), _response(codec_applied="g711_mulaw", codec_ms=3.0)] * 2
    assert _run(api, tree, "--codec", "g711_mulaw") == 0
    posts = api.posts()
    assert len(posts) == 4 and [b'name="codec"' in p for p in posts] == [False, True] * 2
    results = _report(tree)["results"]
    assert [r["codec_requested"] for r in results] == [None, "g711_mulaw"] * 2
    assert results[1]["codec_applied"] == "g711_mulaw"
    assert {s["condition"] for s in _report(tree)["summary"]} == {"as uploaded", "g711_mulaw"}


def test_codec_is_off_by_default(api, tree) -> None:
    assert _run(api, tree) == 0
    assert all(b'name="codec"' not in p for p in api.posts())


def test_unlabelled_files_are_listed_and_never_sent(api, tree) -> None:
    (tree["audio"] / "extra.wav").write_bytes(_wav(seed=9))
    assert _run(api, tree) == 0
    assert _report(tree)["unlabelled_files"] == ["extra.wav"]
    assert len(api.posts()) == 2


# --- reports: private, diagnostic, no audio -----------------------------------------------------


def test_reports_hold_no_audio_paths_or_rates(api, tree) -> None:
    assert _run(api, tree) == 0
    files = sorted(p.suffix for p in tree["out"].iterdir())
    assert files == [".json", ".md"]  # only the two reports; no audio written
    text = "".join(p.read_text() for p in tree["out"].iterdir())
    assert str(tree["root"]) not in text  # no absolute paths (they name the user)
    assert "RIFF" not in text and len(text) < 20_000
    md = next(tree["out"].glob("*.md")).read_text()
    assert "Diagnostic only" in md and "no accuracy, FAR, FRR or EER claim" in md
    assert "%" not in md  # counts, never rates


def test_reports_are_never_overwritten(tree) -> None:
    report = {"run": {"started_utc": "2026-11-17T10:00:00"}, "results": [], "summary": [],
              "model_version_mismatches": [], "unlabelled_files": []}  # fmt: skip
    report["run"] |= {"finished_utc": "x", "api_url": "u", "health": HEALTH, "tool_commit": "c",
                      "labels_sha256": "h", "conditions": [], "stopped_early": None}  # fmt: skip
    sanity.write_reports(tree["out"], report)
    with pytest.raises(FileExistsError):
        sanity.write_reports(tree["out"], report)


def test_reports_are_owner_only_whatever_the_umask(tree) -> None:
    report = {"run": {"started_utc": "2026-11-17T11:00:00"}, "results": [], "summary": [],
              "model_version_mismatches": [], "unlabelled_files": []}  # fmt: skip
    report["run"] |= {"finished_utc": "x", "api_url": "u", "health": HEALTH, "tool_commit": "c",
                      "labels_sha256": "h", "conditions": [], "stopped_early": None}  # fmt: skip
    old = os.umask(0o022)  # the usual default, which gave 0644 in the first private run
    try:
        paths = sanity.write_reports(tree["out"], report)
    finally:
        os.umask(old)
    assert [p.stat().st_mode & 0o777 for p in paths] == [0o600, 0o600]


def test_caution_factor_matches_the_ui() -> None:
    ui_app = pytest.importorskip("voxguard.ui.app")
    if not hasattr(ui_app, "CAUTION_FACTOR"):
        pytest.skip("the UI's caution band is not on this branch yet (#77)")
    assert ui_app.CAUTION_FACTOR == sanity.CAUTION_FACTOR


# --- end to end: the real API app (stub detector) on a scratch database ------------------------


def test_end_to_end_against_the_real_api(tree, tmp_path: Path) -> None:
    uvicorn = pytest.importorskip("uvicorn")
    pytest.importorskip("fastapi")
    from sqlalchemy import create_engine, text

    from voxguard.api.deps import Settings
    from voxguard.api.main import create_app
    from voxguard.inference import StubDetector

    db = tmp_path / "scratch.db"
    app = create_app(StubDetector(threshold=0.5), Settings(db_url=f"sqlite:///{db}"))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.05)
        code = sanity.main([
            "run", "--api-url", f"http://127.0.0.1:{port}", "--labels", str(tree["labels"]),
            "--audio-dir", str(tree["audio"]), "--out", str(tree["out"]), "--allow-stub",
        ])  # fmt: skip
    finally:
        server.should_exit = True
        thread.join(10)
    assert code == 0
    results = _report(tree)["results"]
    assert [r["outcome"] for r in results] == ["ok", "ok"]
    assert [r["ground_truth"] for r in results] == ["genuine", "spoof"]
    with create_engine(f"sqlite:///{db}").connect() as conn:  # stored in the scratch DB only
        assert conn.execute(text("select count(*) from sessions")).scalar() == 2


# --- hardening ---------------------------------------------------------------------------------


@pytest.mark.parametrize("codec", ["g711\r\nX-Evil: 1", "Opus", "a b", "", "x" * 40])
def test_codec_names_are_validated_before_any_request(api, tree, codec: str) -> None:
    assert _run(api, tree, "--codec", codec) == 2
    assert api.requests == []


def test_oversized_files_are_refused_before_any_request(api, tree, monkeypatch) -> None:
    monkeypatch.setattr(sanity, "MAX_FILE_BYTES", 100)
    assert _run(api, tree) == 2 and api.requests == []


def test_markdown_cells_are_escaped(api, tree) -> None:
    _labels(tree["labels"], [{**ROW, "source": "mic | left\nchannel"}])
    api.analyze = [(415, {"detail": "bad | format\nline"})]
    assert _run(api, tree) == 0
    md = next(tree["out"].glob("*.md")).read_text()
    assert "mic \\| left channel" in md and "bad \\| format line" in md
    row = next(line for line in md.splitlines() if line.startswith("| a.wav"))
    assert row.count(" | ") == 10  # still eleven cells
