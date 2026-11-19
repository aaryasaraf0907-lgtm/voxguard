"""scripts/load_test.py: the pure parts (percentiles, checks, parsing, clips, the closed loop)
and one end-to-end run against the real API app (stub detector) on a scratch database.

Audio is synthetic and generated in memory; nothing real is used."""

import importlib.util
import json
import socket
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))  # load_test imports real_audio_sanity from there
SPEC = importlib.util.spec_from_file_location("load_test", ROOT / "scripts/load_test.py")
assert SPEC and SPEC.loader
lt = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = lt  # dataclasses look the module up while it loads
SPEC.loader.exec_module(lt)


def test_percentiles_are_nearest_rank() -> None:
    values = [float(v) for v in range(1, 101)]
    assert lt.pct(values, 50) == 50 and lt.pct(values, 95) == 95 and lt.pct(values, 99) == 99
    assert lt.pct([7.0], 95) == 7.0 and lt.pct([], 95) is None
    assert lt.pct([3.0, 1.0, 2.0], 100) == 3.0


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("1.5GiB / 4GiB|350.25%", (1536.0, 350.25)),
        ("512MiB / 7.6GiB|0.00%", (512.0, 0.0)),
        ("2048KiB / 1GiB|12%", (2.0, 12.0)),
        ("", None),
        ("garbage", None),
    ],
)
def test_docker_stats_parsing(line: str, expected) -> None:
    assert lt.parse_stats(line) == expected


def _r(caller: str, flagged: bool | None, status: int | None = 200) -> object:
    return lt.Result(1, "k", caller, 200, status=status, flagged=flagged)


def test_blocklist_check_matches_hits_to_flagged_responses() -> None:
    results = [_r("p-a", True), _r("p-a", True), _r("p-b", False), _r("p-c", True)]
    ok = [
        {"caller_id": "p-a", "hit_count": 2},
        {"caller_id": "p-c", "hit_count": 1},
        {"caller_id": "other", "hit_count": 9},
    ]
    assert lt.blocklist_check(results, ok, "p-") == {"exact": True, "mismatches": [], "flagged": 3}
    off = [{"caller_id": "p-a", "hit_count": 1}, {"caller_id": "p-b", "hit_count": 1}]
    assert lt.blocklist_check(results, off, "p-")["mismatches"] == ["p-a", "p-b", "p-c"]


def test_blocklist_check_is_inexact_when_a_request_got_no_answer() -> None:
    check = lt.blocklist_check([_r("p-a", None, status=None)], [], "p-")
    assert check["exact"] is False and check["mismatches"] == []


@pytest.mark.parametrize("raw", ["", "0", "1,x", "300", "-1"])
def test_bad_levels_are_refused(raw: str) -> None:
    with pytest.raises(lt.UsageError):
        lt.parse_levels(raw)


def test_levels_parse() -> None:
    assert lt.parse_levels("1, 2,4,48") == [1, 2, 4, 48]


def test_output_must_be_outside_the_repo_and_new(tmp_path: Path) -> None:
    with pytest.raises(lt.UsageError, match="outside the git tree"):
        lt.check_out_dir(str(ROOT / "results-load"))
    (tmp_path / "old.json").write_text("{}")
    with pytest.raises(lt.UsageError, match="new or empty"):
        lt.check_out_dir(str(tmp_path))
    assert lt.check_out_dir(str(tmp_path / "new")) == (tmp_path / "new").resolve()


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8000",
        "http://localhost:8501",
        "http://192.168.1.5:18300",
        "http://example.com:18300",
    ],
)
def test_demo_ports_and_other_machines_are_refused(url: str) -> None:
    with pytest.raises(SystemExit):
        lt.main(["--api-url", url, "--out", "/tmp/never-used"])


def test_mix_d_expects_the_limits() -> None:
    clips = lt.build_clips("D", seed=1, max_upload_mb=25.0)
    assert [c.expect for c in clips] == [200, 413, 422]
    limit = 25 * 1024 * 1024
    assert len(clips[0].data) < limit < len(clips[1].data) == limit + 1


def test_mix_c_covers_lengths_rates_and_codecs() -> None:
    clips = lt.build_clips("C", seed=1)
    wav = [c for c in clips if c.filename.endswith(".wav")]
    assert len(wav) == 20 and sum(c.codec is not None for c in wav) == 10
    assert {c.kind.split()[1] for c in wav} == {"2s", "4s", "10s", "30s", "60s"}


def test_closed_loop_keeps_level_workers_busy_and_meets_the_minimum() -> None:
    clips = lt.build_clips("A", seed=0)
    active, peak, lock = [0], [0], threading.Lock()

    def fake_send(base, clip, caller, level):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.01)
        with lock:
            active[0] -= 1
        return lt.Result(level, clip.kind, caller, clip.expect, status=200, flagged=False)

    results, wall, stopped = lt.run_level("http://x", clips, 4, "p", 0.0, 25, 3, send=fake_send)
    assert not stopped
    assert 25 <= len(results) <= 25 + 4 and peak[0] == 4 and wall > 0
    assert {r.caller for r in results} == {"p-L4-c0", "p-L4-c1", "p-L4-c2"}


def test_a_dead_server_stops_the_level_instead_of_spinning() -> None:
    """Regression: an OOM-killed API turned one level into 22,800 instant connection errors."""
    clips = lt.build_clips("A", seed=0)
    calls = [0]

    def dead(base, clip, caller, level):
        calls[0] += 1
        return lt.Result(level, clip.kind, caller, clip.expect, error="ConnectionRefusedError")

    results, _, stopped = lt.run_level("http://x", clips, 2, "p", 60.0, 30, 3, send=dead)
    assert stopped and lt.MAX_NO_ANSWER_IN_A_ROW <= calls[0] <= lt.MAX_NO_ANSWER_IN_A_ROW + 2


def test_busy_answers_are_counted_but_are_not_failures() -> None:
    results = [
        lt.Result(1, "k", "c", 200, status=200, latency_ms=5.0, flagged=False),
        lt.Result(1, "k", "c", 200, status=503, busy=True),
        lt.Result(1, "k", "c", 200, status=503),  # 503 without Retry-After: a real failure
    ]
    s = lt.summarize(results, 1.0)
    assert s["busy"] == 1 and s["unexpected_status"] == 1 and s["server_errors"] == 1
    assert s["status"] == {"200": 1, "503": 2}


def test_wait_for_api_gives_up() -> None:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]  # nothing listens here once the socket is closed
    t0 = time.monotonic()
    assert lt.wait_for_api(f"http://127.0.0.1:{port}", timeout=0.5) is False
    assert time.monotonic() - t0 < 5


def test_goals_report_section_10_and_the_demo_criterion() -> None:
    def level(n, lat95, client95, bad=0):
        return {
            "level": n,
            "summary": {
                "latency_ms_p95": lat95,
                "client_ms_p95": client95,
                "server_errors": bad,
                "no_answer": 0,
            },
        }

    lines = lt.goals("A", [level(1, 1500.0, 1700.0), level(2, 2500.0, 5200.0)])
    assert "1500 ms, met" in lines[0] and "5200 ms, 0 failures, not met" in lines[1]
    assert lt.goals("C", [level(1, 1500.0, 1700.0)]) == []


# --- end to end: the real API app (stub detector) on a scratch database ------------------------


@pytest.fixture
def stub_api(tmp_path: Path) -> Iterator[str]:
    uvicorn = pytest.importorskip("uvicorn")
    pytest.importorskip("fastapi")
    from voxguard.api.deps import Settings
    from voxguard.api.main import create_app
    from voxguard.inference import StubDetector

    app = create_app(StubDetector(threshold=0.5), Settings(db_url=f"sqlite:///{tmp_path / 's.db'}"))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(10)


def test_end_to_end_with_the_alert_poller(stub_api: str, tmp_path: Path) -> None:
    out = tmp_path / "report"
    code = lt.main(
        [
            "--api-url",
            stub_api,
            "--mix",
            "A",
            "--levels",
            "1,3",
            "--duration",
            "0",
            "--min-requests",
            "12",
            "--callers",
            "4",
            "--poll",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    (report_file,) = out.glob("load-A-*.json")
    report = json.loads(report_file.read_text())
    assert [lv["level"] for lv in report["levels"]] == [1, 3]
    for lv in report["levels"]:
        assert lv["passed"], lv["failures"]
        assert lv["summary"]["status"] == {"200": lv["summary"]["requests"]}
        assert (
            lv["blocklist"]["exact"]
            and lv["alerts"]["seen_by_poller"] == lv["blocklist"]["flagged"]
        )
    md = next(out.glob("load-A-*.md")).read_text()
    assert "| 1 | " in md and "| 3 | " in md and "stub" in md


def test_end_to_end_limits_mix_gets_413_and_422(stub_api: str, tmp_path: Path) -> None:
    code = lt.main(
        [
            "--api-url",
            stub_api,
            "--mix",
            "D",
            "--levels",
            "2",
            "--duration",
            "0",
            "--min-requests",
            "6",
            "--out",
            str(tmp_path / "d"),
        ]
    )
    report = json.loads(next((tmp_path / "d").glob("*.json")).read_text())
    (lv,) = report["levels"]
    assert code == 0, lv["failures"]
    assert lv["summary"]["status"] == {"200": 2, "413": 2, "422": 2}
