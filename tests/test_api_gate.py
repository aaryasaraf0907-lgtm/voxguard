"""The limit on simultaneous analyses (api/gate.py): found necessary by the load test, where 4
long clips or a burst of 48 requests got the API container OOM-killed."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from conftest import FixedDetector, make_wav
from voxguard.api.db import SessionRow
from voxguard.api.deps import Settings
from voxguard.api.gate import AnalysisGate, Busy
from voxguard.api.main import create_app
from voxguard.inference import Prediction


def test_at_most_running_slots_are_held_at_once() -> None:
    gate, active, peak, lock = AnalysisGate(2, 10, wait_s=5), [0], [0], threading.Lock()

    def work(_: int) -> None:
        with gate.slot():
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.02)
            with lock:
                active[0] -= 1

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(work, range(8)))
    assert peak[0] == 2


def test_a_full_queue_is_refused_at_once() -> None:
    gate, release = AnalysisGate(1, 0), threading.Event()

    def hold() -> None:
        with gate.slot():
            release.wait(5)

    holder = threading.Thread(target=hold)
    holder.start()
    time.sleep(0.05)
    t0 = time.monotonic()
    with pytest.raises(Busy, match="queue full"), gate.slot():
        pass
    assert time.monotonic() - t0 < 0.5  # not after a wait
    release.set()
    holder.join()
    with gate.slot():  # the slot came back
        pass


def test_a_waiter_gives_up_after_the_wait() -> None:
    gate, release = AnalysisGate(1, 1, wait_s=0.1), threading.Event()

    def hold() -> None:
        with gate.slot():
            release.wait(5)

    holder = threading.Thread(target=hold)
    holder.start()
    time.sleep(0.05)
    with pytest.raises(Busy, match="waited too long"), gate.slot():
        pass
    release.set()
    holder.join()


def test_the_slot_is_released_when_the_work_fails() -> None:
    gate = AnalysisGate(1, 0)
    with pytest.raises(RuntimeError), gate.slot():
        raise RuntimeError("boom")
    with gate.slot():
        pass


def test_run_uses_only_the_dedicated_threads_and_passes_results_and_errors() -> None:
    """torch keeps per-thread state, so analyses must not spread over the request threads."""
    gate = AnalysisGate(2, 50, wait_s=5)
    with ThreadPoolExecutor(16) as pool:
        names = set(
            pool.map(lambda _: gate.run(lambda: threading.current_thread().name), range(64))
        )
    assert len(names) <= 2 and all(n.startswith("analysis") for n in names)
    assert gate.run(lambda a, b: a + b, 2, 3) == 5
    with pytest.raises(KeyError):
        gate.run(lambda: {}["missing"])
    gate.close()


@pytest.mark.parametrize(("running", "waiting"), [(0, 0), (1, -1)])
def test_invalid_sizes_are_refused(running: int, waiting: int) -> None:
    with pytest.raises(ValueError):
        AnalysisGate(running, waiting)


class SlowDetector(FixedDetector):
    def __init__(self, started: threading.Event, release: threading.Event) -> None:
        super().__init__(0.9)
        self.started, self.release = started, release

    def predict(self, waveform: np.ndarray, sample_rate: int) -> Prediction:
        self.started.set()
        self.release.wait(5)
        return super().predict(waveform, sample_rate)


def test_busy_api_answers_503_with_retry_after_and_stores_nothing(tmp_path) -> None:
    started, release = threading.Event(), threading.Event()
    settings = Settings(
        db_url=f"sqlite:///{tmp_path / 'g.db'}", max_running_analyses=1, max_waiting_analyses=0
    )
    with TestClient(create_app(SlowDetector(started, release), settings)) as c:

        def post():
            return c.post(
                "/v1/analyze", files={"file": ("a.wav", make_wav(0.1))}, data={"caller_id": "x"}
            )

        with ThreadPoolExecutor(1) as pool:
            first = pool.submit(post)
            assert started.wait(5)
            busy = post()
            assert c.get("/health").status_code == 200  # other endpoints still answer
            release.set()
            assert first.result().status_code == 200
        assert busy.status_code == 503 and busy.headers["Retry-After"] == "5"
        assert "Server busy" in busy.json()["detail"]
        with c.app.state.session_factory() as db:  # only the first analysis was stored
            assert db.scalar(select(func.count()).select_from(SessionRow)) == 1


def test_limits_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("VOXGUARD_MAX_RUNNING_ANALYSES", "3")
    monkeypatch.setenv("VOXGUARD_MAX_WAITING_ANALYSES", "0")
    s = Settings.from_env()
    assert (s.max_running_analyses, s.max_waiting_analyses) == (3, 0)
    monkeypatch.delenv("VOXGUARD_MAX_RUNNING_ANALYSES")
    monkeypatch.delenv("VOXGUARD_MAX_WAITING_ANALYSES")
    s = Settings.from_env()
    assert (s.max_running_analyses, s.max_waiting_analyses) == (2, 8)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("VOXGUARD_MAX_RUNNING_ANALYSES", "0"),
        ("VOXGUARD_MAX_RUNNING_ANALYSES", "two"),
        ("VOXGUARD_MAX_WAITING_ANALYSES", "-1"),
        ("VOXGUARD_MAX_WAITING_ANALYSES", "1.5"),
    ],
)
def test_invalid_limits_fail_at_startup(monkeypatch, name: str, value: str) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        Settings.from_env()
