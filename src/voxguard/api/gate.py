"""Limit on simultaneous analyses (decode, phone-line simulation, inference).

Load test, 18 Nov 2026 (real detector, laptop CPU, 4 GB container): throughput stops growing at
about 2.8 analyses/s from 4 at once, while memory keeps growing with each one. Four 60 s clips at
once, or a burst of 48 requests, got the API container OOM-killed. So at most ``running`` analyses
run at once and at most ``waiting`` more wait (up to ``wait_s``) for a turn. Anything beyond that
is refused at once with ``Busy``, which the API answers as 503 with ``Retry-After``. A bounded
queue also keeps threads free for ``/health`` and the lists, since every endpoint shares one
thread pool.

The work itself runs on ``running`` dedicated threads (``run``), not on the request threads. With
the slot limit alone, a burst of 48 still got the container OOM-killed: analyses landed on up to
40 different request threads, and each thread that calls torch keeps its own CPU thread team and
memory arenas. ``MALLOC_ARENA_MAX=2`` in the API image limits the arenas as well.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import TypeVar

T = TypeVar("T")

DEFAULT_RUNNING = 2
DEFAULT_WAITING = 8
WAIT_S = 45.0  # under the UI's 60 s analyze time-out, so a waiting request can still finish
RETRY_AFTER_S = 5


class Busy(Exception):
    """No analysis slot: too many running and waiting, or the wait ran out."""


class AnalysisGate:
    def __init__(self, running: int, waiting: int, wait_s: float = WAIT_S) -> None:
        if running < 1 or waiting < 0:
            raise ValueError("need running >= 1 and waiting >= 0")
        self._slots = threading.BoundedSemaphore(running)
        self._lock = threading.Lock()
        self._waiting = 0
        self.running, self.waiting, self.wait_s = running, waiting, wait_s
        self._workers = ThreadPoolExecutor(max_workers=running, thread_name_prefix="analysis")

    def run(self, fn: Callable[..., T], *args: object, wait_s: float | None = None) -> T:
        """Run ``fn(*args)`` on a dedicated analysis thread once a slot is free; its result or
        exception is passed through. Raises Busy without a slot. ``wait_s`` overrides how long
        to wait for one (a live stream waits at most one hop, then skips that window)."""
        with self.slot(wait_s):  # at most `running` hold a slot, so a worker is always free
            return self._workers.submit(fn, *args).result()

    def close(self) -> None:
        self._workers.shutdown(wait=True)

    @contextmanager
    def slot(self, wait_s: float | None = None) -> Iterator[None]:
        """Hold one analysis slot for the ``with`` block. Raises Busy without one."""
        if not self._slots.acquire(blocking=False):
            with self._lock:
                if self._waiting >= self.waiting:
                    raise Busy("queue full")
                self._waiting += 1
            try:
                got = self._slots.acquire(timeout=self.wait_s if wait_s is None else wait_s)
            finally:
                with self._lock:
                    self._waiting -= 1
            if not got:
                raise Busy("waited too long")
        try:
            yield
        finally:
            self._slots.release()
