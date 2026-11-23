"""Signed alert webhook: lets SMS, e-mail or a security team's systems receive LineTrue alerts
(the header names keep the original ``VoxGuard`` prefix, like the ``VOXGUARD_*`` settings).

With ``VOXGUARD_ALERT_WEBHOOK_URL`` set, each event is POSTed as JSON with
``X-VoxGuard-Event: <event>`` and ``X-VoxGuard-Signature: sha256=<hex>``, the HMAC-SHA256 of the
exact request body with ``VOXGUARD_ALERT_WEBHOOK_SECRET``. The receiver recomputes it and
rejects anything else. Events: ``call_flagged`` (a stored flagged session, upload or live call),
``live_alert`` (the live monitor's first alert in a call), ``transaction_step_up``,
``transaction_held`` and ``transaction_released``.

Delivery runs on one background thread, so no request waits for it; a failed delivery (including
a redirect, which is not followed) is logged (event name and error type only, never the payload)
and not retried. At most MAX_PENDING deliveries wait; beyond that new events are dropped and
logged. Each payload carries ``evidence_head``, the hash at the head of the evidence chain
(api/evidence.py) after the event, so the receiver holds a copy that anchors the chain.
Payloads carry scores, session and decision IDs, the amount and the caller ID when one was given;
never audio.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

TIMEOUT_S = 5.0
MAX_PENDING = 100  # deliveries queued at once; beyond that new events are dropped (and logged)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """A redirect is a failed delivery: urllib would re-send a POST as a bodiless GET."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())


def signature(secret: str, body: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


class Notifier:
    def __init__(self, url: str | None, secret: str | None) -> None:
        self.url, self.secret = url, secret
        self._pool = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="webhook") if url else None
        )
        self._pending = 0
        self._lock = threading.Lock()
        self.dropped = 0

    @property
    def enabled(self) -> bool:
        return self._pool is not None

    def send(self, event: str, payload: dict[str, Any]) -> None:
        if self._pool is None or self.url is None or self.secret is None:
            return
        body = json.dumps(
            {"event": event, "sent_at": datetime.now(UTC).isoformat(), **payload}, sort_keys=True
        ).encode()
        with self._lock:
            if self._pending >= MAX_PENDING:  # a slow receiver must not grow memory without bound
                self.dropped += 1
                logger.warning(
                    "alert webhook %s dropped: %d deliveries pending", event, MAX_PENDING
                )
                return
            self._pending += 1
        self._pool.submit(self._post, event, body, signature(self.secret, body))

    def _post(self, event: str, body: bytes, sig: str) -> None:
        assert self.url is not None
        req = urllib.request.Request(
            self.url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-VoxGuard-Event": event,
                "X-VoxGuard-Signature": sig,
            },
        )
        try:
            with _OPENER.open(req, timeout=TIMEOUT_S) as resp:
                resp.read(1024)
        except Exception as exc:  # never raised to a request; the alert is in the DB anyway
            logger.warning("alert webhook %s not delivered: %s", event, type(exc).__name__)
        finally:
            with self._lock:
                self._pending -= 1

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
