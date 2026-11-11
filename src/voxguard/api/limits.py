"""Request body size limit, enforced before the multipart parser spools anything."""

from __future__ import annotations

from fastapi import HTTPException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class BodySizeLimitMiddleware:
    """413 for POSTs to `path` whose body exceeds `max_body_bytes`.

    Checks Content-Length up front, and also counts streamed bytes so a missing or false
    Content-Length (chunked uploads) cannot get past the limit.
    """

    def __init__(self, app: ASGIApp, max_body_bytes: int, path: str) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes
        self.path = path

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] != self.path:
            await self.app(scope, receive, send)
            return

        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_body_bytes:
            await self._reject(scope, receive, send)
            return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    # HTTPException (not a bare Exception) so FastAPI's body parsing re-raises
                    # it as a 413 instead of wrapping it in "error parsing the body" (400).
                    raise HTTPException(413, "Request body too large")
            return message

        await self.app(scope, limited_receive, send)

    @staticmethod
    async def _reject(scope: Scope, receive: Receive, send: Send) -> None:
        response = JSONResponse(
            {"detail": "Request body too large"},
            status_code=413,
            headers={"Connection": "close"},
        )
        await response(scope, receive, send)
