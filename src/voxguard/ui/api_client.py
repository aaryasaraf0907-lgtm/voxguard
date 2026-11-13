"""Thin HTTP client for the VoxGuard API. No business logic: it sends requests, validates the
responses against the shared schemas and turns every failure into an ``ApiError``."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Literal, TypeVar

import requests
from pydantic import BaseModel, TypeAdapter, ValidationError

from voxguard.schemas import (
    AlertRecord,
    AnalyzeResponse,
    BlocklistEntry,
    CodecsResponse,
    HealthResponse,
    SessionRecord,
)

DEFAULT_API_URL = "http://localhost:8000"
CONNECT_TIMEOUT_S = 3.05
READ_TIMEOUT_S = 10.0
ANALYZE_READ_TIMEOUT_S = 60.0  # decoding plus model inference

ErrorKind = Literal[
    "unreachable",
    "timeout",
    "too_large",
    "unsupported",
    "invalid",
    "not_implemented",
    "unavailable",
    "server",
    "bad_response",
    "other",
]

_STATUS_KIND: dict[int, ErrorKind] = {
    413: "too_large",
    415: "unsupported",
    422: "invalid",
    501: "not_implemented",
    503: "unavailable",
}

T = TypeVar("T", bound=BaseModel)


class ApiError(Exception):
    """A failed API call. ``kind`` is stable (the UI maps it to a translated message);
    ``detail`` is the server's own message when it sent one."""

    def __init__(self, kind: ErrorKind, status: int | None = None, detail: str | None = None):
        super().__init__(f"{kind} (HTTP {status}){': ' + detail if detail else ''}")
        self.kind = kind
        self.status = status
        self.detail = detail


def default_base_url() -> str:
    return os.environ.get("VOXGUARD_API_URL", "").strip() or DEFAULT_API_URL


def _detail(response: requests.Response) -> str | None:
    try:
        body = response.json()
    except ValueError:
        return None
    detail = body.get("detail") if isinstance(body, dict) else None
    return None if detail is None else str(detail)[:300]


def _kind_for(status: int) -> ErrorKind:
    if status in _STATUS_KIND:
        return _STATUS_KIND[status]
    return "server" if status >= 500 else "other"


class VoxGuardClient:
    def __init__(self, base_url: str | None = None, session: requests.Session | None = None):
        self.base_url = (base_url or default_base_url()).rstrip("/")
        self._http = session or requests.Session()

    def _request(
        self, method: str, path: str, *, read_timeout: float = READ_TIMEOUT_S, **kwargs: Any
    ) -> Any:
        try:
            response = self._http.request(
                method,
                f"{self.base_url}{path}",
                timeout=(CONNECT_TIMEOUT_S, read_timeout),
                **kwargs,
            )
        except requests.Timeout:
            raise ApiError("timeout") from None
        except requests.RequestException:
            raise ApiError("unreachable") from None
        if not response.ok:
            raise ApiError(_kind_for(response.status_code), response.status_code, _detail(response))
        try:
            return response.json()
        except ValueError:
            raise ApiError("bad_response", response.status_code) from None

    @staticmethod
    def _parse(model: type[T], payload: Any) -> T:
        try:
            return model.model_validate(payload)
        except ValidationError:
            raise ApiError("bad_response") from None

    @staticmethod
    def _parse_list(model: type[T], payload: Any) -> list[T]:
        try:
            return TypeAdapter(list[model]).validate_python(payload)  # type: ignore[valid-type]
        except ValidationError:
            raise ApiError("bad_response") from None

    def health(self) -> HealthResponse:
        return self._parse(HealthResponse, self._request("GET", "/health"))

    def analyze(
        self,
        audio: bytes,
        filename: str = "audio.wav",
        caller_id: str | None = None,
        simulate_codec: bool = False,
        codec: str | None = None,
    ) -> AnalyzeResponse:
        """POST /v1/analyze. Fields are only sent when set: ``codec`` picks the codec (and implies
        simulation); ``simulate_codec`` alone uses the server's default codec."""
        data: dict[str, str] = {}
        if caller_id:
            data["caller_id"] = caller_id
        if simulate_codec:
            data["simulate_codec"] = "true"
        if codec:
            data["codec"] = codec
        payload = self._request(
            "POST",
            "/v1/analyze",
            read_timeout=ANALYZE_READ_TIMEOUT_S,
            files={"file": (filename, audio)},
            data=data or None,
        )
        return self._parse(AnalyzeResponse, payload)

    def codecs(self) -> CodecsResponse:
        return self._parse(CodecsResponse, self._request("GET", "/v1/codecs"))

    def sessions(self, flagged_only: bool = False, limit: int = 50) -> list[SessionRecord]:
        params = {"flagged_only": str(flagged_only).lower(), "limit": limit}
        return self._parse_list(SessionRecord, self._request("GET", "/v1/sessions", params=params))

    def blocklist(self) -> list[BlocklistEntry]:
        return self._parse_list(BlocklistEntry, self._request("GET", "/v1/blocklist"))

    def alerts(self, since: datetime | None = None, limit: int = 100) -> list[AlertRecord]:
        params: dict[str, Any] = {"limit": limit}
        if since is not None:
            params["since"] = since.isoformat()
        return self._parse_list(AlertRecord, self._request("GET", "/v1/alerts", params=params))
