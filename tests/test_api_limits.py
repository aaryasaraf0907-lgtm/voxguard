import asyncio

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from conftest import make_wav
from voxguard.api.db import SessionRow
from voxguard.api.deps import Settings
from voxguard.api.limits import BodySizeLimitMiddleware

LIMIT = 100


def _run(chunks: list[bytes], headers: list[tuple[bytes, bytes]]):
    """Drive the middleware with a fake app that reads the whole body then answers 200."""
    sent: list[dict] = []
    app_responded = False

    async def app(scope, receive, send):
        nonlocal app_responded
        while (await receive()).get("more_body"):
            pass
        app_responded = True
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    messages = [
        {"type": "http.request", "body": c, "more_body": i < len(chunks) - 1}
        for i, c in enumerate(chunks)
    ]

    async def receive():
        return messages.pop(0)

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "method": "POST", "path": "/x", "headers": headers}
    try:
        asyncio.run(BodySizeLimitMiddleware(app, LIMIT, "/x")(scope, receive, send))
    except HTTPException as exc:  # streamed overflow: raised into the app, rendered by FastAPI
        return exc.status_code, app_responded
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    return status, app_responded


def test_content_length_over_limit_rejected_without_reading() -> None:
    status, app_ran = _run([b"x"], [(b"content-length", str(LIMIT + 1).encode())])
    assert status == 413 and not app_ran


def test_streamed_body_over_limit_rejected_without_content_length() -> None:
    status, app_ran = _run([b"x" * 60, b"x" * 60], [])
    assert status == 413 and not app_ran


def test_lying_content_length_still_rejected_when_streamed() -> None:
    status, app_ran = _run([b"x" * 60, b"x" * 60], [(b"content-length", b"10")])
    assert status == 413 and not app_ran


def test_body_within_limit_passes() -> None:
    status, app_ran = _run([b"x" * 50, b"x" * 50], [])
    assert status == 200 and app_ran


def test_other_paths_are_not_limited() -> None:
    from voxguard.api.main import create_app

    with TestClient(create_app(settings=Settings(db_url="sqlite://", max_upload_bytes=1))) as c:
        assert c.get("/health").status_code == 200


def test_oversized_content_length_via_app_is_413(client_factory) -> None:
    with client_factory(max_upload_bytes=1_000) as c:
        r = c.post("/v1/analyze", files={"file": ("a.wav", make_wav(10.0))})
        assert r.status_code == 413


def test_chunked_oversized_body_via_app_is_413(client_factory) -> None:
    def gen():
        for _ in range(100):
            yield b"x" * 4096

    with client_factory(max_upload_bytes=1_000) as c:
        r = c.post(
            "/v1/analyze",
            content=gen(),
            headers={"content-type": "multipart/form-data; boundary=x"},
        )
        assert r.status_code == 413 and r.json() == {"detail": "Request body too large"}
        with c.app.state.session_factory() as db:
            assert db.query(SessionRow).count() == 0


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("VOXGUARD_MAX_UPLOAD_MB", "abc"),
        ("VOXGUARD_MAX_UPLOAD_MB", "0"),
        ("VOXGUARD_MAX_UPLOAD_MB", "-5"),
        ("VOXGUARD_MAX_AUDIO_SECONDS", "nan"),
        ("VOXGUARD_MAX_AUDIO_SECONDS", "inf"),
    ],
)
def test_invalid_env_fails_fast_naming_variable(monkeypatch, name: str, value: str) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        Settings.from_env()


def test_env_defaults_and_overrides(monkeypatch) -> None:
    for n in ("VOXGUARD_MAX_UPLOAD_MB", "VOXGUARD_MAX_AUDIO_SECONDS", "VOXGUARD_DB_URL"):
        monkeypatch.delenv(n, raising=False)
    s = Settings.from_env()
    assert s.max_upload_bytes == 25 * 1024 * 1024 and s.max_audio_seconds == 60
    assert s.db_url == "sqlite:///./voxguard.db"
    monkeypatch.setenv("VOXGUARD_MAX_UPLOAD_MB", "2")
    monkeypatch.setenv("VOXGUARD_DB_URL", "sqlite:///x.db")
    s = Settings.from_env()
    assert s.max_upload_bytes == 2 * 1024 * 1024 and s.db_url == "sqlite:///x.db"
