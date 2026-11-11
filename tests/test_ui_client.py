import json
import socket
import threading
import time
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
import requests
import uvicorn

from conftest import FixedDetector, make_wav
from voxguard.api.deps import Settings
from voxguard.api.main import create_app
from voxguard.ui.api_client import DEFAULT_API_URL, ApiError, VoxGuardClient, default_base_url

HEALTH = {"status": "ok", "detector": "stub", "model_version": "stub-0"}
ANALYZE = {
    "session_id": "abc", "label": "fake", "fake_score": 0.9, "threshold": 0.5,
    "latency_ms": 1.0, "model_version": "stub-0", "flagged": True, "codec_applied": None,
}  # fmt: skip


def _response(status: int, body: object = None, raw: bytes | None = None) -> requests.Response:
    r = requests.Response()
    r.status_code = status
    r._content = raw if raw is not None else json.dumps(body).encode()
    return r


class FakeSession:
    def __init__(self, response: requests.Response | Exception) -> None:
        self.response = response
        self.calls: list[dict] = []

    def request(self, method: str, url: str, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _client(response) -> tuple[VoxGuardClient, FakeSession]:
    s = FakeSession(response)
    return VoxGuardClient("http://api:9/", session=s), s  # type: ignore[arg-type]


# --- configuration ---------------------------------------------------------------------


def test_default_base_url(monkeypatch) -> None:
    monkeypatch.delenv("VOXGUARD_API_URL", raising=False)
    assert default_base_url() == DEFAULT_API_URL == "http://localhost:8000"
    monkeypatch.setenv("VOXGUARD_API_URL", "http://example:1234")
    assert VoxGuardClient().base_url == "http://example:1234"
    monkeypatch.setenv("VOXGUARD_API_URL", "  ")
    assert default_base_url() == DEFAULT_API_URL


# --- requests and parsing (mocked transport) -------------------------------------------


def test_health_ok() -> None:
    c, s = _client(_response(200, HEALTH))
    assert c.health().detector == "stub"
    assert s.calls[0]["method"] == "GET" and s.calls[0]["url"] == "http://api:9/health"
    connect, read = s.calls[0]["timeout"]
    assert 0 < connect <= 5 and 0 < read <= 30


def test_analyze_sends_multipart_file_and_caller_id_only() -> None:
    c, s = _client(_response(200, ANALYZE))
    result = c.analyze(b"RIFFdata", filename="a.wav", caller_id="alice")
    call = s.calls[0]
    assert (call["method"], call["url"]) == ("POST", "http://api:9/v1/analyze")
    assert call["files"] == {"file": ("a.wav", b"RIFFdata")}
    assert call["data"] == {"caller_id": "alice"}
    assert "simulate_codec" not in json.dumps(call, default=str)  # off by default
    assert result.flagged is True and result.codec_applied is None
    assert call["timeout"][1] > 10  # inference gets a longer read timeout


def test_analyze_sends_simulate_codec_only_when_requested() -> None:
    c, s = _client(_response(200, {**ANALYZE, "codec_applied": "g711_mulaw"}))
    assert c.analyze(b"x", simulate_codec=True).codec_applied == "g711_mulaw"
    assert s.calls[0]["data"] == {"simulate_codec": "true"}
    c.analyze(b"x", caller_id="bob", simulate_codec=True)
    assert s.calls[1]["data"] == {"caller_id": "bob", "simulate_codec": "true"}


def test_analyze_without_caller_sends_no_form_fields() -> None:
    c, s = _client(_response(200, ANALYZE))
    c.analyze(b"x")
    assert s.calls[0]["data"] is None


def test_sessions_query_parameters() -> None:
    c, s = _client(_response(200, []))
    assert c.sessions(flagged_only=True, limit=7) == []
    assert s.calls[0]["params"] == {"flagged_only": "true", "limit": 7}
    c.sessions()
    assert s.calls[1]["params"] == {"flagged_only": "false", "limit": 50}


def test_alerts_query_parameters() -> None:
    c, s = _client(_response(200, []))
    c.alerts()
    assert s.calls[0]["params"] == {"limit": 100}
    since = datetime(2026, 11, 2, 12, 0, tzinfo=UTC)
    c.alerts(since=since, limit=5)
    assert s.calls[1]["params"] == {"limit": 5, "since": since.isoformat()}


def test_blocklist_parsed() -> None:
    body = [{"caller_id": "x", "hit_count": 2, "first_seen": "2026-11-02T12:00:00Z",
             "last_seen": "2026-11-02T12:05:00Z"}]  # fmt: skip
    c, s = _client(_response(200, body))
    (entry,) = c.blocklist()
    assert entry.caller_id == "x" and entry.hit_count == 2 and entry.last_seen.tzinfo
    assert s.calls[0]["url"].endswith("/v1/blocklist")


@pytest.mark.parametrize(
    ("status", "kind"),
    [(413, "too_large"), (415, "unsupported"), (422, "invalid"), (501, "not_implemented"),
     (503, "unavailable"), (500, "server"), (502, "server"), (404, "other"), (400, "other")],
)  # fmt: skip
def test_http_errors_map_to_distinct_kinds(status: int, kind: str) -> None:
    c, _ = _client(_response(status, {"detail": "nope"}))
    with pytest.raises(ApiError) as exc:
        c.health()
    assert (exc.value.kind, exc.value.status, exc.value.detail) == (kind, status, "nope")


def test_validation_error_list_detail_is_stringified() -> None:
    c, _ = _client(_response(422, {"detail": [{"loc": ["query", "limit"], "msg": "bad"}]}))
    with pytest.raises(ApiError) as exc:
        c.sessions()
    assert exc.value.kind == "invalid" and "bad" in exc.value.detail


def test_error_with_non_json_body() -> None:
    c, _ = _client(_response(500, raw=b"<html>boom</html>"))
    with pytest.raises(ApiError) as exc:
        c.health()
    assert exc.value.kind == "server" and exc.value.detail is None


def test_timeout() -> None:
    c, _ = _client(requests.Timeout())
    with pytest.raises(ApiError) as exc:
        c.health()
    assert exc.value.kind == "timeout"


@pytest.mark.parametrize("error", [requests.ConnectionError(), requests.exceptions.SSLError()])
def test_connection_errors(error: Exception) -> None:
    c, _ = _client(error)
    with pytest.raises(ApiError) as exc:
        c.analyze(b"x")
    assert exc.value.kind == "unreachable"


def test_connect_timeout_is_a_timeout() -> None:
    c, _ = _client(requests.ConnectTimeout())
    with pytest.raises(ApiError) as exc:
        c.health()
    assert exc.value.kind == "timeout"


@pytest.mark.parametrize(
    "response",
    [
        _response(200, raw=b"not json"),
        _response(200, {"unexpected": True}),
        _response(200, ["a list, not an object"]),
    ],
)
def test_unexpected_success_bodies_do_not_crash(response) -> None:
    c, _ = _client(response)
    with pytest.raises(ApiError) as exc:
        c.health()
    assert exc.value.kind == "bad_response"


def test_list_endpoint_with_wrong_shape() -> None:
    c, _ = _client(_response(200, {"not": "a list"}))
    with pytest.raises(ApiError) as exc:
        c.blocklist()
    assert exc.value.kind == "bad_response"


# --- against the real API (uvicorn in a thread, random port) ---------------------------


@pytest.fixture
def live_api(tmp_path) -> Iterator[VoxGuardClient]:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    settings = Settings(
        db_url=f"sqlite:///{tmp_path / 'live.db'}",
        max_upload_bytes=200_000,
        max_audio_seconds=5,
        simulate_codec="g711_mulaw",
    )
    app = create_app(FixedDetector(0.9), settings)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    assert server.started, "test API server did not start"
    yield VoxGuardClient(f"http://127.0.0.1:{port}")
    server.should_exit = True
    thread.join(timeout=10)


def test_live_round_trip(live_api: VoxGuardClient) -> None:
    assert live_api.health().model_version == "fixed-0"
    result = live_api.analyze(make_wav(0.5), filename="clip.wav", caller_id=" +91 99 ")
    assert result.flagged and result.label == "fake" and result.fake_score == 0.9
    (session,) = live_api.sessions(flagged_only=True)
    assert session.session_id == result.session_id and session.caller_id == "+91 99"
    (entry,) = live_api.blocklist()
    assert entry.caller_id == "+91 99" and entry.hit_count == 1
    (alert,) = live_api.alerts()
    assert alert.session_id == result.session_id
    assert live_api.alerts(since=alert.created_at) == []


def test_live_error_kinds(live_api: VoxGuardClient) -> None:
    with pytest.raises(ApiError) as exc:
        live_api.analyze(b"definitely not audio")
    assert exc.value.kind == "unsupported" and exc.value.status == 415
    with pytest.raises(ApiError) as exc:
        live_api.analyze(make_wav(6.0, sr=8_000))  # over the 5 s limit, under the byte limit
    assert exc.value.kind == "invalid" and exc.value.status == 422
    with pytest.raises(ApiError) as exc:
        live_api.analyze(make_wav(10.0))  # ~320 KB, over the 200 KB limit
    assert exc.value.kind == "too_large" and exc.value.status == 413
    with pytest.raises(ApiError) as exc:
        live_api.analyze(b"")
    assert exc.value.kind == "other" and exc.value.status == 400


def test_live_phone_line_simulation(live_api: VoxGuardClient) -> None:
    from voxguard.data.codecs import available_codecs, ffmpeg_path

    wav = make_wav(1.0)
    if ffmpeg_path() is None or not available_codecs().get("g711_mulaw"):
        with pytest.raises(ApiError) as exc:  # the UI shows err_unavailable for this
            live_api.analyze(wav, simulate_codec=True)
        assert (exc.value.kind, exc.value.status) == ("unavailable", 503)
        return
    clean = live_api.analyze(wav)
    phone = live_api.analyze(wav, simulate_codec=True)
    assert clean.codec_applied is None and phone.codec_applied == "g711_mulaw"
    sessions = {s.session_id: s.codec_applied for s in live_api.sessions()}
    assert sessions[phone.session_id] == "g711_mulaw" and sessions[clean.session_id] is None
