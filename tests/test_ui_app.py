from datetime import UTC, datetime
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from voxguard.schemas import (
    AlertRecord,
    AnalyzeResponse,
    BlocklistEntry,
    HealthResponse,
    SessionRecord,
)
from voxguard.ui.api_client import ApiError
from voxguard.ui.i18n import t

APP = str(Path(__file__).parents[1] / "src" / "voxguard" / "ui" / "app.py")
T0 = datetime(2026, 11, 2, 12, 30, 5, tzinfo=UTC)


class FakeClient:
    """Stands in for VoxGuardClient: records calls, returns only what the test configures."""

    base_url = "http://fake:8000"

    def __init__(self, detector: str = "stub", down: ApiError | None = None) -> None:
        self.down = down
        self.health_response = HealthResponse(status="ok", detector=detector, model_version="m-1")
        self.sessions_response: list[SessionRecord] = []
        self.blocklist_response: list[BlocklistEntry] = []
        self.alerts_response: list[AlertRecord] = []
        self.analyze_response: AnalyzeResponse | ApiError | None = None
        self.calls: list[tuple] = []

    def _check(self) -> None:
        if self.down:
            raise self.down

    def health(self):
        self._check()
        return self.health_response

    def analyze(self, audio, filename="audio.wav", caller_id=None):
        self.calls.append(("analyze", audio, filename, caller_id))
        if isinstance(self.analyze_response, ApiError):
            raise self.analyze_response
        return self.analyze_response

    def sessions(self, flagged_only=False, limit=50):
        self.calls.append(("sessions", flagged_only, limit))
        self._check()
        return self.sessions_response

    def blocklist(self):
        self._check()
        return self.blocklist_response

    def alerts(self, since=None, limit=100):
        self.calls.append(("alerts", since, limit))
        self._check()
        return self.alerts_response


def _analyze_calls(client: FakeClient) -> list[tuple]:
    return [c for c in client.calls if c[0] == "analyze"]


def _result(flagged: bool, score: float = 0.9, codec: str | None = None) -> AnalyzeResponse:
    return AnalyzeResponse(
        session_id="sess-1", label="fake" if flagged else "real", fake_score=score,
        threshold=0.5, latency_ms=12.34, model_version="m-1", flagged=flagged,
        codec_applied=codec,
    )  # fmt: skip


def _run(client: FakeClient, lang: str = "en") -> AppTest:
    at = AppTest.from_file(APP, default_timeout=20)
    at.session_state["api_client"] = client
    if lang != "en":
        at.session_state["lang"] = lang
    return at.run()


def _text(at: AppTest) -> str:
    kinds = (at.title, at.markdown, at.caption, at.info, at.warning, at.error, at.success)
    parts = [e.value for kind in kinds for e in kind]
    parts += [e.value for kind in (at.header, at.subheader) for e in kind]
    parts += [m.label + " " + m.value for m in at.metric]
    return "\n".join(parts)


# --- rendering -------------------------------------------------------------------------


def test_renders_english() -> None:
    at = _run(FakeClient())
    assert not at.exception
    assert at.title[0].value == t("app_title", "en")
    assert [tab.label for tab in at.tabs] == [
        t(k, "en") for k in ("tab_analyze", "tab_sessions", "tab_blocklist", "tab_alerts")
    ]
    assert t("status_connected", "en") in _text(at)
    assert t("status_detail", "en", detector="stub", model="m-1") in _text(at)


def test_renders_hindi_and_shows_draft_notice() -> None:
    at = _run(FakeClient(), "hi")
    assert not at.exception
    assert [tab.label for tab in at.tabs][0] == t("tab_analyze", "hi")
    text = _text(at)
    assert t("app_subtitle", "hi") in text
    assert t("hi_draft_notice", "hi") in text
    assert t("hi_draft_notice", "en") not in text


def test_language_switch() -> None:
    at = _run(FakeClient())
    assert at.selectbox[0].value == "en"
    at.selectbox[0].set_value("hi").run()
    assert not at.exception and at.tabs[1].label == t("tab_sessions", "hi")
    at.selectbox[0].set_value("en").run()
    assert at.tabs[1].label == t("tab_sessions", "en")
    assert t("hi_draft_notice", "en") not in _text(at)


def test_stub_notice_only_for_stub_detector() -> None:
    assert t("stub_notice", "en") in [w.value for w in _run(FakeClient("stub")).warning]
    assert t("stub_notice", "en") not in [w.value for w in _run(FakeClient("real")).warning]


# --- API down --------------------------------------------------------------------------


def test_api_unavailable_is_handled() -> None:
    at = _run(FakeClient(down=ApiError("unreachable")))
    assert not at.exception
    text = _text(at)
    assert t("status_unavailable", "en") in text and t("err_unreachable", "en") in text
    assert t("status_connected", "en") not in text
    assert not at.metric and not at.dataframe


def test_server_detail_is_shown() -> None:
    at = _run(FakeClient(down=ApiError("server", 500, "db locked")))
    assert t("server_detail", "en", detail="db locked") in _text(at)


# --- empty and populated monitoring ----------------------------------------------------


def test_empty_lists_render_messages_not_tables() -> None:
    at = _run(FakeClient())
    assert not at.exception and not at.dataframe
    text = _text(at)
    for key in ("empty_sessions", "empty_blocklist", "empty_alerts", "utc_note"):
        assert t(key, "en") in text


def test_monitoring_requests_flagged_sessions_only() -> None:
    client = FakeClient()
    _run(client)
    assert ("sessions", True, 50) in client.calls and ("alerts", None, 100) in client.calls


def test_populated_tables_show_api_data_in_utc() -> None:
    client = FakeClient()
    client.sessions_response = [
        SessionRecord(session_id="s1", created_at=T0, caller_id="+91 1", label="fake",
                      fake_score=0.98765, threshold=0.5, flagged=True, model_version="m-1"),
        SessionRecord(session_id="s2", created_at=T0, caller_id=None, label="fake",
                      fake_score=0.7, threshold=0.5, flagged=True, model_version="m-1"),
    ]  # fmt: skip
    client.blocklist_response = [
        BlocklistEntry(caller_id="+91 1", hit_count=3, first_seen=T0, last_seen=T0)
    ]
    client.alerts_response = [
        AlertRecord(alert_id="a1", created_at=T0, session_id="s1", caller_id="+91 1",
                    fake_score=0.98765, message="Flagged call: x")
    ]  # fmt: skip
    at = _run(client)
    assert not at.exception and len(at.dataframe) == 3
    sessions, blocklist, alerts = (d.value for d in at.dataframe)
    assert sessions[t("col_time", "en")].tolist() == ["2026-11-02 12:30:05"] * 2
    assert sessions[t("col_caller", "en")].tolist() == ["+91 1", t("missing_value", "en")]
    assert sessions[t("col_score", "en")].tolist() == [0.9877, 0.7]
    assert sessions[t("col_label", "en")].tolist() == [t("label_fake", "en")] * 2
    assert blocklist[t("col_hits", "en")].tolist() == [3]
    assert alerts[t("col_message", "en")].tolist() == ["Flagged call: x"]


def test_table_headers_follow_language() -> None:
    client = FakeClient()
    client.blocklist_response = [
        BlocklistEntry(caller_id="x", hit_count=1, first_seen=T0, last_seen=T0)
    ]
    at = _run(client, "hi")
    assert t("col_hits", "hi") in at.dataframe[0].value.columns


# --- analysis --------------------------------------------------------------------------


def test_no_result_or_scores_before_analysis() -> None:
    at = _run(FakeClient())
    assert not at.metric
    assert t("no_audio", "en") in _text(at)
    assert not at.button or all(b.label != t("analyze_button", "en") for b in at.button)


def test_simulate_phone_line_is_disabled_and_explained() -> None:
    at = _run(FakeClient())
    (toggle,) = at.toggle
    assert toggle.label == t("simulate_label", "en") and toggle.disabled and not toggle.value
    assert t("simulate_unavailable", "en") in _text(at)


def _analyze(client: FakeClient, lang: str = "en", caller: str = "") -> AppTest:
    at = _run(client, lang)
    if caller:
        at.text_input[0].set_value(caller).run()
    at.file_uploader[0].set_value(("clip.wav", b"RIFFfake", "audio/wav")).run()
    next(b for b in at.button if b.label == t("analyze_button", lang)).click().run()
    return at


def test_flagged_result_is_shown_with_banner() -> None:
    client = FakeClient()
    client.analyze_response = _result(True, 0.9998)
    at = _analyze(client, caller="  alice ")
    assert not at.exception
    assert _analyze_calls(client) == [("analyze", b"RIFFfake", "clip.wav", "alice")]
    assert t("flagged_banner", "en") in [e.value for e in at.error]
    values = {m.label: m.value for m in at.metric}
    assert values[t("m_label", "en")] == t("label_fake", "en")
    assert values[t("m_score", "en")] == "0.9998"
    assert values[t("m_threshold", "en")] == "0.5000"
    assert values[t("m_latency", "en")] == t("unit_ms", "en", value="12.3")
    assert values[t("m_model", "en")] == "m-1"
    assert values[t("m_flagged", "en")] == t("yes", "en")
    assert values[t("m_codec", "en")] == t("none_value", "en")
    assert t("m_session", "en", session_id="sess-1") in _text(at)


def test_unflagged_result_is_shown_without_warning_banner() -> None:
    client = FakeClient()
    client.analyze_response = _result(False, 0.12)
    at = _analyze(client)
    assert t("not_flagged_banner", "en") in [e.value for e in at.success]
    assert t("flagged_banner", "en") not in _text(at)
    values = {m.label: m.value for m in at.metric}
    assert values[t("m_label", "en")] == t("label_real", "en")
    assert values[t("m_flagged", "en")] == t("no", "en")
    assert _analyze_calls(client)[0][3] is None  # blank caller id is not sent


def test_codec_applied_is_shown_only_when_the_api_reports_it() -> None:
    client = FakeClient()
    client.analyze_response = _result(True, codec="g711_mulaw")
    values = {m.label: m.value for m in _analyze(client).metric}
    assert values[t("m_codec", "en")] == "g711_mulaw"


def test_result_survives_language_switch_and_is_translated() -> None:
    client = FakeClient()
    client.analyze_response = _result(True)
    at = _analyze(client)
    at.selectbox[0].set_value("hi").run()
    assert not at.exception
    values = {m.label: m.value for m in at.metric}
    assert values[t("m_label", "hi")] == t("label_fake", "hi")
    assert len(_analyze_calls(client)) == 1  # switching language does not re-analyze


def test_recorded_audio_is_analyzed_too() -> None:
    client = FakeClient()
    client.analyze_response = _result(False, 0.1)
    at = _run(client)
    at.radio[0].set_value("record").run()
    at.audio_input[0].set_value(("rec.wav", b"RIFFmic", "audio/wav")).run()
    next(b for b in at.button if b.label == t("analyze_button", "en")).click().run()
    assert not at.exception
    assert _analyze_calls(client)[0][1] == b"RIFFmic"


@pytest.mark.parametrize(
    "kind", ["too_large", "unsupported", "invalid", "not_implemented", "server", "timeout"]
)
def test_analysis_errors_show_distinct_messages(kind: str) -> None:
    client = FakeClient()
    client.analyze_response = ApiError(kind, 500, "why")
    at = _analyze(client)
    assert not at.exception and not at.metric
    errors = [e.value for e in at.error]
    assert any(t(f"err_{kind}", "en") in e and "why" in e for e in errors)


def test_failed_analysis_clears_previous_result() -> None:
    client = FakeClient()
    client.analyze_response = _result(True)
    at = _analyze(client)
    assert at.metric
    client.analyze_response = ApiError("server", 500)
    next(b for b in at.button if b.label == t("analyze_button", "en")).click().run()
    assert not at.metric


def test_analyze_never_requests_codec_simulation() -> None:
    client = FakeClient()
    client.analyze_response = _result(False)
    _analyze(client)
    calls = _analyze_calls(client)
    assert calls and all(len(c) == 4 for c in calls)  # audio, name, caller only: no codec field
