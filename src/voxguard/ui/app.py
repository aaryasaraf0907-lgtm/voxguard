"""Streamlit demo UI. Run: ``streamlit run src/voxguard/ui/app.py`` (needs the API running).

The UI only talks to the FastAPI backend; it does no inference and no prevention logic.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import streamlit as st

from voxguard.schemas import AnalyzeResponse
from voxguard.ui.api_client import ApiError, VoxGuardClient
from voxguard.ui.i18n import DEFAULT_LANGUAGE, LANGUAGES, t

CLIENT_KEY = "api_client"  # tests put a fake client here before running the script
RESULT_KEY = "analysis_result"
AUDIO_TYPES = ["wav", "flac", "ogg", "mp3"]  # formats the API's decoder (libsndfile) reads
SESSIONS_LIMIT = 50
ALERTS_LIMIT = 100


def get_client() -> Any:
    if CLIENT_KEY not in st.session_state:
        st.session_state[CLIENT_KEY] = VoxGuardClient()
    return st.session_state[CLIENT_KEY]


def error_text(err: ApiError, lang: str) -> str:
    text = t(f"err_{err.kind}" if err.kind != "other" else "err_other", lang)
    if err.detail:
        text += "\n\n" + t("server_detail", lang, detail=err.detail)
    return text


def fmt_time(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")


def render_sidebar(client: Any, lang_default: str) -> tuple[str, str | None]:
    """Language selector and API status. Returns (language, detector name or None if down)."""
    with st.sidebar:
        lang = st.selectbox(
            t("language", DEFAULT_LANGUAGE),
            options=list(LANGUAGES),
            format_func=LANGUAGES.get,
            index=list(LANGUAGES).index(lang_default),
            key="lang",
        )
        st.subheader(t("api_section", lang))
        st.caption(t("api_url", lang, url=client.base_url))
        detector = None
        try:
            health = client.health()
            detector = health.detector
            st.success(t("status_connected", lang))
            st.caption(
                t("status_detail", lang, detector=health.detector, model=health.model_version)
            )
        except ApiError as err:
            st.error(t("status_unavailable", lang))
            st.caption(error_text(err, lang))
        if lang == "hi":
            st.caption(t("hi_draft_notice", lang))
    return lang, detector


def render_result(result: AnalyzeResponse, lang: str) -> None:
    st.subheader(t("result_header", lang))
    if result.flagged:
        st.error(t("flagged_banner", lang))
    else:
        st.success(t("not_flagged_banner", lang))
    cols = st.columns(4)
    cols[0].metric(t("m_label", lang), t(f"label_{result.label}", lang))
    cols[1].metric(t("m_score", lang), f"{result.fake_score:.4f}")
    cols[2].metric(t("m_threshold", lang), f"{result.threshold:.4f}")
    cols[3].metric(t("m_latency", lang), t("unit_ms", lang, value=f"{result.latency_ms:.1f}"))
    cols = st.columns(3)
    cols[0].metric(t("m_model", lang), result.model_version)
    cols[1].metric(t("m_flagged", lang), t("yes" if result.flagged else "no", lang))
    cols[2].metric(t("m_codec", lang), result.codec_applied or t("none_value", lang))
    st.caption(t("m_session", lang, session_id=result.session_id))


def render_analyze(client: Any, lang: str) -> None:
    st.header(t("analyze_header", lang))
    caller_id = st.text_input(
        t("caller_label", lang), max_chars=128, help=t("caller_help", lang), key="caller_id"
    )
    # Disabled on purpose: the API returns 501 for simulate_codec, so it is never sent.
    st.toggle(t("simulate_label", lang), value=False, disabled=True, key="simulate_line")
    st.caption(t("simulate_unavailable", lang))

    source = st.radio(
        t("source_label", lang),
        options=["upload", "record"],
        format_func=lambda k: t(f"source_{k}", lang),
        horizontal=True,
        key="source",
    )
    if source == "upload":
        audio = st.file_uploader(t("upload_label", lang), type=AUDIO_TYPES, key="upload")
    else:
        audio = st.audio_input(t("record_label", lang), key="record")

    if audio is None:
        st.info(t("no_audio", lang))
    else:
        data = audio.getvalue()
        st.audio(data)
        if st.button(t("analyze_button", lang), type="primary", key="analyze"):
            try:
                with st.spinner(t("analyzing", lang)):
                    st.session_state[RESULT_KEY] = client.analyze(
                        data,
                        filename=audio.name or "audio.wav",
                        caller_id=caller_id.strip() or None,
                    )
            except ApiError as err:
                st.session_state.pop(RESULT_KEY, None)
                st.error(error_text(err, lang))

    if RESULT_KEY in st.session_state:
        render_result(st.session_state[RESULT_KEY], lang)


def render_table(rows: list[dict[str, Any]], empty_key: str, lang: str) -> None:
    if rows:
        st.dataframe(rows, hide_index=True, use_container_width=True)
    else:
        st.info(t(empty_key, lang))


def _monitor(lang: str, header_key: str, fetch: Any, to_rows: Any, empty_key: str) -> None:
    st.header(t(header_key, lang))
    st.caption(t("utc_note", lang))
    st.button(t("refresh", lang), key=f"refresh_{header_key}")  # a click reruns the script
    try:
        items = fetch()
    except ApiError as err:
        st.error(error_text(err, lang))
        return
    render_table(to_rows(items), empty_key, lang)


def render_monitoring(client: Any, lang: str, tabs: list[Any]) -> None:
    missing = t("missing_value", lang)

    def session_rows(items: list[Any]) -> list[dict[str, Any]]:
        return [
            {
                t("col_time", lang): fmt_time(s.created_at),
                t("col_caller", lang): s.caller_id or missing,
                t("col_label", lang): t(f"label_{s.label}", lang),
                t("col_score", lang): round(s.fake_score, 4),
                t("col_threshold", lang): round(s.threshold, 4),
                t("col_model", lang): s.model_version,
            }
            for s in items
        ]

    def blocklist_rows(items: list[Any]) -> list[dict[str, Any]]:
        return [
            {
                t("col_caller", lang): b.caller_id,
                t("col_hits", lang): b.hit_count,
                t("col_first_seen", lang): fmt_time(b.first_seen),
                t("col_last_seen", lang): fmt_time(b.last_seen),
            }
            for b in items
        ]

    def alert_rows(items: list[Any]) -> list[dict[str, Any]]:
        return [
            {
                t("col_time", lang): fmt_time(a.created_at),
                t("col_caller", lang): a.caller_id or missing,
                t("col_score", lang): round(a.fake_score, 4),
                t("col_message", lang): a.message,
                t("col_session", lang): a.session_id,
            }
            for a in items
        ]

    with tabs[0]:
        _monitor(
            lang,
            "sessions_header",
            lambda: client.sessions(flagged_only=True, limit=SESSIONS_LIMIT),
            session_rows,
            "empty_sessions",
        )
    with tabs[1]:
        _monitor(lang, "blocklist_header", client.blocklist, blocklist_rows, "empty_blocklist")
    with tabs[2]:
        _monitor(
            lang,
            "alerts_header",
            lambda: client.alerts(limit=ALERTS_LIMIT),
            alert_rows,
            "empty_alerts",
        )


def main() -> None:
    st.set_page_config(page_title=t("app_title", DEFAULT_LANGUAGE), layout="wide")
    client = get_client()
    lang, detector = render_sidebar(client, st.session_state.get("lang", DEFAULT_LANGUAGE))
    st.title(t("app_title", lang))
    st.write(t("app_subtitle", lang))
    if detector == "stub":
        st.warning(t("stub_notice", lang))

    tabs = st.tabs(
        [t(k, lang) for k in ("tab_analyze", "tab_sessions", "tab_blocklist", "tab_alerts")]
    )
    with tabs[0]:
        render_analyze(client, lang)
    render_monitoring(client, lang, tabs[1:])


main()
