"""Streamlit demo UI. Run: ``streamlit run src/voxguard/ui/app.py`` (needs the API running).

The UI only talks to the FastAPI backend; it does no inference and no prevention logic.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import streamlit as st

from voxguard.schemas import AnalyzeResponse
from voxguard.ui.api_client import ApiError, VoxGuardClient
from voxguard.ui.i18n import DEFAULT_LANGUAGE, LANGUAGES, t

CLIENT_KEY = "api_client"  # tests put a fake client here before running the script
RESULTS_KEY = "analysis_results"  # {"audio": digest, "clean": response, "phone": response}
CACHE_KEY = "api_cache"  # {name: (fetched_at, value)}, per browser session
CACHE_TTL_S = 5.0
AUDIO_TYPES = ["wav", "flac", "ogg", "mp3"]  # formats the API's decoder (libsndfile) reads
SESSIONS_LIMIT = 50
ALERTS_LIMIT = 100


def get_client() -> Any:
    if CLIENT_KEY not in st.session_state:
        st.session_state[CLIENT_KEY] = VoxGuardClient()
    return st.session_state[CLIENT_KEY]


def cached(name: str, fetch: Callable[[], Any], ttl: float = CACHE_TTL_S) -> Any:
    """Reuse a successful API read for ``ttl`` seconds within this browser session, so reruns
    (typing, switching language) do not refetch everything. Errors are never cached."""
    cache = st.session_state.setdefault(CACHE_KEY, {})
    hit = cache.get(name)
    if hit is not None and time.monotonic() - hit[0] < ttl:
        return hit[1]
    value = fetch()
    cache[name] = (time.monotonic(), value)
    return value


def invalidate(*names: str) -> None:
    cache = st.session_state.setdefault(CACHE_KEY, {})
    for name in names or tuple(cache):
        cache.pop(name, None)


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
            health = cached("health", client.health)
            detector = health.detector
            st.success(t("status_connected", lang))
            st.caption(
                t("status_detail", lang, detector=health.detector, model=health.model_version)
            )
        except ApiError as err:
            st.error(t("status_unavailable", lang))
            st.caption(error_text(err, lang))
    return lang, detector


def render_result(result: AnalyzeResponse, lang: str, header_key: str = "result_header") -> None:
    st.subheader(t(header_key, lang))
    if result.codec_applied:
        st.caption(t("result_simulated", lang, codec=result.codec_applied))
    else:
        st.caption(t("result_clean", lang))
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


def current_results(digest: str | None) -> dict[str, AnalyzeResponse]:
    """Results that belong to the audio now selected; anything older is discarded."""
    results = st.session_state.get(RESULTS_KEY)
    if digest is None or results is None or results.get("audio") != digest:
        st.session_state.pop(RESULTS_KEY, None)
        return {}
    return {mode: r for mode, r in results.items() if mode != "audio"}


def render_analyze(client: Any, lang: str) -> None:
    st.header(t("analyze_header", lang))
    caller_id = st.text_input(
        t("caller_label", lang), max_chars=128, help=t("caller_help", lang), key="caller_id"
    )
    simulate = st.toggle(
        t("simulate_label", lang), value=False, help=t("simulate_help", lang), key="simulate_line"
    )
    st.caption(t("simulate_on" if simulate else "simulate_off", lang))

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

    data = audio.getvalue() if audio is not None else None
    digest = hashlib.sha256(data).hexdigest() if data else None
    results = current_results(digest)  # a new file or recording drops the previous results

    if audio is None:
        st.info(t("no_audio", lang))
    else:
        st.audio(data)
        mode = "phone" if simulate else "clean"
        if st.button(t("analyze_button", lang), type="primary", key="analyze"):
            results.pop(mode, None)  # never show an older result for this mode after a new run
            try:
                with st.spinner(t("analyzing", lang)):
                    results[mode] = client.analyze(
                        data,
                        filename=audio.name or "audio.wav",
                        caller_id=caller_id.strip() or None,
                        simulate_codec=simulate,
                    )
                invalidate("sessions", "blocklist", "alerts")
            except ApiError as err:
                st.error(error_text(err, lang))
            st.session_state[RESULTS_KEY] = {"audio": digest, **results}

    if "clean" in results and "phone" in results:  # same clip, clean vs phone line
        left, right = st.columns(2)
        with left:
            render_result(results["clean"], lang, "result_header_clean")
        with right:
            render_result(results["phone"], lang, "result_header_phone")
    elif results:
        render_result(next(iter(results.values())), lang)


def render_table(rows: list[dict[str, Any]], empty_key: str, lang: str) -> None:
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")
    else:
        st.info(t(empty_key, lang))


def _monitor(
    lang: str, name: str, header_key: str, fetch: Any, to_rows: Any, empty_key: str
) -> None:
    st.header(t(header_key, lang))
    st.caption(t("utc_note", lang))
    if st.button(t("refresh", lang), key=f"refresh_{header_key}"):
        invalidate(name)
    try:
        items = cached(name, fetch)
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
            "sessions",
            "sessions_header",
            lambda: client.sessions(flagged_only=True, limit=SESSIONS_LIMIT),
            session_rows,
            "empty_sessions",
        )
    with tabs[1]:
        _monitor(
            lang,
            "blocklist",
            "blocklist_header",
            client.blocklist,
            blocklist_rows,
            "empty_blocklist",
        )
    with tabs[2]:
        _monitor(
            lang,
            "alerts",
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
