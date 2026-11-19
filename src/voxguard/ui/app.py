"""Streamlit demo UI. Run: ``streamlit run src/voxguard/ui/app.py`` (needs the API running).

The UI only talks to the FastAPI backend; it does no inference and no prevention logic.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import streamlit as st

from voxguard.schemas import AnalyzeResponse, CodecsResponse
from voxguard.ui.api_client import ApiError, VoxGuardClient
from voxguard.ui.i18n import DEFAULT_LANGUAGE, LANGUAGES, t

CLIENT_KEY = "api_client"  # tests put a fake client here before running the script
RESULTS_KEY = "analysis_results"  # {"audio": digest, "clean": response, "phone": response}
CACHE_KEY = "api_cache"  # {name: (fetched_at, value)}, per browser session
CACHE_TTL_S = 5.0
AUDIO_TYPES = ["wav", "flac", "ogg", "mp3", "m4a"]  # libsndfile reads all but m4a (ffmpeg)
SESSIONS_LIMIT = 50
ALERTS_LIMIT = 100
TRANSACTIONS_LIMIT = 50
# Near-threshold caution (agreed with Aarya, 16 Nov): one-sided, only for flagged results that
# scored less than CAUTION_FACTOR x the threshold. Results under the threshold get no caution.
CAUTION_FACTOR = 1.5
REPLAY_STEP_S = 0.4  # per window when replaying the timeline (tests set 0)
CODEC_NAMES = {  # proper names, the same in every language
    "g711_mulaw": "G.711 µ-law",
    "g711_alaw": "G.711 A-law",
    "amr_nb": "AMR-NB",
    "amr_wb": "AMR-WB",
    "opus": "Opus",
}


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


def error_text(err: ApiError, lang: str, simulated: bool = True) -> str:
    """Translated message for a failed call. A 503 from an analysis without phone-line
    simulation can only come from the audio decoder, so it gets the decoder message."""
    if err.kind == "unavailable" and not simulated:
        key = "err_decoder_unavailable"
    else:
        key = f"err_{err.kind}" if err.kind != "other" else "err_other"
    text = t(key, lang)
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
        st.link_button(
            t("monitor_link", lang), monitor_url(client), help=t("monitor_link_help", lang)
        )
    return lang, detector


def monitor_url(client: Any) -> str:
    """The live call monitor page, as the browser reaches it. In docker compose the UI talks to
    the API by service name (http://api:8000), which the browser cannot reach, so compose sets
    VOXGUARD_MONITOR_URL."""
    return os.environ.get("VOXGUARD_MONITOR_URL", "").strip() or f"{client.base_url}/monitor"


def near_threshold(score: float, threshold: float) -> bool:
    """True for a score at or just above the threshold: threshold <= score < factor x threshold.
    Computed in the UI from the response, so the API and its contract are unchanged."""
    return threshold <= score < threshold * CAUTION_FACTOR


def render_result(result: AnalyzeResponse, lang: str, header_key: str = "result_header") -> None:
    st.subheader(t(header_key, lang))
    if result.codec_applied:
        st.caption(t("result_simulated", lang, codec=result.codec_applied))
    else:
        st.caption(t("result_clean", lang))
    if result.flagged:
        st.error(t("flagged_banner", lang))
        if near_threshold(result.fake_score, result.threshold):
            st.warning(t("caution_near_threshold", lang))
    else:
        st.success(t("not_flagged_banner", lang))
    cols = st.columns(4)
    cols[0].metric(t("m_label", lang), t(f"label_{result.label}", lang))
    # 6 decimals: SSL-AASIST's thresholds are around 0.01, where 4 would hide the difference
    # between a score and the threshold it is compared with.
    cols[1].metric(t("m_score", lang), f"{result.fake_score:.6f}")
    cols[2].metric(t("m_threshold", lang), f"{result.threshold:.6f}")
    cols[3].metric(t("m_latency", lang), t("unit_ms", lang, value=f"{result.latency_ms:.1f}"))
    cols = st.columns(4)
    cols[0].metric(t("m_model", lang), result.model_version)
    cols[1].metric(t("m_flagged", lang), t("yes" if result.flagged else "no", lang))
    cols[2].metric(t("m_codec", lang), result.codec_applied or t("none_value", lang))
    codec_ms = result.codec_ms
    cols[3].metric(
        t("m_codec_ms", lang),
        t("unit_ms", lang, value=f"{codec_ms:.1f}")
        if codec_ms is not None
        else t("none_value", lang),
    )
    st.caption(t("m_session", lang, session_id=result.session_id))
    if result.timeline:
        render_timeline(result, lang, header_key)


SCORE_FLOOR = 1e-4  # log-scale chart: a score of 0 is drawn here


def timeline_spec(result: AnalyzeResponse, lang: str, upto: int | None = None) -> dict:
    """Vega-Lite spec: window scores and the running average (log scale, so a threshold near
    0.01 and scores far above or below it are all readable), the threshold as a dashed line and
    the first moment the running average reached it. Points sit at each window's end time
    ("after N s of audio"). ``upto`` limits the points drawn (replay)."""
    assert result.timeline is not None
    x, y, series = t("timeline_x", lang), t("m_score", lang), t("timeline_series", lang)
    window, running = t("timeline_window", lang), t("timeline_running", lang)
    points = result.timeline[:upto]
    values = [
        {x: p.end_s, series: name, y: max(score, SCORE_FLOOR)}
        for p in points
        for name, score in ((window, p.fake_score), (running, p.running_score))
    ]
    # fixed to the whole recording, so replay does not rescale as points arrive
    every = [s for p in result.timeline for s in (p.fake_score, p.running_score)]
    y_domain = [
        max(min(*every, result.threshold), SCORE_FLOOR) / 1.5,
        min(max(*every, result.threshold) * 1.5, 1.0),
    ]
    x_axis = {
        "field": x,
        "type": "quantitative",
        "scale": {"domain": [0, result.timeline[-1].end_s]},
    }
    layers: list[dict] = [
        {
            "data": {"values": values},
            "mark": {"type": "line", "point": True},
            "encoding": {
                "x": x_axis,
                "y": {
                    "field": y,
                    "type": "quantitative",
                    "scale": {"type": "log", "domain": y_domain},
                },
                "color": {"field": series, "type": "nominal", "sort": [window, running]},
            },
        },
        {
            "data": {"values": [{y: result.threshold, "label": t("timeline_threshold", lang)}]},
            "mark": {"type": "rule", "strokeDash": [6, 4], "color": "#888"},
            "encoding": {
                "y": {"field": y, "type": "quantitative"},
                "tooltip": [{"field": "label"}],
            },
        },
    ]
    if result.first_flag_s is not None and (upto is None or upto >= _flag_index(result)):
        layers.append(
            {
                "data": {"values": [{x: result.first_flag_s}]},
                "mark": {"type": "rule", "color": "#d62728", "strokeDash": [2, 2]},
                "encoding": {"x": {"field": x, "type": "quantitative"}},
            }
        )
    return {"layer": layers, "height": 260}


def _flag_index(result: AnalyzeResponse) -> int:
    """Number of points up to and including the first flagged one."""
    assert result.timeline is not None
    return next(i for i, p in enumerate(result.timeline, start=1) if p.flagged)


def render_timeline(result: AnalyzeResponse, lang: str, key: str) -> None:
    """Window scores and the running average against the threshold, and when a live monitor
    using the same rule would have flagged the call. Replay draws it window by window."""
    assert result.timeline is not None
    st.markdown(f"**{t('timeline_header', lang)}**")
    seconds = f"{result.first_flag_s:.1f}" if result.first_flag_s is not None else ""
    if result.first_flag_s is None:
        st.info(t("timeline_never", lang))
    elif result.flagged:
        st.warning(t("timeline_flagged_at", lang, seconds=seconds))
    else:  # an early alert the rest of the recording did not support
        st.info(t("timeline_flagged_briefly", lang, seconds=seconds))
    st.caption(t("timeline_note", lang))
    chart = st.empty()
    if st.button(t("timeline_replay", lang), key=f"replay_{key}"):
        for n in range(1, len(result.timeline) + 1):
            chart.vega_lite_chart(spec=timeline_spec(result, lang, n), width="stretch")
            time.sleep(REPLAY_STEP_S)
    else:
        chart.vega_lite_chart(spec=timeline_spec(result, lang), width="stretch")


def current_results(digest: str | None) -> dict[str, AnalyzeResponse]:
    """Results that belong to the audio now selected; anything older is discarded."""
    results = st.session_state.get(RESULTS_KEY)
    if digest is None or results is None or results.get("audio") != digest:
        st.session_state.pop(RESULTS_KEY, None)
        return {}
    return {mode: r for mode, r in results.items() if mode != "audio"}


def codec_label(name: str, lang: str) -> str:
    return t("codec_gsm", lang) if name == "gsm" else CODEC_NAMES.get(name, name)


def pick_codec(client: Any, lang: str) -> str | None:
    """Codec dropdown from GET /v1/codecs, offering only codecs this server can run.

    None means no choice was made; the request then uses the server default codec."""
    try:
        info: CodecsResponse = cached("codecs", client.codecs)
    except ApiError:
        st.warning(t("codecs_error", lang))
        return None
    options = [c.name for c in info.codecs if c.available]
    if not options:
        st.warning(t("no_codecs", lang))
        return None
    return st.selectbox(
        t("codec_label", lang),
        options,
        index=options.index(info.default) if info.default in options else 0,
        format_func=lambda name: codec_label(name, lang),
        key="codec",
    )


def render_analyze(client: Any, lang: str) -> None:
    st.header(t("analyze_header", lang))
    caller_id = st.text_input(
        t("caller_label", lang), max_chars=128, help=t("caller_help", lang), key="caller_id"
    )
    simulate = st.toggle(
        t("simulate_label", lang), value=False, help=t("simulate_help", lang), key="simulate_line"
    )
    st.caption(t("simulate_on" if simulate else "simulate_off", lang))
    codec = pick_codec(client, lang) if simulate else None
    show_timeline = st.toggle(
        t("timeline_label", lang), value=False, help=t("timeline_help", lang), key="timeline"
    )

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
    phone = results.get("phone")
    if codec is not None and phone is not None and phone.codec_applied != codec:
        del results["phone"]  # made with another codec: drop it, as for a new clip
        st.session_state[RESULTS_KEY] = {"audio": digest, **results}

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
                        codec=codec,
                        timeline=show_timeline,
                    )
                invalidate("sessions", "blocklist", "alerts", "transactions")
            except ApiError as err:
                st.error(error_text(err, lang, simulated=simulate))
            st.session_state[RESULTS_KEY] = {"audio": digest, **results}

    if "clean" in results and "phone" in results:  # same clip, clean vs phone line
        left, right = st.columns(2)
        with left:
            render_result(results["clean"], lang, "result_header_clean")
        with right:
            render_result(results["phone"], lang, "result_header_phone")
    elif results:
        render_result(next(iter(results.values())), lang)


def report_url(client: Any, session_id: str, lang: str) -> str:
    """The incident report, as the browser reaches it (the API's public address)."""
    base = monitor_url(client).removesuffix("/monitor")
    return f"{base}/v1/sessions/{session_id}/report?lang={lang}"


def render_table(
    rows: list[dict[str, Any]], empty_key: str, lang: str, link_column: str | None = None
) -> None:
    if rows:
        config = (
            {link_column: st.column_config.LinkColumn(display_text=t("col_report", lang))}
            if link_column
            else None
        )
        st.dataframe(rows, hide_index=True, width="stretch", column_config=config)
    else:
        st.info(t(empty_key, lang))


def _monitor(
    lang: str,
    name: str,
    header_key: str,
    fetch: Any,
    to_rows: Any,
    empty_key: str,
    link_column: str | None = None,
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
    render_table(to_rows(items), empty_key, lang, link_column)


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
                t("col_report", lang): report_url(client, a.session_id, lang),
            }
            for a in items
        ]

    def transaction_rows(items: list[Any]) -> list[dict[str, Any]]:
        return [
            {
                t("col_time", lang): fmt_time(d.created_at),
                t("col_amount", lang): round(d.amount_inr, 2),
                t("tx_beneficiary", lang): d.beneficiary or missing,
                t("tx_tier", lang): t(f"tx_tier_{d.tier}", lang),
                t("tx_action", lang): t(f"tx_action_{d.action}", lang),
                t("tx_status", lang): t(f"tx_status_{d.status}", lang),
                t("tx_steps", lang): ", ".join(t(f"tx_method_{m}", lang) for m in d.completed_steps)
                or missing,
                t("col_report", lang): report_url(client, d.session_id, lang),
            }
            for d in items
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
            link_column=t("col_report", lang),
        )
    with tabs[3]:
        _monitor(
            lang,
            "transactions",
            "transactions_header",
            lambda: client.transactions(limit=TRANSACTIONS_LIMIT),
            transaction_rows,
            "empty_transactions",
            link_column=t("col_report", lang),
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
        [
            t(k, lang)
            for k in (
                "tab_analyze",
                "tab_sessions",
                "tab_blocklist",
                "tab_alerts",
                "tab_transactions",
            )
        ]
    )
    with tabs[0]:
        render_analyze(client, lang)
    render_monitoring(client, lang, tabs[1:])


main()
