# ruff: noqa: E501  (the template below is HTML, not Python)
"""Incident report for one session (``GET /v1/sessions/{id}/report?lang=en|hi``).

A printable, self-contained HTML page: the verdict, the live call's per-window scores as a
chart and a table, every transfer decision with its verification steps, how to report the
fraud (1930, cybercrime.gov.in) and a statement that no audio was kept. Nothing is loaded from
the network. Every value is HTML-escaped; strings come from ``voxguard.ui.i18n`` (``report_*``
keys, plus a few shared ones).
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from html import escape

from voxguard.api.db import SessionRow
from voxguard.schemas import CallWindow, TransactionDecision
from voxguard.ui.i18n import t

FLOOR = 1e-4


def _fmt_time(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def _chart(windows: list[CallWindow], threshold: float) -> str:
    """Inline SVG: window scores and the running average on a log scale, with the threshold."""
    w, h, pad = 720, 220, 44
    values = [v for p in windows for v in (p.window_score, p.running_score)] + [threshold]
    lo = max(min(values), FLOOR) / 1.5
    hi = min(max(values) * 1.5, 1.0)
    t_max = max(10.0, max(p.t_s for p in windows))

    def x(s: float) -> float:
        return pad + (w - pad - 10) * s / t_max

    def y(v: float) -> float:
        a, b, c = math.log10(max(v, FLOOR)), math.log10(lo), math.log10(hi)
        return h - 24 - (h - 34) * (a - b) / (c - b)

    def line(key: str, colour: str) -> str:
        pts = " ".join(f"{x(p.t_s):.1f},{y(getattr(p, key)):.1f}" for p in windows)
        return f'<polyline fill="none" stroke="{colour}" stroke-width="2" points="{pts}"/>'

    ty = y(threshold)
    return (
        f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="scores">'
        f'<rect x="0" y="0" width="{w}" height="{h}" fill="#fff" stroke="#e5e7eb"/>'
        f'<line x1="{pad}" x2="{w - 10}" y1="{ty:.1f}" y2="{ty:.1f}" stroke="#888" stroke-dasharray="6 4"/>'
        f"{line('window_score', '#1f6feb')}{line('running_score', '#7cb7ff')}"
        f'<text x="4" y="{ty + 4:.1f}" font-size="11" fill="#555">{threshold:.4f}</text>'
        f'<text x="{pad}" y="{h - 6}" font-size="11" fill="#555">0 s</text>'
        f'<text x="{w - 40}" y="{h - 6}" font-size="11" fill="#555">{t_max:.0f} s</text>'
        "</svg>"
    )


def _evidence(check: dict, session_id: str, lang: str) -> str:
    e = escape
    if check["ok"]:
        status = f'<p class="ok">{e(t("report_evidence_ok", lang, entries=check["entries"]))}</p>'
    else:
        items = "".join(
            f"<li>#{p['seq']} {e(p['kind'])} {e(p['ref'][:12])}: {e(p['problem'])}</li>"
            for p in check["problems"]
        )
        status = f'<p class="bad">{e(t("report_evidence_problem", lang))}</p><ul>{items}</ul>'
    rows = "".join(
        f"<tr><td>{x['seq']}</td><td>{e(t('report_kind_' + x['kind'], lang))}</td>"
        f"<td>{e(_fmt_time(x['created_at']))}</td><td><code>{e(x['hash'][:16])}…</code></td></tr>"
        for x in check.get("session_entries", [])
    )
    table = (
        f"<table><tr><th>#</th><th>{e(t('report_kind', lang))}</th><th>{e(t('col_time', lang))}</th>"
        f"<th>{e(t('report_hash', lang))}</th></tr>{rows}</table>"
        if rows
        else ""
    )
    verify = f"/v1/evidence/verify?session_id={session_id}"
    return (
        f"{status}{table}<p>{e(t('report_evidence_head', lang))}: <code>{e(check['head'])}</code></p>"
        f'<p class="muted">{e(t("report_evidence_note", lang, url=verify))}</p>'
    )


def _actions(lang: str) -> str:
    """Download (the same report as an HTML file) and Print, which also saves a PDF."""
    e = escape
    return (
        '<p class="actions">'
        f'<a href="?lang={e(lang)}&amp;download=true" download>{e(t("report_download", lang))}</a>'
        f'<button type="button" onclick="window.print()">{e(t("report_print", lang))}</button>'
        "</p>"
    )


def render_report(
    row: SessionRow,
    windows: list[CallWindow],
    decisions: list[TransactionDecision],
    lang: str,
    check: dict | None = None,
    actions: bool = True,
    language: tuple[str, bool] | None = None,
) -> str:
    """``actions`` adds the Download and Print buttons (left out of the downloaded file itself)."""
    e = escape
    verdict = t("report_flagged" if row.flagged else "report_not_flagged", lang)
    # speech clearly not in English or Hindi (voxguard/language.py): a note next to the verdict,
    # which is unchanged (a flagged call stays flagged, with its alert and blocklist entry)
    lang_note = ""
    if language is not None and not language[1]:
        lang_note = (
            f'<p class="langnote">{e(t("monitor_lang_unsupported", lang, language=language[0]))} '
            f"{e(t('monitor_lang_unsupported_advice', lang))}</p>"
        )
    facts = [
        (t("report_session", lang), row.session_id),
        (t("report_time", lang), _fmt_time(row.created_at)),
        (t("col_caller", lang), row.caller_id or t("missing_value", lang)),
        (
            t("report_source", lang),
            t("report_source_live" if windows else "report_source_upload", lang),
        ),
        (t("m_score", lang), f"{row.fake_score:.6f}"),
        (t("m_threshold", lang), f"{row.threshold:.6f}"),
        (t("m_codec", lang), row.codec_applied or t("none_value", lang)),
        (t("m_model", lang), row.model_version),
    ]
    if language is not None:  # only when the language check ran (VOXGUARD_LANGUAGE_CHECK)
        name, supported = language
        shown = name if supported else t("report_language_unsupported", lang, language=name)
        facts.append((t("report_language", lang), shown))
    facts_html = "".join(f"<tr><th>{e(k)}</th><td>{e(str(v))}</td></tr>" for k, v in facts)
    if windows:
        flag_at = next((p.t_s for p in windows if p.running_score >= row.threshold), None)
        rows = "".join(
            f"<tr><td>{p.t_s:.1f}</td><td>{p.window_score:.6f}</td><td>{p.running_score:.6f}</td></tr>"
            for p in windows
        )
        first = (
            t("report_first_over", lang, seconds=f"{flag_at:.1f}")
            if flag_at is not None
            else t("report_never_over", lang)
        )
        timeline = (
            f"<p>{e(first)}</p>{_chart(windows, row.threshold)}"
            f"<table><tr><th>{e(t('monitor_seconds', lang))}</th><th>{e(t('monitor_window', lang))}</th>"
            f"<th>{e(t('monitor_running', lang))}</th></tr>{rows}</table>"
        )
    else:
        timeline = f"<p>{e(t('report_no_timeline', lang))}</p>"
    if decisions:
        items = []
        for d in decisions:
            steps = ", ".join(t(f"tx_method_{m}", lang) for m in d.completed_steps) or t(
                "none_value", lang
            )
            items.append(
                "<tr>"
                f"<td>{e(_fmt_time(d.created_at))}</td><td>₹{d.amount_inr:,.2f}</td>"
                f"<td>{e(d.beneficiary or t('missing_value', lang))}</td>"
                f"<td>{e(t(f'tx_tier_{d.tier}', lang))}</td><td>{e(t(f'tx_action_{d.action}', lang))}</td>"
                f"<td>{e(t(f'tx_status_{d.status}', lang))}</td><td>{e(steps)}</td>"
                "</tr>"
            )
        head = "".join(
            f"<th>{e(t(k, lang))}</th>"
            for k in (
                "col_time",
                "tx_amount",
                "tx_beneficiary",
                "tx_tier",
                "tx_action",
                "tx_status",
                "tx_steps",
            )
        )
        transfers = f"<table><tr>{head}</tr>{''.join(items)}</table>"
    else:
        transfers = f"<p>{e(t('report_no_transfers', lang))}</p>"
    guidance = "".join(
        f"<li>{e(t(k, lang))}</li>"
        for k in (
            "monitor_guidance_callback",
            "monitor_guidance_question",
            "monitor_guidance_report",
        )
    )
    return f"""<!doctype html>
<html lang="{e(lang)}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{e(t("report_title", lang))} {e(row.session_id[:8])}</title>
<style>
  body {{ font: 15px/1.5 system-ui, "Segoe UI", "Noto Sans", "Noto Sans Devanagari", sans-serif; color:#1f2430; margin:0; background:#fff; }}
  main {{ max-width: 860px; margin: 0 auto; padding: 24px 16px 48px; }}
  h1 {{ font-size: 1.5rem; margin: 0 0 4px; }} h2 {{ font-size: 1.1rem; margin: 24px 0 8px; }}
  .verdict {{ padding: 12px 14px; border-radius: 8px; background: {"#fee2e2" if row.flagged else "#dcfce7"}; }}
  .langnote {{ padding: 10px 14px; border-radius: 8px; background: #fef3c7; }}
  table {{ border-collapse: collapse; width: 100%; margin: 8px 0; font-size: .92rem; }}
  th, td {{ border: 1px solid #e5e7eb; padding: 4px 8px; text-align: left; vertical-align: top; }}
  th {{ background: #f6f7f9; }}
  .muted {{ color: #6b7280; font-size: .88rem; }}
  .ok {{ color: #14532d; font-weight: 600; }} .bad {{ color: #7f1d1d; font-weight: 600; }}
  code {{ font-size: .85rem; word-break: break-all; }}
  .actions {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 8px 0 16px; }}
  .actions a, .actions button {{ font: inherit; padding: 6px 12px; border: 1px solid #c7ccd4;
    border-radius: 6px; background: #f6f7f9; color: #1f2430; text-decoration: none; cursor: pointer; }}
  @media print {{ main {{ padding: 0; }} .actions {{ display: none; }} }}
</style>
</head>
<body><main>
<h1>{e(t("report_title", lang))}</h1>
<p class="muted">{e(t("report_generated", lang, time=_fmt_time(datetime.now(UTC))))}</p>
{_actions(lang) + chr(10) if actions else ""}<p class="verdict"><strong>{e(verdict)}</strong></p>
{lang_note}
<table>{facts_html}</table>
<h2>{e(t("report_timeline", lang))}</h2>
{timeline}
<h2>{e(t("report_transfers", lang))}</h2>
{transfers}
<h2>{e(t("monitor_guidance_title", lang))}</h2>
<ul>{guidance}</ul>
<h2>{e(t("report_evidence", lang))}</h2>
{_evidence(check, row.session_id, lang) if check else ""}
<p class="muted">{e(t("report_privacy", lang))}</p>
<p class="muted">{e(t("report_disclaimer", lang))}</p>
</main></body>
</html>
"""
