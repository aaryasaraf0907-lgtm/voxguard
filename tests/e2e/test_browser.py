# ruff: noqa: F811  (pytest fixtures imported from browser_stack are used as arguments)
"""Browser end-to-end checks of the Streamlit demo against the real API (stub detector).

Headless Chromium via Playwright. Opt-in: ``VOXGUARD_E2E_TESTS=1`` (``make e2e``, CI job
``e2e``). The stub's scores are hash placeholders; the threshold is set from them (conftest),
so these checks test the application's behaviour, never detection accuracy.
"""

from __future__ import annotations

import json
import os
import shutil
import urllib.request
from typing import Any

import pytest

if os.environ.get("VOXGUARD_E2E_TESTS") != "1":
    pytest.skip("set VOXGUARD_E2E_TESTS=1 (browser tests)", allow_module_level=True)
pytest.importorskip("playwright.sync_api", reason="needs the e2e dependency group")
if shutil.which("ffmpeg") is None:
    pytest.skip("browser tests need ffmpeg (phone line, M4A)", allow_module_level=True)

from browser_stack import browser, page, stack  # noqa: E402, F401
from playwright.sync_api import Page, expect  # noqa: E402

from voxguard.ui.app import CODEC_NAMES  # noqa: E402
from voxguard.ui.i18n import EN, HI  # noqa: E402

pytestmark = pytest.mark.e2e
RESULT_TIMEOUT_MS = 60_000


def squash(text: str) -> str:
    return " ".join(text.split())


def open_app(page: Page, url: str, lang: dict[str, str] = EN) -> None:
    page.goto(url)
    expect(page.get_by_text(lang["status_connected"], exact=True)).to_be_visible()


def upload(page: Page, path: Any) -> None:
    page.locator('input[type="file"]').first.set_input_files(str(path))
    expect(page.get_by_role("button", name=EN["analyze_button"], exact=True)).to_be_visible()


def analyze(page: Page, lang: dict[str, str] = EN) -> None:
    page.get_by_role("button", name=lang["analyze_button"], exact=True).click()


def metric(page: Page, label: str) -> list[str]:
    return (
        page.locator('[data-testid="stMetric"]')
        .filter(has_text=label)
        .locator('[data-testid="stMetricValue"]')
        .all_inner_texts()
    )


def wait_scores(page: Page, n: int, lang: dict[str, str] = EN) -> list[str]:
    expect(page.locator('[data-testid="stMetric"]').filter(has_text=lang["m_score"])).to_have_count(
        n, timeout=RESULT_TIMEOUT_MS
    )
    return metric(page, lang["m_score"])


def toggle_phone_line(page: Page) -> None:
    page.get_by_text(EN["simulate_label"], exact=True).click()


def api_get(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read())


def test_connects_and_says_the_detector_is_a_stub(page: Page, stack: Any) -> None:
    open_app(page, stack.ui_url)
    sidebar = page.locator('[data-testid="stSidebar"]')
    expect(sidebar.get_by_text("stub", exact=False).first).to_be_visible()
    expect(page.get_by_text(squash(EN["stub_notice"]), exact=True)).to_be_visible()


def test_near_threshold_flag_shows_the_caution(page: Page, stack: Any) -> None:
    open_app(page, stack.ui_url)
    upload(page, stack.clips.near)
    analyze(page)
    wait_scores(page, 1)
    expect(page.get_by_text(EN["flagged_banner"], exact=True)).to_be_visible()
    expect(page.get_by_text(squash(EN["caution_near_threshold"]), exact=True)).to_be_visible()
    assert metric(page, EN["m_threshold"]) == [f"{stack.clips.threshold:.6f}"]


def test_clear_flag_has_no_caution_and_low_score_is_not_flagged(page: Page, stack: Any) -> None:
    open_app(page, stack.ui_url)
    upload(page, stack.clips.clear)
    analyze(page)
    wait_scores(page, 1)
    expect(page.get_by_text(EN["flagged_banner"], exact=True)).to_be_visible()
    expect(page.get_by_text(squash(EN["caution_near_threshold"]), exact=True)).to_have_count(0)

    upload(page, stack.clips.below)  # a new clip discards the previous result
    expect(page.get_by_text(EN["flagged_banner"], exact=True)).to_have_count(0)
    analyze(page)
    wait_scores(page, 1)
    expect(page.get_by_text(EN["not_flagged_banner"], exact=True)).to_be_visible()
    expect(page.get_by_text(squash(EN["caution_near_threshold"]), exact=True)).to_have_count(0)


def test_phone_line_side_by_side_and_codec_change_drops_it(page: Page, stack: Any) -> None:
    open_app(page, stack.ui_url)
    upload(page, stack.clips.below)
    analyze(page)
    wait_scores(page, 1)
    toggle_phone_line(page)
    analyze(page)
    wait_scores(page, 2)
    expect(page.get_by_text(EN["result_header_clean"], exact=True)).to_be_visible()
    expect(page.get_by_text(EN["result_header_phone"], exact=True)).to_be_visible()
    assert metric(page, EN["m_codec"]) == [EN["none_value"], "g711_mulaw"]

    page.get_by_role("combobox", name=EN["codec_label"]).click()
    page.get_by_role("option", name=CODEC_NAMES["g711_alaw"]).click()
    wait_scores(page, 1)  # the G.711 µ-law phone result is gone; the original one stays
    expect(page.get_by_text(EN["result_header_phone"], exact=True)).to_have_count(0)


def test_m4a_upload_is_analyzed(page: Page, stack: Any) -> None:
    open_app(page, stack.ui_url)
    upload(page, stack.clips.m4a)
    analyze(page)
    assert len(wait_scores(page, 1)) == 1
    expect(page.get_by_text(EN["err_unsupported"], exact=True)).to_have_count(0)


def test_flagged_caller_reaches_sessions_blocklist_and_alerts(page: Page, stack: Any) -> None:
    caller = "+91-E2E-0001"
    open_app(page, stack.ui_url)
    page.get_by_role("textbox", name=EN["caller_label"]).fill(caller)
    upload(page, stack.clips.clear)
    session = page.get_by_text(EN["m_session"].split("{")[0].strip())
    seen: list[str] = []
    for _ in range(2):
        analyze(page)
        # wait for a new session ID: the previous result stays on screen until the new one arrives
        expect(session).not_to_have_text(seen[-1] if seen else "", timeout=RESULT_TIMEOUT_MS)
        seen.append(session.inner_text())
        expect(page.get_by_text(EN["flagged_banner"], exact=True)).to_be_visible()
    assert len(set(seen)) == 2

    hits = {e["caller_id"]: e["hit_count"] for e in api_get(f"{stack.api_url}/v1/blocklist")}
    assert hits.get(caller) == 2
    for tab, empty in (
        ("tab_sessions", "empty_sessions"),
        ("tab_blocklist", "empty_blocklist"),
        ("tab_alerts", "empty_alerts"),
    ):
        page.get_by_role("tab", name=EN[tab]).click()
        panel = page.get_by_role("tabpanel", name=EN[tab])
        expect(panel.locator('[data-testid="stDataFrame"]')).to_be_visible()
        expect(panel.get_by_text(EN[empty])).to_have_count(0)


def test_hindi_switch_keeps_the_result(page: Page, stack: Any) -> None:
    open_app(page, stack.ui_url)
    upload(page, stack.clips.near)
    analyze(page)
    [score] = wait_scores(page, 1)
    page.locator('[data-testid="stSidebar"] [data-testid="stSelectbox"]').first.click()
    page.get_by_role("option", name="हिन्दी").click()
    expect(page.get_by_text(HI["status_connected"], exact=True)).to_be_visible()
    assert wait_scores(page, 1, HI) == [score]
    expect(page.get_by_text(HI["flagged_banner"], exact=True)).to_be_visible()
    expect(page.get_by_text(squash(HI["caution_near_threshold"]), exact=True)).to_be_visible()


@pytest.mark.parametrize(
    ("clip", "message"), [("corrupt", "err_unsupported"), ("too_long", "err_invalid")]
)
def test_rejected_uploads_show_their_message(
    page: Page, stack: Any, clip: str, message: str
) -> None:
    open_app(page, stack.ui_url)
    upload(page, getattr(stack.clips, clip))
    analyze(page)
    message_text = page.get_by_text(squash(EN[message]), exact=True)
    expect(message_text).to_be_visible(timeout=RESULT_TIMEOUT_MS)
    expect(page.locator('[data-testid="stMetric"]')).to_have_count(0)


def test_narrow_screen_has_no_horizontal_scroll(browser: Any, stack: Any) -> None:
    context = browser.new_context(viewport={"width": 375, "height": 812})
    try:
        page = context.new_page()
        page.set_default_timeout(30_000)
        page.goto(stack.ui_url)  # the sidebar (with "Connected") is collapsed at this width
        expect(page.get_by_text(EN["analyze_header"], exact=True)).to_be_visible()
        upload(page, stack.clips.below)
        analyze(page)
        wait_scores(page, 1)
        overflow = page.evaluate(
            "document.documentElement.scrollWidth - document.documentElement.clientWidth"
        )
        assert overflow <= 0
    finally:
        context.close()


def test_api_down_then_back(page: Page, stack: Any) -> None:
    """Last on purpose: stops and restarts the API the other tests share."""
    open_app(page, stack.ui_url)
    stack.stop_api()
    try:
        page.reload()  # a new browser session: no cached /health
        expect(page.get_by_text(EN["status_unavailable"], exact=True)).to_be_visible()
    finally:
        stack.start_api()
    page.reload()
    expect(page.get_by_text(EN["status_connected"], exact=True)).to_be_visible()
