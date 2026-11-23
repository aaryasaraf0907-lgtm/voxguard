# ruff: noqa: F811  (pytest fixtures imported from browser_stack are used as arguments)
"""The live call monitor page in headless Chromium against the real API (stub detector): a fake
microphone (Chromium plays a generated WAV as the mic), a played file, the phone line, Hindi,
and the link from the Streamlit sidebar. Skipped without the e2e dependency group."""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

if os.environ.get("VOXGUARD_E2E_TESTS") != "1":
    pytest.skip("set VOXGUARD_E2E_TESTS=1 (browser tests)", allow_module_level=True)
pytest.importorskip("playwright.sync_api", reason="needs the e2e dependency group")
if shutil.which("ffmpeg") is None:
    pytest.skip("browser tests need ffmpeg (phone line)", allow_module_level=True)

from browser_stack import browser, page, playwright, stack  # noqa: E402, F401
from playwright.sync_api import Page, expect  # noqa: E402

from voxguard.ui.i18n import EN, HI  # noqa: E402

pytestmark = pytest.mark.e2e


@pytest.fixture(scope="session")
def mic_wav(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("mic") / "mic.wav"
    x = (np.random.default_rng(7).standard_normal(16_000 * 12) * 0.1).astype("float32")
    sf.write(path, x, 16_000, subtype="PCM_16")
    return path


@pytest.fixture(scope="session")
def mic_browser(mic_wav: Path) -> Iterator[Any]:
    """A second Chromium with a fake microphone (the process's shared Playwright instance)."""
    b = playwright().chromium.launch(
        args=[
            "--use-fake-ui-for-media-stream",  # grant the microphone without a prompt
            "--use-fake-device-for-media-stream",
            f"--use-file-for-fake-audio-capture={mic_wav}",
            "--autoplay-policy=no-user-gesture-required",
        ]
    )
    try:
        yield b
    finally:
        b.close()


@pytest.fixture
def monitor(mic_browser: Any, stack: Any, request: pytest.FixtureRequest) -> Iterator[Page]:
    context = mic_browser.new_context(viewport={"width": 1100, "height": 1200})
    pg = context.new_page()
    pg.set_default_timeout(30_000)
    errors: list[str] = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    failed_before = request.session.testsfailed
    pg.goto(f"{stack.api_url}/monitor")
    try:
        yield pg
        assert errors == [], errors  # no JavaScript errors on the page
    finally:
        if request.session.testsfailed > failed_before:
            name = re.sub(r"[^A-Za-z0-9_.-]+", "_", request.node.name)
            with contextlib.suppress(Exception):
                pg.screenshot(path=str(stack.logs / f"{name}.png"), full_page=True)
        context.close()


def sessions(stack: Any) -> list[dict]:
    with urllib.request.urlopen(f"{stack.api_url}/v1/sessions?limit=50", timeout=10) as r:
        return json.loads(r.read())


def wait_final(pg: Page) -> str:
    expect(pg.locator("#final")).to_contain_text(re.compile(r"[0-9a-f]{32}"), timeout=30_000)
    return pg.locator("#final").inner_text()


def test_microphone_call_scores_live_and_is_stored(monitor: Page, stack: Any) -> None:
    monitor.locator("#caller").fill("+91-E2E-MIC")
    monitor.get_by_role("button", name=EN["monitor_start_mic"]).click()
    expect(monitor.locator("#threshold")).not_to_have_text("–")
    expect(monitor.locator("#window")).not_to_have_text("–", timeout=20_000)  # first window
    banner = monitor.locator("#banner")
    expect(banner).to_have_class(re.compile(r"safe|alert|info"))
    monitor.get_by_role("button", name=EN["monitor_stop"]).click()
    text = wait_final(monitor)
    assert "Call ended" in text
    (row,) = [s for s in sessions(stack) if s["caller_id"] == "+91-E2E-MIC"]
    assert row["session_id"] in text


def test_a_played_recording_is_monitored_on_the_phone_line(monitor: Page, stack: Any) -> None:
    monitor.locator("#caller").fill("+91-E2E-FILE")
    monitor.locator("#codec").select_option("g711_mulaw")
    monitor.locator("#file").set_input_files(str(stack.clips.near))
    monitor.get_by_role("button", name=EN["monitor_play"]).click()
    wait_final(monitor)  # the recording ends by itself
    (row,) = [s for s in sessions(stack) if s["caller_id"] == "+91-E2E-FILE"]
    assert row["codec_applied"] == "g711_mulaw"


def test_consent_notice_is_shown_and_guidance_waits_for_an_alert(monitor: Page) -> None:
    expect(monitor.locator("#consent")).to_have_text(EN["monitor_consent"])
    expect(monitor.locator("#guidance")).to_be_hidden()  # shown once an alert has been raised


def test_a_transfer_is_checked_against_the_call_and_verified(monitor: Page, stack: Any) -> None:
    """Approve a transfer for a monitored call: the decision follows the policy; any verification
    it asks for can be recorded until the transfer is released; the report opens."""
    monitor.locator("#file").set_input_files(str(stack.clips.near))
    monitor.get_by_role("button", name=EN["monitor_play"]).click()
    wait_final(monitor)
    monitor.locator("#amount").fill("75000")
    monitor.get_by_role("button", name=EN["monitor_tx_approve"]).click()
    decision = monitor.locator("#decision")
    expect(decision).to_have_class(re.compile(r"pending|held"))  # the e2e policy: never allowed
    for n in range(3):  # at most two steps (a hold, by two different people), then released
        if re.search(r"allowed|released", decision.get_attribute("class") or ""):
            break
        monitor.locator("#verifier").fill(f"E2E Tester {n}")  # shown only when steps are needed
        monitor.locator("#step-buttons button:enabled").first.click()
        monitor.wait_for_timeout(300)
    expect(decision).to_have_class("released")
    href = monitor.locator("#report").get_attribute("href")
    assert href and "/report?lang=en" in href
    with urllib.request.urlopen(f"{stack.api_url}{href}", timeout=10) as r:
        assert r.status == 200 and EN["report_title"] in r.read().decode()
    with urllib.request.urlopen(f"{stack.api_url}/v1/transactions?limit=5", timeout=10) as r:
        assert json.loads(r.read())[0]["amount_inr"] == 75000


def test_the_page_switches_to_hindi(monitor: Page) -> None:
    monitor.locator("#lang").select_option("hi")
    expect(monitor.locator("h1")).to_have_text(HI["monitor_title"])
    expect(monitor.get_by_role("button", name=HI["monitor_start_mic"])).to_be_visible()


def test_the_streamlit_sidebar_links_to_the_monitor(page: Page, stack: Any) -> None:
    page.goto(stack.ui_url)
    link = page.get_by_role("link", name=EN["monitor_link"])
    expect(link).to_have_attribute("href", f"{stack.api_url}/monitor")
