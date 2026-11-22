"""The spoken-language check (VOXGUARD_LANGUAGE_CHECK): a stand-in checker, no Whisper model."""

from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import FixedDetector, make_wav
from voxguard.api.deps import Settings
from voxguard.api.main import create_app
from voxguard.inference import Prediction
from voxguard.language import LanguageResult

THR = 0.5


class _Checker:
    """Says what it is told; records how often it ran and how much audio it saw."""

    def __init__(self, code: str = "ta", supported: bool = False, fail: bool = False) -> None:
        self.code, self.supported, self.fail, self.calls = code, supported, fail, []

    def check(self, waveform: np.ndarray, sample_rate: int) -> LanguageResult | None:
        self.calls.append(len(waveform) / sample_rate)
        if self.fail:
            raise RuntimeError("model broke")
        return LanguageResult(self.code, 0.9 if self.supported else 0.02, self.supported)


def _client(score: float = 0.9, checker: object | None = None) -> TestClient:
    det = FixedDetector(score, threshold=THR)
    return TestClient(create_app(det, Settings(db_url="sqlite://"), language_checker=checker))


def _upload(c, caller: str | None = None) -> dict:
    data = {"caller_id": caller} if caller else {}
    r = c.post("/v1/analyze", files={"file": ("a.wav", make_wav(3.0))}, data=data)
    assert r.status_code == 200, r.text
    return r.json()


def test_unsupported_language_never_switches_off_a_warning() -> None:
    # Aarya's #125 review: a cloned voice speaking Tamil and scoring 0.9 must still be flagged,
    # alerted and blocklisted; the language note is only added to it.
    with _client(0.9, _Checker("ta", supported=False)) as c:
        r = _upload(c, caller="+911234")
        alerts = c.get("/v1/alerts").json()
        blocklist = c.get("/v1/blocklist").json()
        report = c.get(f"/v1/sessions/{r['session_id']}/report").text
        tx = c.post("/v1/transactions/check",
                    json={"session_id": r["session_id"], "amount_inr": 1000}).json()  # fmt: skip
    assert r["label"] == "fake" and r["flagged"] is True
    assert r["language"] == "ta" and r["language_supported"] is False
    assert r["language_name"] in ("Tamil", "ta")
    assert len(alerts) == 1 and blocklist[0]["caller_id"] == "+911234"
    assert "Flagged" in report and "Language not supported yet" in report
    assert "less reliable" in report and tx["tier"] == "high"


def test_unsupported_language_below_the_threshold_says_so_but_keeps_the_verdict() -> None:
    with _client(0.1, _Checker("mr", supported=False)) as c:
        r = _upload(c, caller="+911234")
        report = c.get(f"/v1/sessions/{r['session_id']}/report").text
        alerts = c.get("/v1/alerts").json()
    assert r["label"] == "real" and r["flagged"] is False and alerts == []
    assert r["language_supported"] is False
    assert "Not flagged" in report and "Language not supported yet" in report


def test_supported_language_keeps_the_normal_verdict() -> None:
    with _client(0.9, _Checker("en", supported=True)) as c:
        r = _upload(c, caller="+911234")
        report = c.get(f"/v1/sessions/{r['session_id']}/report").text
        alerts = c.get("/v1/alerts").json()
    assert r["label"] == "fake" and r["flagged"] is True
    assert r["language"] == "en" and r["language_supported"] is True
    assert len(alerts) == 1 and "Spoken language" in report


@pytest.mark.parametrize("checker", [None, _Checker(fail=True)])
def test_no_check_or_a_failed_check_gives_the_normal_verdict(checker) -> None:
    with _client(0.9, checker) as c:
        r = _upload(c)
    assert r["label"] == "fake" and r["flagged"] is True
    assert r["language"] is None and r["language_supported"] is None


def test_the_check_sees_the_original_audio_not_the_phone_line() -> None:
    checker = _Checker("en", supported=True)
    with _client(0.1, checker) as c:
        c.post("/v1/analyze", files={"file": ("a.wav", make_wav(3.0, sr=44_100))})
    assert checker.calls == [pytest.approx(3.0, abs=0.05)]


@pytest.mark.parametrize(
    ("raw", "on"), [("", False), ("off", False), ("0", False), ("on", True), ("TRUE", True)]
)
def test_setting(monkeypatch, raw: str, on: bool) -> None:
    monkeypatch.setenv("VOXGUARD_LANGUAGE_CHECK", raw)
    assert Settings.from_env().language_check is on


def test_bad_setting_fails_fast(monkeypatch) -> None:
    monkeypatch.setenv("VOXGUARD_LANGUAGE_CHECK", "maybe")
    with pytest.raises(ValueError, match="VOXGUARD_LANGUAGE_CHECK"):
        Settings.from_env()


# --- live call ------------------------------------------------------------------------------


class _High(FixedDetector):
    def __init__(self) -> None:
        super().__init__(0.95, threshold=THR)

    def predict(self, waveform: np.ndarray, sample_rate: int) -> Prediction:
        return Prediction(0.95, "fake", THR, 1.0, self.model_version)


_AUDIO = (np.random.default_rng(0).standard_normal(16_000 * 12) * 0.1).astype("<f4")


def _live(checker: object | None) -> tuple[list[dict], dict]:
    app = create_app(_High(), Settings(db_url="sqlite://"), language_checker=checker)
    with TestClient(app) as c:
        with c.websocket_connect("/v1/stream?sample_rate=16000") as ws:
            msgs = [ws.receive_json()]  # ready
            for i in range(0, len(_AUDIO), 4000):
                ws.send_bytes(_AUDIO[i : i + 4000].tobytes())
            ws.send_text(json.dumps({"type": "end"}))
            while msgs[-1]["type"] != "final":
                msgs.append(ws.receive_json())
        alerts = c.get("/v1/alerts").json()
    return msgs, {"alerts": alerts}


def test_live_unsupported_language_is_reported_and_still_alerts() -> None:
    checker = _Checker("mr", supported=False)
    msgs, rest = _live(checker)
    kinds = [m["type"] for m in msgs]
    lang = next(m for m in msgs if m["type"] == "language")
    final = msgs[-1]
    assert lang["language"] == "mr" and lang["language_supported"] is False
    # checked once, on the first 6 s, before the earliest possible alert (two windows, 6.25 s)
    assert len(checker.calls) == 1 and checker.calls[0] == pytest.approx(6.0, abs=0.3)
    assert kinds.index("language") < kinds.index("alert")
    assert "alert" in kinds and len(rest["alerts"]) == 1  # the warning is never switched off
    assert final["flagged"] is True and final["language_supported"] is False


def test_live_supported_language_alerts_as_before() -> None:
    msgs, rest = _live(_Checker("hi", supported=True))
    kinds = [m["type"] for m in msgs]
    assert "alert" in kinds and msgs[-1]["flagged"] is True and len(rest["alerts"]) == 1
    assert msgs[-1]["language"] == "hi" and msgs[-1]["language_supported"] is True
