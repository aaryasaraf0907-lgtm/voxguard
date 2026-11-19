"""scripts/stream_call.py: URL safety, framing, the printed messages, and one call end to end
against the real app (stub detector) under uvicorn. Generated audio only."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("stream_call", ROOT / "scripts/stream_call.py")
assert SPEC and SPEC.loader
sc = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sc
SPEC.loader.exec_module(sc)


@pytest.mark.parametrize(
    "url", ["http://127.0.0.1:8000/v1/stream", "ws://example.com/v1/stream", "ws:///x"]
)
def test_bad_or_remote_urls_are_refused(url: str) -> None:
    with pytest.raises(sc.UsageError):
        sc.check_url(url, allow_remote=False)


def test_local_urls_and_explicit_remote_are_accepted() -> None:
    assert sc.check_url("ws://localhost:8000/v1/stream", False)
    assert sc.check_url("wss://example.com/v1/stream", True)


def test_frames_are_quarter_seconds_and_cover_everything() -> None:
    x = np.arange(48_000 * 2 + 5, dtype=np.float32)
    chunks = sc.frames(x, 48_000)
    assert {len(c) for c in chunks[:-1]} == {12_000} and len(chunks[-1]) == 5
    assert np.array_equal(np.concatenate(chunks), x)


def test_stream_url_sends_only_what_was_given() -> None:
    assert (
        sc.stream_url("ws://h/v1/stream", 16_000, None, None)
        == "ws://h/v1/stream?sample_rate=16000"
    )
    url = sc.stream_url("ws://h/v1/stream", 8000, "g711_mulaw", "+91 1")
    assert "codec=g711_mulaw" in url and "caller_id=%2B91+1" in url


def test_messages_are_readable() -> None:
    alert = {"type": "alert", "t_s": 6.25, "running_score": 0.9, "threshold": 0.5}
    assert "ALERT" in sc.describe({**alert, "level": "high"}) and "6.2" in sc.describe(alert)
    assert "CAUTION" in sc.describe({**alert, "level": "caution"})
    final = {
        "type": "final",
        "stored": True,
        "flagged": False,
        "fake_score": 0.1,
        "threshold": 0.5,
        "windows": 3,
        "session_id": "s1",
    }
    assert "not flagged" in sc.describe(final) and "s1" in sc.describe(final)


def test_a_call_end_to_end(live_server, tmp_path: Path) -> None:
    base, _ = live_server
    wav = tmp_path / "call.wav"
    sf.write(
        wav, (np.random.default_rng(2).standard_normal(16_000 * 9) * 0.1).astype("float32"), 16_000
    )
    out = tmp_path / "m.json"
    code = sc.main(
        [
            str(wav),
            "--url",
            f"{base}/v1/stream",
            "--speed",
            "0",
            "--caller-id",
            "cli",
            "--json",
            str(out),
        ]
    )
    report = json.loads(out.read_text())
    assert (
        code == 0
        and report["final"]["stored"]
        and report["final"]["windows"] == len(report["scores"]) == 3
    )
    assert all(
        s["arrived_after_ms"] is not None and s["arrived_after_ms"] >= 0 for s in report["scores"]
    )
    assert report["duration_s"] == pytest.approx(9.0)
