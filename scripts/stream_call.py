"""Stream a recording to the live call monitor (WS /v1/stream) as if it were a call in progress.

    python scripts/stream_call.py call.wav [--url ws://127.0.0.1:8000/v1/stream]
        [--codec g711_mulaw] [--caller-id +91-DEMO-0001] [--speed 1] [--json out.json]

Sends 0.25 s frames of mono float32 audio at real speed (``--speed 2`` sends twice as fast,
``--speed 0`` as fast as possible) and prints each message as it arrives: the scores every 2 s,
the alert the moment the running average reaches the threshold, and the stored verdict. A
demo fallback for the /monitor page, and a measurement tool: for every score it reports how long
after the window's last audio was sent the score arrived. ``--json`` writes those numbers (no
audio) to a file.

Reads WAV/FLAC/OGG/MP3 with soundfile, anything else (e.g. M4A) through ffmpeg. Only connects to
this machine unless ``--allow-remote`` is given.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
from pathlib import Path

import numpy as np

FRAME_S = 0.25
DEFAULT_URL = "ws://127.0.0.1:8000/v1/stream"


class UsageError(Exception):
    pass


def check_url(url: str, allow_remote: bool) -> str:
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("ws", "wss") or not parts.hostname:
        raise UsageError("--url must be a ws:// or wss:// URL")
    try:
        local = ipaddress.ip_address(parts.hostname).is_loopback
    except ValueError:
        local = parts.hostname == "localhost"
    if not local and not allow_remote:
        raise UsageError("--url is not on this machine; pass --allow-remote if intended")
    return url


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Mono float32 and its sample rate (the server resamples)."""
    import soundfile as sf

    try:
        data, rate = sf.read(path, dtype="float32", always_2d=True)
        return data.mean(axis=1).astype(np.float32), int(rate)
    except (sf.LibsndfileError, RuntimeError):
        exe = shutil.which("ffmpeg")
        if exe is None:
            raise UsageError(f"cannot read {path.name} (no ffmpeg for other formats)") from None
        cmd = [
            exe,
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(path),
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-",
        ]
        raw = subprocess.run(cmd, capture_output=True, check=True, timeout=120).stdout
        return np.frombuffer(raw, dtype="<f4").copy(), 16_000


def frames(x: np.ndarray, rate: int) -> list[np.ndarray]:
    n = max(1, round(FRAME_S * rate))
    return [x[i : i + n] for i in range(0, len(x), n)]


def stream_url(base: str, rate: int, codec: str | None, caller_id: str | None) -> str:
    q = {"sample_rate": str(rate)}
    if codec:
        q["codec"] = codec
    if caller_id:
        q["caller_id"] = caller_id
    return f"{base}?{urllib.parse.urlencode(q)}"


def describe(m: dict) -> str:
    kind = m.get("type")
    if kind == "score":
        state = "OVER threshold" if m["flagged"] else "below threshold"
        return (
            f"{m['t_s']:6.1f} s  window {m['window_score']:.6f}  running {m['running_score']:.6f}"
            f"  ({state})"
        )
    if kind == "alert":
        head = (
            "CAUTION: just over the threshold, verify the caller another way"
            if m.get("level") == "caution"
            else "ALERT: this voice may be AI-generated"
        )
        return (
            f"{m['t_s']:6.1f} s  {head} (running {m['running_score']:.6f} >= {m['threshold']:.6f})"
            "\n         Call back on a number you already have; ask something only the real"
            " person knows;\n         if money or OTPs were asked for, report it: 1930 or"
            " cybercrime.gov.in"
        )
    if kind == "final":
        if not m.get("stored"):
            return "call ended: nothing scored"
        verdict = "FLAGGED" if m["flagged"] else "not flagged"
        return (
            f"call ended: {verdict}, score {m['fake_score']:.6f} vs threshold {m['threshold']:.6f},"
            f" {m['windows']} windows, session {m['session_id']}"
        )
    if kind == "ready":
        return f"connected: threshold {m['threshold']:.6f}, model {m['model_version']}"
    return json.dumps(m)


def stream_samples(
    url: str, x: np.ndarray, rate: int, speed: float = 1.0, quiet: bool = False
) -> dict:
    """Stream mono float32 ``x`` to a ready-made /v1/stream URL. Returns the measurements: every
    score (window score, running average, detector time, how long after its audio it arrived),
    the alert and the ``final`` message. Shared with scripts/live_eval.py."""
    from websockets.sync.client import connect

    chunks = frames(x, rate)
    sent_at: list[tuple[float, float]] = []  # (seconds of audio sent so far, wall time)
    report: dict = {"duration_s": len(x) / rate, "scores": [], "alert": None, "final": None}
    t0 = time.monotonic()

    def sent_time(t_s: float) -> float | None:
        """Wall time at which the audio up to t_s had been sent."""
        return next((w for s, w in sent_at if s >= t_s - 1e-9), None)

    with connect(url, max_size=2**20, open_timeout=10) as ws:

        def receive() -> None:
            for raw in ws:
                m = json.loads(raw)
                now = time.monotonic()
                if not quiet:
                    print(describe(m), flush=True)
                if m["type"] in ("score", "alert"):
                    sent = sent_time(m["t_s"])
                    delay = None if sent is None else round((now - sent) * 1000.0, 1)
                    if m["type"] == "score":
                        report["scores"].append(
                            {
                                "t_s": m["t_s"],
                                "window": m["window_score"],
                                "running": m["running_score"],
                                "detector_ms": m["latency_ms"],
                                "arrived_after_ms": delay,
                            }
                        )
                    else:
                        report["alert"] = {
                            "t_s": m["t_s"],
                            "level": m.get("level"),
                            "arrived_after_ms": delay,
                        }
                if m["type"] == "ready":
                    report["ready"] = m
                if m["type"] in ("final", "error"):
                    report["final"] = m
                    return

        reader = threading.Thread(target=receive, daemon=True)
        reader.start()
        audio_s = 0.0
        for chunk in chunks:
            ws.send(np.ascontiguousarray(chunk, dtype="<f4").tobytes())
            audio_s += len(chunk) / rate
            sent_at.append((audio_s, time.monotonic()))
            if speed > 0:  # pace to (a multiple of) real time
                time.sleep(max(0.0, t0 + audio_s / speed - time.monotonic()))
        ws.send(json.dumps({"type": "end"}))
        reader.join(timeout=120)
    return report


def run(args: argparse.Namespace) -> dict:
    x, rate = load_audio(args.file)
    url = stream_url(check_url(args.url, args.allow_remote), rate, args.codec, args.caller_id)
    return {"file": args.file.name, **stream_samples(url, x, rate, args.speed)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("file", type=Path)
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--codec", help="phone-line codec applied by the server, e.g. g711_mulaw")
    p.add_argument("--caller-id")
    p.add_argument(
        "--speed", type=float, default=1.0, help="1 = real time, 0 = as fast as possible"
    )
    p.add_argument("--json", type=Path, help="write the measurements (no audio) to this file")
    p.add_argument("--allow-remote", action="store_true")
    args = p.parse_args(argv)
    try:
        if not args.file.is_file():
            raise UsageError(f"{args.file} is not a file")
        if args.speed < 0:
            raise UsageError("--speed must be >= 0")
        report = run(args)
    except UsageError as exc:
        p.error(str(exc))
    if args.json:
        args.json.write_text(json.dumps(report, indent=1))
    final = report.get("final") or {}
    return 0 if final.get("type") == "final" else 1


if __name__ == "__main__":
    sys.exit(main())
