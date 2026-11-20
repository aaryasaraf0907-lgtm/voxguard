"""End-to-end evaluation of the live call monitor: whole calls streamed through the deployed API.

    python scripts/live_eval.py --api-url ws://127.0.0.1:18360 --out ~/private/live-eval-1 \\
        --asvspoof-keys .../keys/LA/CM/trial_metadata.txt --asvspoof-audio .../flac \\
        [--codecs none,g711_mulaw,g711_alaw,gsm,opus] [--calls-per-type 20] [--speed 1]
    python scripts/live_eval.py ... --labels labels.csv --audio-dir clips/  # e.g. voice clones

It builds calls of about ``--call-seconds`` from short utterances of **one speaker** each, joined
with 0.25 s pauses:

- ``genuine``: bona fide utterances only;
- ``fake``: spoofed utterances only;
- ``mixed``: bona fide utterances, then spoofed ones of the same speaker (a call that turns into a
  cloned voice); the switch time is known.

Every call is streamed to ``WS /v1/stream`` exactly as the /monitor page or stream_call.py would,
once per phone-line codec (``none`` = audio as captured), and the live messages are recorded:
each window's score and the running average, the alert (time and level), and the stored verdict.
The summary counts, per codec and call type: live alerts under the deployed rule, alerts during
the genuine part of mixed calls, time to alert (from the start for fake calls, from the switch for
mixed ones), calls whose first window alone crossed the threshold, and verdicts. With ``--speed``
above 0 it also reports when each score arrived after its audio.

What it is and is not: a system measurement of the live monitor (the deployed rule, the codec
path, the timing), on a small set, reported as counts and medians, never as accuracy or rates.
The alert rule itself is not tuned here.

Inputs:

- ASVspoof 2021 LA eval (ODC-By; cite Todisco et al. / Yamagishi et al.): only trials whose
  transmission condition is ``none``, so the only codec is the one the API applies. Labels come
  from the official key.
- or a labels CSV in the real-audio sanity format (``file,ground_truth,source,consent``, optional
  ``speaker``), validated the same way, e.g. a consenting speaker's genuine sentences and their
  clones.

No audio is written anywhere; calls are built in memory. Reports (JSON and Markdown, scores and
file names only) go to a new directory outside the repo. Only APIs on this machine are used.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import statistics
import subprocess
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_audio_sanity import inside_git_tree, load_labels  # noqa: E402
from stream_call import UsageError, check_url, stream_samples  # noqa: E402

logger = logging.getLogger("live_eval")

RATE = 16_000
PAUSE_S = 0.25
CALL_TYPES = ("genuine", "fake", "mixed")
DEFAULT_CODECS = ("none", "g711_mulaw", "g711_alaw", "gsm", "opus")
NARROWBAND = frozenset({"g711_mulaw", "g711_alaw", "gsm", "amr_nb"})


@dataclass(frozen=True)
class Utterance:
    path: Path
    label: str  # bonafide | spoof
    speaker: str
    attack: str = "-"


@dataclass
class Call:
    call_id: str
    kind: str  # genuine | fake | mixed
    speaker: str
    segments: list[dict[str, Any]] = field(default_factory=list)  # file, label, start_s, end_s
    switch_s: float | None = None  # mixed: start of the first spoofed utterance

    @property
    def duration_s(self) -> float:
        return self.segments[-1]["end_s"] if self.segments else 0.0


# --- inputs -------------------------------------------------------------------------------------


def read_asvspoof(keys: Path, audio_dir: Path) -> list[Utterance]:
    """ASVspoof 2021 LA eval trials without transmission (condition ``none``) whose file exists."""
    out = []
    for line in keys.read_text().splitlines():
        parts = line.split()
        if len(parts) < 8 or parts[7] != "eval" or parts[2] != "none":
            continue
        path = audio_dir / f"{parts[1]}.flac"
        out.append(Utterance(path, parts[5], parts[0], parts[4]))
    if not out:
        raise UsageError("no ASVspoof eval trials with condition 'none' in the keys")
    return out


def read_labels(labels: Path, audio_dir: Path) -> list[Utterance]:
    """The sanity tool's labels CSV (consent checked there); ``speaker`` column optional."""
    import csv

    entries = load_labels(labels, audio_dir, frozenset())
    with labels.open(newline="") as f:
        speakers = {r["file"]: (r.get("speaker") or "").strip() for r in csv.DictReader(f)}
    return [
        Utterance(
            e.path,
            "bonafide" if e.ground_truth == "genuine" else "spoof",
            speakers.get(e.file) or "speaker",
        )
        for e in entries
    ]


# --- building calls (deterministic, in memory) --------------------------------------------------


def _take(pool: list[Utterance], need_s: float, measure: Any) -> list[Utterance]:
    """Utterances from the end of ``pool`` until they last ``need_s`` (pauses included); [] if
    the pool runs out first. ``measure(u)`` gives an utterance's duration, None if unreadable."""
    picked, total = [], 0.0
    while pool and total < need_s:
        u = pool.pop()
        d = measure(u)
        if d:
            picked.append(u)
            total += d + PAUSE_S
    return picked if total >= need_s else []


def build_calls(
    utterances: list[Utterance], measure: Any, per_type: int, call_s: float, seed: int
) -> list[Call]:
    """``per_type`` calls of each kind, each from one speaker's utterances (none used twice).
    Speakers are visited in a seeded order; a speaker without enough audio left is skipped.
    ``measure`` is only called for utterances a call takes, so only those are decoded."""
    rng = random.Random(seed)
    by_speaker: dict[str, dict[str, list[Utterance]]] = defaultdict(lambda: defaultdict(list))
    for u in sorted(utterances, key=lambda u: str(u.path)):
        by_speaker[u.speaker][u.label].append(u)
    for speaker in sorted(by_speaker):
        for label in sorted(by_speaker[speaker]):
            rng.shuffle(by_speaker[speaker][label])
    speakers = sorted(by_speaker)
    rng.shuffle(speakers)
    calls: list[Call] = []
    for kind in CALL_TYPES:
        made = 0
        for speaker in speakers * 4:  # several calls per speaker when there are few speakers
            if made == per_type:
                break
            pools = by_speaker[speaker]
            if kind == "mixed":
                first = _take(pools["bonafide"], call_s / 2, measure)
                second = _take(pools["spoof"], call_s / 2, measure) if first else []
                parts = first + second if first and second else []
            else:
                label = "bonafide" if kind == "genuine" else "spoof"
                parts = _take(pools[label], call_s, measure)
            if not parts:
                continue
            call = Call(f"{kind}-{made:02d}", kind, speaker)
            t = 0.0
            for u in parts:
                if kind == "mixed" and call.switch_s is None and u.label == "spoof":
                    call.switch_s = round(t, 3)
                end = t + measure(u)
                call.segments.append(
                    {
                        "file": u.path.name,
                        "label": u.label,
                        "attack": u.attack,
                        "start_s": round(t, 3),
                        "end_s": round(end, 3),
                    }
                )
                t = end + PAUSE_S
            calls.append(call)
            made += 1
        if made < per_type:
            logger.warning("only %d %s calls could be built", made, kind)
    return calls


def decode(path: Path) -> np.ndarray:
    """16 kHz mono float32 via ffmpeg (some ASVspoof FLACs make libsndfile lose sync)."""
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise UsageError("live_eval needs ffmpeg on PATH to read the audio")
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
        str(RATE),
        "-",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True, timeout=60).stdout
    return np.frombuffer(raw, dtype="<f4").copy()


def render_call(call: Call, cache: dict[str, np.ndarray]) -> np.ndarray:
    pause = np.zeros(int(PAUSE_S * RATE), dtype=np.float32)
    parts = []
    for i, seg in enumerate(call.segments):
        if i:
            parts.append(pause)
        parts.append(cache[seg["file"]])
    return np.concatenate(parts).astype(np.float32)


# --- measuring ----------------------------------------------------------------------------------


def summarize_call(call: Call, codec: str, measured: dict[str, Any]) -> dict[str, Any]:
    final = measured.get("final") or {}
    threshold = final.get("threshold")
    scores = measured["scores"]
    alert = measured.get("alert")
    out = {
        "call_id": call.call_id,
        "kind": call.kind,
        "speaker": call.speaker,
        "codec": codec,
        "duration_s": round(call.duration_s, 2),
        "switch_s": call.switch_s,
        "windows": len(scores),
        "threshold": threshold,
        "first_window_over": bool(
            scores and threshold is not None and scores[0]["window"] >= threshold
        ),
        "alert_t_s": alert["t_s"] if alert else None,
        "alert_level": alert["level"] if alert else None,
        "verdict_flagged": final.get("flagged"),
        "verdict_score": final.get("fake_score"),
        "arrived_after_ms": [
            s["arrived_after_ms"] for s in scores if s["arrived_after_ms"] is not None
        ],
        "scores": [{k: s[k] for k in ("t_s", "window", "running")} for s in scores],
        "error": final.get("detail") if final.get("type") == "error" else None,
        "ready": measured.get("ready"),
    }
    if call.switch_s is not None and alert:
        out["alert_after_switch_s"] = round(alert["t_s"] - call.switch_s, 2)
    return out


def _median(values: list[float]) -> float | None:
    return round(statistics.median(values), 2) if values else None


def table(results: list[dict[str, Any]], codecs: list[str]) -> list[dict[str, Any]]:
    rows = []
    for codec in codecs:
        for kind in CALL_TYPES:
            rs = [
                r for r in results if r["codec"] == codec and r["kind"] == kind and not r["error"]
            ]
            if not rs:
                continue
            alerted = [r for r in rs if r["alert_t_s"] is not None]
            row: dict[str, Any] = {
                "codec": codec,
                "kind": kind,
                "calls": len(rs),
                "alerted": len(alerted),
                "alerts_high": sum(r["alert_level"] == "high" for r in alerted),
                "first_window_over": sum(r["first_window_over"] for r in rs),
                "verdict_flagged": sum(bool(r["verdict_flagged"]) for r in rs),
            }
            if kind == "fake":
                row["time_to_alert_s"] = _median([r["alert_t_s"] for r in alerted])
            if kind == "mixed":
                after = [
                    r["alert_after_switch_s"] for r in alerted if r["alert_after_switch_s"] >= 0
                ]
                row["alerted_before_switch"] = sum(r["alert_after_switch_s"] < 0 for r in alerted)
                row["alert_after_switch_median_s"] = _median(after)
                row["alert_after_switch_max_s"] = round(max(after), 2) if after else None
            arrivals = [a for r in rs for a in r["arrived_after_ms"]]
            row["score_arrival_median_ms"] = _median(arrivals)
            rows.append(row)
    return rows


def _fmt(v: Any) -> str:
    return "–" if v is None else str(v)


def _secs(v: Any) -> str:
    return "–" if v is None else f"{v} s"


def render_markdown(report: dict[str, Any]) -> str:
    run = report["run"]
    missing = run["unavailable"]
    lines = [
        f"# Live call monitor, end to end ({run['started']})",
        "",
        f"Model `{run['model_version']}`, API {run['api_url']}, commit `{run['commit']}`. "
        f"{run['calls_per_type']} calls of each kind, about {run['call_seconds']:g} s, one "
        f"speaker each; source: {run['source']}. Streamed at speed {run['speed']:g} "
        f"(1 = real time). Rule as deployed: {run['rule']}. Small set: counts and medians "
        "only, not accuracy.",
        "",
        f"Codecs measured: {', '.join(run['codecs'])}."
        + (f" Not available on this server: {', '.join(missing)}." if missing else ""),
        "",
        "| Codec | Kind | Calls | Live alerts (high) | First window alone over "
        "| Flagged verdicts | Fake: time to alert | Mixed: alerts before the switch "
        "| Mixed: after the switch, median / max | Score arrival, median |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in report["table"]:
        fake = _secs(r.get("time_to_alert_s")) if r["kind"] == "fake" else "–"
        before = _fmt(r.get("alerted_before_switch")) if r["kind"] == "mixed" else "–"
        after = "–"
        if r["kind"] == "mixed":
            after = f"{_secs(r.get('alert_after_switch_median_s'))} / "
            after += _secs(r.get("alert_after_switch_max_s"))
        arrival = r["score_arrival_median_ms"]
        arrival_txt = "–" if arrival is None else f"{arrival} ms"
        lines.append(
            f"| {r['codec']} | {r['kind']} | {r['calls']} | {r['alerted']} ({r['alerts_high']}) "
            f"| {r['first_window_over']} | {r['verdict_flagged']} | {fake} | {before} "
            f"| {after} | {arrival_txt} |"
        )
    return "\n".join(lines) + "\n"


# --- main ---------------------------------------------------------------------------------------


def _http_base(ws_url: str) -> str:
    p = urllib.parse.urlsplit(ws_url)
    return f"{'https' if p.scheme == 'wss' else 'http'}://{p.netloc}"


def _get(url: str) -> Any:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=10) as r:
        return json.loads(r.read())


def _commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
            timeout=10,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    return (out.stdout.strip() or "unknown") + ("+uncommitted changes" if dirty else "")


def check_out_dir(raw: str) -> Path:
    out = Path(raw).expanduser().resolve()
    if inside_git_tree(out):
        raise UsageError("--out must be outside the git tree (reports stay private)")
    if out.exists() and any(out.iterdir()):
        raise UsageError("--out must be a new or empty directory")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--api-url", required=True, help="ws://host:port (this machine)")
    p.add_argument("--out", required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--asvspoof-keys", type=Path)
    src.add_argument("--labels", type=Path)
    p.add_argument("--asvspoof-audio", type=Path)
    p.add_argument("--audio-dir", type=Path)
    p.add_argument("--codecs", default=",".join(DEFAULT_CODECS))
    p.add_argument("--calls-per-type", type=int, default=20)
    p.add_argument("--call-seconds", type=float, default=20.0)
    p.add_argument(
        "--speed", type=float, default=1.0, help="1 = real time, 0 = as fast as possible"
    )
    p.add_argument("--parallel", type=int, default=2, help="calls at once (the API allows 4)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    try:
        base = check_url(args.api_url.rstrip("/"), allow_remote=False)
        out = check_out_dir(args.out)
        if (
            args.calls_per_type < 1
            or args.call_seconds < 6
            or args.speed < 0
            or not 1 <= args.parallel <= 4
        ):
            raise UsageError(
                "need --calls-per-type >= 1, --call-seconds >= 6, --speed >= 0, --parallel 1-4"
            )
        if args.asvspoof_keys:
            if not args.asvspoof_audio:
                raise UsageError("--asvspoof-keys needs --asvspoof-audio")
            utterances = read_asvspoof(args.asvspoof_keys, args.asvspoof_audio)
            source = "ASVspoof 2021 LA eval, transmission condition 'none' (official key)"
        else:
            if not args.audio_dir:
                raise UsageError("--labels needs --audio-dir")
            utterances = read_labels(args.labels, args.audio_dir)
            source = f"labels CSV {args.labels.name}"
    except UsageError as exc:
        p.error(str(exc))

    http = _http_base(base)
    health = _get(f"{http}/health")
    codec_info = {c["name"]: c["available"] for c in _get(f"{http}/v1/codecs")["codecs"]}
    wanted = [c.strip() for c in args.codecs.split(",") if c.strip()]
    codecs = [c for c in wanted if c == "none" or codec_info.get(c)]
    unavailable = [c for c in wanted if c not in codecs]

    cache: dict[str, np.ndarray] = {}

    def measure(u: Utterance) -> float | None:
        if u.path.name not in cache:
            try:
                cache[u.path.name] = decode(u.path)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
                return None
        return len(cache[u.path.name]) / RATE

    calls = build_calls(
        [u for u in utterances if u.path.is_file()],
        measure,
        args.calls_per_type,
        args.call_seconds,
        args.seed,
    )
    logger.info(
        "%d calls, %d codecs, model %s", len(calls), len(codecs), health.get("model_version")
    )

    def one(job: tuple[Call, str]) -> dict[str, Any]:
        call, codec = job
        q = {"sample_rate": str(RATE)} | ({"codec": codec} if codec != "none" else {})
        url = f"{base}/v1/stream?{urllib.parse.urlencode(q)}"
        measured = stream_samples(url, render_call(call, cache), RATE, args.speed, quiet=True)
        r = summarize_call(call, codec, measured)
        logger.info(
            "%s %-10s windows %d alert %s verdict %s",
            call.call_id,
            codec,
            r["windows"],
            r["alert_t_s"],
            r["verdict_flagged"],
        )
        return r

    jobs = [(c, codec) for codec in codecs for c in calls]
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        results = list(pool.map(one, jobs))
    ready = next((r["ready"] for r in results if r.get("ready")), {})
    rule = (
        f"an alert once the running average of at least {ready.get('min_alert_windows')} windows "
        f"reaches the threshold; caution below {ready.get('caution_factor')} x it"
    )
    report = {
        "run": {
            "started": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "api_url": base,
            "model_version": health.get("model_version"),
            "commit": _commit(),
            "source": source,
            "calls_per_type": args.calls_per_type,
            "call_seconds": args.call_seconds,
            "speed": args.speed,
            "seed": args.seed,
            "codecs": codecs,
            "unavailable": unavailable,
            "rule": rule,
        },
        "calls": [asdict(c) for c in calls],
        "results": results,
        "table": table(results, codecs),
    }
    out.mkdir(parents=True, exist_ok=True)
    stem = f"live-eval-{report['run']['started'].replace(':', '')}"
    (out / f"{stem}.json").write_text(json.dumps(report, indent=1))
    md = render_markdown(report)
    (out / f"{stem}.md").write_text(md)
    print(md)
    return 0 if all(not r["error"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
