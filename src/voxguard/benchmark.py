"""Reproducible latency benchmark for the detector, codec simulation and the API.

Three times are measured separately and never added together:

- ``detector_inference``: wall time of ``Detector.predict`` on an in-memory waveform (plus the
  detector's own ``Prediction.latency_ms``), after the detector is loaded and warmed up;
- ``codec_preprocessing``: ``api.codec_sim.simulate_phone_line`` per codec (resampling plus an
  ffmpeg encode/decode), without inference;
- ``api_request``: client-side wall time of ``POST /v1/analyze`` over HTTP to a uvicorn server
  in this process (multipart upload, decoding, optional codec, inference, SQLite commit), plus
  the server's own ``latency_ms`` (detector) and, when simulated, ``codec_ms`` per response.

Model loading is timed once, on its own. Warm-up iterations are run and discarded.

    python -m voxguard.benchmark --detector stub --out results/benchmarks/stub-cpu.json
    python -m voxguard.benchmark --detector real --model-repo runs/best.pt --device cuda \\
        --note "RawNet2, untrained weights" --out results/benchmarks/rawnet2-cuda.json
    python -m voxguard.benchmark --render results/benchmarks  # -> docs/benchmarks.md
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import platform
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from voxguard.inference import Detector, StubDetector
from voxguard.metrics import latency_stats

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
Clock = Callable[[], float]  # seconds


def timed_runs(
    fn: Callable[[], object], warmup: int, iterations: int, clock: Clock = time.perf_counter
) -> list[float]:
    """Run ``fn`` ``warmup`` times (discarded), then ``iterations`` times; returns ms per run."""
    if warmup < 0 or iterations < 1:
        raise ValueError("warmup must be >= 0 and iterations >= 1")
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iterations):
        t0 = clock()
        fn()
        samples.append((clock() - t0) * 1000.0)
    return samples


def summarize(samples_ms: list[float], warmup: int) -> dict[str, float | int]:
    """p50/p95/mean (from ``metrics.latency_stats``) plus min/max and the warm-up count."""
    stats = asdict(latency_stats(samples_ms))
    x = np.asarray(samples_ms, dtype=np.float64)
    return {**stats, "min_ms": float(x.min()), "max_ms": float(x.max()), "warmup": warmup}


def make_signal(seconds: float, sample_rate: int, seed: int) -> np.ndarray:
    """Deterministic speech-band mixture plus a little noise, mono float32."""
    n = max(1, int(round(seconds * sample_rate)))
    t = np.arange(n) / sample_rate
    rng = np.random.default_rng(seed)
    x = sum(a * np.sin(2 * np.pi * f * t) for f, a in ((220, 0.3), (880, 0.2), (2500, 0.1)))
    return (x + 0.01 * rng.standard_normal(n)).astype(np.float32)


def wav_bytes(waveform: np.ndarray, sample_rate: int) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, waveform, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# --- measurements ------------------------------------------------------------------------


def bench_detector(
    detector: Detector,
    waveform: np.ndarray,
    sample_rate: int,
    warmup: int,
    iterations: int,
    clock: Clock = time.perf_counter,
) -> dict[str, Any]:
    reported: list[float] = []

    def run() -> None:
        reported.append(detector.predict(waveform, sample_rate).latency_ms)

    wall = timed_runs(run, warmup, iterations, clock)
    return {
        "wall": summarize(wall, warmup),
        "reported_latency_ms": summarize(reported[warmup:], warmup),
    }


def bench_codecs(
    waveform: np.ndarray,
    sample_rate: int,
    codecs: list[str],
    warmup: int,
    iterations: int,
    clock: Clock = time.perf_counter,
) -> dict[str, Any]:
    from voxguard.api import codec_sim

    out: dict[str, Any] = {}
    for codec in codecs:
        try:
            codec_sim.ensure_available(codec)
        except codec_sim.CodecUnavailableError as exc:
            out[codec] = {"skipped": str(exc)}
            continue
        samples = timed_runs(
            lambda c=codec: codec_sim.simulate_phone_line(waveform, sample_rate, c),
            warmup,
            iterations,
            clock,
        )
        out[codec] = summarize(samples, warmup)
    return out


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def bench_api(
    detector: Detector,
    audio: bytes,
    codec: str | None,
    warmup: int,
    iterations: int,
    clock: Clock = time.perf_counter,
) -> dict[str, Any]:
    """Start the real API (uvicorn, temporary SQLite file) in this process and time requests."""
    import requests
    import uvicorn

    from voxguard.api.deps import Settings
    from voxguard.api.main import create_app

    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(db_url=f"sqlite:///{Path(tmp) / 'bench.db'}", simulate_codec=codec)
        port = _free_port()
        config = uvicorn.Config(
            create_app(detector, settings), host="127.0.0.1", port=port, log_level="warning"
        )
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 30
            while not server.started:
                if time.monotonic() > deadline or not thread.is_alive():
                    raise RuntimeError("benchmark API server did not start")
                time.sleep(0.02)
            url = f"http://127.0.0.1:{port}/v1/analyze"
            http = requests.Session()
            out: dict[str, Any] = {}
            modes = [("clean", {})] + ([("simulated", {"simulate_codec": "true"})] if codec else [])
            for name, form in modes:
                bodies: list[dict[str, Any]] = []

                def post(form: dict = form, bodies: list = bodies) -> None:
                    r = http.post(url, files={"file": ("bench.wav", audio)}, data=form, timeout=120)
                    r.raise_for_status()
                    bodies.append(r.json())

                samples = timed_runs(post, warmup, iterations, clock)
                stats: dict[str, Any] = summarize(samples, warmup)
                # one client, one request at a time: not a concurrent load test
                stats["sequential_requests_per_s"] = len(samples) / (sum(samples) / 1000.0)
                timed = bodies[warmup:]  # what the API itself reports, per timed request
                stats["server_latency_ms"] = summarize([b["latency_ms"] for b in timed], warmup)
                if name == "simulated":
                    stats["server_codec_ms"] = summarize([b["codec_ms"] for b in timed], warmup)
                out[name] = stats
            return out
        finally:
            server.should_exit = True
            thread.join(timeout=10)


# --- environment ------------------------------------------------------------------------


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _ffmpeg_version() -> str | None:
    from voxguard.data.codecs import ffmpeg_path

    exe = ffmpeg_path()
    if exe is None:
        return None
    out = subprocess.run([exe, "-version"], capture_output=True, text=True, check=False).stdout
    return out.splitlines()[0] if out else None


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def environment(device: str) -> dict[str, Any]:
    env: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "ffmpeg": _ffmpeg_version(),
        "git_commit": _git_commit(),
        "device": device,
    }
    try:
        import torch
    except ImportError:
        env["torch"] = None
        return env
    env["torch"] = torch.__version__
    env["torch_threads"] = torch.get_num_threads()
    env["cuda_available"] = torch.cuda.is_available()
    if device.startswith("cuda") and torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
        env["cuda"] = torch.version.cuda
    return env


# --- orchestration ----------------------------------------------------------------------


def write_untrained_checkpoint(config_path: Path, out: Path, seed: int = 0) -> Path:
    """Randomly initialized model from a training config's ``model`` section, saved in the
    ``train.py`` checkpoint layout so it loads through ``RealDetector``. For compute-latency
    measurement only: the weights are untrained."""
    import torch
    import yaml

    from voxguard.models import build_model

    params = dict(yaml.safe_load(config_path.read_text())["model"])
    name = params.pop("name")
    if name != "rawnet2":
        raise ValueError("untrained checkpoints are supported for rawnet2 only (no downloads)")
    torch.manual_seed(seed)
    model = build_model(name, **params)
    torch.save(
        {"model": model.state_dict(), "epoch": 0, "config": {"model": {"name": name, **params}}},
        out,
    )
    return out


def load_benchmark_detector(
    kind: str, model_repo: str | None, device: str, clock: Clock = time.perf_counter
) -> tuple[Detector, float]:
    """Build the detector once; returns it and the load time in ms (measured separately)."""
    t0 = clock()
    if kind == "stub":
        detector: Detector = StubDetector(threshold=0.5)
    elif kind == "real":
        if not model_repo:
            raise ValueError("--model-repo is required for --detector real")
        from voxguard.inference import RealDetector

        detector = RealDetector(model_repo, device=device)
    else:
        raise ValueError(f"unknown detector {kind!r}")
    return detector, (clock() - t0) * 1000.0


def run_benchmark(
    detector: Detector,
    *,
    detector_kind: str,
    device: str,
    load_ms: float | None,
    seconds: float,
    sample_rate: int,
    seed: int,
    warmup: int,
    iterations: int,
    codecs: list[str],
    api: bool,
    note: str = "",
    weights: str | None = None,
    clock: Clock | None = None,
) -> dict[str, Any]:
    """Run every measurement. A custom ``clock`` marks the result as mocked."""
    mocked = clock is not None
    clk = clock or time.perf_counter
    x = make_signal(seconds, sample_rate, seed)
    result: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "id": uuid.uuid4().hex[:12],
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "mocked": mocked,
        "note": note,
        "detector": {
            "kind": detector_kind,
            "model_version": detector.model_version,
            "threshold": detector.threshold,
            "device": str(getattr(detector, "device", "cpu")),
            "load_ms": load_ms,
            # "none" for the stub (a hash), "untrained" or "checkpoint" for the real detector
            "weights": weights or ("none" if detector_kind == "stub" else "checkpoint"),
        },
        "input": {"seconds": seconds, "sample_rate": sample_rate, "seed": seed, "channels": 1},
        "method": {"warmup": warmup, "iterations": iterations, "timer": "time.perf_counter"},
        "environment": environment(device),
    }
    logger.info("detector inference (%s, %s)", detector_kind, device)
    result["detector_inference"] = bench_detector(detector, x, sample_rate, warmup, iterations, clk)
    if codecs:
        logger.info("codec preprocessing: %s", ", ".join(codecs))
        result["codec_preprocessing"] = bench_codecs(
            x, sample_rate, codecs, warmup, iterations, clk
        )
    if api:
        codec = next((c for c in codecs if "skipped" not in result["codec_preprocessing"][c]), None)
        logger.info("API requests (codec for simulated requests: %s)", codec)
        result["api_request"] = bench_api(
            detector, wav_bytes(x, sample_rate), codec, warmup, iterations, clk
        )
        result["api_request"]["codec"] = codec
    return result


# --- report -----------------------------------------------------------------------------


def _ms(stats: dict[str, Any] | None) -> str:
    if not stats or "p50_ms" not in stats:
        return "n/a"
    return f"{stats['p50_ms']:.1f} / {stats['p95_ms']:.1f}"


def render_markdown(results: list[dict[str, Any]]) -> str:
    lines = [
        "# Benchmarks",
        "",
        "> Auto-generated by `python -m voxguard.benchmark --render` from"
        " `results/benchmarks/*.json`. Do not edit by hand. Times are p50 / p95 in ms."
        " Method: `docs/benchmarking.md`.",
        "",
        "The CLAUDE.md latency targets (CPU < 2 s, GPU < 300 ms per 4 s clip) apply to Inference"
        " only. API codec_ms is the codec time the API reports for the simulated requests; it is"
        " already inside API simulated and is never added to Inference.",
        "",
        "| Run | Detector | Weights | Device | Load (s) | Inference | Codec | API clean |"
        " API simulated | API codec_ms |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        det, env = r["detector"], r["environment"]
        hardware = env.get("gpu", env["cpu"]) if det["device"].startswith("cuda") else env["cpu"]
        codec = r.get("codec_preprocessing", {})
        codec_cell = ", ".join(
            f"{name} {_ms(s)}" if "skipped" not in s else f"{name} skipped"
            for name, s in codec.items()
        )
        api = r.get("api_request", {})
        load = f"{det['load_ms'] / 1000:.2f}" if det.get("load_ms") is not None else "n/a"
        label = r["id"] + (" (MOCKED)" if r.get("mocked") else "")
        lines.append(
            f"| {label} | {det['model_version']} | {det.get('weights', 'n/a')} | "
            f"{det['device']}: {hardware} | {load} | "
            f"{_ms(r['detector_inference']['wall'])} | {codec_cell or 'n/a'} | "
            f"{_ms(api.get('clean'))} | {_ms(api.get('simulated'))} | "
            f"{_ms((api.get('simulated') or {}).get('server_codec_ms'))} |"
        )
    lines += ["", "Notes per run:", ""]
    for r in results:
        inp, m = r["input"], r["method"]
        lines.append(
            f"- `{r['id']}` ({r['created_at']}, commit {r['environment'].get('git_commit')}): "
            f"{inp['seconds']} s at {inp['sample_rate']} Hz, {m['warmup']} warm-up + "
            f"{m['iterations']} timed runs. {r.get('note') or ''}".rstrip()
        )
    return "\n".join(lines) + "\n"


def load_results(directory: Path) -> list[dict[str, Any]]:
    results = [json.loads(p.read_text()) for p in sorted(directory.glob("*.json"))]
    return [r for r in results if r.get("schema") == SCHEMA_VERSION and not r.get("mocked")]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--detector", choices=["stub", "real"], default="stub")
    p.add_argument("--model-repo", help="checkpoint path or HF repo id (real detector)")
    p.add_argument(
        "--untrained-config",
        type=Path,
        help="build an UNTRAINED rawnet2 checkpoint from this training config (latency only)",
    )
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--seconds", type=float, default=4.0)
    p.add_argument("--sample-rate", type=int, default=16_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--codecs", nargs="*", default=[], help="e.g. g711_mulaw opus")
    p.add_argument("--no-api", action="store_true", help="skip the HTTP request benchmark")
    p.add_argument("--note", default="", help="free text stored with the result")
    p.add_argument("--out", type=Path, help="write the JSON result here")
    p.add_argument("--render", type=Path, metavar="DIR", help="render DIR/*.json to markdown")
    p.add_argument("--render-out", type=Path, default=Path("docs/benchmarks.md"))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.render:
        args.render_out.write_text(render_markdown(load_results(args.render)))
        logger.info("wrote %s", args.render_out)
        return
    note = args.note
    with tempfile.TemporaryDirectory() as tmp:
        if args.untrained_config:
            if args.detector != "real" or args.model_repo:
                p.error("--untrained-config needs --detector real and no --model-repo")
            args.model_repo = str(
                write_untrained_checkpoint(args.untrained_config, Path(tmp) / "best.pt", args.seed)
            )
            note = f"UNTRAINED weights from {args.untrained_config} (latency only). {note}".strip()
        detector, load_ms = load_benchmark_detector(args.detector, args.model_repo, args.device)
        result = run_benchmark(
            detector,
            detector_kind=args.detector,
            device=args.device,
            load_ms=load_ms,
            seconds=args.seconds,
            sample_rate=args.sample_rate,
            seed=args.seed,
            warmup=args.warmup,
            iterations=args.iterations,
            codecs=args.codecs,
            api=not args.no_api,
            note=note,
            weights="untrained" if args.untrained_config else None,
        )
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        logger.info("wrote %s", args.out)
    else:
        sys.stdout.write(text + "\n")


if __name__ == "__main__":
    main()
