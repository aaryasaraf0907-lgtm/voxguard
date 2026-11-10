# Benchmarking

`python -m voxguard.benchmark` measures latency on the machine it runs on. Results are JSON files
in `results/benchmarks/`; `docs/benchmarks.md` is generated from them
(`python -m voxguard.benchmark --render results/benchmarks`) and is never edited by hand.

## What is measured

Three separate times. They are never added up, and only the last one is end-to-end:

| Key | What it times | Includes |
|---|---|---|
| `detector_inference.wall` | `Detector.predict(waveform, sr)` in-process | detector preprocessing (resample, windows) and the model forward pass |
| `detector_inference.reported_latency_ms` | the detector's own `Prediction.latency_ms` (what the API returns as `latency_ms`) | same as above, measured inside `predict` |
| `codec_preprocessing.<codec>` | `api.codec_sim.simulate_phone_line` | resampling to 16 kHz (if needed) and an ffmpeg encode/decode; no inference |
| `api_request.clean` / `.simulated` | client wall time of `POST /v1/analyze` over HTTP to a uvicorn server started by the benchmark (loopback, temporary SQLite file) | multipart upload, decoding, codec (simulated only), inference, prevention and the SQLite commit |

Also recorded:
- `detector.load_ms`: building the detector once (checkpoint load, moving the model to the
  device). It is not part of any inference number.
- `api_request.*.sequential_requests_per_s`: timed requests divided by their summed time, from
  one client sending one request at a time. It is **not** a concurrent load test.

## Method

- **Input:** a deterministic test signal (`make_signal`: 220/880/2500 Hz tones plus seeded
  noise), mono float32. Default 4.0 s at 16 kHz, i.e. one 64,600-sample window for the detector.
  `--seconds`, `--sample-rate` and `--seed` change it, and are recorded.
- **Warm-up:** `--warmup` runs (default 5) are executed and discarded before `--iterations`
  timed runs (default 50). The first CUDA calls (kernel selection, allocator) fall in the warm-up.
- **Timer:** `time.perf_counter`. On CUDA, `RealDetector.predict` copies scores back to the
  CPU, which synchronizes the device, so the wall time covers the GPU work.
- **Statistics:** p50, p95 and mean via `metrics.latency_stats` (Aarya's helper, also used by
  `evaluate.py`), plus min and max.
- **One device per run:** `--device cpu` or `--device cuda`; CPU and GPU numbers are separate
  result files.
- **Metadata:** platform, Python, CPU model and core count, torch version and threads, CUDA
  availability and GPU name, ffmpeg version, git commit (`-dirty` if the tree had changes),
  detector kind, `model_version`, threshold, device, codecs, and a free-text `--note`.
- **Mocked runs:** passing a custom clock (only the unit tests do) sets `"mocked": true`.
  Mocked files are skipped by `--render`.

## Running it

```bash
uv sync --extra api --extra ui --extra ml          # ml only for --detector real

# stub detector (a hash, not a model: shows API and codec overhead only)
uv run python -m voxguard.benchmark --detector stub --codecs g711_mulaw opus \
    --out results/benchmarks/stub-cpu.json

# real detector on a checkpoint, CPU and GPU as separate runs
uv run python -m voxguard.benchmark --detector real --model-repo <path or hf repo> \
    --device cpu --codecs g711_mulaw --note "<which checkpoint>" \
    --out results/benchmarks/<name>-cpu.json
uv run python -m voxguard.benchmark --detector real --model-repo <path or hf repo> \
    --device cuda --codecs g711_mulaw --note "<which checkpoint>" \
    --out results/benchmarks/<name>-cuda.json

uv run python -m voxguard.benchmark --render results/benchmarks   # -> docs/benchmarks.md
```

## Reading the numbers

- Stub numbers measure the plumbing, not a model.
- A checkpoint's weights do not change how long the model takes, but the architecture and its
  configuration do. Until a trained checkpoint exists, the committed RawNet2 numbers come from
  an **untrained** RawNet2 built with the training configuration (`configs/rawnet2.yaml`):
  valid as compute latency for that architecture, meaningless for detection quality. The run's
  `note` says so.
- The CLAUDE.md targets (CPU < 2 s, GPU < 300 ms per 4 s clip) apply to
  `detector_inference`. The phone-line demo path costs `codec_preprocessing` on top, and
  `api_request.simulated` is the end-to-end time a client sees.
- The numbers belong to the recorded machine (here: a laptop under WSL 2), not to a deployment
  target such as a CPU Hugging Face Space.
