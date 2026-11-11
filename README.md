# VoxGuard

Codec-aware voice-deepfake (vishing) detector for telephone-quality audio, with a thin
prevention layer and a demo app. University project (Engineering Design & Innovation),
aligned with SIH 2026 problem statement SIH26104 (detection and prevention of voice-cloning
attacks, with latency as a reported metric).

> **Private repository.** Do not publish or describe the method externally. See `CLAUDE.md` section 15.

## What it does

Most audio-deepfake detectors are trained and tested on clean studio audio, but scam calls pass
through lossy telephone codecs (G.711, AMR, GSM, Opus) that remove the artifacts those
detectors rely on. VoxGuard trains and evaluates detectors under simulated phone channels, and
wraps them in an application:

- **Detection:** RawNet2 (baseline) and SSL-AASIST (XLS-R + AASIST) models behind one
  `Detector` interface, plus a deterministic stub for development.
- **Prevention:** every analysis is stored; a score at or above the threshold flags the call,
  raises an alert and adds the caller to a blocklist.
- **Demo:** a Streamlit app (English and Hindi) to upload or record audio, optionally pass it
  through a simulated phone line, and review flagged calls, the blocklist and alerts.

## Status

- Data pipeline, codec simulation, training, evaluation, API, prevention layer, demo UI, Docker
  and benchmarking are implemented.
- First RawNet2 results (2x2 matrix, per codec, held-out GSM) are in
  [docs/results.md](docs/results.md). Aarya's STATUS notes a class-weight bug in those runs (both
  arms alike); RawNet2 is being retrained with the fix. SSL-AASIST training is in progress.
- Progress and known issues per person: [docs/STATUS.md](docs/STATUS.md). Scope and roadmap:
  `CLAUDE.md`.

## Architecture

```
Streamlit UI (ui/)  --HTTP-->  FastAPI (api/)  -->  Detector (inference.py: Stub | Real)
  EN/HI, upload/mic,            decode audio        RealDetector: RawNet2 / SSL-AASIST
  phone-line toggle,            codec simulation    checkpoint from a path or HF Hub
  flagged calls, blocklist,     (ffmpeg)
  alerts                        prevention --> SQLite (sessions, alerts, blocklist)

Training side: data/ (download, manifests, codecs, augment) -> train.py -> evaluate.py
```

Details: [docs/architecture.md](docs/architecture.md).

## Technology

Python 3.11, `uv`, PyTorch + torchaudio, Hugging Face Transformers (XLS-R), FastAPI, Pydantic
v2, SQLAlchemy 2 + SQLite, Streamlit, ffmpeg, Docker / docker compose, pytest, ruff, GitHub
Actions.

## Repository layout

```
src/voxguard/
  api/          FastAPI app, audio decoding, codec simulation, prevention, SQLite models
  ui/           Streamlit app, API client, English/Hindi strings
  data/         dataset download, manifests, codec chain, augmentation, datasets
  models/       RawNet2, AASIST, SSL-AASIST
  inference.py  Detector contract: Prediction, StubDetector, RealDetector
  train.py, evaluate.py, metrics.py, benchmark.py
configs/        training and evaluation configs
docker/         Dockerfiles (API, UI, Hugging Face Space) and the Space start script
docs/           architecture, deployment, benchmarking, results, status
results/        evaluation and benchmark JSON (docs are generated from these)
tests/          pytest suite
```

## Setup

```bash
make setup                          # uv sync (Python 3.11)
uv sync --extra api --extra ui      # what the API and demo need
uv sync --extra api --extra ui --extra ml   # add torch etc. for the real detector (several GB)
```

Phone-line simulation needs `ffmpeg` on `PATH` (`python -m voxguard.data.codecs` lists the
codecs your ffmpeg supports).

## Configuration

All variables, with defaults, are in [`.env.example`](.env.example). The main ones:

| Variable | Default | Meaning |
|---|---|---|
| `VOXGUARD_DETECTOR` | `stub` | `stub` or `real` |
| `VOXGUARD_MODEL_REPO` | (none) | real detector: a `best.pt` path, a directory holding it, or a Hugging Face repo id |
| `HF_TOKEN` | (none) | needed to download from the private Hugging Face org |
| `VOXGUARD_DEVICE` | `cpu` | real detector device: `cpu`, `cuda` or `auto` |
| `VOXGUARD_THRESHOLD` | `0.5` | decision threshold (see below) |
| `VOXGUARD_SIMULATE_CODEC` | (none) | codec used when the phone-line toggle is on, e.g. `g711_mulaw` |
| `VOXGUARD_DB_URL` | `sqlite:///./voxguard.db` | SQLite only |
| `VOXGUARD_API_URL` | `http://localhost:8000` | where the UI finds the API |

Invalid values stop the API at startup with an error naming the variable.

## Run the demo locally

```bash
# terminal 1: API on http://localhost:8000 (interactive docs at /docs)
uv run uvicorn voxguard.api.main:create_app --factory --port 8000

# terminal 2: Streamlit UI on http://localhost:8501
uv run streamlit run src/voxguard/ui/app.py
```

With the default stub detector no weights or GPU are needed; its scores are deterministic
placeholders, not detection results, and the UI says so.

Using the demo:

1. **Language:** the selector at the top of the sidebar switches between English and Hindi.
   The sidebar also shows whether the API is reachable and which detector it runs.
2. **Analyze:** upload a WAV/FLAC/OGG/MP3 file (up to 25 MB, 60 s) or record with the
   microphone, optionally enter a caller ID, and press **Analyze**. The result shows the label,
   score, threshold, detector latency, model version and whether the call was flagged.
3. **Phone line:** turn on **Simulate phone line** and analyze again to score the same clip after
   a telephone codec; when both results exist they are shown side by side. The API must have
   `VOXGUARD_SIMULATE_CODEC` set and ffmpeg with that codec, otherwise the UI explains why
   nothing was analyzed.
4. **Monitoring:** the other tabs list flagged sessions, the caller blocklist and alerts
   (times in UTC). A new file or recording clears the previous result.

## Run with Docker

```bash
cp .env.example .env               # optional; e.g. VOXGUARD_SIMULATE_CODEC=g711_mulaw
docker compose up --build          # UI http://localhost:8501, API http://localhost:8000
```

The API image includes ffmpeg (G.711, GSM and Opus; no AMR) and the stub detector; SQLite lives
on the `voxguard-data` volume. The real detector image (`INSTALL_ML=1`), the Hugging Face Space
image, every variable and troubleshooting are in [docs/deployment.md](docs/deployment.md).

## Using the real detector

```bash
export VOXGUARD_DETECTOR=real
export VOXGUARD_MODEL_REPO=<path to best.pt, or an HF repo id>
export HF_TOKEN=<token>              # only for the private Hugging Face org
uv run uvicorn voxguard.api.main:create_app --factory --port 8000
```

- Trained RawNet2 checkpoints are in the private Hugging Face org
  (`voxguard-team/voxguard-rawnet2-clean`, `voxguard-team/voxguard-rawnet2-codec-aug`); access
  needs a token with permission for that org. SSL-AASIST checkpoints follow when training
  finishes.
- The model is loaded once at startup; `/health` then reports `"detector": "real"` and the
  checkpoint's `model_version`. If loading fails the API exits with an error that says what to
  fix; it never falls back to the stub.
- **Threshold:** `VOXGUARD_THRESHOLD` defaults to 0.5, which is not an evaluated operating
  point. `results/<model>.json` records an `eer_threshold` per checkpoint and test condition;
  set the one that matches your checkpoint. `VOXGUARD_THRESHOLD_MODE=low_far` with
  `VOXGUARD_LOW_FAR_THRESHOLD` uses a manually chosen threshold instead.

## Tests and checks

```bash
make lint                         # ruff check + ruff format --check
make test                         # pytest; torch, ffmpeg and Docker tests skip when unavailable
make docker-smoke                 # builds the images and runs the Docker smoke tests
```

CI runs ruff and pytest (with ffmpeg, without torch) and the Docker smoke tests on every PR.

## Documentation

- [docs/architecture.md](docs/architecture.md): components, codec simulation, latency
- [docs/deployment.md](docs/deployment.md): Docker, compose, real detector, Hugging Face Space
- [docs/benchmarking.md](docs/benchmarking.md) and [docs/benchmarks.md](docs/benchmarks.md):
  latency method and measured results
- [docs/results.md](docs/results.md): evaluation results (generated)
- [docs/STATUS.md](docs/STATUS.md): progress, limitations, next steps
