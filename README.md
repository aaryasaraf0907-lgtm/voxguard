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
  [docs/results.md](docs/results.md). They are **preliminary**: Aarya's STATUS notes that those
  runs used inverted class weights (both arms alike), and RawNet2 is being retrained with the
  fix. Corrected results will replace them; SSL-AASIST training is in progress.
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

Prerequisites:

- `git` and `make`.
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/), e.g.
  `curl -LsSf https://astral.sh/uv/install.sh | sh`. You do not need to install Python
  yourself: uv uses Python 3.11 (from `.python-version`) and downloads it if needed.
- Optional: `ffmpeg` for the phone-line simulation (e.g. `sudo apt install ffmpeg`), and Docker
  with compose for the Docker run. Neither is needed for the basic demo.

```bash
git clone https://github.com/aaryasaraf0907-lgtm/voxguard.git && cd voxguard
git checkout dev                    # current work; main holds the last milestone
make setup                          # uv sync
uv sync --extra api --extra ui      # what the API and demo need
uv sync --extra api --extra ui --extra ml   # add torch etc. for the real detector (several GB)
```

The repository's `.python-version` makes `uv sync` and `make setup` use Python 3.11, the
version CI and the Docker images use. The package itself also supports 3.12+ (`requires-python`
is `>=3.11`); `uv sync --python 3.12` (or newer) overrides the pin.

Phone-line simulation needs `ffmpeg` on `PATH` (`uv run python -m voxguard.data.codecs` lists
the codecs your ffmpeg supports).

## Configuration

All variables, with defaults, are in [`.env.example`](.env.example). `docker compose` reads
`.env` itself; the local run below does not, so export the variables in the shell that starts
the API, or load the file there with `set -a; . ./.env; set +a`. The main ones:

| Variable | Default | Meaning |
|---|---|---|
| `VOXGUARD_DETECTOR` | `stub` | `stub` or `real` |
| `VOXGUARD_MODEL_REPO` | (none) | real detector: a `best.pt` path, a directory holding it, or a Hugging Face repo id |
| `HF_TOKEN` | (none) | needed to download from the private Hugging Face org |
| `VOXGUARD_DEVICE` | `cpu` | real detector device: `cpu`, `cuda` or `auto` |
| `VOXGUARD_THRESHOLD` | `0.5` | decision threshold (see below) |
| `VOXGUARD_SIMULATE_CODEC` | (none) | default codec for the phone-line toggle, e.g. `g711_mulaw` |
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

If port 8000 is taken, start the API with another `--port` and point the UI at it, e.g.
`VOXGUARD_API_URL=http://localhost:8001 uv run streamlit run src/voxguard/ui/app.py`
(`--server.port` changes the UI's own port).

For the phone-line toggle, start the API with a codec configured (and ffmpeg installed):

```bash
VOXGUARD_SIMULATE_CODEC=g711_mulaw uv run uvicorn voxguard.api.main:create_app --factory --port 8000
```

Using the demo:

1. **Language:** the selector at the top of the sidebar switches between English and Hindi.
   The sidebar also shows whether the API is reachable and which detector it runs.
2. **Analyze:** upload a WAV/FLAC/OGG/MP3 file (up to 25 MB, 60 s) or record with the
   microphone, optionally enter a caller ID, and press **Analyze**. The result shows the label,
   score, threshold, detector latency, model version and whether the call was flagged.
3. **Phone line:** turn on **Simulate phone line**, pick a codec (the list shows the codecs this
   server's ffmpeg has; GSM is marked "not used in training", the held-out codec) and analyze
   again to score the same clip after that codec, at the bitrate used in evaluation. When both
   results exist they are shown side by side, with detector latency and codec processing time
   separately. Without ffmpeg (or any codec) the UI explains why nothing was analyzed.
4. **Monitoring:** the other tabs list flagged sessions, the caller blocklist and alerts
   (times in UTC). A new file or recording clears the previous result.

## Run with Docker

```bash
cp .env.example .env               # optional; e.g. VOXGUARD_SIMULATE_CODEC=g711_mulaw
docker compose up --build          # UI http://localhost:8501, API http://localhost:8000
# ports taken? VOXGUARD_API_PORT=8001 VOXGUARD_UI_PORT=8502 docker compose up --build
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
  needs a token with permission for that org. They come from the runs with the class-weight bug
  above and will be replaced by the retrained ones; SSL-AASIST checkpoints follow when training
  finishes. Aarya has loaded one through the API (PR #23 review); other environments have not
  verified them yet.
- The model is loaded once at startup; `/health` then reports `"detector": "real"` and the
  checkpoint's `model_version`. If loading fails the API exits with an error that says what to
  fix; it never falls back to the stub.
- **Threshold:** `VOXGUARD_THRESHOLD` defaults to 0.5. That is a default, not a tuned or
  validated operating point. `results/<model>.json` records, per checkpoint and test condition,
  the `eer_threshold`: the score at which false accepts equal false rejects on that evaluation
  set. It is a reasonable starting value for the matching checkpoint, but it was measured on the
  evaluation data itself (not on separate validation data), and the current values belong to
  the pre-fix RawNet2 runs. `VOXGUARD_THRESHOLD_MODE=low_far` with `VOXGUARD_LOW_FAR_THRESHOLD`
  uses a manually chosen threshold instead.
- **Do not run a trained checkpoint at 0.5.** The committed RawNet2 `eer_threshold` values range
  from 0.23 to 0.9999 across checkpoints and test conditions, and Aarya reports ~0.001 for the
  first SSL-AASIST checkpoint (53 % accuracy at 0.5; see her section of
  [docs/STATUS.md](docs/STATUS.md)). Set `VOXGUARD_THRESHOLD` from the checkpoint's
  `eer_threshold`. The API logs a warning at startup when the real detector runs on the 0.5
  default in `eer` mode.

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
