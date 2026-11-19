# LineTrue

> Formerly VoxGuard (renamed 19 Nov 2026). Only the display name changed: the code, the `voxguard`
> package, the `VOXGUARD_*` settings and the repository keep the old name.

Codec-aware voice-deepfake (vishing) detector for telephone-quality audio, with a thin
prevention layer and a demo app. University project (Engineering Design & Innovation),
aligned with SIH 2026 problem statement SIH26104 (detection and prevention of voice-cloning
attacks, with latency as a reported metric).

> **Private repository.** Do not publish or describe the method externally. See `CLAUDE.md` section 15.

## What it does

Most audio-deepfake detectors are trained and tested on clean studio audio, but scam calls pass
through lossy telephone codecs (G.711, AMR, GSM, Opus) that remove the artifacts those
detectors rely on. LineTrue trains and evaluates detectors under simulated phone channels, and
wraps them in an application:

- **Detection:** RawNet2 (baseline) and SSL-AASIST (XLS-R + AASIST) models behind one
  `Detector` interface, plus a deterministic stub for development.
- **Prevention:** every analysis is stored; a score at or above the threshold flags the call,
  raises an alert and adds the caller to a blocklist.
- **Demo:** a Streamlit app (English and Hindi) to upload or record audio, optionally pass it
  through a simulated phone line, and review flagged calls, the blocklist and alerts.
- **Live call monitor:** a page (http://localhost:8000/monitor, linked from the app's sidebar)
  that listens through the microphone, or plays a recording, and updates the risk every 2 s
  while the call is going on, warning before the caller is done talking. It listens to a
  microphone or a file, not a phone line.
- **Transfer checks:** during the call, a transfer approval is checked against the call's risk
  and allowed, sent for verification (call-back, OTP or a second approver) or held, with an
  audit trail and a printable incident report per call (no audio kept).

## Quick start: the demo with the real model

You need only `git` and Docker with compose v2 (on Windows, Docker Desktop with WSL
integration), plus a Hugging Face account in the private `voxguard-team` organisation. No Python,
`uv` or `make` is needed for this path.

1. **Token:** on Hugging Face, Settings → Access Tokens → Create new token, type **Read** (not
   Write, not Fine-grained), any name.
2. **Clone:**
   ```bash
   git clone https://github.com/aaryasaraf0907-lgtm/voxguard.git && cd voxguard && git checkout dev
   cp .env.example .env
   ```
3. **Edit `.env`:** change these six lines (they are all already in the file):
   ```bash
   INSTALL_ML=1
   VOXGUARD_DETECTOR=real
   VOXGUARD_MODEL_REPO=voxguard-team/voxguard-ssl-aasist-codec-aug
   HF_TOKEN=<your read token>
   VOXGUARD_API_START_PERIOD=600s
   VOXGUARD_SIMULATE_CODEC=g711_mulaw
   ```
4. **Start:** `docker compose up --build`. Ports 8000 or 8501 already in use? Run
   `VOXGUARD_API_PORT=8001 VOXGUARD_UI_PORT=8502 docker compose up --build` and open port 8502
   below instead.
5. **Open** http://localhost:8501 once the `ui` container has started. It starts only after the
   API is healthy, and the first start downloads the 1.26 GB checkpoint, so this takes a few
   minutes; opening the page earlier gives "connection refused". The sidebar should say
   "Connected" and "Detector: real". Upload a WAV, FLAC, OGG, MP3 or M4A file and press
   **Analyze**. For the live call monitor, use **Open the live call monitor** in the sidebar
   (http://localhost:8000/monitor).

For development (tests, the local run without Docker, training), see [Setup](#setup).

## Status

- Data pipeline, codec simulation, training, evaluation, API, prevention layer, demo UI, Docker
  and benchmarking are implemented.
- **SSL-AASIST** is trained (clean and codec-augmented); the codec-augmented checkpoint is the
  demo model, with its dev-set thresholds published next to it.
- The RawNet2 results in [docs/results.md](docs/results.md) (2x2 matrix, per codec, held-out
  GSM) are **preliminary**: those runs used inverted class weights (both arms alike), and RawNet2
  is being retrained with the fix. The final four-model results will replace them.
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

For development and the local run without Docker (the
[Quick start](#quick-start-the-demo-with-the-real-model) needs none of this):

- `git` and `make`.
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/), e.g.
  `curl -LsSf https://astral.sh/uv/install.sh | sh`. You do not need to install Python
  yourself: uv uses Python 3.11 (from `.python-version`) and downloads it if needed.
- Optional: `ffmpeg` (with `ffprobe`) for the phone-line simulation and for M4A uploads (e.g.
  `sudo apt install ffmpeg`). Not needed for the basic demo with WAV, FLAC, OGG or MP3.
- For the deployment: Docker with compose v2 (on Windows, Docker Desktop with WSL integration).
  For the real detector also a Hugging Face account in the private `voxguard-team` organisation
  with a **read** token (see [Run with Docker Compose](#run-with-docker-compose-the-deployment)).

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
| `VOXGUARD_THRESHOLD` | (none) | explicit decision threshold; overrides the checkpoint's `thresholds.json` (see below); without either, 0.5 |
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
2. **Analyze:** upload a WAV/FLAC/OGG/MP3/M4A file (up to 25 MB, 60 s) or record with the
   microphone, optionally enter a caller ID, and press **Analyze**. The result shows the label,
   score, threshold, detector latency, model version and whether the call was flagged. A
   flagged score less than 1.5 times the threshold also shows a caution that the result is
   uncertain.
3. **Phone line:** turn on **Simulate phone line**, pick a codec (the list shows the codecs this
   server's ffmpeg has; GSM is marked "not used in training", the held-out codec) and analyze
   again to score the same clip after that codec, at the bitrate used in evaluation. When both
   results exist they are shown side by side, with detector latency and codec processing time
   separately. Without ffmpeg (or any codec) the UI explains why nothing was analyzed.
4. **Monitoring:** the other tabs list flagged sessions, the caller blocklist and alerts
   (times in UTC). A new file or recording clears the previous result.

## Run with Docker Compose (the deployment)

Local Docker Compose is how LineTrue is deployed: the API and the UI in two containers on your
machine. With the stub detector:

```bash
cp .env.example .env               # optional; e.g. VOXGUARD_SIMULATE_CODEC=g711_mulaw
docker compose up --build          # UI http://localhost:8501, API http://localhost:8000
# ports taken? VOXGUARD_API_PORT=8001 VOXGUARD_UI_PORT=8502 docker compose up --build
```

With the real detector, set these values in `.env` (from `.env.example`), then run
`docker compose up --build` (again, if the stub stack was already built):

```bash
INSTALL_ML=1
VOXGUARD_DETECTOR=real
VOXGUARD_MODEL_REPO=voxguard-team/voxguard-ssl-aasist-codec-aug
HF_TOKEN=<your read token for voxguard-team>   # Hugging Face: Settings, Access Tokens, type Read
VOXGUARD_API_START_PERIOD=600s
VOXGUARD_SIMULATE_CODEC=g711_mulaw
```

`INSTALL_ML=1` builds the API image with a CPU-only torch (about 0.4 GB of Python wheels on
amd64); keep it in `.env` so a later `--build` does not rebuild the image without torch. The
first start downloads the checkpoint (SSL-AASIST: 1.26 GB), hence the longer health-check grace
period. The UI container starts only once the API is healthy, so open http://localhost:8501
after the `ui` container has started (with the ports-taken variant, http://localhost:8502).
`.env` is git-ignored: never commit it or share the token.

`curl localhost:8000/health` should then show `"detector": "real"`, and the UI sidebar
"Detector: real". `voxguard-ssl-aasist-codec-aug` is the demo model (its `thresholds.json` sets the
operating threshold); the RawNet2 repos are for integration and latency checks only (see Status).
This path has been run end to end on amd64 (Windows with WSL2 and Docker Desktop): with the
SSL-AASIST demo model, from a fresh clone with nothing cached, the first analysis in the UI came
3 min 28 s after `git clone` (18 Nov 2026; an earlier RawNet2 run on a slower connection took
10 min 51 s). That is machine time only: installing Docker, getting Hugging Face access and
editing `.env` are not included, and the time depends mostly on the network. arm64 (Apple
Silicon) has not been run yet. Prerequisites, what was verified, timings and
troubleshooting are in [docs/deployment.md](docs/deployment.md#real-detector).

The API image includes ffmpeg (G.711, GSM and Opus; no AMR); SQLite and the downloaded
checkpoint live on the `voxguard-data` volume. A hosted Hugging Face Space is not part of the
deployment: Docker Spaces need a paid Hugging Face plan, so it is kept only as an optional route
in [docs/deployment.md](docs/deployment.md).

## Using the real detector without Docker

```bash
export VOXGUARD_DETECTOR=real
export VOXGUARD_MODEL_REPO=<path to best.pt, or an HF repo id>
export HF_TOKEN=<token>              # only for the private Hugging Face org
uv run uvicorn voxguard.api.main:create_app --factory --port 8000
```

- Trained checkpoints are in the private Hugging Face org; access needs a token with permission
  for that org. SSL-AASIST: `voxguard-team/voxguard-ssl-aasist-codec-aug` (the demo model) and
  `voxguard-team/voxguard-ssl-aasist-clean`, each with its `thresholds.json`. RawNet2:
  `voxguard-team/voxguard-rawnet2-clean` and `voxguard-team/voxguard-rawnet2-codec-aug`, from the
  runs with the class-weight bug above, to be replaced by the retrained ones. The codec-aug RawNet2 checkpoint (epoch 17) has been loaded through the API and
  the demo on CUDA (13 Nov). Because of the class-weight bug its scores are for checking
  integration and latency only, not detection results.
- The model is loaded once at startup; `/health` then reports `"detector": "real"` and the
  checkpoint's `model_version`. If loading fails the API exits with an error that says what to
  fix; it never falls back to the stub.
- **Threshold:** each checkpoint's thresholds are chosen on the dev set and published as
  `thresholds.json` next to `best.pt` (a `clean` and a `phone` value). The API reads it at
  startup and uses the **phone** threshold for uploads of unknown origin and for simulated phone
  lines; the clean one is only for audio known to be clean (no request can say that yet). An
  explicit `VOXGUARD_THRESHOLD` overrides the file; `low_far` mode uses its own threshold.
  Without a file the detector's default of 0.5 applies, which is not an operating point for any
  trained checkpoint (their scores sit on very different scales; see `docs/results.md`), and the
  API logs a warning. Both SSL-AASIST repos publish their `thresholds.json`; the RawNet2 repos
  do not yet, so with them the API decides at 0.5 and warns.

## Tests and checks

```bash
make lint                         # ruff check + ruff format --check
make test                         # pytest; torch, ffmpeg and Docker tests skip when unavailable
make docker-smoke                 # builds the images and runs the Docker smoke tests
make coverage                     # pytest with a coverage report (fails below the floor)
make audit                        # known vulnerabilities in every locked package
make e2e                          # the demo in headless Chromium (stub detector; needs ffmpeg)
```

CI runs on every PR to `dev` and `main`:

- **lint-test:** ruff, the dependency audit, and pytest with coverage (with ffmpeg, without
  torch). The coverage table is in the job summary.
- **e2e:** the browser tests: uploads (WAV, M4A), the phone line side by side, the
  near-threshold caution, the prevention tabs, Hindi, error messages, a narrow screen, and the
  API going down and back.
- **docker** and **docker-ml:** the Docker smoke tests, then a Trivy scan of each image (fails
  on fixable critical vulnerabilities; fixable high ones are listed in the job summary).

Accepted exceptions, each with a reason, go in `.github/pip-audit-ignore.txt` and
`.trivyignore`. Dependabot opens weekly update PRs to `dev` for the Actions, the lock file and
the Docker base images.

## Documentation

- [docs/architecture.md](docs/architecture.md): components, codec simulation, latency
- [docs/deployment.md](docs/deployment.md): Docker, compose, real detector, Hugging Face Space
- [docs/demo-script.md](docs/demo-script.md): how to present the demo live and record the video
- [docs/benchmarking.md](docs/benchmarking.md) and [docs/benchmarks.md](docs/benchmarks.md):
  latency method and measured results
- [docs/results.md](docs/results.md): evaluation results (generated)
- [docs/real-audio-testing.md](docs/real-audio-testing.md): private sanity check on consented
  real recordings (diagnostic only)
- [docs/STATUS.md](docs/STATUS.md): progress, limitations, next steps
