# Deployment

LineTrue ships three images, all on `python:3.11-slim-bookworm`, installed with `uv sync --frozen`
from the committed `uv.lock`, and run as the non-root user `voxguard` (uid 1000):

| Image | Dockerfile | Contents | Port |
|---|---|---|---|
| API | `docker/Dockerfile.api` | FastAPI + SQLite + ffmpeg (codec simulation and M4A decoding), `api` extra | 8000 |
| UI | `docker/Dockerfile.ui` | Streamlit, `ui` extra (no ffmpeg, no ML stack) | 8501 |
| Space | `docker/Dockerfile.space` | API (internal, 127.0.0.1:8000) + UI, one container | 7860 |

The API and Space images take `--build-arg INSTALL_ML=1` to add the `ml` extra (torch,
torchaudio, transformers, huggingface-hub) that `VOXGUARD_DETECTOR=real` needs. Without it the
images run the stub detector only. The ML variant installs CPU-only torch and torchaudio, not
the CUDA build that `uv.lock` describes for Linux (see
[CPU-only torch](#cpu-only-torch-in-the-ml-images)).

No secrets, tokens, `.env` files or model weights are copied into any image (`.dockerignore`).

**The deployment is local Docker Compose** (below): API + UI with the real detector, as agreed
for the submission (CLAUDE.md section 3, MVP item 9). A hosted Hugging Face Docker Space is only
an **optional route that needs a paid plan**: Hugging Face requires one to create Docker Spaces
(PRO for a personal account, Team or Enterprise for an organisation), and this project uses free
tools only. The Space image and its steps are kept, unchanged, in
[Private Hugging Face Space](#private-hugging-face-space-optional-future-route-needs-a-paid-plan).

## Local: docker compose (the deployment)

With the stub detector (no model, no Hugging Face access, small images):

```bash
cp .env.example .env                  # optional; every setting has a default
docker compose up --build             # or: make docker-up (detached, waits for health)
```

With the real detector, see [Real detector](#real-detector) below.

- UI: http://localhost:8501, API: http://localhost:8000 (`/health`, `/docs`).
- The UI container reaches the API as `http://api:8000` (compose service name). Compose sets
  this itself, so `VOXGUARD_API_URL` in `.env` does not affect the containers.
- The UI starts only after the API's health check passes.
- Stop with `docker compose down` (data kept) or `docker compose down --volumes` (data deleted).

### Persistence

The API stores SQLite at `/data/voxguard.db` on the named volume `voxguard-data`, together with
the Hugging Face download cache (`HF_HOME=/data/hf-cache`). Compose fixes `VOXGUARD_DB_URL`, so a
`VOXGUARD_DB_URL` copied into `.env` cannot move the database off the volume. Sessions, alerts and
the blocklist survive `docker compose down` / `up` and container restarts.

### Configuration

Compose passes `.env` (or the file named by `VOXGUARD_ENV_FILE`) to the API container; a local
`uvicorn` run does not read it (see the README). All variables are listed in `.env.example`; the
ones that matter for deployment:

| Variable | Default | Notes |
|---|---|---|
| `VOXGUARD_DETECTOR` | `stub` | `real` needs an image built with `INSTALL_ML=1` and `VOXGUARD_MODEL_REPO` |
| `VOXGUARD_MODEL_REPO` | (none) | local path inside the container or a Hugging Face repo id |
| `HF_TOKEN` | (none) | only for a private model repo; read by `huggingface_hub` at runtime |
| `VOXGUARD_THRESHOLD` | (none) | explicit threshold; overrides the checkpoint's `thresholds.json`. Without either the detector's 0.5 default applies, which is not a tuned operating point |
| `VOXGUARD_THRESHOLD_MODE` / `VOXGUARD_LOW_FAR_THRESHOLD` | `eer` / (none) | `low_far` refuses to start without an explicit threshold |
| `VOXGUARD_SIMULATE_CODEC` | (none) | codec for `simulate_codec=true`; unset means the API answers 501. With these images use `g711_mulaw` (or `gsm`/`opus`); AMR is not available |
| `VOXGUARD_DEVICE` | `cpu` | real detector device: `cpu`, `cuda` or `auto` |
| `VOXGUARD_API_START_PERIOD` | `20s` | API health-check grace period (compose); `600s` for the real detector |
| `VOXGUARD_API_START_TIMEOUT` | 600 (real) / 60 | Space image only: seconds `space_start.sh` waits for the API |
| `VOXGUARD_MAX_UPLOAD_MB` / `VOXGUARD_MAX_AUDIO_SECONDS` | `25` / `60` | API limits |
| `VOXGUARD_HIGH_VALUE_INR` | `50000` | transfers at or above this are high value (stricter actions; see `docs/architecture.md`) |
| `VOXGUARD_TRANSACTION_POLICY` | empty (the default table) | JSON, every tier `[normal, high-value]` action: `allow`, `step_up` or `hold` |
| `VOXGUARD_ALERT_WEBHOOK_URL` / `VOXGUARD_ALERT_WEBHOOK_SECRET` | empty | optional signed alert webhook; the secret (16+ characters) is required with the URL |
| `VOXGUARD_MAX_STREAMS` / `VOXGUARD_MAX_STREAM_SECONDS` | `4` / `600` | live calls at once on `/v1/stream` and the longest call, in seconds of audio |
| `VOXGUARD_STREAM_ORIGINS` | empty | extra browser origins allowed to open `/v1/stream` (comma-separated); the API's own `/monitor` page always may |
| `VOXGUARD_MONITOR_URL` (UI) | `<VOXGUARD_API_URL>/monitor` | the sidebar link to the live call monitor; compose sets `http://localhost:${VOXGUARD_API_PORT}/monitor`, since the browser cannot reach `http://api:8000` |
| `VOXGUARD_MAX_RUNNING_ANALYSES` / `VOXGUARD_MAX_WAITING_ANALYSES` | `2` / `8` | analyses at once and how many more may wait (up to 45 s); beyond that 503 "busy" with `Retry-After`. Raise them only with more memory (see "Load test") |
| `VOXGUARD_UI_MAX_UPLOAD_MB` | `25` | UI uploader limit (whole MB); keep it at or below the API limit |
| `INSTALL_ML` | `0` | build arg for the API image |
| `VOXGUARD_API_PORT` / `VOXGUARD_UI_PORT` | `8000` / `8501` | host ports |
| `VOXGUARD_BIND_ADDRESS` | `127.0.0.1` | host address the two ports are published on. The default is this machine only, because the API has no authentication; `0.0.0.0` also serves other devices on the network (use only on a network you trust) |

Invalid values stop the API at startup with an error naming the variable (see
Troubleshooting).

### Codec simulation

The API image installs Debian's ffmpeg, which also provides `ffprobe`. Both are needed to
accept M4A uploads (and FLAC files libsndfile rejects); without them those uploads get a 503
and WAV, FLAC, OGG and MP3 still work. Check which encoders it has with:

```bash
docker compose exec api python -m voxguard.data.codecs
```

In CI (Debian bookworm's ffmpeg in the API image) the report shows `g711_mulaw`, `g711_alaw`,
`gsm` and `opus` as OK and `amr_nb` / `amr_wb` as missing: Debian's ffmpeg is built without the
OpenCORE AMR encoders, so AMR simulation needs a different ffmpeg build. G.711 is built into
ffmpeg and always available. Set
`VOXGUARD_SIMULATE_CODEC=g711_mulaw` (or another codec the report lists as OK) in `.env` and
restart; then:

```bash
curl -F file=@call.wav -F simulate_codec=true http://localhost:8000/v1/analyze
```

The response's `codec_applied` names the codec. Behaviour and errors are in
`docs/architecture.md`.

### Real detector

**Before you start:**
- Docker with compose v2 (on Windows: Docker Desktop with WSL integration for your distro).
- A Hugging Face account that is a member of the private `voxguard-team` organisation (ask
  Aarya), and a **read** token for it (huggingface.co > Settings > Access Tokens). A read token
  is enough; never use a write token here.
- Network and disk for the downloads: the `INSTALL_ML=1` API image installs about **0.4 GB**
  of Python wheels on amd64 (CPU-only torch 210 MB, torchaudio, and the other `ml` and `api`
  packages, compressed), on top of the Python base image and Debian's ffmpeg. The first start
  downloads the checkpoint (SSL-AASIST: 1.26 GB; RawNet2: 212 MB) and, for SSL-AASIST, the small
  XLS-R `config.json`. The built image takes about
  2.6 GB (sum of its layers; Docker Desktop reports 3.2 GB), so leave several GB free, Docker
  Desktop's disk limit included.

**Steps:**

1. `cp .env.example .env`, then set in `.env` (keep each comment on its own line; values only):

   ```bash
   INSTALL_ML=1
   VOXGUARD_DETECTOR=real
   VOXGUARD_MODEL_REPO=voxguard-team/voxguard-ssl-aasist-codec-aug
   HF_TOKEN=<your read token>
   VOXGUARD_API_START_PERIOD=600s
   VOXGUARD_SIMULATE_CODEC=g711_mulaw
   ```

   Leave `VOXGUARD_THRESHOLD` empty so the checkpoint's `thresholds.json` is used. Set
   `INSTALL_ML=1` **in `.env`**, not only on the command line: compose reads it from `.env` on
   every `--build`, and a later `docker compose up --build` or `make docker-up` with the
   `.env.example` value `0` would rebuild the API image without torch (the API then exits with
   `needs the 'ml' extra`). `.env` is git-ignored; never commit it or paste the token anywhere.
2. `docker compose up --build` (or `make docker-up`). The first build downloads the wheels; the
   first start downloads the checkpoint into the `voxguard-data` volume. The UI starts once the
   API is healthy.
3. Check: `curl http://localhost:8000/health` shows `"detector": "real"` and the checkpoint's
   `model_version`; the UI sidebar shows "Connected" and "Detector: real".

`voxguard-team/voxguard-ssl-aasist-codec-aug` (SSL-AASIST trained with codec augmentation) is
the model for the demo; `voxguard-team/voxguard-ssl-aasist-clean` is the clean-trained
comparison. Both have a `thresholds.json` chosen on the dev set. The two RawNet2 repos hold
checkpoints trained with the class-weight bug (README, Status): use them only to check that the
pipeline runs and how long it takes; they have no `thresholds.json`, so the API warns that it
uses the 0.5 default.

**Verified with SSL-AASIST on amd64 (15 Nov 2026; same machine as below, Docker Desktop
29.8.2):** `voxguard-team/voxguard-ssl-aasist-codec-aug` (HF revision `f376f4f`) in compose:
- the API logged `loaded ssl_aasist/ssl-aasist-codec-aug-v2:voxguard-ssl-aasist-codec-aug:epoch11
  on cpu`, the warm-up, and the `thresholds.json` it read (clean and phone values); `/health`
  reported that `model_version`. The API container's memory peaked at about 1.9 GiB (sampled
  every second during startup);
- `thresholds.json` (model `ssl_aasist/ssl-aasist-codec-aug-v2`, epoch 11, selected on dev) and
  the XLS-R `config.json` (`facebook/wav2vec2-xls-r-300m`, only that file) were downloaded into
  `/data/hf-cache` on the volume; every `/v1/analyze` response and stored session carried the
  file's **phone** threshold, clean audio and phone lines alike;
- codecs: G.711 µ-law and A-law, GSM and Opus simulated; AMR-NB and AMR-WB answered 503 (this
  image's ffmpeg has no AMR encoders);
- the UI showed "Detector: real", the model version and the threshold used, and clean and
  phone-line results side by side (headless Chromium);
- sessions, alerts and the blocklist survived `docker compose down` / `up`; with the cache filled,
  the API restarted without network (checkpoint, `thresholds.json` and XLS-R config all from the
  cache);
- startup stopped (exit code 3, no stub fallback) for a `thresholds.json` naming another epoch,
  a malformed `thresholds.json`, a missing checkpoint, and an invalid token on an empty cache.

On CPU in this container, one 17.7 s recording (five 4 s windows) took about 2.2 s of detector
time per request. These are timings only; detection results are in `docs/results.md`.


**Verified on amd64 (14 Nov 2026; Windows with WSL2, Docker Desktop 29.8.2, i5-13450HX):**
with the `.env` above, `docker compose up --build` built the CPU-only ML image, downloaded
`voxguard-team/voxguard-rawnet2-codec-aug` through a read token and served it:
- `/health` reported `"detector": "real"` and
  `rawnet2/rawnet2-codec-aug:voxguard-rawnet2-codec-aug:epoch17`;
- `/v1/analyze` on a 17.7 s, 16 kHz speech recording: clean, and through G.711 µ-law, G.711
  A-law, GSM and Opus (`codec_applied` set, `codec_ms` reported); AMR-NB and AMR-WB answered
  503, as this ffmpeg has no AMR encoders;
- latency on CPU, 20 requests each, p50 (p95): detector `latency_ms` 98 ms (111) clean and 97 ms
  (103) with G.711; `codec_ms` 95 ms (103); the whole request 114 ms (124) clean and 205 ms (209)
  with G.711. These are timings only: the checkpoint's scores are not detection results;
- the UI (headless Chromium): "Connected", "Detector: real", clean and phone-line results side
  by side, the Hindi UI; no errors;
- persistence: after `docker compose down` and `up`, all sessions, alerts and the blocklist were
  still there and the checkpoint was not downloaded again (healthy in 12 s); with the cache
  filled, the API also starts without network;
- startup failures stop the API with exit code 3 and a message naming the cause, never falling
  back to the stub: an invalid token, no token, a wrong repo id (`could not download the
  checkpoint from the Hugging Face Hub`), no network with an empty cache (`checkpoint not
  found`), and `VOXGUARD_DETECTOR=real` on an image built without `INSTALL_ML=1`.

**15-minute check (CLAUDE.md section 17).** Both runs: a fresh `git clone` of `dev`, `.env` from
`.env.example` with the six values above (written by a script), then `docker compose up --build`,
with an empty build cache (a separate buildx builder) and an empty volume, on an i5-13450HX
laptop (WSL2, Docker Desktop). Time from `git clone` to the first successful analysis in the UI:

| Run | Checkpoint | Clone to first analysis | Build | API start to healthy | Slow steps |
|---|---|---|---|---|---|
| 18 Nov 2026, `dev` @ `056f4fc` | SSL-AASIST codec-aug (the demo model, 1.26 GB) | **3 min 28 s** | 2 min 7 s | 1 min 14 s (download and load 67.6 s, warm-up 0.9 s) | Debian ffmpeg 40 s; image export 41 s; CPU torch 23 s; other Python packages 13 s |
| 14 Nov 2026 | RawNet2 (212 MB) | 10 min 51 s | about 10 min | | Debian ffmpeg 4 min 12 s; Python packages other than torch 4 min 42 s; CPU torch 19 s |

The same steps took very different times on the two days: the 14 Nov run spent most of its time
fetching Debian and Python packages, so the total depends mainly on the network and the package
mirrors, not on the checkpoint size. On 18 Nov the API container used at most 3.4 GiB of memory,
the host kept at least 3.2 GB available and swap stayed under 20 MB; the first analysis (a 4 s
clip) reported 617 ms of detector latency. The separate builder adds image export time (41 s here)
that a normal build mostly skips. **Not included:** installing Docker, getting access to
`voxguard-team` and a read token, and the time a person needs to read the README and edit
`.env`. These are machine runs, one per day on one connection; a slower network or mirror takes
longer. A run by a person following the README has not been timed yet.

**Not verified:** arm64 (Apple Silicon) has not been built or run. Its CPU wheels are pinned with
their own hashes and resolve without CUDA packages, but no arm64 image exists yet. GPU in Docker
is not set up (see below).

**Notes:**

- The detector is built **once**, when the API starts, and requests reuse it. It then runs one
  warm-up inference on 4 s of 16 kHz silence (one window), so the first request's `latency_ms`
  does not include CUDA/cuDNN start-up. Only that path is warmed: the first clip at another
  sample rate (resampling) or longer than about 5 s (several windows at once) can still be
  slower once. Measured on CUDA with RawNet2: a 4 s 16 kHz first request took 15.8 ms (1163.8 ms
  before the warm-up existed); a 6 s 44.1 kHz stereo one took 46.9 ms first, then 15.7 ms.
- If the detector cannot be built or fails its warm-up, the API exits with an error naming the
  problem (missing `ml` extra, missing `VOXGUARD_MODEL_REPO`, checkpoint not found, Hugging Face
  download error, corrupt checkpoint, `VOXGUARD_DEVICE=cuda` without a GPU, warm-up inference
  failed). It never falls back to the stub.
- The checkpoint is downloaded at startup into the `/data/hf-cache` volume, so later restarts
  reuse it. A full SSL-AASIST checkpoint is GBs, which is why `VOXGUARD_API_START_PERIOD`
  (health-check grace period, default 20 s) should be raised for the real detector.
- **SSL-AASIST loads slowly on CPU.** Aarya measured it on a Mac CPU (13 Nov): about 10 s to
  load, then a 235 ms warm-up (RawNet2: 53 ms). On a first start the multi-GB download comes
  before that, so expect tens of seconds or more on a small CPU host (e.g. a Space's CPU Basic,
  not measured). The warm-up adds well under a second. `VOXGUARD_API_START_PERIOD=600s` (compose;
  the 20 s default is too short) and the Space image's `VOXGUARD_API_START_TIMEOUT` (600 s by
  default for the real detector) cover this.
- `/health` then reports `"detector": "real"` and the checkpoint's `model_version`
  (`<model>[/<run>]:<source>:epoch<N>`, e.g.
  `ssl_aasist/ssl-aasist-codec-aug-v2:voxguard-ssl-aasist-codec-aug:epoch11`).
- **Threshold:** at startup the API reads `thresholds.json` next to the checkpoint (the same
  directory, or the same HF model repo) and logs its `clean` and `phone` values
  (`INFO voxguard.api.detector_loader: thresholds for ...`). It uses the
  phone threshold for every request today (uploads of unknown origin and simulated phone lines).
  An explicit `VOXGUARD_THRESHOLD` overrides the file, so keep it empty in `.env`. A missing file
  keeps the 0.5 default and logs `VOXGUARD_THRESHOLD is the default 0.50 ...`; a malformed file,
  out-of-range values or a `model_version` naming a different model or epoch than the loaded
  checkpoint stop startup (where the checkpoint was loaded from is ignored).
- To use a local checkpoint instead, mount it, e.g. add `- ./runs/best:/models:ro` under the
  API's `volumes` and set `VOXGUARD_MODEL_REPO=/models` (a directory holding `best.pt`, or the
  file itself).
- `VOXGUARD_DEVICE` (`cpu` default, `cuda`, `auto`) picks the device. The compose file does not
  request a GPU: running on CUDA in Docker additionally needs the NVIDIA Container Toolkit and a
  `deploy.resources.reservations.devices` entry, which this project has not set up or tested.

### CPU-only torch in the ML images

Agreed with Aarya (13-14 Nov 2026). `uv.lock` describes torch 2.15.0 from PyPI, which on Linux is
the CUDA build: about 3.0 GB of compressed wheels (torch 539 MB, cuDNN 519 MB, cuBLAS 405 MB,
triton 286 MB, other `nvidia-*` wheels), although the compose deployment runs on CPU. The ML
images therefore install the CPU builds instead, in the Dockerfiles only, so `pyproject.toml` and
`uv.lock` stay as they are for local, Kaggle and Colab setups:

- `docker/torch-cpu.txt` pins `torch==2.15.0+cpu` and `torchaudio==2.11.0+cpu` from
  `https://download.pytorch.org/whl/cpu`, each with one sha256 per architecture (amd64 and
  arm64); `docker/ml_cpu.sh` installs them with `--require-hashes --no-deps`.
- `uv sync` in the image skips torch, torchaudio and the 19 CUDA packages torch pulls in
  (`docker/ml-cpu-skip.txt`), and runs `--inexact` so it keeps the pinned pair.
- **torchaudio is 2.11.0+cpu while `uv.lock` has 2.11.1:** PyPI's 2.11.1 Linux wheels are CUDA
  builds (they need `libcudart.so.13` and `libtorch_cuda.so`, so they cannot load with a CPU
  torch), and the CPU index has no 2.11.1. On amd64 the pinned pair gave bit-identical
  resampling output and RawNet2 scores to the locked pair (14 Nov 2026).
- The build fails unless the result is CPU-only (`docker/check_ml_cpu.py`, run after the torch
  install and again after the last `uv sync`): `torch.version.cuda is None`, `import torchaudio`
  works, the installed versions match the pins, no CUDA, NVIDIA or Triton package is installed,
  and a 44.1 kHz to 16 kHz resample runs.
- Nothing in the running container brings the CUDA build back: `UV_NO_SYNC=1` makes `uv run`
  use the environment as built, and the default user cannot write `/opt/venv`, so a `uv sync`
  fails. Only a root shell in the container (`docker exec -u 0 ... uv sync`) could replace torch
  with uv.lock's CUDA build; do not do that.
- `tests/test_docker_ml_pins.py` (normal CI) keeps the pins and the skip list consistent with
  `uv.lock`; `tests/test_docker_ml.py` (CI job `docker-ml`, or `make docker-ml-smoke`) builds the
  amd64 ML image, audits it, and starts the real detector on an untrained RawNet2 checkpoint.
- The `ml` extra also brings training-only packages (wandb, scikit-learn, numba/llvmlite for
  librosa; about 320 MB installed). Leaving them out would need a `pyproject.toml` change.

## M2 demo runbook (real model in compose)

For a live demo with `voxguard-team/voxguard-ssl-aasist-codec-aug`. Set up the
[real detector](#real-detector) first (`INSTALL_ML=1` and the other values in `.env`).

### 1. Get the model into the volume

Pick one:
- **Read token (the documented path).** Put `HF_TOKEN=<read token>` in `.env`. The first start
  downloads the 1.26 GB checkpoint into the `voxguard-data` volume.
- **From a local Hugging Face cache, offline, no token.** Use this on a machine that already
  downloaded the checkpoint (for example with `huggingface-cli download` or an earlier local run).
  1. Create the volume without starting anything:
     `docker compose create`.
  2. Copy the snapshot files into the volume. `cp -L` turns the cache's symlinks into real
     files, and the container has no network:

     ```bash
     docker run --rm --network none -v voxguard_voxguard-data:/data \
       -v ~/.cache/huggingface/hub:/src:ro --entrypoint sh voxguard-api:local -c '
       for repo in models--voxguard-team--voxguard-ssl-aasist-codec-aug models--facebook--wav2vec2-xls-r-300m; do
         rev=$(cat /src/$repo/refs/main); dst=/data/hf-cache/hub/$repo
         mkdir -p $dst/refs $dst/snapshots && cp /src/$repo/refs/main $dst/refs/main
         cp -rL /src/$repo/snapshots/$rev $dst/snapshots/$rev
       done'
     ```

  3. Add `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` to `.env`, and leave `HF_TOKEN` out.

  The copy needs only `best.pt`, `thresholds.json` and the model card from the checkpoint repo,
  and XLS-R's `config.json`.

### 2. Start and pre-flight check

```bash
docker compose up -d --wait
curl -s http://localhost:8000/health
docker compose logs api | grep -E "loaded|warm-up|thresholds"
```

- `/health` shows `"detector": "real"` and
  `ssl_aasist/ssl-aasist-codec-aug-v2:voxguard-ssl-aasist-codec-aug:epoch11`.
- The log shows `thresholds for ... (selected on dev): clean 0.01378, phone 0.01268`. Every
  response then carries the phone threshold, 0.012676. Keep `VOXGUARD_THRESHOLD` empty.
- `curl -s http://localhost:8000/v1/codecs` lists G.711, GSM and Opus as available. AMR is
  unavailable in this image's ffmpeg.
- The UI sidebar (http://localhost:8501) shows "Connected" and the detector and model.
- **Run one analysis just before presenting, then one more.** The first request after the stack
  has been idle can be slower; the next one is back to normal (see "Idle latency" below). Repeat
  this before the live part if the stack has been idle for ten minutes or more, keep the laptop
  plugged in on a performance power mode, and close heavy apps. The full presenting sequence is
  in the [demo script](demo-script.md).

### 3. During the demo

- **Use clear cases.** A flagged score below 1.5× the threshold also shows the yellow caution
  "Close to the decision threshold: this result is uncertain". Say so; such results are hints,
  not verdicts.
- **Choose the genuine demo clips beforehand** with the
  [sanity tool](real-audio-testing.md), on the same model, and check how each one scores. The
  codec-aug model flags a noticeable share of off-domain real speech as fake at the phone
  threshold (Aarya's In-the-Wild evaluation; numbers to follow in `docs/results.md`). An
  unchecked live voice could be flagged during the demo.
- **Phone line:** turn on **Simulate phone line** to score the same clip after a codec. The two
  results are shown side by side.
- **Formats:** WAV, FLAC, OGG, MP3 and M4A uploads; at most 25 MB and 60 s.
- **Privacy:** uploaded audio is not stored. The database keeps only each session's score,
  threshold, label, flag, codec, model version, latency and optional caller ID. Decoding temp
  files are removed after each request.

### 4. Reset the demo data (keeps the model)

Before presenting, clear old sessions, alerts and the blocklist without deleting the model
cache:

```bash
docker compose stop
docker compose run --rm --no-deps --entrypoint sh api -c \
  'rm -f /data/voxguard.db /data/voxguard.db-journal /data/voxguard.db-wal /data/voxguard.db-shm'
docker compose up -d --wait
```

The API creates an empty database on start. **Do not use `docker compose down -v` for this:** it
deletes the whole volume, model cache included.

### 5. If something stops

- **The API exits on its own:** compose restarts it (`restart: unless-stopped`). Measured: healthy
  again 13 s after an in-process exit, data intact.
- **A container stopped with `docker stop` or `docker kill`:** Docker treats this as a manual
  stop and does not restart it. Run `docker compose up -d --wait`.
- **After `docker compose down` / `up`:** sessions, alerts and the blocklist persist. Measured:
  healthy again in 22 s.

### 6. Rehearse without touching the demo data

Run a second stack under another project name. Give it its own image tags in a small override
file (so `voxguard-*:local` are not retagged) and its own ports:

```bash
VOXGUARD_API_PORT=18100 VOXGUARD_UI_PORT=18601 \
  docker compose -p voxguard-rehearsal -f docker-compose.yml -f rehearsal.override.yml up -d --wait
```

with `rehearsal.override.yml` (kept outside the repo, or git-ignored):

```yaml
services:
  api:
    image: voxguard-api:rehearsal
  ui:
    image: voxguard-ui:rehearsal
```

It gets its own `voxguard-rehearsal_voxguard-data` volume and database. Copy the model in as in
step 1, using that volume name and `voxguard-api:rehearsal`.

**Verified (17 Nov 2026, `dev` @ `51c1acf`; i5-13450HX, WSL2, Docker Desktop 29.8.2).** Run in
an isolated project as in step 6, with the model copied from a local cache and offline. The run
covered:
- **The model:** the copied files were byte-identical to the cache. The image built in 79 s with
  a warm build cache.
- **Start-up:** healthy in 17 s; load 7.4 s, warm-up 566 ms. `/health`, the log and every
  response showed the model and the phone threshold above.
- **22 API checks**, all passing:
  - WAV, FLAC, M4A with its index at the end, and a FLAC that libsndfile rejects;
  - every codec, with AMR answering 503;
  - unknown codec 422, unsupported bytes 415, over 60 s 422, empty upload 400, over 25 MB 413;
  - the lists, and a blocklist entry.

  Detector `latency_ms` was 467–535 ms per request for 3–5 s clips on CPU, and `codec_ms`
  86–139 ms. These are single requests, not a benchmark.
- **15 UI checks in headless Chromium**, English and Hindi, all passing:
  - the live caution on a consented recording that scored 1.013× the threshold, and 1.077×
    through G.711;
  - side by side; an M4A upload; the monitoring tabs.
- **Operations:**
  - no uploaded audio found in the database, the volume or the container;
  - persistence across `down` / `up`;
  - an automatic restart after an in-process exit;
  - no restart after `docker kill`, as described above;
  - the reset in step 4: lists empty afterwards, model cache unchanged.
- **Manual check in Chrome on Windows** (Aditya, same stack):
  - the English and Hindi interfaces render correctly, Devanagari included;
  - the flagged-calls, blocklist and alerts tabs load;
  - a consented M4A recording was analysed by the real model: fake score 0.012911 against the
    threshold 0.012676, flagged, with the red flagged banner and the yellow near-threshold
    caution.

  The UI showed a detector latency of 6681.1 ms for that request. It came about 12 minutes after
  the last restart, with the API idle in between. Three requests right afterwards took 671,
  522 and 551 ms, so it did not happen again. It is one observation, not a benchmark; the
  measurement below followed it up.

**Idle latency (measured 18 Nov 2026, `dev` @ `76813dd`; same machine, image and model, CPU,
offline, an isolated compose project).** 49 requests in three conditions, each repeated three
times, with the container's major page faults, refaults, memory and CPU pressure and the host's
swap recorded around every request:
- **Fresh start, no idle:** the first request of each clip length (1, 2 and 3 windows, and the
  M4A recording) was no slower than the second (ratio 0.93–1.14 in all 12 cases). The startup
  warm-up already covers this, so a warm-up per clip length would not help.
- **After 13–15 minutes idle:** the first request's detector latency was 574–2223 ms (median
  713 ms), against about 490 ms when warm; every following request was back to 450–530 ms.
  None of the slow requests had major page faults, refaults, swap use or memory pressure, so
  the model's memory was not paged out.
- **M4A after idle:** the ffmpeg decoder's first start after idle read its files from disk again
  (about 210 major faults), adding about 0.4–0.8 s to the whole request (1.0–1.4 s instead of
  about 0.6 s). This is not part of the detector latency.
- **The 6.7 s of 17 Nov was not reproduced** (the slowest was 2.2 s).

The slowdown after idle is consistent with the CPU or the WSL2 virtual machine leaving a
power-saving state, but CPU frequency was not measured, so its cause is not established. One
warm-up request removes it, which is why the pre-flight step runs one analysis and then another.
One machine and one run; other hosts may differ.

## Private Hugging Face Space (optional future route; needs a paid plan)

**Not part of the MVP and not deployed.** Hugging Face requires a paid plan to create Docker
Spaces: PRO for a personal account, Team or Enterprise for an organisation (the CPU Basic
hardware itself is free). On 13 Nov 2026 creating a Docker Space under `voxguard-team` was still
locked, even with the write role in the organisation. The project uses free tools only and is not
buying a plan for this, so local Docker Compose (above) is the selected deployment. The image and
steps below are kept so the Space can be deployed later if a plan becomes available; nothing in
them has changed.

`docker/Dockerfile.space` runs the API on 127.0.0.1:8000 inside the container and Streamlit on
port 7860, which is what a Docker Space exposes. `docker/space_start.sh` starts the API, waits
for `/health`, and exits with the API's error if it fails to start, so the Space shows it in its
logs. The wait is `VOXGUARD_API_START_TIMEOUT` seconds: 600 by default with
`VOXGUARD_DETECTOR=real` (the first start downloads the checkpoint), 60 otherwise.

Manual steps (need your Hugging Face account and a plan that allows Docker Spaces; nothing here
has been deployed):

1. On huggingface.co: **New Space**, SDK **Docker**, visibility **Private** (CLAUDE.md section 15),
   hardware **CPU basic**. Creating the Space and pushing to it need **write** access: do step 1
   on the website, and use a token with write access to that Space for the `git push` in step 6.
   A read-only token cannot push.
2. Clone the Space repo and copy the files the image needs:

   ```bash
   git clone https://huggingface.co/spaces/<hf-user>/<space-name> space
   cp -r pyproject.toml uv.lock LICENSE src docker .dockerignore space/
   cp docker/Dockerfile.space space/Dockerfile
   ```

3. Replace `space/README.md` with the Space metadata (the Space's own README, not this repo's;
   hatchling only needs the file to exist):

   ```markdown
   ---
   title: LineTrue
   sdk: docker
   app_port: 7860
   ---
   Private LineTrue demo.
   ```

4. In the Space **Settings > Variables and secrets**: add `VOXGUARD_SIMULATE_CODEC` (e.g.
   `g711_mulaw`) and any other variable from the table above as **Variables**; add `HF_TOKEN` as
   a **Secret** only if the real detector loads a private model repo. That token needs only
   **read** access to the `voxguard-team` org; Spaces pass secrets to the container as
   environment variables, and it is never written to the image or the repo.
5. For the real detector, set the build arg by changing `ARG INSTALL_ML=0` to `ARG INSTALL_ML=1`
   in the Space's `Dockerfile` (Spaces do not take build args), and set `VOXGUARD_DETECTOR=real`
   and `VOXGUARD_MODEL_REPO` (e.g. `voxguard-team/voxguard-ssl-aasist-codec-aug`). Leave
   `VOXGUARD_DEVICE` unset (CPU). The locked torch is the CUDA build from PyPI, so this image is
   several GB and builds slowly even on CPU hardware (see Real detector above).
6. `git add . && git commit -m "Deploy LineTrue" && git push`. The Space builds and starts; open
   it and check the sidebar shows "Connected" and, for the real detector, "Detector: real" with
   the checkpoint's model version. If the API cannot load the model, the Space log shows the
   API's error and the container stops; it never falls back to the stub.

Limitations on a Space:
- Without Space persistent storage (a paid add-on mounted at `/data`), the SQLite database and
  the checkpoint cache are lost on every restart or rebuild.
- If the API process dies after startup, the UI keeps running and shows "API unavailable"
  until the Space restarts.

## Load test

`scripts/load_test.py` (`make load-test`) sends concurrent analyses through the public API in a
closed loop: N workers, each sending the next request as soon as the last one answers. Per level
it runs two warm-up analyses, then at least `--duration` seconds and `--min-requests` requests.
It checks every status, that each caller's blocklist count equals its flagged responses, and,
with `--poll`, that a client polling `/v1/alerts?since=` sees every alert. With `--container`, it
also checks that the container did not restart, stayed healthy and left no ffmpeg process or
temp directory behind. The audio is synthetic (seeded tone plus noise) and is never written to
disk. It refuses demo ports and other machines: run it against a scratch API
(`make sanity-api`) or a scratch compose project, never the demo database. Reports go to a new
directory outside the repo.

| Mix | Requests |
|---|---|
| A | 4 s, 16 kHz WAV, no codec (the section 10 clip) |
| B | A with `codec=g711_mulaw` |
| C | 2 to 60 s, 16 or 44.1 kHz, half with a codec, about 5 % M4A |
| D | limits: a valid 23 MB upload (200), one byte over the upload limit (413), 61 s (422) |

```bash
# scratch compose project with the real detector on another port, then:
make load-test LOAD_URL=http://127.0.0.1:18310 OUT=$HOME/private/load-1 \
    LOAD_ARGS="--mix A --levels 1,2,4,8 --container <project>-api-1 --poll"
```

### Results (19 Nov 2026)

Real detector (`ssl_aasist/ssl-aasist-codec-aug-v2`, epoch 11) in an isolated compose project
built from this branch: laptop CPU (i5-13450HX, 16 threads, 7.6 GB RAM for WSL), container
limited to 4 GB, model loaded offline from a cache. 60 s per level. Times are per request in ms
(p50 / p95). "Detector" is `latency_ms`, the model only; "client" is the whole request as the
caller sees it, including waiting for an analysis slot. Every level below passed every check:
no 5xx, no time-out, no restart, exact blocklist and alert counts, nothing left behind.

| Mix | At once | Requests | Req/s | Client p50 / p95 | Detector p50 / p95 | Codec p95 | Peak memory |
|---|---|---|---|---|---|---|---|
| A | 1 | 124 | 2.1 | 484 / 516 | 474 / 506 | | 1.83 GiB |
| A | 2 | 158 | 2.6 | 738 / 901 | 728 / 890 | | 1.90 GiB |
| A | 4 | 168 | 2.8 | 1448 / 1513 | 721 / 753 | | 1.85 GiB |
| A | 8 | 173 | 2.8 | 2869 / 2996 | 715 / 754 | | 1.85 GiB |
| B | 1 | 92 | 1.5 | 654 / 687 | 561 / 592 | 90 | 1.91 GiB |
| B | 8 | 156 | 2.5 | 3212 / 3408 | 705 / 737 | 118 | 1.96 GiB |
| C | 1 | 34 | 0.5 | 720 / 6036 | 618 / 5999 | 194 | 2.16 GiB |
| C | 4 | 41 | 0.6 | 3716 / 18825 | 1788 / 9486 | 273 | 2.64 GiB |
| D | 2 | 34 | 1.1 | 67 / 5292 | 5012 / 5171 | | 3.11 GiB |
| A, burst | 48 | 15,702 | 248 | 140 / 190 | 1217 / 1389 | | 2.29 GiB |

- **Section 10 goal** (4 s clip, one at a time, detector p95 under 2 s on CPU): met, 506 ms.
- **Demo** (two people at once, whole request p95 under 5 s, no errors): met, 901 ms.
- **Capacity:** about 2.75 analyses per second on this laptop. More people at once only makes
  each wait longer (8 at once: about 3 s per 4 s clip). Long clips are the slow ones: a 60 s
  clip takes about 6 s alone.
- **Burst:** 104 analysed, 15,598 answered at once with 503 "busy" and `Retry-After: 5`
  (p95 190 ms), memory 2.3 GiB, `/health` and the lists still answering (p95 under 0.25 s).

### What the load test found and changed

Before these changes, at the same settings:

| Problem | Seen | Fix |
|---|---|---|
| 16 or more analyses at once gave 500 "Could not store session" (SQLite "database is locked": readers blocked the commit and the 5 s lock wait ran out) | stub detector, 17 of 27,134 requests | WAL journal and a 30 s lock wait (`api/db.py`) |
| A client polling `/v1/alerts?since=` missed alerts: timestamps were taken before the commit, so rows committed out of timestamp order (up to 897 in one 20 s level) | stub detector, 2 to 48 at once: up to 12 % missed | the write lock is taken first, then the timestamp (`BEGIN IMMEDIATE`) |
| Four 60 s clips at once, or a burst of 48, got the API OOM-killed (over 4 GB) | real detector, 5 kills | at most 2 analyses run at once and 8 wait; more get 503 "busy" (`api/gate.py`, `VOXGUARD_MAX_RUNNING_ANALYSES` / `VOXGUARD_MAX_WAITING_ANALYSES`) |
| With that limit alone, the burst still reached 4 GB: analyses ran on up to 40 different request threads, and each thread that calls torch keeps its own thread team and memory arenas | real detector, 1 kill | analyses run on 2 dedicated threads; `MALLOC_ARENA_MAX=2` in the API image |

Throughput did not drop with the limit (2.75 analyses/s at 4 at once, before and after), because
the model already used every core for one analysis. Raise the limits only with more memory: a
60 s clip alone peaks around 2.2 GiB in the container.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `api` exits at startup with `VOXGUARD_DETECTOR=real needs the 'ml' extra` | The image was built without the ML stack: `INSTALL_ML=1 docker compose build api`. |
| `api` exits with `VOXGUARD_... must be ...` | An invalid setting in `.env`; the message names the variable. `docker compose logs api` shows it. |
| `dependency failed to start: container voxguard-api-1 is unhealthy` | The API did not pass `/health`: see `docker compose logs api`. With the real detector, the checkpoint download may still be running: raise `VOXGUARD_API_START_PERIOD` (e.g. `600s`). |
| `api` exits with `could not download the checkpoint from the Hugging Face Hub` (401, 403 or repository not found) | `HF_TOKEN` is missing from `.env`, mistyped, or belongs to an account that is not in `voxguard-team`; or `VOXGUARD_MODEL_REPO` is misspelt. Check the token on huggingface.co, fix `.env`, then `docker compose up`. |
| `api` exits with `VOXGUARD_MODEL_REPO checkpoint not found` for a Hugging Face repo id | The container could not reach huggingface.co and the checkpoint is not in the `voxguard-data` cache yet: check the network or proxy, then start again. |
| Build fails in `sh docker/ml_cpu.sh torch` or `... check` | The CPU-only check failed (the message names the cause: a CUDA package, a version that differs from `docker/torch-cpu.txt`, or a wheel whose hash does not match). Do not loosen the check; see [CPU-only torch](#cpu-only-torch-in-the-ml-images). |
| API log shows `Temporary failure in name resolution ... thresholds.json` at startup | No network, but the checkpoint is cached: the API retries the `thresholds.json` lookup, then starts normally. |
| Build fails with `no space left on device` | The ML image needs several GB: free disk, or raise Docker Desktop's disk limit; `docker system df` shows what Docker uses. |
| Space log: `LineTrue API did not become healthy within 60 s` | Raise `VOXGUARD_API_START_TIMEOUT` in the Space variables. |
| Real detector flags almost everything or nothing; API log warns `VOXGUARD_THRESHOLD is the default` | The checkpoint has no `thresholds.json` (or `VOXGUARD_THRESHOLD` is set): upload its `thresholds.json`, or set `VOXGUARD_THRESHOLD` explicitly. |
| `api` exits with `unusable thresholds for VOXGUARD_MODEL_REPO=...` | `thresholds.json` is malformed, has a value outside [0, 1], or its `model_version` names a different model or epoch than the loaded checkpoint: fix the file. |
| UI sidebar shows "API unavailable" | `docker compose ps` should show `api` as healthy. In compose the UI must use `http://api:8000`, not `localhost` (each container has its own localhost). |
| `simulate_codec=true` returns 501 | `VOXGUARD_SIMULATE_CODEC` is not set for the API. |
| `simulate_codec=true` returns 503 | This ffmpeg has no encoder for the chosen codec; pick one the codec report lists as OK. |
| Uploads above 25 MB fail | The API limit (`VOXGUARD_MAX_UPLOAD_MB`); raise both it and `VOXGUARD_UI_MAX_UPLOAD_MB`. |
| `port is already allocated` | Change `VOXGUARD_API_PORT` / `VOXGUARD_UI_PORT`. |
| The API stays stopped after `docker kill` / `docker stop` | A manual stop: the restart policy does not apply. Run `docker compose up -d --wait`. |
| A script uploading more than 25 MB gets a connection reset instead of a 413 | The API rejects an oversized body before reading it. Clients that send `Expect: 100-continue` (curl) receive the 413, and the UI caps uploads at 25 MB. Keep script uploads under the limit. |
| `docker: command not found` in WSL | Enable Docker Desktop > Settings > Resources > WSL integration for your distro. |

## Verifying a deployment

`VOXGUARD_DOCKER_TESTS=1 uv run pytest tests/test_docker_smoke.py -v` (or `make docker-smoke`)
builds the images, starts compose on ports 18000/18501 under a throwaway project name, and
checks health, UI-to-API networking, non-root users, ffmpeg encoders, analyze with and without
codec simulation, persistence across `down`/`up`, startup errors, and the Space image. It removes
its containers and volume afterwards. CI runs it on every PR.
