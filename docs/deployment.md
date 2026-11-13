# Deployment

VoxGuard ships three images, all on `python:3.11-slim-bookworm`, installed with `uv sync --frozen`
from the committed `uv.lock`, and run as the non-root user `voxguard` (uid 1000):

| Image | Dockerfile | Contents | Port |
|---|---|---|---|
| API | `docker/Dockerfile.api` | FastAPI + SQLite + ffmpeg, `api` extra | 8000 |
| UI | `docker/Dockerfile.ui` | Streamlit, `ui` extra (no ffmpeg, no ML stack) | 8501 |
| Space | `docker/Dockerfile.space` | API (internal, 127.0.0.1:8000) + UI, one container | 7860 |

The API and Space images take `--build-arg INSTALL_ML=1` to add the `ml` extra (torch,
torchaudio, transformers, huggingface-hub) that `VOXGUARD_DETECTOR=real` needs. Without it the
images run the stub detector only. The ML variant is several GB larger (torch from PyPI
includes CUDA libraries).

No secrets, tokens, `.env` files or model weights are copied into any image (`.dockerignore`).

**Selected deployment for the MVP: local Docker Compose** (below), the CLAUDE.md MVP item 9
option that needs no paid service. A hosted Hugging Face Docker Space is an **optional future
route**: Hugging Face requires a paid plan to create Docker Spaces (PRO for a personal account,
Team or Enterprise for an organisation), and this project uses free tools only. The Space image
and its steps are kept, unchanged, in
[Private Hugging Face Space](#private-hugging-face-space-optional-future-route-needs-a-paid-plan).

## Local: docker compose (the selected MVP deployment)

```bash
cp .env.example .env                  # optional; every setting has a default
docker compose up --build             # or: make docker-up (detached, waits for health)
```

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
| `VOXGUARD_UI_MAX_UPLOAD_MB` | `25` | UI uploader limit (whole MB); keep it at or below the API limit |
| `INSTALL_ML` | `0` | build arg for the API image |
| `VOXGUARD_API_PORT` / `VOXGUARD_UI_PORT` | `8000` / `8501` | host ports |

Invalid values stop the API at startup with an error naming the variable (see
Troubleshooting).

### Codec simulation

The API image installs Debian's ffmpeg. Check which encoders it has with:

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

```bash
INSTALL_ML=1 docker compose build api     # large download (see below)
# .env:
#   VOXGUARD_DETECTOR=real
#   VOXGUARD_MODEL_REPO=voxguard-team/voxguard-rawnet2-codec-aug   (or -clean, or a mounted path)
#   HF_TOKEN=<token with read access>                    (only for a private repo)
#   VOXGUARD_THRESHOLD=                                  (leave empty: thresholds.json is used)
#   VOXGUARD_API_START_PERIOD=600s                       (first start downloads the checkpoint)
docker compose up
```

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
  not measured). The warm-up adds well under a second. The default `600s` grace period for the
  real detector (and `VOXGUARD_API_START_TIMEOUT` in the Space image) covers this.
- `/health` then reports `"detector": "real"` and the checkpoint's `model_version`
  (`<model>:<source>:epoch<N>`).
- **Threshold:** at startup the API reads `thresholds.json` next to the checkpoint (the same
  directory, or the same HF model repo) and logs its `clean` and `phone` values. It uses the
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
- **Image size:** `INSTALL_ML=1` installs the locked torch from PyPI, which on Linux pulls the
  CUDA build and its `nvidia-*` wheels (several GB), even for CPU-only use. A CPU wheel index
  would need a `pyproject.toml` source entry and a re-lock (shared files; to be decided with
  Aarya). The ML image has not been built yet.

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
   title: VoxGuard
   sdk: docker
   app_port: 7860
   ---
   Private VoxGuard demo.
   ```

4. In the Space **Settings > Variables and secrets**: add `VOXGUARD_SIMULATE_CODEC` (e.g.
   `g711_mulaw`) and any other variable from the table above as **Variables**; add `HF_TOKEN` as
   a **Secret** only if the real detector loads a private model repo. That token needs only
   **read** access to the `voxguard-team` org; Spaces pass secrets to the container as
   environment variables, and it is never written to the image or the repo.
5. For the real detector, set the build arg by changing `ARG INSTALL_ML=0` to `ARG INSTALL_ML=1`
   in the Space's `Dockerfile` (Spaces do not take build args), and set `VOXGUARD_DETECTOR=real`
   and `VOXGUARD_MODEL_REPO` (e.g. `voxguard-team/voxguard-rawnet2-codec-aug`). Leave
   `VOXGUARD_DEVICE` unset (CPU). The locked torch is the CUDA build from PyPI, so this image is
   several GB and builds slowly even on CPU hardware (see Real detector above).
6. `git add . && git commit -m "Deploy VoxGuard" && git push`. The Space builds and starts; open
   it and check the sidebar shows "Connected" and, for the real detector, "Detector: real" with
   the checkpoint's model version. If the API cannot load the model, the Space log shows the
   API's error and the container stops; it never falls back to the stub.

Limitations on a Space:
- Without Space persistent storage (a paid add-on mounted at `/data`), the SQLite database and
  the checkpoint cache are lost on every restart or rebuild.
- If the API process dies after startup, the UI keeps running and shows "API unavailable"
  until the Space restarts.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `api` exits at startup with `VOXGUARD_DETECTOR=real needs the 'ml' extra` | The image was built without the ML stack: `INSTALL_ML=1 docker compose build api`. |
| `api` exits with `VOXGUARD_... must be ...` | An invalid setting in `.env`; the message names the variable. `docker compose logs api` shows it. |
| `dependency failed to start: container voxguard-api-1 is unhealthy` | The API did not pass `/health`: see `docker compose logs api`. With the real detector, the checkpoint download may still be running: raise `VOXGUARD_API_START_PERIOD` (e.g. `600s`). |
| Space log: `VoxGuard API did not become healthy within 60 s` | Raise `VOXGUARD_API_START_TIMEOUT` in the Space variables. |
| Real detector flags almost everything or nothing; API log warns `VOXGUARD_THRESHOLD is the default` | The checkpoint has no `thresholds.json` (or `VOXGUARD_THRESHOLD` is set): upload its `thresholds.json`, or set `VOXGUARD_THRESHOLD` explicitly. |
| `api` exits with `unusable thresholds for VOXGUARD_MODEL_REPO=...` | `thresholds.json` is malformed, has a value outside [0, 1], or its `model_version` names a different model or epoch than the loaded checkpoint: fix the file. |
| UI sidebar shows "API unavailable" | `docker compose ps` should show `api` as healthy. In compose the UI must use `http://api:8000`, not `localhost` (each container has its own localhost). |
| `simulate_codec=true` returns 501 | `VOXGUARD_SIMULATE_CODEC` is not set for the API. |
| `simulate_codec=true` returns 503 | This ffmpeg has no encoder for the chosen codec; pick one the codec report lists as OK. |
| Uploads above 25 MB fail | The API limit (`VOXGUARD_MAX_UPLOAD_MB`); raise both it and `VOXGUARD_UI_MAX_UPLOAD_MB`. |
| `port is already allocated` | Change `VOXGUARD_API_PORT` / `VOXGUARD_UI_PORT`. |
| `docker: command not found` in WSL | Enable Docker Desktop > Settings > Resources > WSL integration for your distro. |

## Verifying a deployment

`VOXGUARD_DOCKER_TESTS=1 uv run pytest tests/test_docker_smoke.py -v` (or `make docker-smoke`)
builds the images, starts compose on ports 18000/18501 under a throwaway project name, and
checks health, UI-to-API networking, non-root users, ffmpeg encoders, analyze with and without
codec simulation, persistence across `down`/`up`, startup errors, and the Space image. It removes
its containers and volume afterwards. CI runs it on every PR.
