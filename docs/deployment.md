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

## Local: docker compose (the supported fallback)

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

The API reads `.env` (or the file named by `VOXGUARD_ENV_FILE`). All variables are listed in
`.env.example`; the ones that matter for deployment:

| Variable | Default | Notes |
|---|---|---|
| `VOXGUARD_DETECTOR` | `stub` | `real` needs an image built with `INSTALL_ML=1` and `VOXGUARD_MODEL_REPO` |
| `VOXGUARD_MODEL_REPO` | (none) | local path inside the container or a Hugging Face repo id |
| `HF_TOKEN` | (none) | only for a private model repo; read by `huggingface_hub` at runtime |
| `VOXGUARD_THRESHOLD` | `0.5` | placeholder until evaluation results give an EER threshold |
| `VOXGUARD_THRESHOLD_MODE` / `VOXGUARD_LOW_FAR_THRESHOLD` | `eer` / (none) | `low_far` refuses to start without an explicit threshold |
| `VOXGUARD_SIMULATE_CODEC` | (none) | codec for `simulate_codec=true`; unset means the API answers 501 |
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

G.711 (`g711_mulaw`, `g711_alaw`) is built into ffmpeg and always available. Set
`VOXGUARD_SIMULATE_CODEC=g711_mulaw` (or another codec the report lists as OK) in `.env` and
restart; then:

```bash
curl -F file=@call.wav -F simulate_codec=true http://localhost:8000/v1/analyze
```

The response's `codec_applied` names the codec. Behaviour and errors are in
`docs/architecture.md`.

### Real detector

```bash
INSTALL_ML=1 docker compose build api     # large download
# .env:
#   VOXGUARD_DETECTOR=real
#   VOXGUARD_MODEL_REPO=<hf-user>/<private-model-repo>   (or a path mounted into the container)
#   HF_TOKEN=<token with read access>                    (only for a private repo)
docker compose up
```

The checkpoint is downloaded at startup into the `/data/hf-cache` volume, so later restarts reuse
it. `/health` then reports `"detector": "real"` and the checkpoint's `model_version`. To use a
local checkpoint instead, mount it, e.g. add `- ./runs/best:/models:ro` under the API's
`volumes` and set `VOXGUARD_MODEL_REPO=/models`.

## Private Hugging Face Space (prepared, not deployed)

`docker/Dockerfile.space` runs the API on 127.0.0.1:8000 inside the container and Streamlit on
port 7860, which is what a Docker Space exposes. `docker/space_start.sh` starts the API, waits
for `/health`, and exits with the API's error if it fails to start, so the Space shows it in its
logs.

Manual steps (need your Hugging Face account; nothing here has been deployed):

1. On huggingface.co: **New Space**, SDK **Docker**, visibility **Private** (CLAUDE.md section 15),
   hardware **CPU basic**.
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
   a **Secret** only if the real detector loads a private model repo.
5. For the real detector, set the build arg by changing `ARG INSTALL_ML=0` to `ARG INSTALL_ML=1`
   in the Space's `Dockerfile` (Spaces do not take build args), and set `VOXGUARD_DETECTOR=real`
   and `VOXGUARD_MODEL_REPO`.
6. `git add . && git commit -m "Deploy VoxGuard" && git push`. The Space builds and starts; open
   it and check the sidebar shows "Connected".

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
| `dependency failed to start: container voxguard-api-1 is unhealthy` | The API did not pass `/health`: see `docker compose logs api`. |
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
