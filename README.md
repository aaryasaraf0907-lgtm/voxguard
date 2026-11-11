# VoxGuard

Codec-aware voice-deepfake (vishing) detector for telephone-quality audio, with a thin
prevention layer and a demo app. University project (Engineering Design & Innovation).

> **Private repository.** Do not publish or describe the method externally. See `CLAUDE.md` section 15.

## Status

Early development: data pipeline, models, API and demo UI exist; no trained model results yet. See [docs/STATUS.md](docs/STATUS.md) and `CLAUDE.md` for scope and roadmap.

## Quickstart

```bash
make setup   # uv sync (Python 3.11)
make lint
make test
```

Optional extras: `uv sync --extra ml`, `--extra api`, `--extra ui`.

The default detector is a deterministic stub (`VOXGUARD_DETECTOR=stub`), so everything
runs without weights or GPU.

## Run the demo locally

The default detector is the stub, so no weights or GPU are needed. Scores it returns are
deterministic placeholders, not real detection results (the UI says so when it is active).

```bash
uv sync --extra api --extra ui

# terminal 1: API on http://localhost:8000
uv run uvicorn voxguard.api.main:create_app --factory --port 8000

# terminal 2: Streamlit UI on http://localhost:8501
uv run streamlit run src/voxguard/ui/app.py
```

- **API address:** the UI reads `VOXGUARD_API_URL` (default `http://localhost:8000`), e.g.
  `VOXGUARD_API_URL=http://localhost:9000 uv run streamlit run src/voxguard/ui/app.py`.
  Other settings are listed in `.env.example`.
- **Language:** use the selector at the top of the sidebar to switch between English and
  Hindi. The Hindi text is a draft that has not been reviewed yet.
- **Using it:** upload a WAV/FLAC/OGG/MP3 file or record with the microphone, then press
  Analyze. The other tabs list flagged sessions, the caller blocklist and alerts (all times UTC).
- **Phone-line simulation:** the API supports it (`simulate_codec=true`, with the codec set by
  `VOXGUARD_SIMULATE_CODEC` and ffmpeg installed; see `docs/architecture.md`). The UI toggle is
  still shown disabled, so recordings analyzed from the UI are used exactly as provided.
- **Database:** SQLite only (`VOXGUARD_DB_URL=sqlite:///...`); other database URLs are rejected
  at startup.

## Run with Docker

```bash
cp .env.example .env               # optional: settings such as VOXGUARD_SIMULATE_CODEC
docker compose up --build          # UI http://localhost:8501, API http://localhost:8000
```

The API image includes ffmpeg and the stub detector; the SQLite database lives on the
`voxguard-data` volume. The real detector, the Hugging Face Space image, every variable and
troubleshooting are covered in [docs/deployment.md](docs/deployment.md).

## Layout

See `CLAUDE.md` section 6 and [docs/architecture.md](docs/architecture.md).
