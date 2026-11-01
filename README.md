# VoxGuard

Codec-aware voice-deepfake (vishing) detector for telephone-quality audio, with a thin
prevention layer and a demo app. University project (Engineering Design & Innovation).

> **Private repository.** Do not publish or describe the method externally. See `CLAUDE.md` section 15.

## Status

Scaffold only. See [docs/STATUS.md](docs/STATUS.md) and `CLAUDE.md` for scope and roadmap.

## Quickstart

```bash
make setup   # uv sync (Python 3.11)
make lint
make test
```

Optional extras: `uv sync --extra ml`, `--extra api`, `--extra ui`.

The default detector is a deterministic stub (`VOXGUARD_DETECTOR=stub`), so everything
runs without weights or GPU.

## Layout

See `CLAUDE.md` section 6 and [docs/architecture.md](docs/architecture.md).
