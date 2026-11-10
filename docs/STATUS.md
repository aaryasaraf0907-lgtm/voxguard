# Status

Each person updates ONLY their own section at the end of a working session.

## Person 1 - ML Core + repo owner (Aarya)

- **Done:** scaffold, 3a-3g (data, codecs, datasets/metrics, RawNet2 + train, SSL-AASIST + Kaggle guide, evaluate.py, RealDetector) merged; CLAUDE.md fixes (DataShare, merge policy). Reviewed and merged Person 2's #3, #6, #14, #15, #18. Branch ruleset protects `main` and `dev`. ASVspoof 2019 LA downloaded and verified (train 25,380 / dev 24,844 / eval 71,237). Training-set codec variants built locally (2 per clip; G.711 µ/A-law, AMR-NB, Opus; 76,140 rows, 2.3 GB). #17 Kaggle CLI runner: code, LA and prep data are private Kaggle datasets; a GPU smoke run passed (T4: ~62 clips/s).
- **In progress:** RawNet2 full runs on Kaggle, 20 epochs each: `voxguard-rawnet2-clean` (~4.5 h) and `voxguard-rawnet2-codec-aug` (~9 h, queued behind the clean run).
- **Blocked:** none.
- **Next:** fetch checkpoints; eval-set codec copies (per-codec, incl. a held-out codec) and `evaluate.py` for the RawNet2 2x2 matrix; faster data loading + both T4s before SSL-AASIST (which likely needs next week's GPU quota, resets 14 Nov); M1 promotion on 15 Nov.
- **Environment note:** local ffmpeg has G.711, Opus and AMR-NB; AMR-WB and GSM are missing locally.

## Person 2 - Product & Platform

- **Done:** PR #3 API skeleton, PR #6 prevention layer (alerts, blocklist, `/v1/sessions`, `/v1/blocklist`, `/v1/alerts`, `eer`/`low_far` threshold modes), PR #14 Streamlit demo (EN/HI; upload or microphone; flagged sessions, blocklist and alerts tabs; 4-decimal alert messages; SQLite-only `VOXGUARD_DB_URL`), PR #15 codec simulation (`simulate_codec=true` runs a real ffmpeg encode/decode with the codec from `VOXGUARD_SIMULATE_CODEC`; non-16 kHz input resampled with ffmpeg's band-limited resampler) and PR #18 Docker infrastructure (API/UI/Space images, compose with a `/data` volume, ffmpeg and Docker smoke tests in CI, `docs/deployment.md`).
- **In progress:** PR `feat/realdetector-benchmarking` (open, awaiting Aarya's review): the API builds `RealDetector` once at startup with `VOXGUARD_DEVICE` (cpu/cuda/auto) and clear startup errors; `python -m voxguard.benchmark` (detector inference, codec preprocessing and HTTP API request times kept separate; results in `results/benchmarks/`, table in `docs/benchmarks.md`); configurable startup waits for the real detector (`VOXGUARD_API_START_PERIOD`, `VOXGUARD_API_START_TIMEOUT`).
- **Blocked:** real-model verification and trained-model benchmarks wait for Aarya's first checkpoint (RawNet2 runs in progress on Kaggle). Committed RawNet2 timings use untrained weights at the training configuration (compute latency only).
- **Limitations:** Hindi strings: Aarya's suggested edits are in the PR #14 review; keep the draft notice until they land. The UI's phone-line toggle is still disabled. `low_far` is a manually configured threshold; no evaluation-derived operating point yet. `latency_ms` is detector-only (codec time excluded; benchmarked separately). The `INSTALL_ML=1` image has not been built (multi-GB CUDA torch; CPU wheel index needs a joint `pyproject.toml` decision). GPU in Docker is not configured. No DB migrations.
- **Next:** benchmark and verify Aarya's trained checkpoint (CPU and GPU) when it is published; PR #14 review follow-ups (stale result card, Hindi edits, refetch caching) and enabling the phone-line toggle with a 503 `unavailable` message; deploy the private Space once an HF account is set up; with Aarya: CPU torch index for the ML image, `codec_ms` in the response, request-level codec choice; load `eer_threshold` from `results/*.json` once real results exist.
