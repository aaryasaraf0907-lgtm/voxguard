# Status

Each person updates ONLY their own section at the end of a working session.

## Person 1 - ML Core + repo owner (Aarya)

- **Done:** scaffold, 3a-3g (data, codecs, datasets/metrics, RawNet2 + train, SSL-AASIST + Kaggle guide, evaluate.py, RealDetector) merged; CLAUDE.md fixes (DataShare, merge policy). Reviewed and merged Person 2's #3, #6, #14, #15, #18. Branch ruleset protects `main` and `dev`. ASVspoof 2019 LA downloaded and verified (train 25,380 / dev 24,844 / eval 71,237). Training-set codec variants built locally (2 per clip; G.711 µ/A-law, AMR-NB, Opus; 76,140 rows, 2.3 GB). #17 Kaggle CLI runner: code, LA and prep data are private Kaggle datasets; a GPU smoke run passed (T4: ~62 clips/s).
- **In progress:** RawNet2 full runs on Kaggle, 20 epochs each: `voxguard-rawnet2-clean` (~4.5 h) and `voxguard-rawnet2-codec-aug` (~9 h, queued behind the clean run).
- **Blocked:** none.
- **Next:** fetch checkpoints; eval-set codec copies (per-codec, incl. a held-out codec) and `evaluate.py` for the RawNet2 2x2 matrix; faster data loading + both T4s before SSL-AASIST (which likely needs next week's GPU quota, resets 14 Nov); M1 promotion on 15 Nov.
- **Environment note:** local ffmpeg has G.711, Opus and AMR-NB; AMR-WB and GSM are missing locally.

## Person 2 - Product & Platform

- **Done:** PR #3 API skeleton, PR #6 prevention layer (alerts, blocklist, `/v1/sessions`, `/v1/blocklist`, `/v1/alerts`, `eer`/`low_far` threshold modes), PR #14 Streamlit demo (EN/HI; upload or microphone; flagged sessions, blocklist and alerts tabs; 4-decimal alert messages; SQLite-only `VOXGUARD_DB_URL`) and PR #15 codec simulation (`simulate_codec=true` runs a real ffmpeg encode/decode with the codec from `VOXGUARD_SIMULATE_CODEC`; unset 501, missing ffmpeg 503, failure 500; non-16 kHz input resampled with ffmpeg's band-limited resampler; 30 s timeout per ffmpeg call in `data/codecs.py`).
- **In progress:** PR `feat/infrastructure-deployment` (open, awaiting review): Dockerfiles for API (ffmpeg, stub by default, `INSTALL_ML=1` for the real detector), UI and a single-container Hugging Face Space image; `docker-compose.yml` with a `/data` volume for SQLite and the HF cache, health checks, non-root containers; `.streamlit/config.toml` upload limit 25 MB (PR #14 review item 1); ffmpeg in CI (PR #15 review item 2) and a Docker smoke-test CI job; clearer startup error when the real detector lacks the ml extra; `docs/deployment.md`. The Space is prepared but not deployed (needs an HF account).
- **Blocked:** none.
- **Limitations:** Hindi strings: Aarya's suggested edits are in the PR #14 review; keep the draft notice until they land. The UI's phone-line toggle is still disabled. `low_far` is a manually configured threshold; no evaluation-derived operating point yet. `latency_ms` is detector-only (codec time excluded). No DB migrations.
- **Next:** PR #14 review follow-ups (stale result card, Hindi edits, refetch caching) and enabling the phone-line toggle with a 503 `unavailable` message; deploy the private Space once an HF account is set up; `codec_ms` in the response (schema change, needs Aarya); load `eer_threshold` from `results/*.json` once real results exist.
