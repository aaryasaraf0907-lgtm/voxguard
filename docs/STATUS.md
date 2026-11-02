# Status

Each person updates ONLY their own section at the end of a working session.

## Person 1 - ML Core + repo owner (Aarya)

- **Done:** scaffold (PR #1) merged; 3a download script + manifests (PR #2, open); 3b codecs + augment (open, stacked on 3a). Reviewed and merged Person 2's API skeleton (PR #3).
- **In progress:** LA.zip download (7.64 GB, slow link, resumable); review of PRs #2 and 3b.
- **Blocked:** none.
- **Next:** 3c datasets + metrics, then 3d RawNet2 + train.
- **Environment note:** local ffmpeg has G.711, Opus and AMR-NB; AMR-WB and GSM encoders are missing (run precompute on Kaggle/Colab for the full set).

## Person 2 - Product & Platform

- **Done:** repo audit; PR 1 `feat/api-skeleton` opened (FastAPI app factory, `/health`, `/v1/analyze`, audio decoding, SQLite `sessions` table, upload/duration limits, API tests, CI runs the `api` extra).
- **In progress:** awaiting review of PR 1.
- **Blocked:** none. `simulate_codec=true` returns 501 until `data/codecs.py` lands.
- **Next:** PR 2 prevention layer (flagged sessions, alerts, blocklist, `/v1/sessions`, `/v1/blocklist`, `/v1/alerts`); PR 3 Streamlit EN/HI UI.
