# Status

Each person updates ONLY their own section at the end of a working session.

## Person 1 - ML Core + repo owner (Aarya)

- **Done:** scaffold merged (PR #1); 3a download script + manifests (`feat/data-download`) opened.
- **In progress:** awaiting review of 3a; LA.zip (7.64 GB) not yet downloaded, waiting for go-ahead.
- **Blocked:** none.
- **Next:** run the LA download, then 3b codecs + augmentation.
- **Environment note:** local ffmpeg has G.711, Opus and AMR-NB; AMR-WB and GSM encoders are missing (build on Kaggle/Colab for the full set).

## Person 2 - Product & Platform

- **Done:** repo audit; PR 1 `feat/api-skeleton` opened (FastAPI app factory, `/health`, `/v1/analyze`, audio decoding, SQLite `sessions` table, upload/duration limits, API tests, CI runs the `api` extra).
- **In progress:** awaiting review of PR 1.
- **Blocked:** none. `simulate_codec=true` returns 501 until `data/codecs.py` lands.
- **Next:** PR 2 prevention layer (flagged sessions, alerts, blocklist, `/v1/sessions`, `/v1/blocklist`, `/v1/alerts`); PR 3 Streamlit EN/HI UI.
