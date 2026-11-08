# Status

Each person updates ONLY their own section at the end of a working session.

## Person 1 - ML Core + repo owner (Aarya)

- **Done:** scaffold, 3a download + manifests, 3b codecs + augment, 3c datasets + metrics merged; 3d RawNet2 + train loop merged; 3e SSL-AASIST + Kaggle guide merged; 3f evaluate.py merged; 3g RealDetector opened (API serves it end to end on a smoke checkpoint). Reviewed and merged Person 2's API skeleton (PR #3).
- **In progress:** getting LA.zip. Edinburgh DataShare returns 403 to Kaggle and Colab IPs, so it is downloaded on the Mac and uploaded to Kaggle as a private dataset (`--from-zip`).
- **Blocked:** none.
- **Next:** LA.zip to Kaggle, first real RawNet2 runs (clean vs codec-aug), then SSL-AASIST; M1 check on 15 Nov.
- **Environment note:** local ffmpeg has G.711, Opus and AMR-NB; AMR-WB and GSM are missing locally (Kaggle's ffmpeg 6.1.1 to be checked with `python -m voxguard.data.codecs`).

## Person 2 - Product & Platform

- **Done:** repo audit; PR 1 `feat/api-skeleton` opened (FastAPI app factory, `/health`, `/v1/analyze`, audio decoding, SQLite `sessions` table, upload/duration limits, API tests, CI runs the `api` extra).
- **In progress:** awaiting review of PR 1.
- **Blocked:** none. `simulate_codec=true` returns 501 until `data/codecs.py` lands.
- **Next:** PR 2 prevention layer (flagged sessions, alerts, blocklist, `/v1/sessions`, `/v1/blocklist`, `/v1/alerts`); PR 3 Streamlit EN/HI UI.
