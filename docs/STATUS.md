# Status

Each person updates ONLY their own section at the end of a working session.

## Person 1 - ML Core + repo owner (Aarya)

- **Done:** scaffold, 3a download + manifests, 3b codecs + augment, 3c datasets + metrics merged; 3d RawNet2 + train loop merged; 3e SSL-AASIST + Kaggle guide merged; 3f evaluate.py merged; 3g RealDetector opened (API serves it end to end on a smoke checkpoint). Reviewed and merged Person 2's API skeleton (PR #3).
- **In progress:** getting LA.zip. Edinburgh DataShare returns 403 to Kaggle and Colab IPs, so it is downloaded on the Mac and uploaded to Kaggle as a private dataset (`--from-zip`).
- **Blocked:** none.
- **Next:** LA.zip to Kaggle, first real RawNet2 runs (clean vs codec-aug), then SSL-AASIST; M1 check on 15 Nov.
- **Environment note:** local ffmpeg has G.711, Opus and AMR-NB; AMR-WB and GSM are missing locally (Kaggle's ffmpeg 6.1.1 to be checked with `python -m voxguard.data.codecs`).

## Person 2 - Product & Platform

- **Done:** PR #3 API skeleton and PR #6 prevention layer (alerts, blocklist, `/v1/sessions`, `/v1/blocklist`, `/v1/alerts`, `eer`/`low_far` threshold modes) merged. PR `feat/streamlit-ui` opened: Streamlit demo (`ui/app.py`, `ui/api_client.py`, `ui/i18n.py`) in English and Hindi against the existing API; analyze via upload or microphone, flagged sessions, blocklist and alerts tabs; "simulate phone line" shown disabled. Two small API fixes from the PR #6 review: alert messages use 4 decimals, non-SQLite `VOXGUARD_DB_URL` is rejected at startup. Tests run locally (ruff clean, no torch): 247 passed, 29 skipped (Aarya's torch/ffmpeg tests).
- **In progress:** awaiting review of the UI PR.
- **Blocked:** none.
- **Limitations:** Hindi strings are a draft and need Aarya's review. `simulate_codec` is still 501 in the API and is never sent by the UI. `low_far` is a manually configured threshold; no evaluation-derived operating point exists yet. `/health` still maps the detector name from the class (an explicit `name` is Aarya's change to the contract). The UI was tested with Streamlit's `AppTest` and a real API process, not in a browser.
- **Next:** codec simulation (`simulate_codec` as a codec name, needs ffmpeg), Docker/compose, deploy; load `eer_threshold` from `results/*.json` once real results exist.
