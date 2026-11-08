# Status

Each person updates ONLY their own section at the end of a working session.

## Person 1 - ML Core + repo owner (Aarya)

- **Done:** scaffold, 3a download + manifests, 3b codecs + augment, 3c datasets + metrics merged; 3d RawNet2 + train loop merged; 3e SSL-AASIST + Kaggle guide merged; 3f evaluate.py merged; 3g RealDetector opened (API serves it end to end on a smoke checkpoint). Reviewed and merged Person 2's API skeleton (PR #3).
- **In progress:** getting LA.zip. Edinburgh DataShare returns 403 to Kaggle and Colab IPs, so it is downloaded on the Mac and uploaded to Kaggle as a private dataset (`--from-zip`).
- **Blocked:** none.
- **Next:** LA.zip to Kaggle, first real RawNet2 runs (clean vs codec-aug), then SSL-AASIST; M1 check on 15 Nov.
- **Environment note:** local ffmpeg has G.711, Opus and AMR-NB; AMR-WB and GSM are missing locally (Kaggle's ffmpeg 6.1.1 to be checked with `python -m voxguard.data.codecs`).

## Person 2 - Product & Platform

- **Done:** PR #3 API skeleton (reviewed and merged by Aarya); PR `feat/prevention-layer` opened: prevention layer in `api/prevention.py` (`flagged = fake_score >= effective threshold`, one rule in one place), `alerts` and `blocklist` tables with atomic blocklist upsert, one transaction per analysis, SQLite foreign keys on, UTC-aware timestamps, `GET /v1/sessions`, `/v1/blocklist`, `/v1/alerts?since=`, threshold modes `eer` (default) and `low_far`. Tests run locally: 152 passed, 8 skipped (ffmpeg/torch not installed here).
- **In progress:** awaiting review of the prevention-layer PR.
- **Blocked:** none.
- **Limitations:** `low_far` uses a manually configured `VOXGUARD_LOW_FAR_THRESHOLD`; there is no evaluation-derived operating point yet (needs Aarya's results). The default detector threshold is 0.5, not an EER-derived value. `simulate_codec=true` still returns 501 (`data/codecs.py` exists now; wiring it needs a codec-name field and ffmpeg). No migrations: existing databases get the new tables via `create_all`, but later column changes to `sessions` would not apply.
- **Next:** Streamlit EN/HI UI; `simulate_codec` wiring; Docker/compose; `/health` detector `name` once Aarya adds it to the detector.
- **Codec simulation (PR `feat/codec-simulation`, awaiting Aarya's joint review with the Streamlit UI PR #14):** `simulate_codec=true` now runs a real ffmpeg encode/decode (`data.codecs.apply_codec`) before inference instead of returning 501; the codec is chosen by `VOXGUARD_SIMULATE_CODEC` (no default; unset gives 501, missing ffmpeg 503, failure 500). `codec_applied` stays `str | None` and carries the codec name. One change in Aarya's `data/codecs.py`: a 30 s timeout on each ffmpeg call (new `FFMPEG_TIMEOUT_S`, `_run(..., timeout=)`). Tests run locally with a static ffmpeg 7.0.2 (user-level, no gsm encoder): 212 passed, 23 skipped (before the PR #14 UI tests).
