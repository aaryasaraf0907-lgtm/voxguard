# Status

Each person updates ONLY their own section at the end of a working session.

## Person 1 - ML Core + repo owner (Aarya)

- **Done:** scaffold, 3a download + manifests, 3b codecs + augment, 3c datasets + metrics merged; 3d RawNet2 + train loop merged; 3e SSL-AASIST + Kaggle guide merged; 3f evaluate.py merged; 3g RealDetector opened (API serves it end to end on a smoke checkpoint). Reviewed and merged Person 2's API skeleton (PR #3).
- **In progress:** getting LA.zip. Edinburgh DataShare returns 403 to Kaggle and Colab IPs, so it is downloaded on the Mac and uploaded to Kaggle as a private dataset (`--from-zip`).
- **Blocked:** none.
- **Next:** LA.zip to Kaggle, first real RawNet2 runs (clean vs codec-aug), then SSL-AASIST; M1 check on 15 Nov.
- **Environment note:** local ffmpeg has G.711, Opus and AMR-NB; AMR-WB and GSM are missing locally (Kaggle's ffmpeg 6.1.1 to be checked with `python -m voxguard.data.codecs`).

## Person 2 - Product & Platform

- **Done:** PR #3 API skeleton, PR #6 prevention layer (alerts, blocklist, `/v1/sessions`, `/v1/blocklist`, `/v1/alerts`, `eer`/`low_far` threshold modes), PR #14 Streamlit demo (EN/HI; upload or microphone; flagged sessions, blocklist and alerts tabs; 4-decimal alert messages; SQLite-only `VOXGUARD_DB_URL`) and PR #15 codec simulation (`simulate_codec=true` runs a real ffmpeg encode/decode with the codec from `VOXGUARD_SIMULATE_CODEC`; unset 501, missing ffmpeg 503, failure 500; non-16 kHz input resampled with ffmpeg's band-limited resampler; 30 s timeout per ffmpeg call in `data/codecs.py`).
- **In progress:** none.
- **Blocked:** none.
- **Limitations:** Hindi strings: Aarya's suggested edits are in the PR #14 review; keep the draft notice until they land. The UI's phone-line toggle is still disabled. `low_far` is a manually configured threshold; no evaluation-derived operating point yet. `latency_ms` is detector-only (codec time excluded). No DB migrations.
- **Next:** PR #14 review follow-ups (upload limit, stale result, Hindi edits) and enabling the phone-line toggle; Docker/compose; deploy; load `eer_threshold` from `results/*.json` once real results exist.
