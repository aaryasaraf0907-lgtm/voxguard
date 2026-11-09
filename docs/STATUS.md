# Status

Each person updates ONLY their own section at the end of a working session.

## Person 1 - ML Core + repo owner (Aarya)

- **Done:** scaffold, 3a-3g (data, codecs, datasets/metrics, RawNet2 + train, SSL-AASIST + Kaggle guide, evaluate.py, RealDetector) merged; CLAUDE.md fixes (DataShare, merge policy). Reviewed and merged Person 2's #3, #6, #14, #15 (Hindi review in #14). Branch ruleset now actually protects `main` and `dev` (lint-test required).
- **In progress:** LA.zip download on the Mac (resumable; download retries now reset on progress); then upload to Kaggle as a private dataset.
- **Blocked:** none.
- **Next:** manifests + codec variants on Kaggle, first real RawNet2 runs (clean vs codec-aug), then SSL-AASIST; M1 promotion on 15 Nov.
- **Environment note:** local ffmpeg has G.711, Opus and AMR-NB; AMR-WB and GSM are missing locally.

## Person 2 - Product & Platform

- **Done:** PR #3 API skeleton, PR #6 prevention layer (alerts, blocklist, `/v1/sessions`, `/v1/blocklist`, `/v1/alerts`, `eer`/`low_far` threshold modes), PR #14 Streamlit demo (EN/HI; upload or microphone; flagged sessions, blocklist and alerts tabs; 4-decimal alert messages; SQLite-only `VOXGUARD_DB_URL`) and PR #15 codec simulation (`simulate_codec=true` runs a real ffmpeg encode/decode with the codec from `VOXGUARD_SIMULATE_CODEC`; unset 501, missing ffmpeg 503, failure 500; non-16 kHz input resampled with ffmpeg's band-limited resampler; 30 s timeout per ffmpeg call in `data/codecs.py`).
- **In progress:** none.
- **Blocked:** none.
- **Limitations:** Hindi strings: Aarya's suggested edits are in the PR #14 review; keep the draft notice until they land. The UI's phone-line toggle is still disabled. `low_far` is a manually configured threshold; no evaluation-derived operating point yet. `latency_ms` is detector-only (codec time excluded). No DB migrations.
- **Next:** PR #14 review follow-ups (upload limit, stale result, Hindi edits) and enabling the phone-line toggle; Docker/compose; deploy; load `eer_threshold` from `results/*.json` once real results exist.
