# CLAUDE.md — LineTrue

> Shared project brief for BOTH teammates' Claude Code sessions. Read it fully before doing anything.
> If it conflicts with a direct instruction from the user in the session, follow the user and flag the conflict.
> Decisions change: update this file via PR and tell the other person.
> External facts below (dataset URLs, codec availability, model names, licenses) are best-effort. VERIFY them at execution time and report anything that differs.

---

## 1. What we are building

**LineTrue** (display name since 19 Nov 2026; formerly VoxGuard, renamed because a public SIH26104 repo uses that name. Code identifiers stay `voxguard`: the package, `VOXGUARD_*` variables, image, repo and HF repo names) is a codec-aware voice-deepfake (vishing) detector for telephone-quality audio, plus a thin "prevention" layer and a demo app.

- University semester project: Engineering Design & Innovation. Team of **2**. Final submission **mid-December 2026** (~6 weeks from 1 Nov). **All tools must be free.**
- Aligned with SIH 2026 problem statement **SIH26104** (AICTE): "AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks" -> detection AND prevention, and REAL-TIME (latency is a reported metric).
- Secondary goal: portfolio quality for big-tech internship recruiting. Engineering quality counts: typed code, tests, CI, clean small PRs, reproducible scripts, readable README.
- Tertiary goal: a patent-ready technical disclosure (see section 15).

## 2. The core idea (novelty)

Most audio-deepfake detectors are trained and benchmarked on clean, studio-quality speech. Real scam calls pass through lossy telephony codecs (AMR-NB/WB, GSM, G.711, low-bitrate Opus) that strip the spectral artifacts those detectors rely on, so lab-accurate models degrade on real calls.

**Our contribution = a codec-aware training and evaluation regime**: simulate telephony channels during training (randomised codec chain, bitrate, bandwidth limiting, noise), and evaluate under clean AND phone-degraded conditions, including codecs unseen at training time. The architecture itself (XLS-R + AASIST) is NOT the novelty; the regime is.

**Headline result we must produce - the 2x2 matrix**, for each model:

| | Test: clean | Test: phone-degraded |
|---|---|---|
| Train: clean only | EER | EER (expected to blow up) |
| Train: codec-augmented | EER | EER (expected to hold) |

Stretch evidence: train on a subset of codecs, test on an unseen codec.

## 3. Scope

**MVP (must ship):**
1. ASVspoof 2019 LA data pipeline + codec-degradation pipeline (offline precomputed variants + manifests)
2. RawNet2 baseline (from scratch)
3. SSL-AASIST: XLS-R 300M front-end (fine-tuned) + AASIST back-end
4. Evaluation: EER, accuracy, 2x2 matrix per codec, latency p50/p95 per clip
5. `Detector` interface + real detector + stub detector
6. FastAPI service with analyze endpoint, SQLite persistence, **prevention layer** (flagged sessions + blocklist + alerts)
7. Streamlit demo (English + Hindi UI), incl. "simulate phone line" toggle (same clip, clean vs codec-degraded)
8. Docker, GitHub Actions CI (ruff + pytest), README, architecture + results docs
9. Local docker-compose deployment (API + UI, real detector). A private HF Space is optional only: since 2026, Docker Spaces need a paid HF plan (PRO / Team / Enterprise), which the free-only rule excludes (checked 13 Nov 2026)

**Stretch (only after MVP is green, in this order):**
1. Unseen-codec generalization experiment
2. Hindi generalization test (see section 11)
3. Out-of-domain eval on In-the-Wild dataset
4. Second SSL front-end (WavLM-Large or HuBERT) + score-average fusion
5. ONNX export / dynamic quantization for CPU latency
6. WebSocket alerts, packet-loss simulation, load test

**Explicitly dropped (do NOT add unless the user asks):** Redis/Celery, Postgres, Kubernetes, Prometheus/Grafana, Twilio/live telephony, React/Next frontend, DVC. Rationale: 2 people, ~6 weeks, free-only. Inference is ~1 s so nothing needs queueing; SQLite is enough; checkpoints go to Hugging Face Hub instead of DVC.

## 4. Architecture

    [ASVspoof LA (+ extra eval sets)]
            |
      data/download.py -> manifests (csv)
            |
      data/codecs.py (ffmpeg chain) -> degraded variants + manifests
            |
      train.py (RawNet2 | SSL-AASIST), config-driven, resumable
            |
      checkpoints -> Hugging Face Hub (private model repo)
            |
      inference.py  Detector (Real | Stub)  <-- THE CONTRACT (section 7)
            |
      api/ FastAPI  --> SQLite (sessions, blocklist, alerts)
            |
      ui/ Streamlit (EN/HI)  <-- upload / mic record / simulate phone line / flagged list

## 5. Tech stack (final, all free)

| Layer | Choice |
|---|---|
| Language / env | Python 3.11 for dev/CI, `uv` (fallback pip + venv), `pyproject.toml`, src layout; package also runs on 3.12+ (Kaggle ships 3.13) |
| ML | PyTorch, HuggingFace Transformers (`facebook/wav2vec2-xls-r-300m`), torchaudio/soundfile/librosa |
| Codec simulation | FFmpeg via subprocess (G.711 mu-law/A-law native; AMR-NB `libopencore_amrnb`, AMR-WB `libvo_amrwbenc`, GSM `libgsm`, Opus `libopus` if available) |
| Baselines/refs | RawNet2 (ASVspoof 2021 baseline), AASIST (clovaai/aasist), SSL-AASIST (Tak et al. 2022, TakHemlata/SSL_Anti-spoofing). Check licences; implement/adapt with attribution rather than blind copy |
| Tracking | Weights & Biases free tier (optional, must work with `--no-wandb`) |
| Model hosting | Hugging Face Hub (private model repo) |
| API | FastAPI, Pydantic v2, SQLAlchemy 2 + SQLite |
| UI | Streamlit (>=1.39 for `st.audio_input`) |
| Quality | pytest, pytest-benchmark, ruff (lint + format), pre-commit (optional) |
| CI/CD | GitHub Actions (ruff + pytest on PRs to `dev`/`main`) |
| Packaging/deploy | Docker + docker-compose (local); HF Space only if a free route exists again (Docker Spaces need a paid plan as of Nov 2026) |
| Training compute | Kaggle Notebooks / Colab free GPUs; MacBook for dev + smoke tests only |

## 6. Repo layout

    voxguard/
    |- CLAUDE.md
    |- README.md
    |- pyproject.toml
    |- Makefile                  # setup, lint, test, train targets
    |- .env.example
    |- configs/                  # yaml: data, codecs, rawnet2, ssl_aasist, eval
    |- scripts/                  # kaggle/colab runners, one-off helpers
    |- src/voxguard/
    |  |- data/                  # download.py, codecs.py, augment.py, datasets.py, manifests.py
    |  |- models/                # rawnet2.py, aasist.py, ssl_aasist.py
    |  |- metrics.py             # EER, accuracy, latency helpers
    |  |- train.py
    |  |- evaluate.py
    |  |- inference.py           # Prediction, Detector, StubDetector, RealDetector
    |  |- schemas.py             # pydantic API schemas (contract)
    |  |- api/                   # main.py, db.py, prevention.py, deps.py
    |  `- ui/                    # app.py, i18n.py
    |- tests/
    |- docker/                   # Dockerfile.api, Dockerfile.ui (+ docker-compose.yml at root)
    |- docs/                     # STATUS.md, architecture.md, results.md, invention-disclosure.md
    `- .github/                  # workflows/ci.yml, pull_request_template.md

Git ignores (never commit): `data/raw/`, `data/processed/`, `*.pt/*.pth/*.ckpt`, `models/checkpoints/`, `.env`, `.venv/`, `wandb/`, results caches. Datasets are reproduced by scripts; checkpoints live on HF Hub.

## 7. Interface contract (do not change without telling the other person + PR)

Audio conventions: 16 kHz mono float32. Detector does its own preprocessing: resample -> mono -> crop/pad to 64,600 samples (~4 s); clips longer than 4 s are scored with non-overlapping windows and averaged; a leftover of at least ~1 s (16,150 samples) is scored as the clip's last 64,600 samples (overlapping the previous window), never repeat-padded, and a shorter leftover is dropped (changed 15 Nov: padded tails scored well above the other windows on real speech), except in a 4.0-5.05 s clip, where the last 64,600 samples are always scored as a second window (changed 20 Nov, #103: on real speech the first 4 s alone scored above the last 4 s and flagged short genuine clips). Only clips shorter than 4 s are repeat-padded, as in training.

    # src/voxguard/inference.py
    @dataclass(frozen=True)
    class Prediction:
        fake_score: float        # 0.0-1.0, HIGHER = more likely AI-generated
        label: Literal["real", "fake"]
        threshold: float
        latency_ms: float
        model_version: str

    class Detector(Protocol):
        model_version: str
        threshold: float
        def predict(self, waveform: np.ndarray, sample_rate: int) -> Prediction: ...

- `StubDetector`: deterministic fake_score from a hash of the waveform (so tests are repeatable). Used by default in CI and by Person 2 until a checkpoint exists.
- `RealDetector`: loads weights from a local path or HF Hub repo id.
- Selected by env `VOXGUARD_DETECTOR=stub|real`; `VOXGUARD_MODEL_REPO`, `VOXGUARD_THRESHOLD` configure the real one.

REST API (schemas in `schemas.py`):

| Method + path | Purpose |
|---|---|
| `GET /health` | `{status, detector, model_version}` |
| `POST /v1/analyze` | multipart `file`; optional form `caller_id`, `simulate_codec`, `codec` -> `{session_id, label, fake_score, threshold, latency_ms, model_version, flagged, codec_applied, codec_ms}` |
| `GET /v1/codecs` | `{default, codecs: [{name, available, bitrate}]}`: every codec in `data/codecs.CODECS`; `available` = this server's ffmpeg has the encoder; `bitrate` = `data.augment.default_spec(name).bitrate` (null for fixed-rate G.711); `default` = `VOXGUARD_SIMULATE_CODEC` or null |
| `GET /v1/sessions?flagged_only=&limit=` | stored sessions |
| `GET /v1/blocklist` | caller ids auto-blocked, with hit counts |
| `GET /v1/alerts?since=` | alerts raised by the prevention layer |

Phone-line simulation in `/v1/analyze` (agreed in #44's review):
- `codec=<name>` selects the codec and implies simulation, whatever `simulate_codec` says. Without `codec`, `simulate_codec=true` uses `VOXGUARD_SIMULATE_CODEC` (501 if unset). An empty `codec` counts as omitted. An unknown name is 422; a known codec this server's ffmpeg lacks is 503. Another codec is never substituted, and nothing is scored or stored on these errors.
- The codec runs at `default_spec(codec)`'s bitrate, the evaluation sets' condition (no noise in the demo), after resampling to 16 kHz.
- Busy server (added 19 Nov, #92): at most `VOXGUARD_MAX_RUNNING_ANALYSES` (2) analyses run and `VOXGUARD_MAX_WAITING_ANALYSES` (8) wait, up to 45 s; beyond that `/v1/analyze` answers **503 with `Retry-After`**, nothing scored or stored. A 503 without `Retry-After` is still a missing codec or decoder. Session/alert `created_at` is taken under the write lock, so it follows commit order (what `since=` relies on).
- Live-monitoring timeline (added 19 Nov): optional form field `timeline=true` adds `timeline` (per detector window: `start_s`, `end_s`, `fake_score`, `running_score`, `flagged`) and `first_flag_s` (seconds of audio after which the running average first reached the threshold, or null) to the response; both response-only, null without it. Each window is scored with `Detector.predict` on the detector's own windows; the clip score is their mean, so verdict and stored session are unchanged.
- Live call monitor (added 19 Nov): `WS /v1/stream?sample_rate=<8000-48000>[&caller_id=&codec=]` takes mono float32 LE frames and `{"type": "end"}`; replies `ready`, `score` (every 2 s once a 4 s window is in: `t_s`, `window_score`, `running_score`, `flagged`), `alert` (first time the running average of at least 2 windows reaches 1.5x the threshold, `ALERT_FACTOR` in `api/risk.py`, changed 20 Nov; the stored verdict still uses the threshold; `level` is always `high`, kept for clients and webhooks; `ready` carries `alert_factor`), `busy`, `limit`, `final`, `error`. Each window is scored as `/v1/analyze` scores a 4 s clip; at the end one session is stored through the prevention layer with the running average as its score. No audio kept beyond the latest window. `GET /monitor` serves its browser page.
- Transfer checks (added 19 Nov, `api/policy.py`): `POST /v1/transactions/check` `{session_id, amount_inr, beneficiary?}` decides from the call's risk, live (running average of the stream so far) or stored: tier `unknown` (live, fewer than 2 windows) / `low` / `caution` (threshold to 1.5x) / `high`, one tier up if the caller was flagged on an earlier call; action from `VOXGUARD_TRANSACTION_POLICY` per tier and value (`high_value` at or above `VOXGUARD_HIGH_VALUE_INR`, 50,000): `allow`, `step_up` (any one of `callback`, `otp`, `second_approver`) or `hold` (`second_approver` and `callback`). `POST /v1/transactions/{decision_id}/verify` `{method, verified_by, note?}` records a step and releases when done; `GET /v1/transactions[?session_id=]` lists decisions with an append-only audit trail. `GET /v1/sessions/{id}/timeline` returns a live call's stored per-window scores (no audio); `GET /v1/sessions/{id}/report?lang=` a printable incident report. Optional signed webhook (`VOXGUARD_ALERT_WEBHOOK_URL`, HMAC-SHA256 with `VOXGUARD_ALERT_WEBHOOK_SECRET`) for `call_flagged`, `live_alert`, `transaction_step_up`, `transaction_held`, `transaction_released`.
- Evidence trail (added 19 Nov, `api/evidence.py`): every stored session, a live call's windows, each transfer decision and each verification step add a hash-chained entry to `evidence_log` in the same transaction; `GET /v1/evidence/verify[?session_id=]` returns `{ok, entries, head, problems, session_entries}`; webhook payloads carry `evidence_head`. A hold's second approver and call-back need different `verified_by` names (422 otherwise).
- `codec_applied` is the codec actually used, or null. `codec_ms` is the wall time of the whole simulation (resample + ffmpeg encode/decode), null without simulation; response-only, not stored. `latency_ms` stays detector inference only.

Prevention rule (MVP): `fake_score >= threshold` -> session stored as flagged, alert created, `caller_id` (if given) upserted into blocklist with a counter. Threshold modes: `eer` (default) and `low_far` (operating point shifted to reduce missed fakes).

Thresholds (agreed 13 Nov): `thresholds.json` next to each checkpoint (`{model_version, selected_on: "dev", clean, phone, low_far: {clean, phone}, per_codec: {...}}`), chosen on dev by `evaluate.py`; GSM is never used to choose one. In `eer` mode the API uses, in order: an explicit `VOXGUARD_THRESHOLD`; else the file's **phone** threshold for audio of unknown origin and for simulated phone lines, its **clean** threshold only for audio known to be clean (no request field says that yet); else the detector's threshold (0.5). `per_codec` and `low_far` are for reporting only. The file's `model_version` must name the same model and epoch as the loaded checkpoint: `<name>[/<run>]:<source>:epoch<N>` (the run is the training run, e.g. `ssl_aasist/ssl-aasist-codec-aug-v2`, so clean and codec-augmented checkpoints differ; added 14 Nov) is compared on its first and last parts only, since `<source>` is the load location (`best.pt`, a mounted directory, the HF repo name) and differs between `evaluate.py` and the API. A malformed file or a different model or epoch stops startup; a missing file keeps the old behaviour. The response's `threshold` is the one applied.

## 8. Data and codec spec

- **Primary dataset:** ASVspoof 2019 LA (train/dev/eval), English, 16 kHz. Source: Edinburgh DataShare (handle 10283/3336) - verify URL, size, licence. Ask the user before downloading; check free disk first (`df -h`).
  - Verified 2026-11-01/08: `LA.zip` is 7,640,952,520 bytes; licence ODC-By (attribution; cite Todisco et al., Interspeech 2019); no checksum published.
  - DataShare returns **HTTP 403 to Kaggle and Colab IP ranges** (even the landing page), and caps throughput per client at roughly 0.3-0.6 MB/s (parallel connections do not help). Download once on a personal machine (`python -m voxguard.data.download --yes --keep-zip`, resumable), upload `LA.zip` to Kaggle as a **private** dataset, and build manifests there with `--from-zip` (see `scripts/kaggle_setup.md`).
- **Stretch eval sets:** In-the-Wild (out-of-domain). Hindi set per section 11.
- **Codec chain:** resample to codec rate (8 kHz for NB codecs, 16 kHz for WB) -> encode -> decode via ffmpeg -> resample back to 16 kHz. Randomise codec and bitrate per sample (seeded). Optional additive noise. `G.711` must always work; other codecs are skipped with a clear warning if ffmpeg lacks the encoder (`ffmpeg -encoders | grep -i -E "amr|gsm|opus"`). Report codec availability in the first status update.
- **Offline precompute by default:** for each training file keep the clean version + N random codec variants; write manifests (`path,label,split,codec,bitrate,source`). Eval sets: deterministic degraded copies per codec so numbers are reproducible. On-the-fly augmentation is optional (subprocess ffmpeg per sample is slow).
- Optional RawBoost augmentation (as in SSL-AASIST) is a config flag, not default.

## 9. Models and training

- **RawNet2 baseline:** from scratch; input 64,600 samples; trains on CPU/MPS for smoke tests, free GPU for the real run.
- **SSL-AASIST (primary):** XLS-R 300M front-end fine-tuned (config: full vs last-N layers), AASIST back-end; weighted cross-entropy (bona fide vs spoof); start from the SSL-AASIST paper's hyperparameters and document deviations.
- Engineering requirements: device-agnostic (`cuda|mps|cpu`), mixed precision on CUDA, **checkpoint every epoch + resume** (free GPU sessions die), seeds, config-driven (`python -m voxguard.train --config configs/ssl_aasist.yaml`), `--max-samples` for smoke tests, optional W&B.
- Fallback if free compute is too slow: freeze most of XLS-R, fine-tune last N layers + AASIST head, shorter crops. Document it honestly.

## 10. Evaluation protocol

- Scores = P(fake). **EER** via ROC curve (FPR = FNR crossing; interpolate). Also accuracy at the EER threshold and at 0.5, plus confusion matrix.
- For each model: the 2x2 matrix (section 2), with the degraded column broken down per codec.
- **Latency:** p50/p95 per 4 s clip, CPU and GPU where available, reported via `pytest-benchmark` and `evaluate.py`. Targets (goals, not guarantees): CPU < 2 s, GPU < 300 ms.
- Outputs: `results/*.json` and an auto-generated `docs/results.md` (tables + plots). Never hand-type numbers into docs.

## 11. Languages

- **UI: English + Hindi (MVP).** All user-visible strings go through `i18n.t(key, lang)` with `en` and `hi` dictionaries; language toggle in the sidebar. Aarya checks new Hindi strings when she reviews the PR; corrections go in a follow-up PR, so this never blocks a merge.
- **Detection: English is the trained/evaluated language** (ASVspoof is English).
- **Hindi detection is a STRETCH generalization test**, not a feature claim: real Hindi speech from open corpora (e.g. Google FLEURS `hi_in`; verify access), fake Hindi generated with open TTS (e.g. XTTS-v2 Hindi, `facebook/mms-tts-hin`; verify availability and non-commercial licences, which are fine for academic use but must be noted). XLS-R is multilingual so it is a reasonable bet; WavLM is English-centric.
- In every doc, claim only what was measured ("evaluated on a small Hindi set"), never "supports Hindi".

## 12. Work split

| | **Person 1 - ML Core + repo owner (Aarya)** | **Person 2 - Product & Platform (teammate)** |
|---|---|---|
| Owns | `data/`, `models/`, `metrics.py`, `train.py`, `evaluate.py`, `inference.py` (RealDetector), `configs/`, `scripts/`, results docs | `api/`, `ui/`, `schemas.py`, `docker/`, `.github/`, API/UI tests, README, `architecture.md`, deployment |
| Delivers | pipeline, 2 models, 2x2 results, latency numbers, checkpoint on HF Hub | working API + prevention layer, EN/HI demo, CI, Docker, local compose deployment, demo script/video |
| Shared | `CLAUDE.md`, `pyproject.toml`, `docs/STATUS.md`, `docs/invention-disclosure.md`, the section-7 contract | |

Person 1 lands the scaffold + contract first (small PR, within a day) so Person 2 is never blocked. Person 2 builds against `StubDetector` until the real checkpoint exists.
Each Claude Code session updates ONLY its own section of `docs/STATUS.md` at the end of a working session: done, in progress, blocked, next.

## 13. Git workflow

- Everything reaches `main` and `dev` through a PR: never push to `main`/`dev` directly, never force-push or rewrite history. Work on `feat/<task>`, `fix/<task>`, `docs/<task>` branches from `dev`; open PRs with `gh pr create --base dev`. Never bypass the branch ruleset or use admin overrides.
- **Merging into `dev` (team decision, 8 Nov 2026):** the author may merge their own PR once (1) CI is green on the PR's **latest** commit, (2) ruff and the test suite pass locally, and (3) if `dev` has moved since the PR's CI run, the PR has been re-tested merged onto current `dev`. Review by the other person is welcome but not required. Changes to the section 7 contract, shared preprocessing, or this file must still be called out to the other person in the PR description. The other person's PRs may be merged after a review and the same three checks. Work inside your own area (section 12) needs no one else's go-ahead: routine rebuilds, image pulls, dev tooling, smoke tests and merging your own PRs. Only section 7 contract changes, shared preprocessing and this file need the other person first.
- Small PRs (aim < 400 changed lines), imperative commit messages, CI green before merging or asking for review, fill in the PR template.
- `dev` -> `main` only at milestones, by PR: **M1 15 Nov** (pipeline + stub end-to-end), **M2 29 Nov** (real model in the demo), **M3 8 Dec** (feature freeze). **Brought forward (18 Nov, then 22 Nov): the whole project, including the final `dev` -> `main` promotion, is complete by 24 Nov 2026, 23:59 IST** (see section 16).
- Never commit secrets, `.env`, data, checkpoints. Never print tokens. Use `.env.example` for variable names only.

## 14. Coding conventions

Python 3.11, type hints everywhere, ruff (lint + format), pydantic v2, `logging` not `print`, no global mutable state, seeds set and logged, every script runnable via `python -m voxguard...` or a `make` target. Tests: pytest; anything needing weights or big data is marked `slow` and skipped in CI. Prefer boring and readable over clever. Do not add dependencies without a reason in the PR description.

## 15. IP and security rules

- This is meant to be patentable. **Keep the GitHub repo PRIVATE and any HF Space/model repo PRIVATE** until the college IPR/TBI cell confirms the provisional filing. No public README, blog, demo, talk or paper describing the codec-aware method before then. (Not legal advice; the user checks with the IPR cell.)
- Maintain `docs/invention-disclosure.md`: dated notes on what was conceived and by whom (both teammates), the problem, the technical solution, the experiments that show the technical effect (the 2x2 matrix), and prior art found (Google Patents, InPASS).
- Use consented/synthetic/open data only. No real customer call recordings. Respect dataset licences.

## 16. Roadmap (2 people, from 2 Nov)

| Week | Person 1 | Person 2 |
|---|---|---|
| 1 (2-8 Nov) | scaffold PR + contract; ASVspoof download; codec pipeline | review scaffold; FastAPI + SQLite skeleton on StubDetector; UI skeleton |
| 2 (9-15 Nov) | dataset loaders, EER, RawNet2 + smoke run; SSL-AASIST code | prevention layer, alerts, UI (EN/HI), API tests, CI. **M1** |
| 3 (16-22 Nov) | full training runs (clean vs codec-aug) on Kaggle/Colab | Docker/compose, benchmark harness, "simulate phone line" toggle (uses `codecs.py`) |
| 4 (23-29 Nov) | evaluation matrix, per-codec results, RealDetector + checkpoint to HF Hub. **M2** | integrate real detector, compose deployment with the real detector, demo polish |
| 5 (30 Nov-6 Dec) | stretch: unseen codec / Hindi; results write-up | docs, README, demo video/script; **M3 (8 Dec) feature freeze** |
| 6 (7-13 Dec) | final numbers, report sections, invention disclosure | final report/presentation support, rehearsal, buffer |

**Compressed plan (decided 18 Nov; deadline moved to 24 Nov 23:59 IST on 22 Nov): everything above is done by 24 Nov 2026, 23:59 IST.** The only blocked work is GPU training after the Kaggle quota reset (Sat 21 Nov 05:30 IST). 21 Nov: RawNet2 codec-aug v2 and the SSL realistic fine-tune run in parallel. 22-23 Nov: thresholds, evaluation, the four-model `docs/results.md`, checkpoints published, demo model and operating point fixed. Person 2 in parallel: demo script and video, the 15-minute README check with a person, retrained RawNet2 benchmarks, re-running the sanity clips. 23 Nov: Person 2's demo video, README check with a person, Firefox check and sanity clips. 24 Nov: final docs and STATUS, then the final `dev` -> `main` promotion, by 23:59 IST. The week rows above are kept for history.

If behind schedule, cut stretch items first, then the second SSL model, then Hindi eval. Never cut: the 2x2 matrix, the working demo, tests/CI.

## 17. Definition of done

1. `docker compose up` (or documented local run) serves API + UI; analyze works with the real checkpoint.
2. `docs/results.md` contains the 2x2 matrix, per-codec breakdown, latency table, generated by scripts.
3. CI green on `main`; tests cover metrics, codecs, API, prevention rule, i18n keys.
4. README lets a stranger set up and run the demo in under 15 minutes.
5. `docs/invention-disclosure.md` is complete and reviewed by both teammates.

## 18. Working agreements for Claude Code

- Plan first for any task touching more than ~3 files and state the plan; wait for a go-ahead only if it changes the section 7 contract, shared preprocessing or this file.
- Ask before: deleting data or work you did not create, spending limited GPU quota (Kaggle/Colab), downloading datasets > 1 GB, changing the contract. Routine downloads (Docker base images, Python packages, the team's own checkpoints) and installing dev tools need no approval; mention them in the PR.
- Prefer the smallest change that works; no scope creep beyond sections 3 and 12.
- After each PR: give a short summary, how to run/test it, and what is next.
