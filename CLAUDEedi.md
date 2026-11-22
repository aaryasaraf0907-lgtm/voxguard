# CLAUDE.md — VoxGuard

> Shared project brief for BOTH teammates' Claude Code sessions. Read it fully before doing anything.
> If it conflicts with a direct instruction from the user in the session, follow the user and flag the conflict.
> Decisions change: update this file via PR and tell the other person.
> External facts below (dataset URLs, codec availability, model names, licenses) are best-effort. VERIFY them at execution time and report anything that differs.

---

## 1. What we are building

**VoxGuard** is a codec-aware voice-deepfake (vishing) detector for telephone-quality audio, plus a thin "prevention" layer and a demo app.

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
9. Private deployment (HF Space, CPU) or local docker-compose demo as fallback

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
| Language / env | Python 3.11, `uv` (fallback pip + venv), `pyproject.toml`, src layout |
| ML | PyTorch, HuggingFace Transformers (`facebook/wav2vec2-xls-r-300m`), torchaudio/soundfile/librosa |
| Codec simulation | FFmpeg via subprocess (G.711 mu-law/A-law native; AMR-NB `libopencore_amrnb`, AMR-WB `libvo_amrwbenc`, GSM `libgsm`, Opus `libopus` if available) |
| Baselines/refs | RawNet2 (ASVspoof 2021 baseline), AASIST (clovaai/aasist), SSL-AASIST (Tak et al. 2022, TakHemlata/SSL_Anti-spoofing). Check licences; implement/adapt with attribution rather than blind copy |
| Tracking | Weights & Biases free tier (optional, must work with `--no-wandb`) |
| Model hosting | Hugging Face Hub (private model repo) |
| API | FastAPI, Pydantic v2, SQLAlchemy 2 + SQLite |
| UI | Streamlit (>=1.39 for `st.audio_input`) |
| Quality | pytest, pytest-benchmark, ruff (lint + format), pre-commit (optional) |
| CI/CD | GitHub Actions (ruff + pytest on PRs to `dev`/`main`) |
| Packaging/deploy | Docker + docker-compose; HF Space (CPU, **private**) |
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

Audio conventions: 16 kHz mono float32. Detector does its own preprocessing: resample -> mono -> crop/pad to 64,600 samples (~4 s); clips longer than 4 s are scored with non-overlapping windows and averaged.

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
| `POST /v1/analyze` | multipart `file`; optional form `caller_id`, `simulate_codec` -> `{session_id, label, fake_score, threshold, latency_ms, model_version, flagged, codec_applied}` |
| `GET /v1/sessions?flagged_only=&limit=` | stored sessions |
| `GET /v1/blocklist` | caller ids auto-blocked, with hit counts |
| `GET /v1/alerts?since=` | alerts raised by the prevention layer |

Prevention rule (MVP): `fake_score >= threshold` -> session stored as flagged, alert created, `caller_id` (if given) upserted into blocklist with a counter. Threshold modes: `eer` (default) and `low_far` (operating point shifted to reduce missed fakes).

## 8. Data and codec spec

- **Primary dataset:** ASVspoof 2019 LA (train/dev/eval), English, 16 kHz. Source: Edinburgh DataShare (handle 10283/3336) - verify URL, size, licence. Ask the user before downloading; check free disk first (`df -h`).
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

- **UI: English + Hindi (MVP).** All user-visible strings go through `i18n.t(key, lang)` with `en` and `hi` dictionaries; language toggle in the sidebar. Aarya reviews the Hindi strings.
- **Detection: English is the trained/evaluated language** (ASVspoof is English).
- **Hindi detection is a STRETCH generalization test**, not a feature claim: real Hindi speech from open corpora (e.g. Google FLEURS `hi_in`; verify access), fake Hindi generated with open TTS (e.g. XTTS-v2 Hindi, `facebook/mms-tts-hin`; verify availability and non-commercial licences, which are fine for academic use but must be noted). XLS-R is multilingual so it is a reasonable bet; WavLM is English-centric.
- In every doc, claim only what was measured ("evaluated on a small Hindi set"), never "supports Hindi".

## 12. Work split

| | **Person 1 - ML Core + repo owner (Aarya)** | **Person 2 - Product & Platform (teammate)** |
|---|---|---|
| Owns | `data/`, `models/`, `metrics.py`, `train.py`, `evaluate.py`, `inference.py` (RealDetector), `configs/`, `scripts/`, results docs | `api/`, `ui/`, `schemas.py`, `docker/`, `.github/`, API/UI tests, README, `architecture.md`, deployment |
| Delivers | pipeline, 2 models, 2x2 results, latency numbers, checkpoint on HF Hub | working API + prevention layer, EN/HI demo, CI, Docker, private deploy, demo script/video |
| Shared | `CLAUDE.md`, `pyproject.toml`, `docs/STATUS.md`, `docs/invention-disclosure.md`, the section-7 contract | |

Person 1 lands the scaffold + contract first (small PR, within a day) so Person 2 is never blocked. Person 2 builds against `StubDetector` until the real checkpoint exists.
Each Claude Code session updates ONLY its own section of `docs/STATUS.md` at the end of a working session: done, in progress, blocked, next.

## 13. Git workflow

- `main` and `dev` are protected by a ruleset: **PR required + 1 approval**. So: never push to `main`/`dev`, never force-push, never merge your own PR. Work on `feat/<task>`, `fix/<task>`, `docs/<task>` branches from `dev`; open PRs with `gh pr create --base dev`; the OTHER person reviews.
- Small PRs (aim < 400 changed lines), imperative commit messages, CI green before asking for review, fill in the PR template.
- `dev` -> `main` only at milestones, by PR: **M1 15 Nov** (pipeline + stub end-to-end), **M2 29 Nov** (real model in the demo), **M3 8 Dec** (feature freeze).
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
| 4 (23-29 Nov) | evaluation matrix, per-codec results, RealDetector + checkpoint to HF Hub. **M2** | integrate real detector, private deploy, demo polish |
| 5 (30 Nov-6 Dec) | stretch: unseen codec / Hindi; results write-up | docs, README, demo video/script; **M3 (8 Dec) feature freeze** |
| 6 (7-13 Dec) | final numbers, report sections, invention disclosure | final report/presentation support, rehearsal, buffer |

If behind schedule, cut stretch items first, then the second SSL model, then Hindi eval. Never cut: the 2x2 matrix, the working demo, tests/CI.

## 17. Definition of done

1. `docker compose up` (or documented local run) serves API + UI; analyze works with the real checkpoint.
2. `docs/results.md` contains the 2x2 matrix, per-codec breakdown, latency table, generated by scripts.
3. CI green on `main`; tests cover metrics, codecs, API, prevention rule, i18n keys.
4. README lets a stranger set up and run the demo in under 15 minutes.
5. `docs/invention-disclosure.md` is complete and reviewed by both teammates.

## 18. Working agreements for Claude Code

- Plan first for any task touching more than ~3 files; wait for the user's go-ahead.
- Ask before: downloading > 1 GB, installing system packages, spending quota, deleting anything, changing the contract.
- Prefer the smallest change that works; no scope creep beyond sections 3 and 12.
- After each PR: give a short summary, how to run/test it, and what is next.
