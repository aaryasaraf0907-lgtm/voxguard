# Architecture

What is implemented on `dev`. The overall pipeline diagram is in `CLAUDE.md` section 4.

## Components

| Component | Code | Role |
|---|---|---|
| Streamlit UI | `src/voxguard/ui/app.py`, `api_client.py`, `i18n.py` | Upload or record audio, phone-line toggle, results, flagged sessions, blocklist, alerts. Talks to the API over HTTP only; no inference or prevention logic. All strings through `i18n.t(key, lang)` (English, Hindi). |
| FastAPI backend | `src/voxguard/api/main.py` (`create_app`) | `/health`, `/v1/analyze`, `/v1/codecs`, `/v1/sessions`, `/v1/blocklist`, `/v1/alerts`; schemas in `schemas.py` (CLAUDE.md section 7). |
| Audio decoding | `api/audio.py`, `api/audio_ffmpeg.py`, `api/limits.py` | Upload limit (25 MB, enforced before multipart parsing), soundfile decoding to mono float32 at the native rate, duration/channel/rate checks; an ffmpeg fallback for M4A (AAC) and FLAC files libsndfile cannot read (below). |
| Codec simulation | `api/codec_sim.py` -> `data/codecs.py` | Optional ffmpeg encode/decode before inference (below). |
| Detector | `inference.py` (Aarya), `api/detector_loader.py` | `Detector` protocol; `StubDetector` (hash, default) or `RealDetector` (RawNet2 / SSL-AASIST checkpoint from a path or Hugging Face Hub). Built once at API startup, then one warm-up inference (startup below); `VOXGUARD_DEVICE` picks cpu/cuda/auto. The detector does its own resampling and windowing. |
| Thresholds | `api/thresholds.py` | Reads and validates the checkpoint's `thresholds.json` (chosen on dev by `evaluate.py`) and matches it to the loaded checkpoint (below). |
| Prevention | `api/prevention.py` | One rule, `fake_score >= threshold`, with the threshold chosen as below. A flagged analysis writes the session, an alert and an atomic blocklist upsert in one transaction. |
| Persistence | `api/db.py` | SQLite (SQLAlchemy 2): `sessions`, `alerts`, `blocklist`; UTC-aware timestamps; foreign keys on; no migrations (`create_all`). |
| Deployment | `docker/`, `docker-compose.yml` | Local Docker Compose is the deployment: API and UI images, a `/data` volume (SQLite database and Hugging Face cache). `INSTALL_ML=1` adds CPU-only torch to the API image for the real detector. A single-container Hugging Face Space image is kept as an optional route only (Docker Spaces need a paid plan). See `docs/deployment.md`. |
| Benchmarks | `benchmark.py` | Detector, codec and API latency measured separately. See `docs/benchmarking.md`. |
| Real-audio sanity tool | `scripts/real_audio_sanity.py` | A diagnostic run on consented, labelled recordings through the public API only (`/health`, `/v1/analyze`), against a scratch API and database; private reports with counts only. See `docs/real-audio-testing.md`. |

### Startup

1. Settings are read from `VOXGUARD_*` variables when the app is created; an invalid value
   stops startup.
2. The SQLite tables are created if missing.
3. The detector is built (`VOXGUARD_DETECTOR=stub|real`). For the real one, the checkpoint is
   loaded from `VOXGUARD_MODEL_REPO` (a local path or a Hugging Face repo, cached under
   `/data/hf-cache` in compose), then one inference on 4 s of silence runs, so CUDA or CPU
   start-up is not reported as the first request's latency. Any failure here (missing ML extra,
   unavailable device, missing or unreadable checkpoint, failed warm-up) stops startup; there is
   no fallback to the stub.
4. Real detector only: `thresholds.json` is looked up next to the checkpoint. Its
   `model_version` must name the same model and epoch as the loaded checkpoint
   (`<name>[/<run>]:<source>:epoch<N>`, compared on the first and last parts only, since
   `<source>` is the load location). A malformed or mismatched file, or a Hub error while
   fetching it, stops startup. A missing file leaves the detector's own threshold, with a warning
   when that would be the untuned 0.5 (`eer` mode, no `VOXGUARD_THRESHOLD`).

### Which threshold applies

In order:

1. `VOXGUARD_THRESHOLD_MODE=low_far`: the explicit `VOXGUARD_LOW_FAR_THRESHOLD` (required in that
   mode).
2. An explicit `VOXGUARD_THRESHOLD`.
3. The checkpoint's `thresholds.json`: its **phone** threshold for uploads of unknown origin and
   for simulated phone lines, its **clean** threshold only for audio known to be clean. No request
   field says that yet, so every request currently uses the phone threshold. The file's
   `low_far` and `per_codec` values are validated but used for reporting only.
4. The detector's own threshold (0.5).

The response's `threshold` is the one applied, and the session stores it.

### Request flow (`POST /v1/analyze`)

1. Body size check (413), bounded read, decode (400/415/422; the ffmpeg fallback can also give
   500/503). A sample rate whose resampling to 16 kHz would need a filter of over a million
   elements (e.g. 44101 Hz) is refused with 422: torchaudio's filter size depends on the
   rate's common factor with 16000, and such a rate would need gigabytes. Standard rates pass.
2. If `codec` is given or `simulate_codec=true`: check the codec is known (422), configured
   (501, server default only) and available (503), then
   resample to 16 kHz and encode/decode with ffmpeg (500 on failure; never falls back to the
   original audio).
   Steps 1 (decode onwards) to 3 hold one of a limited number of analysis slots
   (`api/gate.py`): at most `VOXGUARD_MAX_RUNNING_ANALYSES` (default 2) run at once and at most
   `VOXGUARD_MAX_WAITING_ANALYSES` (default 8) wait, up to 45 s, for a turn. Beyond that the
   answer is 503 with `Retry-After: 5` at once, and nothing is scored or stored. The load test
   showed why: throughput stops growing at about 2.8 analyses/s from 4 at once on a laptop CPU,
   while memory keeps growing, and 4 long clips or a burst of 48 requests got the API
   OOM-killed (docs/deployment.md, "Load test").
3. `Detector.predict(waveform, sample_rate)`. With `timeline=true` (`api/timeline.py`), the
   audio is resampled once with the detector's own resampler, split with `inference.to_windows`
   (the detector's 4 s windows, the last overlapping the one before), and each window is scored
   with `predict`. The clip score is their mean, which is how the detector averages them anyway,
   so the verdict and the stored session are the same (scores equal to within float rounding on
   the real model). The response adds `timeline` (per window: start, end, score, running average,
   at or over the threshold) and `first_flag_s` (when the running average first reached the
   threshold), both response-only. It is a view of a recording, not phone streaming. Measured on
   the real model: 0 to 13 % slower than a plain analysis, since windows are not batched.
4. Prevention decision at the threshold chosen as above; store session (+ alert + blocklist
   when flagged) in one transaction. The transaction starts with SQLite's write lock
   (`BEGIN IMMEDIATE`) and the timestamps are taken after it, so they follow commit order: an
   `/v1/alerts?since=` poller never skips an alert that committed late. The database runs in
   WAL mode with a 30 s lock wait, so reads do not block writes.
5. Return `AnalyzeResponse` (label, score, threshold, detector latency, model version, flagged,
   `codec_applied`, `codec_ms`).

### UI state

- **Near-threshold caution** (agreed with Aarya, 16 Nov): a flagged result whose score is below
  `CAUTION_FACTOR` (1.5) times the threshold also shows "Close to the decision threshold: this
  result is uncertain". The band is one-sided (`threshold <= fake_score < 1.5 x threshold`;
  about 0.0127-0.019 at the demo threshold): results under the threshold get no caution, and a
  clearly high score reads as a clear fake. It is computed in the UI from the response's
  `fake_score` and `threshold`; the label, the flag and the API response are unchanged.
- **Show live-monitoring timeline** asks for `timeline=true`. The chart (log scale, so a threshold
  near 0.01 and scores far from it are readable) shows each window's score and the running
  average, a dashed threshold line and a marker where a live monitor would have flagged the call;
  **Replay** draws it window by window. If the running average reached the threshold early but
  the full recording is below it, the UI says so instead of claiming the call would be flagged.
- Results are kept per audio clip (SHA-256 of the bytes) and per mode (original, phone line).
  Selecting a different file or recording discards them, so a result is never shown next to
  audio it was not computed from. When a clip has both results they are shown side by side.
- API reads (health, the three lists) are reused for 5 s within a browser session, so typing or
  switching language does not refetch them; **Refresh** and a new analysis refetch at once.
  Errors are never cached.
- HTTP errors map to messages by kind: 413, 415, 422, 501 (simulation not configured), 503
  with `Retry-After` (server busy: try again in a few seconds), other 503 (simulation
  unavailable; for an analysis without simulation, the decoder's ffmpeg is missing), other 5xx,
  timeouts, unreachable API.

## Audio decoding and the ffmpeg fallback

`decode_audio` tries libsndfile first, in memory (WAV, FLAC, OGG, MP3), exactly as before. Only
when libsndfile cannot open or read the upload does `api/audio_ffmpeg.py` try ffmpeg, and only
for two formats, recognised from the file's first bytes: FLAC (`fLaC`) and MP4/M4A (an `ftyp`
box with brand `M4A `, `mp42`, `isom`, `iso2` or `mp41`). Everything else gets 415
(`Unsupported or undecodable audio`; libsndfile's own message is not returned). File names and
client content types are never used.

1. The upload limit is checked again before anything is written to disk.
2. The bytes go to a `mkstemp` file (0600) in a private `TemporaryDirectory` (0700). This is the
   only ffmpeg input path: an M4A with its index (`moov`) at the end, as laptop and phone
   recorders write it, cannot be read from a pipe because ffmpeg has to seek back to the samples.
3. `ffprobe` checks the container, exactly one audio stream with the expected codec (FLAC or
   AAC), at most 2 channels, at most 192 kHz and the duration limit, before any decoding.
4. `ffmpeg` decodes to 16 kHz mono float32 PCM on stdout.
5. The directory is removed on every path, after each process has been killed (if still running)
   and reaped.

Security measures:

- Argument lists only, never a shell. `-nostdin`, `-protocol_whitelist file`, the demuxer forced
  with `-f` (so a file can never be read as an HLS or concat playlist that pulls in other files)
  and `-enable_drefs 0` for MP4. The only input is the temp file.
- Output is read in 64 KiB chunks under a hard cap (duration limit plus 0.5 s for AAC priming
  and padding); past it the process is killed (422). The decoded audio is then trimmed to the
  duration limit (the same frame count as the libsndfile path), so the allowance only absorbs
  encoder padding and the Detector never receives more than the limit. A file whose probe
  reports no duration is still decoded, within the same cap and trim. Each process has a 30 s
  timeout. stderr is discarded.
- Clients get fixed messages. Logs get a category, the exit code and the elapsed time, never
  ffmpeg's output, audio bytes or paths.

| Case | Status |
|---|---|
| Not FLAC or MP4 by signature (including playlists), or the probe disagrees with the signature or the codec is not allowed | 415 |
| ffmpeg cannot decode the file | 415 |
| Over the duration, channel or rate limits (from the probe, or the output cap), or non-finite samples | 422 |
| `ffmpeg` or `ffprobe` missing | 503 |
| Timeout, or malformed output | 500 |

Verified against the API image's own ffmpeg (Debian 12, 5.1.9) by the Docker smoke tests: an
M4A with its index at the end decodes as the image's user (uid 1000), the temp directory and file
are 0700/0600 while ffmpeg runs, nothing is left in `/tmp`, and the M4A analyzes end to end
through `/v1/analyze`, with and without phone-line simulation. Locally and in CI the decoder
tests have also passed with ffmpeg 7.0, 8.0 and 9.0.

Files decoded this way are downmixed and resampled to 16 kHz by ffmpeg, not by the Detector, so
their scores can differ slightly from the same audio converted to WAV and uploaded.

## Codec simulation in `/v1/analyze`

`POST /v1/analyze` accepts the optional form fields `simulate_codec` (boolean, default false)
and `codec` (a codec name; it implies simulation). `GET /v1/codecs` lists every codec with its
availability on this server and bitrate; the UI's codec dropdown offers only available ones.

- **neither:** the audio goes to the detector unchanged and `codec_applied` is `null`.
- **simulated:** after decoding and before inference, the waveform is resampled to 16 kHz mono (only
  if it is not already) and passed through a real encode/decode cycle by
  `voxguard.data.codecs.apply_codec`, the same ffmpeg function used to build the degraded
  training and evaluation sets. The detector then receives the degraded 16 kHz waveform.
  `codec_applied` is set to the codec name, and only after the transformation succeeded; the
  session row stores it too. Prevention (flagging, alerts, blocklist) works on the degraded
  audio's score exactly as for normal requests.

**Which codec:** the request's `codec` field, else the server default `VOXGUARD_SIMULATE_CODEC`
(`g711_mulaw`, `g711_alaw`, `amr_nb`, `amr_wb`, `gsm` or `opus`; any other value stops startup;
no default is defined). A requested codec is never replaced by another one. **Bitrate:** the
evaluation sets' `voxguard.data.augment.default_spec(codec)` (the middle bitrate of the codec's
list, e.g. Opus 12 kbit/s, GSM 13; G.711 is fixed-rate), so the demo's phone line is the
condition the reported results were measured on.

**Requirements:** `ffmpeg` on `PATH` with an encoder for the chosen codec
(`python -m voxguard.data.codecs` prints what is available). G.711 needs no extra libraries.

**Errors (nothing is scored or stored in any of them, and the original audio is never used as
a fallback):**

| Situation | Status |
|---|---|
| `codec` is not a known codec name | 422 |
| `simulate_codec=true`, no `codec`, `VOXGUARD_SIMULATE_CODEC` unset | 501 |
| ffmpeg or the encoder is missing | 503 |
| The clip is shorter than 20 ms at 16 kHz (codecs return silence for a few samples) | 422 |
| ffmpeg fails or times out (30 s per process), or returns wrong-length, non-finite or silent output for non-silent input | 500 |

Example:

```bash
VOXGUARD_SIMULATE_CODEC=g711_mulaw uv run uvicorn voxguard.api.main:create_app --factory
curl -F file=@call.wav -F simulate_codec=true http://localhost:8000/v1/analyze
```

**Limitations**
- The simulation moves audio through ffmpeg pipes and creates no temporary files (only the
  decoder fallback above uses a private temporary file).
- Non-16 kHz input is first resampled to 16 kHz with ffmpeg's resampler (libswresample, which
  low-pass filters before decimation), so content above 8 kHz is removed instead of aliasing
  into the band; input already at 16 kHz is not resampled. This reuses the ffmpeg this feature
  already requires and adds no Python dependency. It is separate from the detector's own
  preprocessing, which still resamples when simulation is off.
- One fixed bitrate per codec and no noise: this is the deterministic evaluation condition, not
  the randomized training augmentation.
- The set of available encoders is read once per process; install ffmpeg before starting the API.

## Live call monitor (`WS /v1/stream`, `/monitor`)

`api/stream.py` scores a call while it is still going on; `/monitor` (`api/monitor_page.py`, plain
HTML and JavaScript, no framework, nothing from the network) is its browser client, and
`scripts/stream_call.py` streams a file at real speed for demos and measurements.

1. The client opens `ws://<api>/v1/stream?sample_rate=<Hz>` (8-48 kHz; optional `caller_id`,
   `codec`) and sends mono float32 frames of about 0.25 s. The page captures at the browser's own
   rate and the server resamples, so Firefox works too.
2. Once one detector window has arrived (64,600 samples at 16 kHz, about 4 s), the latest window
   is scored every 2 s of audio, exactly as `POST /v1/analyze` scores a 4 s clip (codec first,
   then `Detector.predict`), through the same analysis-slot limit; a window that gets no slot
   within one step is skipped and reported as `busy`. If scoring falls behind, only the newest
   window is scored.
3. Each `score` message carries the window score and the running average. An `alert` is sent
   the first time the running average of **at least two windows** reaches **1.5 x the
   threshold** (`ALERT_FACTOR`, `api/risk.py`): on the real model, a genuine recording's first
   window sat just over the threshold (0.012836 against 0.012676), and in 300 streamed ASVspoof
   calls 35 of 100 genuine calls ran just over it, against 2 of 100 at 1.5 x, with every fake
   call still alerted at 6.25 s (`docs/real-audio-testing.md`, section 4). Between the threshold
   and the alert level the page says the risk is elevated and to keep listening, without an
   alert. The stored verdict still uses the threshold itself (the prevention rule). After an
   alert, the page grades it: if the running average falls back under 1.5 x, it asks to verify
   the caller another way; above it, to hang up and call back. Once an alert is
   raised the page lists what to do (call back on a known number, ask something only the real
   person knows, report at 1930, India's National Cyber Crime Helpline, or cybercrime.gov.in). A
   consent notice asks the user to tell the other party that the call is being checked.
4. On `{"type": "end"}`, a hang-up or the length limit, one session is stored through the usual
   prevention layer (alert and blocklist when flagged). Its score is the running average of the
   live windows. A call shorter than one window is scored once at the end if it is at least 1 s.
5. Only the latest window of audio is kept, in memory; nothing is written to disk.

Limits: `VOXGUARD_MAX_STREAMS` (4) calls at once, `VOXGUARD_MAX_STREAM_SECONDS` (600) per call,
1 MB frames, 30 s without a frame. Browsers send `Origin` on a WebSocket handshake: only the
API's own origin (the `/monitor` page) and `VOXGUARD_STREAM_ORIGINS` may open a stream, so a page
on another site cannot use a local API; clients without `Origin` (`stream_call.py`) are allowed.

Measured on the real model (laptop CPU, isolated compose project, private clips): a score
arrives 0.45-0.57 s after its last audio; the detector needs about 0.46 s per 2 s step. On a
genuine recording followed by a fake one, the alert came 1.3 s after the fake part began
(1.7 s in the browser), with none during the genuine part; four calls at once: no window
skipped, scores within 1.6 s, 2.0 GiB.

What it is not: it listens to a microphone or a file, not a phone line (telephony is out of
scope); the codec runs per window, not across the whole call; the first score needs one window
and an alert two (about 6 s). Browsers allow the microphone only on localhost or HTTPS.

## Transfer checks, evidence and alerts (`api/policy.py`, `api/transactions.py`)

Graded verification, not a binary accusation: a transfer asked for during or after a checked
call is decided from the call's risk.

| Tier | When | Normal value | High value (≥ `VOXGUARD_HIGH_VALUE_INR`, 50,000) |
|---|---|---|---|
| unknown | live call, fewer than 2 windows heard | step-up | step-up |
| low | below the threshold | allow | allow |
| caution | threshold to 1.5 x it | step-up | hold |
| high | 1.5 x the threshold or more | hold | hold |

- A caller flagged on an earlier call moves one tier up. The table can be replaced with
  `VOXGUARD_TRANSACTION_POLICY` (JSON, every tier: `[normal, high-value]`).
- **Step-up:** any one of a call-back on a number already on file, an OTP, or a second
  approver. **Hold:** a second approver and a call-back. Steps are attestations recorded with
  the name of whoever did them; the prototype sends no OTP and places no call.
- A live call is read from the monitor's running average (the API keeps live calls in memory
  while they last); a finished one from its stored session. Decisions and every step are stored
  with an append-only audit trail (`transactions`, `transaction_events`).
- **Feature-level log:** a live call's per-window scores are stored with its session
  (`call_windows`, numbers only, no audio) and returned by `/v1/sessions/{id}/timeline`.
- **Incident report** (`/v1/sessions/{id}/report?lang=en|hi`, `api/report_page.py`): the
  verdict, the timeline as a chart and a table, every transfer decision and step, the 1930 /
  cybercrime.gov.in guidance and a no-audio statement. Self-contained HTML, every value escaped.
  Linked from the monitor page and the Alerts and Transactions tabs. The page has **Download
  report (HTML)** and **Print / Save as PDF** buttons (hidden when printed); `&download=true`
  returns the same report as a file (`Content-Disposition: attachment`,
  `linetrue-report-<session>-<lang>.html`) without the buttons, and the monitor links to it
  directly ("Download the report"). PDF comes from the browser's print dialog, so no PDF library
  is added.
- **Tamper-evident evidence trail** (`api/evidence.py`): every stored session, a live call's
  per-window scores, each transfer decision and each verification step add one entry to
  `evidence_log` in the same transaction: the SHA-256 of the record's canonical fields, the
  previous entry's hash and its own hash over both. Writers hold the write lock, so the chain is
  strictly ordered. `GET /v1/evidence/verify[?session_id=]` walks the chain and recomputes it
  from the records as they are now: an edited or deleted record, an edited or removed entry, or a
  broken link is reported. The chain head is printed on the incident report and sent with every
  webhook event, so a copy outside the database anchors it. It makes tampering detectable; it
  does not prevent it, and it is not a claim of legal admissibility.
- **A hold needs two people:** its second approver and call-back must be recorded under
  different names (not authenticated: a prototype has no logins).
- **Signed webhook** (`api/webhook.py`): with `VOXGUARD_ALERT_WEBHOOK_URL`, events are POSTed
  as JSON with `X-VoxGuard-Signature: sha256=<HMAC of the body>` from one background thread;
  failures are logged by type only and never slow a request. Redirects are not followed (a
  redirect is a failed delivery), and at most 100 deliveries wait; beyond that new events are
  dropped and logged.

Measured on the real model (private clips, browser, a ₹75,000 transfer during a genuine-then-fake
call): at 3.5 s (no window yet) *verify first*; at 9.6 s (genuine part, risk low) *allowed*; at
16.7 s (3.8 s into the fake part, risk high) *held*, released only after a second approver and a
call-back.

## Latency

Three different times exist, and the API reports the first two separately:

- **Detector inference latency:** `latency_ms` in the response (and in the stored session) is
  `Prediction.latency_ms`, measured inside the detector around `predict`. With simulation on,
  it is measured on the degraded audio.
- **Codec preprocessing time:** `codec_ms` in the response (not stored), the wall time of the
  whole simulation (resampling plus encode/decode through ffmpeg); `null` without simulation and
  not included in `latency_ms`.
- **Total request time** (upload read, decoding, codec, inference, database commit): not
  measured by the API either, so `latency_ms` must not be read as end-to-end latency.
