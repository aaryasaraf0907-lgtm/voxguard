# Architecture

What is implemented on `dev`. The overall pipeline diagram is in `CLAUDE.md` section 4.

## Components

| Component | Code | Role |
|---|---|---|
| Streamlit UI | `src/voxguard/ui/app.py`, `api_client.py`, `i18n.py` | Upload or record audio, phone-line toggle, results, flagged sessions, blocklist, alerts. Talks to the API over HTTP only; no inference or prevention logic. All strings through `i18n.t(key, lang)` (English, Hindi). |
| FastAPI backend | `src/voxguard/api/main.py` (`create_app`) | `/health`, `/v1/analyze`, `/v1/sessions`, `/v1/blocklist`, `/v1/alerts`; schemas in `schemas.py` (CLAUDE.md section 7). |
| Audio decoding | `api/audio.py`, `api/limits.py` | Upload limit (25 MB, enforced before multipart parsing), soundfile decoding to mono float32 at the native rate, duration/channel/rate checks. |
| Codec simulation | `api/codec_sim.py` -> `data/codecs.py` | Optional ffmpeg encode/decode before inference (below). |
| Detector | `inference.py` (Aarya), `api/detector_loader.py` | `Detector` protocol; `StubDetector` (hash, default) or `RealDetector` (RawNet2 / SSL-AASIST checkpoint from a path or Hugging Face Hub). Built once at API startup; `VOXGUARD_DEVICE` picks cpu/cuda/auto. The detector does its own resampling and windowing. |
| Prevention | `api/prevention.py` | One rule, `fake_score >= threshold` (threshold from the detector, or `VOXGUARD_LOW_FAR_THRESHOLD` in `low_far` mode). A flagged analysis writes the session, an alert and an atomic blocklist upsert in one transaction. |
| Persistence | `api/db.py` | SQLite (SQLAlchemy 2): `sessions`, `alerts`, `blocklist`; UTC-aware timestamps; foreign keys on; no migrations (`create_all`). |
| Deployment | `docker/`, `docker-compose.yml` | API and UI images, compose with a `/data` volume, single-container Hugging Face Space image. See `docs/deployment.md`. |
| Benchmarks | `benchmark.py` | Detector, codec and API latency measured separately. See `docs/benchmarking.md`. |

### Request flow (`POST /v1/analyze`)

1. Body size check (413), bounded read, decode (400/415/422).
2. If `codec` is given or `simulate_codec=true`: check the codec is known (422), configured
   (501, server default only) and available (503), then
   resample to 16 kHz and encode/decode with ffmpeg (500 on failure; never falls back to the
   original audio).
3. `Detector.predict(waveform, sample_rate)`.
4. Prevention decision; store session (+ alert + blocklist when flagged) in one transaction.
5. Return `AnalyzeResponse` (label, score, threshold, detector latency, model version, flagged,
   `codec_applied`, `codec_ms`).

### UI state

- Results are kept per audio clip (SHA-256 of the bytes) and per mode (original, phone line).
  Selecting a different file or recording discards them, so a result is never shown next to
  audio it was not computed from. When a clip has both results they are shown side by side.
- API reads (health, the three lists) are reused for 5 s within a browser session, so typing or
  switching language does not refetch them; **Refresh** and a new analysis refetch at once.
  Errors are never cached.
- HTTP errors map to messages by kind: 413, 415, 422, 501 (simulation not configured), 503
  (simulation unavailable), other 5xx, timeouts, unreachable API.

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
| ffmpeg fails or times out (30 s per process), or returns wrong-length, non-finite or silent output for non-silent input | 500 |

Example:

```bash
VOXGUARD_SIMULATE_CODEC=g711_mulaw uv run uvicorn voxguard.api.main:create_app --factory
curl -F file=@call.wav -F simulate_codec=true http://localhost:8000/v1/analyze
```

**Limitations**
- Audio moves through ffmpeg pipes; no temporary files are created.
- Non-16 kHz input is first resampled to 16 kHz with ffmpeg's resampler (libswresample, which
  low-pass filters before decimation), so content above 8 kHz is removed instead of aliasing
  into the band; input already at 16 kHz is not resampled. This reuses the ffmpeg this feature
  already requires and adds no Python dependency. It is separate from the detector's own
  preprocessing, which still resamples when simulation is off.
- One fixed bitrate per codec and no noise: this is the deterministic evaluation condition, not
  the randomized training augmentation.
- The set of available encoders is read once per process; install ffmpeg before starting the API.

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
