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
3. `Detector.predict(waveform, sample_rate)`.
4. Prevention decision at the threshold chosen as above; store session (+ alert + blocklist
   when flagged) in one transaction.
5. Return `AnalyzeResponse` (label, score, threshold, detector latency, model version, flagged,
   `codec_applied`, `codec_ms`).

### UI state

- **Near-threshold caution** (agreed with Aarya, 16 Nov): a flagged result whose score is below
  `CAUTION_FACTOR` (1.5) times the threshold also shows "Close to the decision threshold: this
  result is uncertain". The band is one-sided (`threshold <= fake_score < 1.5 x threshold`;
  about 0.0127-0.019 at the demo threshold): results under the threshold get no caution, and a
  clearly high score reads as a clear fake. It is computed in the UI from the response's
  `fake_score` and `threshold`; the label, the flag and the API response are unchanged.
- Results are kept per audio clip (SHA-256 of the bytes) and per mode (original, phone line).
  Selecting a different file or recording discards them, so a result is never shown next to
  audio it was not computed from. When a clip has both results they are shown side by side.
- API reads (health, the three lists) are reused for 5 s within a browser session, so typing or
  switching language does not refetch them; **Refresh** and a new analysis refetch at once.
  Errors are never cached.
- HTTP errors map to messages by kind: 413, 415, 422, 501 (simulation not configured), 503
  (simulation unavailable; for an analysis without simulation, the decoder's ffmpeg is
  missing), other 5xx, timeouts, unreachable API.

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
