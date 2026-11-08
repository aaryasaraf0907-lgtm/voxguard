# Architecture

The pipeline diagram lives in `CLAUDE.md` section 4. This document covers the parts of the
API that need more than a diagram.

## Codec simulation in `/v1/analyze`

`POST /v1/analyze` accepts the optional form field `simulate_codec` (boolean, default false).

- **false or absent:** the audio goes to the detector unchanged and `codec_applied` is `null`.
- **true:** after decoding and before inference, the waveform is resampled to 16 kHz mono (only
  if it is not already) and passed through a real encode/decode cycle by
  `voxguard.data.codecs.apply_codec`, the same ffmpeg function used to build the degraded
  training and evaluation sets. The detector then receives the degraded 16 kHz waveform.
  `codec_applied` is set to the codec name, and only after the transformation succeeded; the
  session row stores it too. Prevention (flagging, alerts, blocklist) works on the degraded
  audio's score exactly as for normal requests.

**Which codec:** the repository does not define a default, so the operator chooses one with
`VOXGUARD_SIMULATE_CODEC` (`g711_mulaw`, `g711_alaw`, `amr_nb`, `amr_wb`, `gsm` or `opus`; any
other value stops startup). The request cannot pick a codec. Bitrate is the codec's default.

**Requirements:** `ffmpeg` on `PATH` with an encoder for the chosen codec
(`python -m voxguard.data.codecs` prints what is available). G.711 needs no extra libraries.

**Errors (nothing is scored or stored in any of them, and the original audio is never used as
a fallback):**

| Situation | Status |
|---|---|
| `simulate_codec=true` but `VOXGUARD_SIMULATE_CODEC` unset | 501 |
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
- The codec is fixed per server, not per request, and there is no noise or bitrate variation;
  this is a deterministic single-codec demo, not the randomized training augmentation.
- The set of available encoders is read once per process; install ffmpeg before starting the API.

## Latency

Three different times exist, and the API reports only the first:

- **Detector inference latency:** `latency_ms` in the response (and in the stored session) is
  `Prediction.latency_ms`, measured inside the detector around `predict`. With simulation on,
  it is measured on the degraded audio.
- **Codec preprocessing time** (resampling plus encode/decode through ffmpeg): not measured and
  not included in `latency_ms`.
- **Total request time** (upload read, decoding, codec, inference, database commit): not
  measured by the API either, so `latency_ms` must not be read as end-to-end latency.
