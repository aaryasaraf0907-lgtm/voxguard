import numpy as np
import pytest

from voxguard.api import codec_sim
from voxguard.api.codec_sim import CodecSimulationError, CodecUnavailableError, resample_to_16k
from voxguard.data.codecs import CodecError, available_codecs, ffmpeg_path
from voxguard.inference import linear_resample

needs_ffmpeg = pytest.mark.skipif(ffmpeg_path() is None, reason="ffmpeg not installed")


def tone(freq: float, sr: int, seconds: float = 1.0, amp: float = 0.5) -> np.ndarray:
    n = int(sr * seconds)
    return (amp * np.sin(2 * np.pi * freq * np.arange(n) / sr)).astype(np.float32)


def band_power(x: np.ndarray, freq: float, sr: int = 16_000, half_width: float = 100) -> float:
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1 / sr)
    return float(spectrum[(f > freq - half_width) & (f < freq + half_width)].sum())


def rel_db(x: np.ndarray, freq: float, reference_amp: float = 0.5) -> float:
    """Power of ``x`` around ``freq`` relative to a 16 kHz reference tone of that frequency."""
    ref = tone(freq, 16_000, len(x) / 16_000, reference_amp)
    return 10 * np.log10(band_power(x, freq) / band_power(ref, freq) + 1e-30)


# --- real ffmpeg ------------------------------------------------------------------------


@needs_ffmpeg
@pytest.mark.parametrize("sr", [8_000, 22_050, 32_000, 44_100, 48_000])
def test_output_length_dtype_and_mono(sr: int) -> None:
    y = resample_to_16k(tone(1000, sr, 1.5), sr)
    assert y.dtype == np.float32 and y.ndim == 1
    assert len(y) == round(1.5 * sr * 16_000 / sr) == 24_000
    assert np.isfinite(y).all()


@needs_ffmpeg
def test_16k_input_is_returned_unchanged_without_ffmpeg_call(monkeypatch) -> None:
    monkeypatch.setattr(codec_sim, "_run", lambda *a, **k: pytest.fail("ffmpeg must not run"))
    x = tone(1000, 16_000)
    y = resample_to_16k(x, 16_000)
    assert np.array_equal(y, x) and y.dtype == np.float32


@needs_ffmpeg
@pytest.mark.parametrize("sr", [8_000, 44_100, 48_000])
@pytest.mark.parametrize("freq", [1_000, 3_000])
def test_in_band_content_is_preserved_in_amplitude_and_phase(sr: int, freq: int) -> None:
    y = resample_to_16k(tone(freq, sr, 1.0), sr)
    ref = tone(freq, 16_000, 1.0)
    edge = 200  # ignore filter edge effects (12 ms)
    assert abs(rel_db(y, freq)) < 0.1  # level within 0.1 dB
    assert np.abs(y[edge:-edge] - ref[edge:-edge]).max() < 2e-3  # no delay, no distortion


@needs_ffmpeg
@pytest.mark.parametrize(
    ("sr", "freq", "alias"), [(48_000, 12_000, 4_000), (44_100, 10_000, 6_000)]
)
def test_content_above_the_target_nyquist_is_suppressed_not_aliased(sr, freq, alias) -> None:
    x = tone(freq, sr)  # above 8 kHz: must vanish, not fold down to `alias`
    y = resample_to_16k(x, sr)
    assert rel_db(y, alias) < -60
    # the previous linear-interpolation resampler folds it back at nearly full level
    assert rel_db(linear_resample(x, sr, 16_000), alias) > -6


@needs_ffmpeg
def test_mixed_signal_keeps_voice_band_and_drops_ultrasonic_content() -> None:
    sr = 48_000
    x = tone(500, sr, amp=0.3) + tone(2_500, sr, amp=0.2) + tone(15_000, sr, amp=0.4)
    y = resample_to_16k(x, sr)
    assert abs(rel_db(y, 500, 0.3)) < 0.1 and abs(rel_db(y, 2_500, 0.2)) < 0.1
    # 15 kHz would alias to 1 kHz at 16 kHz sampling
    assert rel_db(y, 1_000, 0.4) < -60


@needs_ffmpeg
@pytest.mark.parametrize("sr", [8_000, 44_100, 48_000])
@pytest.mark.parametrize("n", [1, 2, 3, 10, 50, 200])
def test_very_short_inputs_keep_exact_length_and_are_finite(sr: int, n: int) -> None:
    y = resample_to_16k(np.full(n, 0.3, dtype=np.float32), sr)
    assert len(y) == max(1, round(n * 16_000 / sr))
    assert y.dtype == np.float32 and np.isfinite(y).all()


@needs_ffmpeg
def test_short_constant_signal_is_not_silenced() -> None:
    y = resample_to_16k(np.full(441, 0.3, dtype=np.float32), 44_100)  # 10 ms
    assert len(y) == 160 and abs(float(y[60:100].mean()) - 0.3) < 0.02


@needs_ffmpeg
def test_full_scale_square_wave_stays_finite() -> None:
    x = np.sign(tone(1_000, 48_000)).astype(np.float32)
    y = resample_to_16k(x, 48_000)
    assert np.isfinite(y).all() and np.abs(y).max() < 1.3  # filter overshoot only


@needs_ffmpeg
@pytest.mark.skipif(not available_codecs().get("g711_mulaw"), reason="no g711 encoder")
def test_pipeline_resample_then_codec_for_non_16k_input() -> None:
    sr = 44_100
    x = tone(1_000, sr, amp=0.3) + tone(6_000, sr, amp=0.3)
    y, out_sr = codec_sim.simulate_phone_line(x, sr, "g711_mulaw")
    assert out_sr == 16_000 and y.dtype == np.float32 and y.shape == (16_000,)
    assert np.isfinite(y).all()
    assert rel_db(y, 1_000, 0.3) > -1.0  # voice-band tone survives resample + codec
    assert rel_db(y, 6_000, 0.3) < -40  # above the 4 kHz telephone band: gone


# --- argument handling (no ffmpeg needed) ------------------------------------------------


@pytest.mark.parametrize("bad", [np.zeros(0, dtype=np.float32), np.zeros((4, 2), dtype=np.float32)])
def test_rejects_empty_and_non_mono_input(bad: np.ndarray) -> None:
    with pytest.raises(CodecSimulationError):
        resample_to_16k(bad, 8_000)


def test_missing_ffmpeg_raises_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(codec_sim, "ffmpeg_path", lambda: None)
    with pytest.raises(CodecUnavailableError):
        resample_to_16k(tone(1000, 8_000), 8_000)


def _patch_run(monkeypatch, response):
    seen = {}
    monkeypatch.setattr(codec_sim, "ffmpeg_path", lambda: "/usr/bin/ffmpeg")

    def fake(cmd, data, *a, **k):
        seen.update(cmd=cmd, data=data)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(codec_sim, "_run", fake)
    return seen


def test_ffmpeg_command_and_flush_padding(monkeypatch) -> None:
    out = np.zeros(8_000, dtype="<f4").tobytes()
    seen = _patch_run(monkeypatch, out)
    y = resample_to_16k(tone(1000, 8_000, 0.5), 8_000)
    cmd = seen["cmd"]
    assert isinstance(cmd, list)
    rates = [cmd[i + 1] for i, token in enumerate(cmd) if token == "-ar"]
    assert rates == ["8000", "16000"]  # input rate, then output rate
    assert len(seen["data"]) > 4_000 * 4  # signal plus the zero flush tail
    assert len(y) == 8_000


@pytest.mark.parametrize(
    "response",
    [np.zeros(10, dtype="<f4").tobytes(), np.full(8_000, np.nan, dtype="<f4").tobytes(), b""],
    ids=["too_short", "nan", "empty"],
)
def test_malformed_resampler_output_is_rejected(monkeypatch, response) -> None:
    _patch_run(monkeypatch, response)
    with pytest.raises(CodecSimulationError, match="malformed"):
        resample_to_16k(tone(1000, 8_000, 0.5), 8_000)


def test_resampler_failure_does_not_leak_stderr(monkeypatch) -> None:
    _patch_run(monkeypatch, CodecError("ffmpeg failed: /secret/path"))
    with pytest.raises(CodecSimulationError) as exc:
        resample_to_16k(tone(1000, 8_000, 0.5), 8_000)
    assert "secret" not in str(exc.value)
