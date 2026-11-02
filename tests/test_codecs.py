import numpy as np
import pytest

from audio_helpers import SAMPLE_RATE, synthetic
from voxguard.data.codecs import (
    CODECS,
    CodecError,
    apply_codec,
    available_codecs,
    codec_report,
    ffmpeg_path,
    usable_codecs,
)

needs_ffmpeg = pytest.mark.skipif(ffmpeg_path() is None, reason="ffmpeg not installed")


def band_energy(x: np.ndarray, lo: float, hi: float) -> float:
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / SAMPLE_RATE)
    return float(spec[(freqs >= lo) & (freqs < hi)].sum())


def test_codec_table_sane() -> None:
    assert CODECS["g711_mulaw"].required and CODECS["g711_alaw"].required
    assert CODECS["amr_nb"].codec_rate == 8000 and CODECS["amr_wb"].codec_rate == 16000


def test_report_lists_every_codec() -> None:
    report = codec_report()
    assert all(name in report for name in CODECS)


def test_unknown_codec_and_bad_bitrate() -> None:
    with pytest.raises(CodecError):
        apply_codec(synthetic(), "nope")
    with pytest.raises(CodecError):
        apply_codec(synthetic(), "amr_nb", 12345)
    with pytest.raises(CodecError):
        usable_codecs(["nope"]) if ffmpeg_path() else (_ for _ in ()).throw(CodecError("x"))


@needs_ffmpeg
@pytest.mark.parametrize("codec", ["g711_mulaw", "g711_alaw"])
def test_g711_always_works_and_removes_wideband_content(codec: str) -> None:
    assert available_codecs()[codec]
    x = synthetic()
    y = apply_codec(x, codec)
    assert y.shape == x.shape and y.dtype == np.float32
    assert np.isfinite(y).all()
    # 8 kHz telephony rate: the 3 kHz tone survives, content above 4 kHz is gone.
    assert band_energy(y, 2800, 3200) > 0.5 * band_energy(x, 2800, 3200)
    assert band_energy(y, 4500, 8000) < 0.1 * band_energy(x, 4500, 8000) + 1e-9
    assert np.corrcoef(x, y)[0, 1] > 0.9


@needs_ffmpeg
def test_g711_is_deterministic() -> None:
    x = synthetic()
    assert np.array_equal(apply_codec(x, "g711_mulaw"), apply_codec(x, "g711_mulaw"))


@needs_ffmpeg
@pytest.mark.parametrize("codec", [c for c, ok in available_codecs().items() if ok])
def test_every_available_codec_preserves_length_and_signal(codec: str) -> None:
    x = synthetic(1.37)  # odd length: not a multiple of any codec frame
    y = apply_codec(x, codec)
    assert len(y) == len(x) and np.isfinite(y).all()
    assert band_energy(y, 300, 600) > 0.3 * band_energy(x, 300, 600)  # 440 Hz tone survives


@needs_ffmpeg
def test_usable_codecs_skips_missing_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    names = usable_codecs()
    assert "g711_mulaw" in names and "g711_alaw" in names
    for name, ok in available_codecs().items():
        assert (name in names) == ok
    if not all(available_codecs().values()):
        assert "skipping codec" in caplog.text
