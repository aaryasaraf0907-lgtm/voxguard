import numpy as np
import pytest
import soundfile as sf
from fastapi import HTTPException

from conftest import make_wav
from voxguard.api.audio import MAX_SAMPLE_RATE, decode_audio


def test_mono_16k() -> None:
    x, sr = decode_audio(make_wav(1.0), 60)
    assert sr == 16_000 and x.shape == (16_000,) and x.dtype == np.float32


def test_native_rate_preserved() -> None:
    _, sr = decode_audio(make_wav(1.0, sr=8_000), 60)
    assert sr == 8_000


def test_stereo_downmixed_to_mono() -> None:
    x, sr = decode_audio(make_wav(1.0, sr=44_100, channels=2), 60)
    assert sr == 44_100 and x.shape == (44_100,)


def test_very_short_clip_accepted() -> None:
    x, _ = decode_audio(make_wav(0.01), 60)
    assert x.shape == (160,)


def test_truncated_wav_decodes_available_frames() -> None:
    # libsndfile clamps the declared length to the bytes present (verified, not an error)
    data = make_wav(1.0)
    x, _ = decode_audio(data[: len(data) // 2], 60)
    assert 0 < len(x) < 16_000


@pytest.mark.parametrize("data", [b"hello world", b"\x00" * 100, b"RIFF"])
def test_undecodable_is_415(data: bytes) -> None:
    with pytest.raises(HTTPException) as exc:
        decode_audio(data, 60)
    assert exc.value.status_code == 415


def test_header_only_wav_is_422() -> None:
    with pytest.raises(HTTPException) as exc:
        decode_audio(make_wav(1.0)[:44], 60)
    assert exc.value.status_code == 422


def test_over_duration_is_422() -> None:
    with pytest.raises(HTTPException) as exc:
        decode_audio(make_wav(2.0), 1.0)
    assert exc.value.status_code == 422


def test_exactly_at_duration_limit_ok() -> None:
    x, _ = decode_audio(make_wav(1.0), 1.0)
    assert len(x) == 16_000


def test_too_many_channels_is_422() -> None:
    with pytest.raises(HTTPException) as exc:
        decode_audio(make_wav(0.1, channels=3), 60)
    assert exc.value.status_code == 422


def test_sample_rate_above_limit_is_422() -> None:
    with pytest.raises(HTTPException) as exc:
        decode_audio(make_wav(0.01, sr=MAX_SAMPLE_RATE + 1000), 60)
    assert exc.value.status_code == 422


@pytest.mark.skipif("FLAC" not in sf.available_formats(), reason="no FLAC support")
def test_flac_decodes() -> None:
    import io

    buf = io.BytesIO()
    sf.write(buf, np.zeros(1600, dtype=np.float32), 16_000, format="FLAC")
    x, sr = decode_audio(buf.getvalue(), 60)
    assert sr == 16_000 and len(x) == 1600
