import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import make_wav
from voxguard.api.db import SessionRow
from voxguard.inference import Prediction, StubDetector
from voxguard.schemas import AnalyzeResponse


def _post(client: TestClient, data: bytes, **form: str):
    return client.post("/v1/analyze", files={"file": ("a.wav", data)}, data=form)


def _rows(client: TestClient) -> list[SessionRow]:
    with client.app.state.session_factory() as db:
        return list(db.query(SessionRow).all())


def test_analyze_ok_and_persisted(client: TestClient) -> None:
    r = _post(client, make_wav(1.0), caller_id="+911234")
    assert r.status_code == 200
    body = AnalyzeResponse.model_validate(r.json())
    assert body.codec_applied is None and body.model_version == "stub-0"
    (row,) = _rows(client)
    assert row.session_id == body.session_id and row.caller_id == "+911234"
    assert row.fake_score == body.fake_score and row.flagged == body.flagged


def test_same_audio_same_score(client: TestClient) -> None:
    data = make_wav(1.0)
    assert _post(client, data).json()["fake_score"] == _post(client, data).json()["fake_score"]


def test_flagged_follows_score_vs_threshold(client_factory) -> None:
    data = make_wav(1.0)
    with client_factory() as c:
        score = _post(c, data).json()["fake_score"]
    for threshold, expected in [(score, True), (min(score + 1e-6, 1.0), score >= 1.0)]:
        with client_factory(StubDetector(threshold=threshold)) as c:
            assert _post(c, data).json()["flagged"] is expected


def test_blank_caller_id_stored_as_none(client: TestClient) -> None:
    _post(client, make_wav(0.5), caller_id="")
    assert _rows(client)[0].caller_id is None


def test_simulate_codec_true_is_501_and_stores_nothing(client: TestClient) -> None:
    r = _post(client, make_wav(0.5), simulate_codec="true")
    assert r.status_code == 501 and _rows(client) == []


def test_simulate_codec_false_ok(client: TestClient) -> None:
    r = _post(client, make_wav(0.5), simulate_codec="false")
    assert r.status_code == 200 and r.json()["codec_applied"] is None


def test_missing_file_is_422(client: TestClient) -> None:
    assert client.post("/v1/analyze", data={"caller_id": "x"}).status_code == 422


def test_empty_upload_is_400(client: TestClient) -> None:
    assert _post(client, b"").status_code == 400


def test_non_audio_is_415(client: TestClient) -> None:
    r = _post(client, b"not audio at all")
    assert r.status_code == 415 and _rows(client) == []


def test_over_duration_is_422(client_factory) -> None:
    with client_factory(max_audio_seconds=1.0) as c:
        assert _post(c, make_wav(2.0)).status_code == 422


def test_upload_over_limit_is_413_and_stores_nothing(client_factory) -> None:
    with client_factory(max_upload_bytes=1_000) as c:
        # within multipart slack, so the handler's own bounded read must catch it
        assert _post(c, make_wav(1.0)).status_code == 413
        assert _rows(c) == []


def test_detector_receives_native_rate_and_full_waveform(client_factory) -> None:
    seen = {}

    class Spy(StubDetector):
        def predict(self, waveform: np.ndarray, sample_rate: int) -> Prediction:
            seen.update(n=len(waveform), sr=sample_rate, dtype=waveform.dtype, ndim=waveform.ndim)
            return super().predict(waveform, sample_rate)

    with client_factory(Spy(threshold=0.5)) as c:
        assert _post(c, make_wav(6.0, sr=8_000, channels=2)).status_code == 200
    assert seen == {"n": 48_000, "sr": 8_000, "dtype": np.dtype("float32"), "ndim": 1}


def test_detector_failure_is_500_and_stores_nothing(client_factory) -> None:
    class Broken(StubDetector):
        def predict(self, waveform: np.ndarray, sample_rate: int) -> Prediction:
            raise RuntimeError("boom")

    with client_factory(Broken(threshold=0.5)) as c:
        r = _post(c, make_wav(0.5))
        assert r.status_code == 500 and "boom" not in r.text
        assert _rows(c) == []


@pytest.mark.parametrize("bad", ["x" * 129])
def test_overlong_caller_id_is_422(client: TestClient, bad: str) -> None:
    assert _post(client, make_wav(0.5), caller_id=bad).status_code == 422
