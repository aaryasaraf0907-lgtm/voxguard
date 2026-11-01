from datetime import UTC, datetime

import numpy as np
import pytest
from pydantic import ValidationError

from voxguard.inference import StubDetector
from voxguard.schemas import (
    AlertRecord,
    AnalyzeResponse,
    BlocklistEntry,
    HealthResponse,
    SessionRecord,
)


def test_analyze_response_from_prediction() -> None:
    pred = StubDetector(threshold=0.5).predict(np.ones(16_000, dtype=np.float32), 16_000)
    resp = AnalyzeResponse(
        session_id="s1",
        label=pred.label,
        fake_score=pred.fake_score,
        threshold=pred.threshold,
        latency_ms=pred.latency_ms,
        model_version=pred.model_version,
        flagged=pred.fake_score >= pred.threshold,
    )
    assert resp.codec_applied is None
    assert set(resp.model_dump()) == {
        "session_id", "label", "fake_score", "threshold", "latency_ms",
        "model_version", "flagged", "codec_applied",
    }  # fmt: skip


def test_analyze_response_rejects_bad_values() -> None:
    base = dict(
        session_id="s", label="real", fake_score=0.1, threshold=0.5,
        latency_ms=1.0, model_version="v", flagged=False,
    )  # fmt: skip
    AnalyzeResponse(**base)
    with pytest.raises(ValidationError):
        AnalyzeResponse(**{**base, "fake_score": 1.5})
    with pytest.raises(ValidationError):
        AnalyzeResponse(**{**base, "label": "maybe"})


def test_other_models_construct() -> None:
    now = datetime.now(UTC)
    HealthResponse(status="ok", detector="stub", model_version="stub-0")
    SessionRecord(
        session_id="s", created_at=now, label="fake", fake_score=0.9, threshold=0.5,
        flagged=True, model_version="v",
    )  # fmt: skip
    BlocklistEntry(caller_id="+911234567890", hit_count=2, first_seen=now, last_seen=now)
    AlertRecord(alert_id="a", created_at=now, session_id="s", fake_score=0.9, message="flagged")
