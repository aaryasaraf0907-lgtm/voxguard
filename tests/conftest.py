import io
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from voxguard.api.deps import Settings
from voxguard.api.main import create_app
from voxguard.inference import Detector, StubDetector


def make_wav(seconds: float = 1.0, sr: int = 16_000, channels: int = 1, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((int(seconds * sr), channels)) * 0.1).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@pytest.fixture
def client_factory() -> Callable[..., object]:
    @contextmanager
    def factory(detector: Detector | None = None, **settings: object) -> Iterator[TestClient]:
        cfg = Settings(db_url="sqlite://", **settings)  # type: ignore[arg-type]
        app = create_app(detector or StubDetector(threshold=0.5), cfg)
        with TestClient(app) as client:
            yield client

    return factory


@pytest.fixture
def client(client_factory: Callable[..., object]) -> Iterator[TestClient]:
    with client_factory() as c:  # type: ignore[attr-defined]
        yield c
