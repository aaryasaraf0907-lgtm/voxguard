import io
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from voxguard.api.deps import Settings
from voxguard.api.main import create_app
from voxguard.inference import Detector, Prediction, StubDetector


def make_wav(seconds: float = 1.0, sr: int = 16_000, channels: int = 1, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((int(seconds * sr), channels)) * 0.1).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "docker: builds and runs Docker images (opt-in)")


class FixedDetector:
    """Returns a chosen fake_score; the label follows its own threshold."""

    model_version = "fixed-0"

    def __init__(self, score: float, threshold: float = 0.5) -> None:
        self.score = score
        self.threshold = threshold

    def predict(self, waveform: np.ndarray, sample_rate: int) -> Prediction:
        label = "fake" if self.score >= self.threshold else "real"
        return Prediction(self.score, label, self.threshold, 1.0, self.model_version)


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


@pytest.fixture
def live_server(tmp_path):
    """The real app (stub detector) under uvicorn on a free 127.0.0.1 port, in a thread.
    Yields (ws:// base URL, session factory)."""
    import socket
    import threading
    import time

    uvicorn = pytest.importorskip("uvicorn")
    pytest.importorskip("websockets")
    app = create_app(
        StubDetector(threshold=0.5), Settings(db_url=f"sqlite:///{tmp_path / 'live.db'}")
    )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.05)
    yield f"ws://127.0.0.1:{port}", lambda: app.state.session_factory()
    server.should_exit = True
    thread.join(10)
