"""Docker smoke test of the INSTALL_ML=1 API image: CPU-only torch and real-detector startup.

Opt-in (builds the ML image, ~0.4 GB of wheels, takes minutes):

    VOXGUARD_DOCKER_ML_TESTS=1 uv run pytest tests/test_docker_ml.py -v

The real detector runs on an UNTRAINED RawNet2 checkpoint written inside the container from
configs/rawnet2.yaml, so no Hugging Face token or private checkpoint is needed. It checks that the
detector loads and serves through the API, never what it scores.
"""

import json
import os
import shutil
import subprocess
import time
import uuid
from collections.abc import Iterator

import pytest

from test_docker_smoke import ROOT, _get, _post_wav

PORT = 18100

pytestmark = [
    pytest.mark.docker,
    pytest.mark.skipif(
        os.environ.get("VOXGUARD_DOCKER_ML_TESTS") != "1" or shutil.which("docker") is None,
        reason="set VOXGUARD_DOCKER_ML_TESTS=1 (builds the INSTALL_ML=1 image; needs docker)",
    ),
]

AUDIT = r"""
import importlib.metadata as md, json, os, torch, torchaudio, torchaudio.functional as AF
y = AF.resample(torch.sin(torch.arange(44100) / 44100 * 6.2832 * 440), 44100, 16000)
print(json.dumps({
    "cuda": torch.version.cuda,
    "torch": md.version("torch"),
    "torchaudio": md.version("torchaudio"),
    "cuda_dists": [d.metadata["Name"] for d in md.distributions()
                   if d.metadata["Name"].lower().startswith(("nvidia", "cuda", "triton"))],
    "resampled": list(y.shape),
    "venv_writable": os.access("/opt/venv/lib/python3.11/site-packages", os.W_OK),
}))
"""


def _docker(*args: str, check: bool = True, timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", *args], cwd=ROOT, capture_output=True, text=True, check=check, timeout=timeout
    )


@pytest.fixture(scope="module")
def image() -> Iterator[str]:
    tag = f"voxguard-api:ml-smoke-{uuid.uuid4().hex[:8]}"
    _docker(
        "build", "--build-arg", "INSTALL_ML=1", "-f", "docker/Dockerfile.api", "-t", tag, ".",
        timeout=1800,
    )  # fmt: skip
    try:
        yield tag
    finally:
        _docker("rmi", "-f", tag, check=False)


@pytest.fixture(scope="module")
def checkpoint_volume(image: str) -> Iterator[str]:
    """A volume holding an untrained RawNet2 best.pt (written as root, mounted read-only later)."""
    volume = f"voxguard-ml-smoke-{uuid.uuid4().hex[:8]}"
    _docker("volume", "create", volume)
    code = (
        "from pathlib import Path; from voxguard.benchmark import write_untrained_checkpoint; "
        "write_untrained_checkpoint(Path('/configs/rawnet2.yaml'), Path('/models/best.pt'))"
    )
    try:
        _docker(
            "run", "--rm", "-u", "0", "-v", f"{volume}:/models",
            "-v", f"{ROOT / 'configs'}:/configs:ro", image, "python", "-c", code,
        )  # fmt: skip
        yield volume
    finally:
        _docker("volume", "rm", "-f", volume, check=False)


def test_image_is_cpu_only_and_cannot_be_resynced_to_cuda(image: str) -> None:
    # The same check the build runs (a cached build does not print it).
    check = _docker("run", "--rm", image, "python", "docker/check_ml_cpu.py",
                    "uv.lock", "docker/torch-cpu.txt")  # fmt: skip
    assert "CPU-only ML environment OK" in check.stdout, check.stdout + check.stderr
    audit = json.loads(_docker("run", "--rm", image, "python", "-c", AUDIT).stdout)
    assert audit == {
        "cuda": None,
        "torch": "2.15.0+cpu",
        "torchaudio": "2.11.0+cpu",
        "cuda_dists": [],
        "resampled": [16000],
        "venv_writable": False,  # the default user cannot `uv sync` CUDA torch back in
    }, audit
    # `uv run` must not sync the environment back to uv.lock's CUDA torch (UV_NO_SYNC=1).
    r = _docker("run", "--rm", image, "uv", "run", "python", "-c",
                "import torch; print(torch.version.cuda, torch.__version__)")  # fmt: skip
    assert r.stdout.split() == ["None", "2.15.0+cpu"], r.stderr[-2000:]


def test_real_detector_starts_and_serves_from_the_ml_image(
    image: str, checkpoint_volume: str
) -> None:
    name = f"voxguard-ml-smoke-{uuid.uuid4().hex[:8]}"
    _docker(
        "run", "-d", "--name", name, "-p", f"{PORT}:8000",
        "-v", f"{checkpoint_volume}:/models:ro",
        "-e", "VOXGUARD_DETECTOR=real", "-e", "VOXGUARD_MODEL_REPO=/models/best.pt",
        "-e", "VOXGUARD_SIMULATE_CODEC=g711_mulaw", image,
    )  # fmt: skip
    try:
        deadline = time.monotonic() + 180
        while True:
            try:
                status, body = _get(f"http://127.0.0.1:{PORT}/health")
                break
            except OSError:
                if time.monotonic() > deadline:
                    pytest.fail(_docker("logs", name, check=False).stderr[-3000:])
                time.sleep(1)
        assert status == 200 and b'"detector":"real"' in body.replace(b" ", b""), body
        assert b"rawnet2:best.pt:epoch0" in body, body
        code, clean = _post_wav(f"http://127.0.0.1:{PORT}/v1/analyze")
        assert code == 200 and clean["model_version"] == "rawnet2:best.pt:epoch0", clean
        assert clean["latency_ms"] > 0 and clean["codec_applied"] is None
        code, phone = _post_wav(f"http://127.0.0.1:{PORT}/v1/analyze", codec="g711_mulaw")
        assert code == 200 and phone["codec_applied"] == "g711_mulaw" and phone["codec_ms"] > 0
    finally:
        print(_docker("logs", name, check=False).stderr[-3000:])
        _docker("rm", "-f", name, check=False)


def test_missing_checkpoint_stops_startup_without_a_stub_fallback(
    image: str, checkpoint_volume: str
) -> None:
    r = _docker(
        "run", "--rm", "-v", f"{checkpoint_volume}:/models:ro",
        "-e", "VOXGUARD_DETECTOR=real", "-e", "VOXGUARD_MODEL_REPO=/models/missing.pt", image,
        check=False, timeout=180,
    )  # fmt: skip
    out = r.stdout + r.stderr
    assert r.returncode != 0, out[-2000:]
    assert "VOXGUARD_MODEL_REPO checkpoint not found" in out, out[-2000:]
    assert "Uvicorn running" not in out
