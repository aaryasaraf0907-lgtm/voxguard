"""Docker smoke tests: build the images, run docker compose and the Space image, and exercise
the real containers. Opt-in (they build images and take minutes):

    VOXGUARD_DOCKER_TESTS=1 uv run pytest tests/test_docker_smoke.py -v
"""

import io
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).parents[1]
API_PORT, UI_PORT, SPACE_PORT = 18000, 18501, 17860

pytestmark = [
    pytest.mark.docker,
    pytest.mark.skipif(
        os.environ.get("VOXGUARD_DOCKER_TESTS") != "1" or shutil.which("docker") is None,
        reason="set VOXGUARD_DOCKER_TESTS=1 (needs a working docker with compose)",
    ),
]


def _wav() -> bytes:
    t = np.arange(16_000) / 16_000
    x = (0.3 * np.sin(2 * np.pi * 1000 * t) + 0.3 * np.sin(2 * np.pi * 6000 * t)).astype("f4")
    buf = io.BytesIO()
    sf.write(buf, x, 16_000, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _get(url: str) -> tuple[int, bytes]:
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.status, r.read()


def _post_wav(url: str, **form: str) -> tuple[int, dict]:
    boundary = uuid.uuid4().hex
    parts = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
        for k, v in form.items()
    ]
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n'
        "Content-Type: audio/wav\r\n\r\n".encode()
        + _wav()
        + b"\r\n"
    )
    body = b"".join(parts) + f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


class Compose:
    def __init__(self, env_file: Path) -> None:
        self.project = f"voxguard-smoke-{uuid.uuid4().hex[:8]}"
        self.env = {
            **os.environ,
            "VOXGUARD_ENV_FILE": str(env_file),
            "VOXGUARD_API_PORT": str(API_PORT),
            "VOXGUARD_UI_PORT": str(UI_PORT),
        }

    def __call__(self, *args: str, check: bool = True, timeout: int = 900):
        cmd = ["docker", "compose", "-p", self.project, "-f", str(ROOT / "docker-compose.yml")]
        return subprocess.run(
            [*cmd, *args], cwd=ROOT, env=self.env, capture_output=True, text=True,
            check=check, timeout=timeout,
        )  # fmt: skip


@pytest.fixture(scope="module")
def compose(tmp_path_factory) -> Iterator[Compose]:
    env_file = tmp_path_factory.mktemp("env") / "smoke.env"
    env_file.write_text("VOXGUARD_DETECTOR=stub\nVOXGUARD_SIMULATE_CODEC=g711_mulaw\n")
    c = Compose(env_file)
    try:
        c("up", "--build", "--detach", "--wait", "--wait-timeout", "180")
        yield c
    finally:
        logs = c("logs", "--no-color", check=False).stdout
        c("down", "--volumes", "--remove-orphans", check=False)
        print(logs[-4000:])


def test_compose_config_is_valid(compose: Compose) -> None:
    cfg = json.loads(compose("config", "--format", "json").stdout)
    assert set(cfg["services"]) == {"api", "ui"}
    assert cfg["services"]["ui"]["environment"]["VOXGUARD_API_URL"] == "http://api:8000"
    assert cfg["services"]["api"]["environment"]["VOXGUARD_DB_URL"] == "sqlite:////data/voxguard.db"
    # .env.example uses inline comments; compose must strip them, and the fixed DB URL must win
    env = {**compose.env, "VOXGUARD_ENV_FILE": str(ROOT / ".env.example")}
    out = subprocess.run(
        ["docker", "compose", "-f", str(ROOT / "docker-compose.yml"), "config", "--format", "json"],
        cwd=ROOT, env=env, capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    api_env = json.loads(out)["services"]["api"]["environment"]
    assert api_env["VOXGUARD_DETECTOR"] == "stub" and api_env["VOXGUARD_THRESHOLD"] == "0.5"
    assert api_env["VOXGUARD_DB_URL"] == "sqlite:////data/voxguard.db"


def test_api_health_from_host(compose: Compose) -> None:
    status, body = _get(f"http://127.0.0.1:{API_PORT}/health")
    assert status == 200
    assert json.loads(body) == {"status": "ok", "detector": "stub", "model_version": "stub-0"}


def test_ui_is_served(compose: Compose) -> None:
    assert _get(f"http://127.0.0.1:{UI_PORT}/_stcore/health") == (200, b"ok")
    assert _get(f"http://127.0.0.1:{UI_PORT}/")[0] == 200


def test_ui_container_reaches_api_by_service_name(compose: Compose) -> None:
    code = (
        "from voxguard.ui.api_client import VoxGuardClient; c = VoxGuardClient(); "
        "print(c.base_url, c.health().detector)"
    )
    out = compose("exec", "-T", "ui", "python", "-c", code).stdout.split()
    assert out == ["http://api:8000", "stub"]


def test_ui_client_phone_line_through_containers(compose: Compose) -> None:
    """The UI's own client, inside the UI container, asks the API for a simulated analysis."""
    code = (  # the UI image has no soundfile: build the WAV with the standard library
        "import io, wave, numpy as np; "
        "from voxguard.ui.api_client import VoxGuardClient; "
        "b = io.BytesIO(); w = wave.open(b, 'wb'); w.setnchannels(1); w.setsampwidth(2); "
        "w.setframerate(16000); "
        "w.writeframes((8000 * np.sin(np.arange(16000) / 3)).astype('<i2').tobytes()); "
        "w.close(); c = VoxGuardClient(); "
        "print(c.analyze(b.getvalue(), simulate_codec=True).codec_applied, "
        "c.analyze(b.getvalue()).codec_applied)"
    )
    r = compose("exec", "-T", "ui", "python", "-c", code, check=False)
    assert r.returncode == 0, r.stderr[-2000:]
    out = r.stdout.split()
    assert out == ["g711_mulaw", "None"]


def test_containers_run_as_non_root(compose: Compose) -> None:
    for service in ("api", "ui"):
        assert compose("exec", "-T", service, "id", "-u").stdout.strip() == "1000"


def test_ffmpeg_and_encoders_in_api_container(compose: Compose) -> None:
    report = compose("exec", "-T", "api", "python", "-m", "voxguard.data.codecs").stdout
    print(report)
    for codec in ("g711_mulaw", "g711_alaw"):  # required by data/codecs.py
        assert any(
            line.split()[:1] == [codec] and line.endswith("OK") for line in report.splitlines()
        )


def test_analyze_and_codec_simulation_through_container(compose: Compose) -> None:
    url = f"http://127.0.0.1:{API_PORT}/v1/analyze"
    status, plain = _post_wav(url, caller_id="smoke-caller")
    assert status == 200 and plain["codec_applied"] is None
    status, sim = _post_wav(url, simulate_codec="true", caller_id="smoke-caller")
    assert status == 200, sim
    assert sim["codec_applied"] == "g711_mulaw"
    assert sim["fake_score"] != plain["fake_score"]  # the detector saw different (degraded) audio
    status, sessions = _get(f"http://127.0.0.1:{API_PORT}/v1/sessions?limit=10")
    codecs = {s["codec_applied"] for s in json.loads(sessions)}
    assert codecs == {None, "g711_mulaw"}
    for path in ("/v1/sessions?flagged_only=true", "/v1/blocklist", "/v1/alerts"):
        assert _get(f"http://127.0.0.1:{API_PORT}{path}")[0] == 200


def test_sqlite_survives_container_recreation(compose: Compose) -> None:
    def session_ids() -> set[str]:
        _, body = _get(f"http://127.0.0.1:{API_PORT}/v1/sessions?limit=500")
        return {s["session_id"] for s in json.loads(body)}

    status, created = _post_wav(f"http://127.0.0.1:{API_PORT}/v1/analyze")
    assert status == 200
    before = session_ids()
    assert created["session_id"] in before
    compose("down")  # removes the containers, keeps the named volume
    compose("up", "--detach", "--wait", "--wait-timeout", "180")
    assert session_ids() == before


def test_bad_configuration_fails_with_a_clear_error(compose: Compose) -> None:
    cases = {
        "VOXGUARD_DETECTOR=real": "INSTALL_ML=1",  # default image has no ml extra
        "VOXGUARD_SIMULATE_CODEC=bogus": "VOXGUARD_SIMULATE_CODEC must be one of",
        "VOXGUARD_THRESHOLD_MODE=low_far": "VOXGUARD_LOW_FAR_THRESHOLD",
    }
    for env, expected in cases.items():
        r = compose("run", "--rm", "--no-deps", "-e", env, "api", check=False, timeout=120)
        assert r.returncode != 0, env
        assert expected in r.stdout + r.stderr, (env, (r.stdout + r.stderr)[-2000:])


@pytest.fixture(scope="module")
def space() -> Iterator[tuple[str, str]]:
    tag = f"voxguard-space:smoke-{uuid.uuid4().hex[:8]}"
    name = tag.replace(":", "-")
    subprocess.run(
        ["docker", "build", "-f", "docker/Dockerfile.space", "-t", tag, "."],
        cwd=ROOT, check=True, capture_output=True, text=True, timeout=900,
    )  # fmt: skip
    subprocess.run(
        ["docker", "run", "-d", "--name", name, "-p", f"{SPACE_PORT}:7860",
         "-e", "VOXGUARD_SIMULATE_CODEC=g711_mulaw", tag],
        check=True, capture_output=True, text=True,
    )  # fmt: skip
    try:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            try:
                if _get(f"http://127.0.0.1:{SPACE_PORT}/_stcore/health")[0] == 200:
                    break
            except OSError:
                time.sleep(1)
        else:
            pytest.fail("Space container did not become ready")
        yield name, tag
    finally:
        print(subprocess.run(["docker", "logs", name], capture_output=True, text=True).stdout)
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        subprocess.run(["docker", "rmi", "-f", tag], capture_output=True)


def test_space_image_serves_ui_on_7860_with_internal_api(space: tuple[str, str]) -> None:
    space, _ = space
    assert _get(f"http://127.0.0.1:{SPACE_PORT}/_stcore/health") == (200, b"ok")
    code = (
        "from voxguard.ui.api_client import VoxGuardClient; c = VoxGuardClient(); "
        "print(c.base_url, c.health().detector)"
    )
    r = subprocess.run(
        ["docker", "exec", space, "python", "-c", code], capture_output=True, text=True
    )
    assert r.stdout.split() == ["http://127.0.0.1:8000", "stub"], r.stderr
    r = subprocess.run(["docker", "exec", space, "id", "-u"], capture_output=True, text=True)
    assert r.stdout.strip() == "1000"


def test_space_exits_early_when_the_api_cannot_start(space: tuple[str, str]) -> None:
    """space_start.sh must stop as soon as the API dies, not wait out its start timeout."""
    _, tag = space
    started = time.monotonic()
    r = subprocess.run(
        ["docker", "run", "--rm", "-e", "VOXGUARD_SIMULATE_CODEC=bogus",
         "-e", "VOXGUARD_API_START_TIMEOUT=300", tag],
        capture_output=True, text=True, timeout=120,
    )  # fmt: skip
    elapsed = time.monotonic() - started
    out = r.stdout + r.stderr
    assert r.returncode != 0 and elapsed < 60, (elapsed, out[-2000:])
    assert "VOXGUARD_SIMULATE_CODEC must be one of" in out
    assert "VoxGuard API exited during startup" in out


def test_space_start_timeout_is_validated(space: tuple[str, str]) -> None:
    _, tag = space
    r = subprocess.run(
        ["docker", "run", "--rm", "-e", "VOXGUARD_API_START_TIMEOUT=soon", tag],
        capture_output=True, text=True, timeout=120,
    )  # fmt: skip
    assert r.returncode != 0
    assert "VOXGUARD_API_START_TIMEOUT must be a number" in r.stdout + r.stderr
