"""The CPU-only torch pins of the INSTALL_ML=1 images stay consistent with uv.lock.

docker/torch-cpu.txt pins torch and torchaudio (CPU builds, one hash per architecture) and
docker/ml-cpu-skip.txt lists what `uv sync` must not install. A torch bump in uv.lock, or a new
CUDA dependency of torch, must fail here rather than in a Docker build.
"""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[1]
PINS = ROOT / "docker" / "torch-cpu.txt"
SKIP = ROOT / "docker" / "ml-cpu-skip.txt"
CUDA_PREFIXES = ("nvidia-", "cuda-", "triton")


def _lock() -> dict[str, dict]:
    with open(ROOT / "uv.lock", "rb") as f:
        return {p["name"]: p for p in tomllib.load(f)["package"]}


def _pins() -> dict[str, tuple[str, list[str]]]:
    """{name: (version, [sha256, ...])} from the pip requirements file."""
    text = PINS.read_text().replace("\\\n", " ")
    pins = {}
    for line in text.splitlines():
        m = re.match(r"^([A-Za-z0-9_.-]+)==(\S+)(.*)$", line.strip())
        if m:
            pins[m[1]] = (m[2], re.findall(r"--hash=sha256:([0-9a-f]{64})", m[3]))
    return pins


def _skip() -> set[str]:
    return {
        line.split("#")[0].strip()
        for line in SKIP.read_text().splitlines()
        if line.split("#")[0].strip()
    }


def _cuda_deps_of(name: str, lock: dict[str, dict]) -> set[str]:
    """CUDA packages reachable from ``name`` in uv.lock (any platform marker)."""
    seen, stack = set(), [name]
    while stack:
        for dep in lock[stack.pop()].get("dependencies", []):
            dep_name = dep["name"]
            if dep_name.startswith(CUDA_PREFIXES) and dep_name not in seen:
                seen.add(dep_name)
                stack.append(dep_name)
    return seen


def test_pins_are_cpu_builds_with_one_hash_per_architecture() -> None:
    pins = _pins()
    assert set(pins) == {"torch", "torchaudio"}
    for name, (version, hashes) in pins.items():
        assert version.endswith("+cpu"), (name, version)
        assert len(hashes) == len(set(hashes)) == 2, (name, hashes)  # amd64 + arm64


def test_pinned_torch_is_the_locked_version() -> None:
    assert _pins()["torch"][0].split("+")[0] == _lock()["torch"]["version"]


def test_pinned_torchaudio_is_the_agreed_deviation_from_the_lock() -> None:
    """2.11.0+cpu while uv.lock has 2.11.1 (its Linux wheel is a CUDA build; see the pins file).
    If uv.lock's torchaudio changes, this pin has to be re-checked, not silently kept."""
    assert _pins()["torchaudio"][0] == "2.11.0+cpu"
    assert _lock()["torchaudio"]["version"] == "2.11.1"


def test_skip_list_covers_torch_torchaudio_and_every_cuda_dependency() -> None:
    lock = _lock()
    cuda = _cuda_deps_of("torch", lock) | _cuda_deps_of("torchaudio", lock)
    assert cuda, "uv.lock's torch no longer has CUDA dependencies: revisit the CPU pins"
    assert {"torch", "torchaudio"} | cuda <= _skip()


def test_skip_list_holds_only_torch_and_cuda_packages() -> None:
    lock = _lock()
    for name in _skip():
        assert name in lock, f"{name} is not in uv.lock"
        assert name in ("torch", "torchaudio") or name.startswith(CUDA_PREFIXES), name


def test_dockerfiles_install_through_the_cpu_script() -> None:
    for dockerfile in ("Dockerfile.api", "Dockerfile.space"):
        text = (ROOT / "docker" / dockerfile).read_text()
        assert "uv sync" not in text, f"{dockerfile} must sync through docker/ml_cpu.sh"
        assert text.count("sh docker/ml_cpu.sh sync") == 2
        assert "sh docker/ml_cpu.sh torch" in text and "sh docker/ml_cpu.sh check" in text
        assert "UV_NO_SYNC=1" in text
