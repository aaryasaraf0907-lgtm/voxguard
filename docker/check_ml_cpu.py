"""Build-time check of the INSTALL_ML=1 images: fails the build unless the environment is CPU-only.

    python docker/check_ml_cpu.py uv.lock docker/torch-cpu.txt

- torch is a CPU build (``torch.version.cuda is None``; wheel names differ per architecture, so
  the version suffix is not relied on) at uv.lock's version and at the pinned version;
- ``import torchaudio`` works and torchaudio is the pinned version (2.11.0+cpu; uv.lock's
  2.11.1 Linux wheel is a CUDA build, see docker/torch-cpu.txt);
- no CUDA, NVIDIA or Triton distribution is installed;
- torchaudio resampling actually runs (44.1 kHz -> 16 kHz, as the detector does).
"""

from __future__ import annotations

import importlib.metadata as md
import platform
import re
import sys
import tomllib

CUDA_PREFIXES = ("nvidia-", "nvidia_", "cuda-", "cuda_", "triton")


def pinned_versions(pins_path: str) -> dict[str, str]:
    """``{name: version}`` from the ``name==version`` lines of docker/torch-cpu.txt."""
    with open(pins_path) as f:
        text = f.read()
    return dict(re.findall(r"^([A-Za-z0-9_.-]+)==(\S+)", text, flags=re.M))


def main(lock_path: str, pins_path: str) -> None:
    import torch
    import torchaudio  # noqa: F401  (the import itself is a check: it loads the native library)
    import torchaudio.functional as AF

    with open(lock_path, "rb") as f:
        locked = {p["name"]: p["version"] for p in tomllib.load(f)["package"]}
    pins = pinned_versions(pins_path)

    assert torch.version.cuda is None, f"torch is a CUDA build ({torch.version.cuda})"
    assert not torch.backends.cudnn.is_available(), "cuDNN is available"
    installed = {n: md.version(n) for n in ("torch", "torchaudio")}
    for name, version in installed.items():
        assert version == pins[name], f"{name} {version} != pinned {pins[name]}"
    assert installed["torch"].split("+")[0] == locked["torch"], (
        f"torch {installed['torch']} != uv.lock {locked['torch']}"
    )

    names = sorted({(d.metadata["Name"] or "").lower() for d in md.distributions()})
    cuda = [n for n in names if n.startswith(CUDA_PREFIXES)]
    assert not cuda, f"CUDA/NVIDIA/Triton packages installed: {cuda}"

    t = torch.arange(44_100, dtype=torch.float32) / 44_100
    y = AF.resample(torch.sin(2 * torch.pi * 440 * t), 44_100, 16_000)
    assert y.shape == (16_000,) and bool(torch.isfinite(y).all()), y.shape
    peak = float(y.abs().max())
    assert 0.9 < peak < 1.1, f"resampled 440 Hz tone has peak {peak}"

    print(
        f"CPU-only ML environment OK on {platform.machine()}: torch {installed['torch']} "
        f"(cuda={torch.version.cuda}), torchaudio {installed['torchaudio']} "
        f"(uv.lock: {locked['torchaudio']}), resample 44100->16000 Hz peak {peak:.3f}, "
        f"{len(names)} distributions, none CUDA/NVIDIA/Triton"
    )


if __name__ == "__main__":
    main(*(sys.argv[1:3] if len(sys.argv) >= 3 else ("uv.lock", "docker/torch-cpu.txt")))
