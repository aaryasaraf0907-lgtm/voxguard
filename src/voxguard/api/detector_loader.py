"""Detector construction for the API: one detector per process, built at startup.

``VOXGUARD_DETECTOR=stub`` (default) goes through ``inference.get_detector()`` unchanged. For
``real`` the API builds ``RealDetector`` itself so it can pass ``VOXGUARD_DEVICE``
(``get_detector()`` always uses the CPU). Every failure becomes a ``RuntimeError`` that says what
to fix; the API never falls back to the stub when the real detector was requested.
"""

from __future__ import annotations

import logging
import os
import time

from voxguard.inference import Detector, get_detector

logger = logging.getLogger(__name__)


def _missing_ml(exc: ModuleNotFoundError) -> RuntimeError:
    return RuntimeError(
        f"VOXGUARD_DETECTOR=real needs the 'ml' extra, but {exc.name!r} is not installed. "
        "Install it with `uv sync --extra api --extra ml`, or build the Docker API image "
        "with INSTALL_ML=1."
    )


def _check_device(device: str) -> None:
    if device != "cuda":
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("VOXGUARD_DEVICE=cuda but torch reports no CUDA device; use cpu or auto")


def load_detector(device: str = "cpu") -> Detector:
    kind = os.environ.get("VOXGUARD_DETECTOR", "stub").strip().lower()
    if kind != "real":
        return get_detector()  # stub, or its own clear error for unknown values
    repo = os.environ.get("VOXGUARD_MODEL_REPO", "").strip()
    if not repo:
        raise RuntimeError(
            "VOXGUARD_DETECTOR=real needs VOXGUARD_MODEL_REPO (a local checkpoint path or a "
            "Hugging Face repo id)"
        )
    t0 = time.perf_counter()
    try:
        from voxguard.inference import RealDetector

        _check_device(device)
        detector = RealDetector(repo, device=device)
    except ModuleNotFoundError as exc:
        raise _missing_ml(exc) from exc
    except RuntimeError:
        raise
    except FileNotFoundError as exc:
        raise RuntimeError(f"VOXGUARD_MODEL_REPO checkpoint not found: {exc}") from exc
    except ValueError:
        raise  # RealDetector's own messages (threshold, missing source) are already clear
    except Exception as exc:  # hub errors, corrupt or incompatible checkpoints
        origin = type(exc).__module__.split(".")[0]
        where = (
            "download it from the Hugging Face Hub" if origin == "huggingface_hub" else "load it"
        )
        raise RuntimeError(
            f"could not {where} (VOXGUARD_MODEL_REPO={repo!r}): {type(exc).__name__}: {exc}"
        ) from exc
    logger.info(
        "loaded %s on %s in %.1f s",
        detector.model_version,
        getattr(detector, "device", device),
        time.perf_counter() - t0,
    )
    return detector
