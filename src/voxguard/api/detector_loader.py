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

import numpy as np

from voxguard.api import thresholds as thresholds_mod
from voxguard.api.thresholds import ThresholdsError, ThresholdSet
from voxguard.inference import DEFAULT_THRESHOLD, NUM_SAMPLES, TARGET_SR, Detector, get_detector

logger = logging.getLogger(__name__)


def _missing_ml(exc: ModuleNotFoundError) -> RuntimeError:
    return RuntimeError(
        f"VOXGUARD_DETECTOR=real needs the 'ml' extra, but {exc.name!r} is not installed. "
        "Install it with `uv sync --extra api --extra ml`, or build the Docker API image "
        "with INSTALL_ML=1."
    )


class DeviceUnavailableError(RuntimeError):
    """``VOXGUARD_DEVICE`` asks for a device this machine does not have."""


def _check_device(device: str) -> None:
    if device != "cuda":
        return
    import torch

    if not torch.cuda.is_available():
        raise DeviceUnavailableError(
            "VOXGUARD_DEVICE=cuda but torch reports no CUDA device; use cpu or auto"
        )


def load_detector(device: str = "cpu") -> Detector:
    kind = os.environ.get("VOXGUARD_DETECTOR", "stub").strip().lower()
    if kind != "real":
        return get_detector()  # stub, or its own clear error for unknown values
    try:  # the missing ML stack is the more basic problem, so report it first
        import torch  # noqa: F401
    except ModuleNotFoundError as exc:
        raise _missing_ml(exc) from exc
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
    except DeviceUnavailableError:
        raise
    except FileNotFoundError as exc:
        raise RuntimeError(f"VOXGUARD_MODEL_REPO checkpoint not found: {exc}") from exc
    except ValueError:
        raise  # RealDetector's own messages (threshold, missing source) are already clear
    except Exception as exc:  # hub errors; corrupt or incompatible checkpoints (torch errors)
        origin = type(exc).__module__.split(".")[0]
        where = (
            "download the checkpoint from the Hugging Face Hub"
            if origin == "huggingface_hub"
            else "load the checkpoint"
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
    _warm_up(detector)
    return detector


def _warm_up(detector: Detector) -> None:
    """One inference on silence before serving, so the first request does not pay for CUDA /
    cuDNN initialisation (about 1 s on a laptop GPU) and report it as detector latency."""
    t0 = time.perf_counter()
    try:
        detector.predict(np.zeros(NUM_SAMPLES, dtype=np.float32), TARGET_SR)
    except Exception as exc:  # a broken model fails the first request with a clear 500 anyway
        raise RuntimeError(f"the real detector failed its warm-up inference: {exc}") from exc
    logger.info("warm-up inference took %.0f ms", (time.perf_counter() - t0) * 1000)


def load_model_thresholds(detector: Detector) -> ThresholdSet | None:
    """The real detector's thresholds.json (next to its checkpoint), or None.

    The stub has none. A missing file keeps the legacy single threshold; an unusable one
    (malformed, out of range, for another model_version) stops startup.
    """
    if os.environ.get("VOXGUARD_DETECTOR", "stub").strip().lower() != "real":
        return None
    repo = os.environ.get("VOXGUARD_MODEL_REPO", "").strip()
    try:
        found = thresholds_mod.load_thresholds(repo, detector.model_version)
    except ThresholdsError as exc:
        raise RuntimeError(f"unusable thresholds for VOXGUARD_MODEL_REPO={repo!r}: {exc}") from exc
    except Exception as exc:  # hub/network errors while looking for thresholds.json
        raise RuntimeError(
            f"could not fetch {thresholds_mod.FILENAME} (VOXGUARD_MODEL_REPO={repo!r}): "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if found is None:
        logger.warning("%s has no %s; using one threshold", repo, thresholds_mod.FILENAME)
    else:
        logger.info(
            "thresholds for %s (selected on %s): clean %.4g, phone %.4g",
            found.model_version, found.selected_on, found.clean, found.phone,
        )  # fmt: skip
    return found


def warn_if_default_threshold(
    detector: Detector, thresholds: ThresholdSet | None, threshold_mode: str, override: float | None
) -> None:
    """Warn when the real detector would decide at the untuned 0.5 default."""
    real = os.environ.get("VOXGUARD_DETECTOR", "stub").strip().lower() == "real"
    if real and threshold_mode == "eer" and override is None and thresholds is None:
        logger.warning(
            "VOXGUARD_THRESHOLD is the default %.2f, which is not an operating point for any "
            "trained checkpoint, and the checkpoint has no %s. Upload its %s, set "
            "VOXGUARD_THRESHOLD to its eer_threshold, or use VOXGUARD_THRESHOLD_MODE=low_far.",
            DEFAULT_THRESHOLD, thresholds_mod.FILENAME, thresholds_mod.FILENAME,
        )  # fmt: skip
