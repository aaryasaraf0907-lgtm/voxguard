"""Per-checkpoint decision thresholds from ``thresholds.json`` (agreed with Aarya, 13 Nov 2026).

The file sits next to ``best.pt`` (a local directory, or the checkpoint's Hugging Face model
repo) and is chosen on the dev set by ``evaluate.py``:

    {"model_version": "...", "selected_on": "dev", "clean": 0.0, "phone": 0.0,
     "low_far": {"clean": 0.0, "phone": 0.0}, "per_codec": {"opus": 0.0}}

Reading it needs no torch. The API uses ``clean`` and ``phone``; ``low_far`` and
``per_codec`` are validated and kept for reporting but not used for decisions.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FILENAME = "thresholds.json"


class ThresholdsError(ValueError):
    """thresholds.json exists but cannot be used (malformed, out of range, wrong model)."""


@dataclass(frozen=True)
class ThresholdSet:
    model_version: str
    clean: float
    phone: float
    selected_on: str | None = None
    low_far: dict[str, float] = field(default_factory=dict)
    per_codec: dict[str, float] = field(default_factory=dict)


def _threshold(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ThresholdsError(f"{where} must be a number in [0, 1], got {value!r}")
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ThresholdsError(f"{where} must be a number in [0, 1], got {value!r}")
    return value


def _threshold_map(value: Any, where: str) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ThresholdsError(f"{where} must be an object, got {type(value).__name__}")
    return {str(k): _threshold(v, f"{where}.{k}") for k, v in value.items()}


def parse_thresholds(data: Any, expected_model_version: str) -> ThresholdSet:
    """Validate a decoded thresholds.json against the loaded detector's model_version."""
    if not isinstance(data, dict):
        raise ThresholdsError(f"{FILENAME} must hold a JSON object")
    version = data.get("model_version")
    if not isinstance(version, str) or not version:
        raise ThresholdsError(f"{FILENAME}: model_version must be a non-empty string")
    if version != expected_model_version:
        raise ThresholdsError(
            f"{FILENAME} is for model_version {version!r}, but the loaded checkpoint is "
            f"{expected_model_version!r}"
        )
    selected_on = data.get("selected_on")
    if selected_on is not None and not isinstance(selected_on, str):
        raise ThresholdsError(f"{FILENAME}: selected_on must be a string")
    return ThresholdSet(
        model_version=version,
        clean=_threshold(data.get("clean"), f"{FILENAME}: clean"),
        phone=_threshold(data.get("phone"), f"{FILENAME}: phone"),
        selected_on=selected_on,
        low_far=_threshold_map(data.get("low_far"), f"{FILENAME}: low_far"),
        per_codec=_threshold_map(data.get("per_codec"), f"{FILENAME}: per_codec"),
    )


def _local_file(model_repo: str) -> Path | None:
    """thresholds.json for a local checkpoint file or directory; None if the repo is not local."""
    local = Path(model_repo).expanduser()
    if local.is_file():
        return local.with_name(FILENAME)
    if local.is_dir():
        return local / FILENAME
    return None


def _hub_file(model_repo: str) -> Path | None:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    try:
        return Path(hf_hub_download(repo_id=model_repo, filename=FILENAME))
    except EntryNotFoundError:  # the repo exists but has no thresholds.json (yet)
        return None


def load_thresholds(model_repo: str, model_version: str) -> ThresholdSet | None:
    """The checkpoint's thresholds, or None when it has no thresholds.json.

    A file that exists but is unusable raises ThresholdsError: the API must not run on
    thresholds it cannot trust. Hub errors other than "file not found" propagate.
    """
    path = _local_file(model_repo)
    if path is None:
        path = _hub_file(model_repo)
    if path is None or not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ThresholdsError(f"cannot read {path}: {exc}") from None
    return parse_thresholds(data, model_version)
