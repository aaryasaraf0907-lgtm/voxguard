"""Model definitions (RawNet2, AASIST, SSL-AASIST), a registry and a checkpoint loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_model(name: str, **params: Any):
    """Build a model by config name. Imports are lazy so torch is only needed when used."""
    if name == "rawnet2":
        from voxguard.models.rawnet2 import RawNet2

        return RawNet2(**params)
    if name == "ssl_aasist":
        from voxguard.models.ssl_aasist import SSLAASIST

        return SSLAASIST(**params)
    raise ValueError(f"unknown model {name!r}; known: ['rawnet2', 'ssl_aasist']")


def load_from_checkpoint(path: Path, device: str = "cpu"):
    """Rebuild a model from a ``train.py`` checkpoint; returns ``(model in eval mode, info)``.

    For SSL-AASIST the pretrained XLS-R weights are not downloaded again: only its small
    ``config.json`` is fetched to build the architecture, then the checkpoint supplies every
    weight.
    """
    import torch

    ckpt = torch.load(path, map_location=device, weights_only=False)
    params = dict(ckpt["config"]["model"])
    name = params.pop("name")
    build_params = dict(params)
    if name == "ssl_aasist" and build_params.get("pretrained"):
        from transformers import Wav2Vec2Config

        overrides = Wav2Vec2Config.from_pretrained(build_params["pretrained"]).to_dict()
        build_params.update(pretrained=None, ssl_config_overrides=overrides)
    model = build_model(name, **build_params)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), {"name": name, "epoch": ckpt["epoch"], **params}
