"""Model definitions (RawNet2, AASIST, SSL-AASIST) and a name -> constructor registry."""

from __future__ import annotations

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
