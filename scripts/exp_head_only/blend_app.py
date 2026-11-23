"""A scratch API serving a gate candidate C (0.75 x p3 + 0.25 x a top-layer run), for the
test-only checks (README "Test-only checks").

The API's own ``RealDetector`` loads p3 as usual (``VOXGUARD_MODEL_REPO``); its model is then
wrapped so one XLS-R pass gives both p3's logits (from the last hidden state, p3's own head) and
the run's (``hidden_states[22]`` through the run's ``TopHead``), and the blended P(fake) comes out
as log-probabilities, so the detector's softmax returns it unchanged. Nothing else in the API
changes. Usage (after ``real_audio_sanity.py serve``'s environment)::

    BLEND_RUN=fix-2-top-init-l2-gentle BLEND_THRESHOLD=0.01649 \
        python -m uvicorn blend_app:create --factory --port 8101
"""

from __future__ import annotations

import os

import torch
from head_lib import TopHead, X
from torch import nn

P3_WEIGHT = 0.75  # candidate C (README "Round 2b")


class Blend(nn.Module):
    def __init__(self, p3: nn.Module, top: TopHead, w: float = P3_WEIGHT) -> None:
        super().__init__()
        self.p3, self.top, self.w = p3, top, w
        self.k = len(p3.frontend.model.encoder.layers) - len(top.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.p3
        x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-7)
        out = m.frontend.model(x, output_hidden_states=True)
        z = nn.functional.max_pool2d(m.proj(out.last_hidden_state).transpose(1, 2).unsqueeze(1), 3)
        p_p3 = torch.softmax(m.backend(m.encoder(nn.functional.selu(m.first_bn(z)))), -1)[:, 1]
        p_top = torch.softmax(self.top(out.hidden_states[self.k]), -1)[:, 1]
        p = (self.w * p_p3.float() + (1 - self.w) * p_top.float()).clamp(1e-7, 1 - 1e-7)
        return torch.log(torch.stack([1 - p, p], dim=-1))


def wrap(det, run: str) -> None:
    """Replace ``det.model`` (p3) by the blend with ``run``'s top-layer head, in place."""
    top = TopHead(det.model)
    top.load_state_dict(torch.load(X / "runs" / run / "best_head.pt", map_location="cpu"))
    dev = next(det.model.parameters()).device
    det.model = Blend(det.model, top.to(dev).eval()).eval()
    det.model_version = f"{det.model_version}+C:{run}"


def create():
    from voxguard.api.detector_loader import load_detector
    from voxguard.api.main import create_app

    det = load_detector(os.environ.get("VOXGUARD_DEVICE", "cpu"))
    wrap(det, os.environ["BLEND_RUN"])
    det.threshold = float(os.environ["BLEND_THRESHOLD"])
    return create_app(detector=det)
