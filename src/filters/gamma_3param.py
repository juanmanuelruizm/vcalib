"""Gamma filter (3 params, per-channel): I'_c = I_c ** gamma_c.

Non-linear tone-response correction. Covers: sensor/log-gamma response differences,
and non-linear residuals that the linear (affine / matrix) filters cannot model.
``gamma_c = 1`` is identity. The input is clamped to >= ``EPS`` before the power to
keep gradients well-defined at zero; the base class then reclamps output to [0, 1].

This is the "curves tier" from the spec, built upfront as part of the full library.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .base import Filter, clamp_param

GAMMA_RANGE = (0.5, 2.5)
EPS = 1e-6


class Gamma(Filter):
    """Learnable per-channel gamma (tone curve): ``I'_c = I_c ** gamma_c`` (3 params)."""

    def __init__(self, init_identity: bool = True) -> None:
        super().__init__()
        init = 1.0 if init_identity else 0.5 * (GAMMA_RANGE[0] + GAMMA_RANGE[1])
        self.gamma = nn.Parameter(torch.full((3,), float(init)))

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        g = clamp_param(self.gamma, *GAMMA_RANGE).view(1, 3, 1, 1)
        base = torch.clamp(x, min=EPS)
        return torch.pow(base, g)

    def get_params(self) -> Dict[str, torch.Tensor]:
        return {"gamma": clamp_param(self.gamma, *GAMMA_RANGE).detach()}
