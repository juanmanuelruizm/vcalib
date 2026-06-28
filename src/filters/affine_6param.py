"""Affine per-channel filter (6 params): I'_c = a_c * I_c + b_c.

The flagship linear photometric correction. Covers sensor gain, white balance,
exposure, and per-channel black-level offset. Identity init = no-op.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .base import Filter, clamp_param

GAIN_RANGE = (0.1, 2.0)
OFFSET_RANGE = (-1.0, 1.0)


class Affine6Param(Filter):
    """Learnable per-channel affine: ``I'_c = a_c * I_c + b_c`` (6 params)."""

    def __init__(self, init_identity: bool = True) -> None:
        super().__init__()
        g_init = 1.0 if init_identity else 0.5 * (GAIN_RANGE[0] + GAIN_RANGE[1])
        b_init = 0.0 if init_identity else 0.5 * (OFFSET_RANGE[0] + OFFSET_RANGE[1])
        self.gains = nn.Parameter(torch.full((3,), float(g_init)))
        self.offsets = nn.Parameter(torch.full((3,), float(b_init)))

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        g = clamp_param(self.gains, *GAIN_RANGE).view(1, 3, 1, 1)
        b = clamp_param(self.offsets, *OFFSET_RANGE).view(1, 3, 1, 1)
        return g * x + b

    def get_params(self) -> Dict[str, torch.Tensor]:
        return {
            "gains": clamp_param(self.gains, *GAIN_RANGE).detach(),
            "offsets": clamp_param(self.offsets, *OFFSET_RANGE).detach(),
        }
