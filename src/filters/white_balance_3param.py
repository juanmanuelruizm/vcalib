"""White-balance filter (3 params): I'_c = a_c * I_c, per-channel gains, no offset.

Covers: color-temperature / white-balance shifts (e.g. tungsten vs daylight). A
subset of the per-channel affine with zero offsets. Cheaper than affine when only
the light source spectrum changes (no black-level offset).
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .base import Filter, clamp_param

GAIN_RANGE = (0.5, 2.0)


class WhiteBalance(Filter):
    """Learnable per-channel white-balance gains: ``I'_c = a_c * I_c`` (3 params)."""

    def __init__(self, init_identity: bool = True) -> None:
        super().__init__()
        init = 1.0 if init_identity else 0.5 * (GAIN_RANGE[0] + GAIN_RANGE[1])
        self.gains = nn.Parameter(torch.full((3,), float(init)))

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        g = clamp_param(self.gains, *GAIN_RANGE).view(1, 3, 1, 1)
        return g * x

    def get_params(self) -> Dict[str, torch.Tensor]:
        return {"gains": clamp_param(self.gains, *GAIN_RANGE).detach()}
