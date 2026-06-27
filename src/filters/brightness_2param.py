"""Brightness filter (2 params): I' = a * I + b, same gain/offset for all channels.

Covers: global exposure / illumination intensity changes. The simplest photometric
correction; a subset of the per-channel affine (with a_R=a_G=a_B and b_R=b_G=b_B).
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .base import Filter, clamp_param

GAIN_RANGE = (0.5, 2.0)
OFFSET_RANGE = (-0.5, 0.5)


class Brightness(Filter):
    """Learnable global brightness: ``I' = a * I + b`` (2 parameters)."""

    def __init__(self, init_identity: bool = True) -> None:
        super().__init__()
        gain_init = 1.0 if init_identity else 0.5 * (GAIN_RANGE[0] + GAIN_RANGE[1])
        offset_init = 0.0 if init_identity else 0.5 * (OFFSET_RANGE[0] + OFFSET_RANGE[1])
        self.gain = nn.Parameter(torch.tensor(float(gain_init)))
        self.offset = nn.Parameter(torch.tensor(float(offset_init)))

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        a = clamp_param(self.gain, *GAIN_RANGE)
        b = clamp_param(self.offset, *OFFSET_RANGE)
        return a * x + b

    def get_params(self) -> Dict[str, torch.Tensor]:
        return {
            "gain": clamp_param(self.gain, *GAIN_RANGE).detach(),
            "offset": clamp_param(self.offset, *OFFSET_RANGE).detach(),
        }
