"""Saturation filter (1 param): I' = L + s * (I - L), L = luma.

Covers: color vividness / desaturation under flat or colored illumination. ``s=1``
is identity, ``s=0`` is grayscale (luma), ``s>1`` boosts saturation. Luma uses the
ITU-R BT.601 weights ``L = 0.299 R + 0.587 G + 0.114 B``.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .base import Filter, clamp_param

SAT_RANGE = (0.0, 2.0)
_LUMA = (0.299, 0.587, 0.114)


class Saturation(Filter):
    """Learnable global saturation toward luma: ``I' = L + s * (I - L)`` (1 param)."""

    def __init__(self, init_identity: bool = True) -> None:
        super().__init__()
        init = 1.0 if init_identity else 0.5 * (SAT_RANGE[0] + SAT_RANGE[1])
        self.sat = nn.Parameter(torch.tensor(float(init)))

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        s = clamp_param(self.sat, *SAT_RANGE)
        luma = x[:, 0] * _LUMA[0] + x[:, 1] * _LUMA[1] + x[:, 2] * _LUMA[2]
        luma = luma.unsqueeze(1).expand_as(x)
        return luma + s * (x - luma)

    def get_params(self) -> Dict[str, torch.Tensor]:
        return {"saturation": clamp_param(self.sat, *SAT_RANGE).detach()}
