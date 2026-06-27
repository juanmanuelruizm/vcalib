"""LMS / Bradford chromatic-adaptation white balance (F4).

Physically-grounded color-temperature / illuminant correction via cone-response
transform: RGB → LMS (fixed Bradford matrix), learnable adaptation in LMS space
(diagonal gains or full 3×3), LMS → RGB. Identity init = identity adaptation.

Params: 3 (diagonal) or 9 (full). ``mode`` configurable.
Corrects: color-temperature / illuminant shifts. Low overfit risk (physically
constrained). The well-conditioned, interpretable color-temp specialist.
reg_loss: 0 by default (low-capacity); optional L2 toward identity.
"""

from __future__ import annotations

from typing import Dict, cast

import torch
import torch.nn as nn

from .base import Filter, clamp_param

_GAIN_RANGE = (0.5, 2.0)
_MATRIX_RANGE = (-2.0, 2.0)

_BRADFORD = torch.tensor(
    [[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]],
    dtype=torch.float32,
)
_BRADFORD_INV = torch.linalg.inv(_BRADFORD)


class ChromaticAdaptation(Filter):
    """Learnable chromatic adaptation in LMS space via the Bradford transform."""

    def __init__(self, mode: str = "diagonal", init_identity: bool = True) -> None:
        super().__init__()
        if mode not in ("diagonal", "full"):
            raise ValueError(f"mode must be 'diagonal' or 'full', got {mode!r}")
        self.mode = mode
        if mode == "diagonal":
            init = 1.0 if init_identity else 0.5 * (_GAIN_RANGE[0] + _GAIN_RANGE[1])
            self.gains = nn.Parameter(torch.full((3,), float(init)))
        else:
            init_m = torch.eye(3) if init_identity else torch.zeros(3, 3)
            self.matrix = nn.Parameter(init_m)
        self.register_buffer("_bradford", _BRADFORD)
        self.register_buffer("_bradford_inv", _BRADFORD_INV)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        flat = x.reshape(B, 3, -1)  # (B, 3, H*W)
        lms = torch.matmul(cast(torch.Tensor, self._bradford), flat)
        if self.mode == "diagonal":
            g = clamp_param(self.gains, *_GAIN_RANGE).view(1, 3, 1)
            lms = lms * g
        else:
            m = clamp_param(self.matrix, *_MATRIX_RANGE)
            lms = torch.matmul(m, lms)
        out = torch.matmul(cast(torch.Tensor, self._bradford_inv), lms)
        return out.reshape(B, 3, H, W)

    def get_params(self) -> Dict[str, torch.Tensor]:
        if self.mode == "diagonal":
            return {"gains": clamp_param(self.gains, *_GAIN_RANGE).detach()}
        return {"matrix": clamp_param(self.matrix, *_MATRIX_RANGE).detach()}

    def reg_loss(self) -> torch.Tensor:
        if self.mode == "diagonal":
            return (self.gains - 1.0).pow(2).mean()
        return (
            (self.matrix - torch.eye(3, device=self.matrix.device, dtype=self.matrix.dtype))
            .pow(2)
            .mean()
        )
