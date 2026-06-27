"""Full 3x3 color-correction matrix + offset filter (12 params): I' = M @ I + b.

The flagship linear filter with cross-channel coupling. Covers: ISP color-correction
matrix (CCM), white balance, cross-channel crosstalk, metamerism. ``M`` init = identity
and ``b`` init = 0 (identity = no-op). Still linear, so it cannot model gamma/clipping
residuals — pair with :class:`Gamma` via :class:`CompositeFilter` for that.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .base import Filter, clamp_param

MATRIX_RANGE = (-2.0, 2.0)
OFFSET_RANGE = (-1.0, 1.0)


class Matrix12Param(Filter):
    """Learnable 3x3 CCM + offset: ``I' = M @ I + b`` (12 params)."""

    def __init__(self, init_identity: bool = True) -> None:
        super().__init__()
        if init_identity:
            m_init = torch.eye(3)
            b_init = torch.zeros(3)
        else:
            m_init = torch.zeros(3, 3)
            b_init = torch.full((3,), 0.5 * (OFFSET_RANGE[0] + OFFSET_RANGE[1]))
        self.matrix = nn.Parameter(m_init)
        self.offset = nn.Parameter(b_init)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        m = clamp_param(self.matrix, *MATRIX_RANGE)
        b = clamp_param(self.offset, *OFFSET_RANGE).view(1, 3, 1, 1)
        # x: (B, 3, H, W) -> apply 3x3 over the channel dim.
        return torch.einsum("cd,bdhw->bchw", m, x) + b

    def get_params(self) -> Dict[str, torch.Tensor]:
        return {
            "matrix": clamp_param(self.matrix, *MATRIX_RANGE).detach(),
            "offset": clamp_param(self.offset, *OFFSET_RANGE).detach(),
        }
