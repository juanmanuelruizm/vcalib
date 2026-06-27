"""Full differentiable 3D LUT filter — the headline high-capacity filter.

Maps RGB → RGB via a ``(3, N, N, N)`` vertex grid with trilinear interpolation
(``F.grid_sample`` in 3D). The most expressive single filter: models channel coupling
(brightness + color-temperature) and non-linearity simultaneously. Identity init = the
identity LUT (vertices on the ``r=g=b`` diagonal map to themselves).

Params = ``3·N³`` (N=9 → 2,187; N=17 → 14,739; N=33 → 107,811). N configurable via
``build_filter({"type": "lut_3d", "size": N})``.

Overfit risk: HIGH — can memorize any per-pixel-ish mapping on one calibration pair.
Mitigated by ``reg_loss`` (TV smoothness + identity-distance) + the held-out val
overfit gate. No ``clamp_param`` on the vertices — the base final clamp handles [0,1],
keeping gradients alive everywhere.
"""

from __future__ import annotations

from typing import Dict, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Filter

LUT_RANGE = (0.0, 1.0)


def _identity_lut(size: int) -> torch.Tensor:
    """Build the identity 3D LUT: out_c = in_c. Shape (3, N, N, N) = (C, D, H, W).

    grid_sample coords are (x, y, z) = (W, H, D), so:
      R output (ch 0) varies along W (dim 2),
      G output (ch 1) varies along H (dim 1),
      B output (ch 2) varies along D (dim 0).
    """
    lin = torch.linspace(0.0, 1.0, size)
    r = lin.view(1, 1, size).expand(size, size, size)  # varies along W
    g = lin.view(1, size, 1).expand(size, size, size)  # varies along H
    b = lin.view(size, 1, 1).expand(size, size, size)  # varies along D
    return torch.stack([r, g, b], dim=0).contiguous()


class LUT3D(Filter):
    """Differentiable 3D LUT with trilinear interpolation. ``size`` = N (grid vertices per axis)."""

    def __init__(self, size: int = 9, init_identity: bool = True) -> None:
        super().__init__()
        if size < 2:
            raise ValueError(f"size must be >= 2, got {size}")
        self.size = size
        self.register_buffer("_identity", _identity_lut(size))
        if init_identity:
            init = _identity_lut(size).clone()
        else:
            init = torch.zeros(3, size, size, size)
        self.lut = nn.Parameter(init)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        # Grid: (B, 3, H, W) → normalize to [-1, 1] → (1, B*H*W, 1, 1, 3)
        grid = (2.0 * x - 1.0).permute(0, 2, 3, 1).reshape(1, B * H * W, 1, 1, 3)
        lut = self.lut.unsqueeze(0)  # (1, 3, N, N, N)
        out = F.grid_sample(lut, grid, mode="bilinear", padding_mode="border", align_corners=True)
        return out.reshape(B, 3, H, W)

    def get_params(self) -> Dict[str, torch.Tensor]:
        return {"lut": self.lut.detach().clamp(*LUT_RANGE)}

    def reg_loss(self) -> torch.Tensor:
        """Total-variation smoothness + identity-anchoring on the LUT grid."""
        lut = self.lut
        tv = (lut[..., 1:, :, :] - lut[..., :-1, :, :]).pow(2).mean()
        tv = tv + (lut[..., :, 1:, :] - lut[..., :, :-1, :]).pow(2).mean()
        tv = tv + (lut[..., :, :, 1:] - lut[..., :, :, :-1]).pow(2).mean()
        id_dist = (lut - cast(torch.Tensor, self._identity)).pow(2).mean()
        return tv + id_dist
