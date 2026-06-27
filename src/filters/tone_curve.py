"""Per-channel multi-control-point monotone tone curves (F2).

Each channel has a learnable tone curve defined by P control points. Monotonicity is
guaranteed by construction: the curve is the cumulative sum of ``softplus(deltas)``,
normalized to [0, 1]. Identity init = linear ramp (equal deltas). Linear interpolation
between control points via ``F.grid_sample`` (1D).

Params = ``3·P`` (P=16 → 48; P=32 → 96). ``P`` configurable.
Corrects: per-channel non-linear tone (brightness/exposure non-linearity, per-channel
color response). No cross-channel coupling.
reg_loss: 2nd-difference (curvature) + identity-distance.
"""

from __future__ import annotations

import math
from typing import Dict, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Filter


class ToneCurve(Filter):
    """Learnable per-channel monotone tone curve with P control points."""

    def __init__(self, P: int = 16, init_identity: bool = True) -> None:
        super().__init__()
        if P < 2:
            raise ValueError(f"P must be >= 2, got {P}")
        self.P = P
        if init_identity:
            target = 1.0 / (P - 1)
            delta_init = math.log(math.exp(target) - 1.0)  # inverse softplus → equal deltas
            deltas = torch.full((3, P - 1), float(delta_init))
        else:
            deltas = torch.zeros(3, P - 1)
        self.deltas = nn.Parameter(deltas)
        lin = torch.linspace(0.0, 1.0, P)
        self.register_buffer("_identity_cp", lin.view(1, P).expand(3, P).contiguous())

    def _curves(self) -> torch.Tensor:
        """Return (3, P) monotone curves in [0, 1] on the current device/dtype."""
        d = F.softplus(self.deltas)  # (3, P-1), all > 0
        curves = torch.cat(
            [torch.zeros(3, 1, device=d.device, dtype=d.dtype), torch.cumsum(d, dim=1)], dim=1
        )
        return curves / curves[:, -1:].clamp(min=1e-8)  # normalize to [0, 1]

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        curves = self._curves()  # (3, P)
        P = self.P
        # 1D interp via 2D grid_sample: curves as (B*3, 1, P, 1), grid (B*3, H, W, 2)
        curves_img = curves.view(3, 1, P, 1).expand(B, 3, 1, P, 1).reshape(B * 3, 1, P, 1)
        x_vals = x.reshape(B * 3, 1, H, W)
        # Grid: y-coord = 2*val-1 (sample along P axis = height), x-coord = 0 (W=1)
        grid = torch.zeros(B * 3, H, W, 2, device=x.device, dtype=x.dtype)
        grid[..., 1] = 2.0 * x_vals.squeeze(1) - 1.0
        out = F.grid_sample(
            curves_img, grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        return out.reshape(B, 3, H, W)

    def get_params(self) -> Dict[str, torch.Tensor]:
        return {"curves": self._curves().detach(), "deltas": self.deltas.detach()}

    def reg_loss(self) -> torch.Tensor:
        cp = self._curves()  # (3, P)
        curvature = (cp[:, 2:] - 2 * cp[:, 1:-1] + cp[:, :-2]).pow(2).mean()
        id_dist = (cp - cast(torch.Tensor, self._identity_cp)).pow(2).mean()
        return curvature + id_dist
