"""Large-K spatial tone curve (F5) — zone-dependent non-linear tone.

Reuses ``SpatialFilter``'s bilinear K×K control grid + the monotone-curve
parameterization from ``ToneCurve``, per channel, per zone. Different zones get
different non-linear tone curves, smoothly interpolated (no seams).

Params = ``3·P·K²`` (P=8,K=3 → 216; P=16,K=4 → 768). ``P`` and ``grid_size`` configurable.
Corrects: zone-dependent non-linear tone (uneven lighting + per-zone exposure response).
Overfit risk: HIGH at large K·P — gated by held-out val + reg_loss.
reg_loss: spatial TV on the curve-control grid + curve curvature + identity-distance.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F

from .spatial import SpatialFilter


class SpatialToneCurve(SpatialFilter):
    """Spatially-varying per-channel monotone tone curves via a K×K control grid."""

    def __init__(self, P: int = 8, grid_size: int = 3, init_identity: bool = True) -> None:
        super().__init__(grid_size, n_field_channels=3 * (P - 1), fill_value=0.0)
        if P < 2:
            raise ValueError(f"P must be >= 2, got {P}")
        self.P = P
        if init_identity:
            target = 1.0 / (P - 1)
            delta_val = math.log(math.exp(target) - 1.0)
            with torch.no_grad():
                self.control_grid.fill_(delta_val)
        lin = torch.linspace(0.0, 1.0, P)
        self.register_buffer("_identity_cp", lin.view(1, P, 1, 1).expand(3, P, 1, 1).contiguous())

    def _spatial_curves(
        self, H: int, W: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Return (3, P, H, W) monotone curves per pixel (spatially interpolated)."""
        K = self.grid_size
        P = self.P
        deltas = F.softplus(self.control_grid).view(3, P - 1, K, K)
        curves = torch.cat(
            [torch.zeros(3, 1, K, K, device=device, dtype=dtype), torch.cumsum(deltas, dim=1)],
            dim=1,
        )
        curves = curves / curves[:, -1:].clamp(min=1e-8)  # (3, P, K, K)
        # Spatial upsample each of the 3*P control-point maps
        field = self._sample(curves.view(3 * P, K, K), H, W, device, dtype)  # (1, 3*P, H, W)
        return field.view(3, P, H, W)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        cp = self._spatial_curves(H, W, x.device, x.dtype)  # (3, P, H, W)
        P = self.P
        # 1D interp: gather control points at floor/ceil indices, blend
        idx = x * (P - 1)  # (B, 3, H, W) in [0, P-1]
        i0 = idx.floor().long().clamp(0, P - 2)
        i1 = i0 + 1
        t = idx - i0.float()  # (B, 3, H, W)
        cp_exp = cp.unsqueeze(0).expand(B, 3, P, H, W)  # (B, 3, P, H, W)
        cp0 = cp_exp.gather(2, i0.unsqueeze(2).expand(B, 3, 1, H, W)).squeeze(2)
        cp1 = cp_exp.gather(2, i1.unsqueeze(2).expand(B, 3, 1, H, W)).squeeze(2)
        return (1 - t) * cp0 + t * cp1

    def get_params(self) -> Dict[str, torch.Tensor]:
        return {"control_grid": self.control_grid.detach()}

    def reg_loss(self) -> torch.Tensor:
        K = self.grid_size
        P = self.P
        g = self.control_grid.view(3, P - 1, K, K)
        # Spatial TV on the delta grid
        tv = (g[..., 1:, :] - g[..., :-1, :]).pow(2).mean()
        tv = tv + (g[..., :, 1:] - g[..., :, :-1]).pow(2).mean()
        # Curvature of the mean curve
        d = F.softplus(self.control_grid).view(3, P - 1, K, K)
        curves = torch.cumsum(d, dim=1)  # (3, P-1, K, K)
        curvature = (curves[:, 2:] - 2 * curves[:, 1:-1] + curves[:, :-2]).pow(2).mean()
        # Identity distance
        id_dist = (g - 0.0).pow(2).mean()  # deltas → 0 means equal deltas → linear ramp
        return tv + curvature + 0.1 * id_dist
