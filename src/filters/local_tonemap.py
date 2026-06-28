"""CLAHE-like local tone mapping (F6) — guided-filter approximation.

Differentiable, fast approximation of CLAHE: local-mean (guided-filter-style)
normalization with a learnable K×K gain field. ``I' = μ_local + g(zone) · (I − μ_local)``
where ``μ_local`` is a box-filter local mean and ``g`` is bilinearly interpolated from
the K×K grid. Identity init: g=1 → no-op. True differentiable CLAHE (soft sliding-
window histograms) is future stretch, not this filter.

Params: ``K²`` (gain field) — K=4 → 16. ``grid_size`` configurable.
Corrects: local contrast / spatially-varying exposure (the CLAHE intent) without soft
histograms. Large-K spatial curve + local-mean normalization.
reg_loss: TV on the gain field + identity-anchoring (gain→1).
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from .base import clamp_param
from .spatial import SpatialFilter

_GAIN_RANGE = (0.0, 3.0)


class LocalTonemap(SpatialFilter):
    """CLAHE-like local tone mapping via guided-filter local mean + K×K gain field."""

    def __init__(self, grid_size: int = 4, radius: int = 0, init_identity: bool = True) -> None:
        super().__init__(grid_size, n_field_channels=1, fill_value=1.0)
        self.radius = radius  # 0 = auto (min(H,W)//16)

    def _local_mean(self, x: torch.Tensor) -> torch.Tensor:
        r = self.radius
        if r <= 0:
            r = max(1, min(x.shape[2], x.shape[3]) // 16)
        return F.avg_pool2d(x, kernel_size=2 * r + 1, stride=1, padding=r, count_include_pad=False)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        mu = self._local_mean(x)  # (B, 3, H, W)
        gain = clamp_param(self.control_grid, *_GAIN_RANGE)  # (1, K, K)
        gain_field = self._sample(gain, H, W, x.device, x.dtype)  # (1, 1, H, W)
        return mu + gain_field.expand(B, 3, H, W) * (x - mu)

    def get_params(self) -> Dict[str, torch.Tensor]:
        return {"gain": clamp_param(self.control_grid, *_GAIN_RANGE).detach()}

    def reg_loss(self) -> torch.Tensor:
        g = self.control_grid  # (1, K, K)
        tv = (g[..., 1:, :] - g[..., :-1, :]).pow(2).mean()
        tv = tv + (g[..., :, 1:] - g[..., :, :-1]).pow(2).mean()
        id_dist = (g - 1.0).pow(2).mean()  # gain→1 = identity
        return tv + id_dist
