"""Spatially-varying parametric filters (zone-dependent illumination correction).

The global filters in this package apply one correction to the whole frame. Real
illumination is rarely uniform — vignetting, directional/uneven lighting, partial
shadow, and mixed light sources all vary across the field. These spatial filters
attach a **bilinear control-point grid**: a K×K mesh of control points whose params
are bilinearly interpolated (via ``torch.nn.functional.grid_sample``) to a smooth
per-pixel param field. Different zones of the image therefore receive different
corrections, with C0 continuity (no seams).

Param count scales as ``K² · (params per pixel)``:
  K=2 → 4 control points; K=3 → 9; K=4 → 16; K=8 → 64.
Larger K = finer spatial control but more params to calibrate (higher edge cost).

Scope note: this relaxes the original "6–12 params" budget. Inference stays <1ms
(``grid_sample`` is cheap); calibration is offline and still <100 steps, but with
more params the optimizer may need more steps to converge. Use the smallest K that
recovers the distance — start at K=2 and escalate only if a global filter plateaus.

Filters (n_field = channels in the control grid):
- ``SpatialBrightness``  (n_field=2: gain, offset; broadcast to RGB)  — K=2 → 8 params
- ``SpatialWhiteBalance`` (n_field=3: per-channel gains)              — K=2 → 12 params
- ``SpatialAffine``      (n_field=6: 3 gains + 3 offsets)             — K=2 → 24 params
- ``SpatialGamma``       (n_field=3: per-channel gamma)               — K=2 → 12 params
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Filter, clamp_param
from .gamma_3param import EPS, GAMMA_RANGE

# Reuse the global filters' physical ranges for consistency.
GAIN_RANGE = (0.1, 2.0)
BRIGHT_GAIN_RANGE = (0.5, 2.0)
BRIGHT_OFFSET_RANGE = (-0.5, 0.5)
WB_GAIN_RANGE = (0.5, 2.0)
OFFSET_RANGE = (-1.0, 1.0)


class SpatialFilter(Filter):
    """Base: holds a (n_field, K, K) control grid + bilinear upsample to per-pixel field."""

    def __init__(self, grid_size: int, n_field_channels: int, fill_value: float = 1.0) -> None:
        super().__init__()
        if grid_size < 2:
            raise ValueError(f"grid_size must be >= 2, got {grid_size}")
        self.grid_size = grid_size
        self.n_field_channels = n_field_channels
        init = torch.full((n_field_channels, grid_size, grid_size), float(fill_value))
        self.control_grid = nn.Parameter(init)

    def _coord_grid(self, H: int, W: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        ys = torch.linspace(-1, 1, H, device=device, dtype=dtype)
        xs = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([gx, gy], dim=-1).unsqueeze(0)  # (1, H, W, 2), order (x, y)

    def _sample(
        self, grid: torch.Tensor, H: int, W: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Bilinearly upsample a (n_field, K, K) grid to (1, n_field, H, W)."""
        g = grid.unsqueeze(0).to(device=device, dtype=dtype)  # (1, n_field, K, K)
        coords = self._coord_grid(H, W, device, dtype)  # (1, H, W, 2)
        field = F.grid_sample(g, coords, mode="bilinear", padding_mode="border", align_corners=True)
        return field  # (1, n_field, H, W)

    def get_params(self) -> Dict[str, torch.Tensor]:
        return {"control_grid": self.control_grid.detach()}


class SpatialBrightness(SpatialFilter):
    """Spatial brightness: per-pixel gain + offset broadcast to RGB. ``I' = g·I + b``."""

    def __init__(self, grid_size: int = 2, init_identity: bool = True) -> None:
        super().__init__(grid_size, n_field_channels=2, fill_value=1.0)
        if init_identity:
            with torch.no_grad():
                self.control_grid[1] = 0.0  # offset channel = 0

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        gains = clamp_param(self.control_grid[0:1], *BRIGHT_GAIN_RANGE)
        offsets = clamp_param(self.control_grid[1:2], *BRIGHT_OFFSET_RANGE)
        grid = torch.cat([gains, offsets], dim=0)
        field = self._sample(grid, H, W, x.device, x.dtype)  # (1, 2, H, W)
        g = field[:, 0:1].expand(B, 3, H, W)
        b = field[:, 1:2].expand(B, 3, H, W)
        return g * x + b

    def get_params(self) -> Dict[str, torch.Tensor]:
        return {
            "gains": clamp_param(self.control_grid[0:1], *BRIGHT_GAIN_RANGE).detach(),
            "offsets": clamp_param(self.control_grid[1:2], *BRIGHT_OFFSET_RANGE).detach(),
        }


class SpatialWhiteBalance(SpatialFilter):
    """Spatial white balance: per-pixel per-channel gains. ``I'_c = g_c·I_c``."""

    def __init__(self, grid_size: int = 2, init_identity: bool = True) -> None:
        super().__init__(grid_size, n_field_channels=3, fill_value=1.0)
        # identity: gains = 1 (already the fill value)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        grid = clamp_param(self.control_grid, *WB_GAIN_RANGE)
        field = self._sample(grid, H, W, x.device, x.dtype)  # (1, 3, H, W)
        return field.expand(B, 3, H, W) * x

    def get_params(self) -> Dict[str, torch.Tensor]:
        return {"gains": clamp_param(self.control_grid, *WB_GAIN_RANGE).detach()}


class SpatialAffine(SpatialFilter):
    """Spatial affine: per-pixel per-channel gains + offsets. ``I'_c = g_c·I_c + b_c``."""

    def __init__(self, grid_size: int = 2, init_identity: bool = True) -> None:
        super().__init__(grid_size, n_field_channels=6, fill_value=1.0)
        if init_identity:
            with torch.no_grad():
                self.control_grid[3:6] = 0.0  # offsets = 0

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        gains = clamp_param(self.control_grid[0:3], *GAIN_RANGE)
        offsets = clamp_param(self.control_grid[3:6], *OFFSET_RANGE)
        grid = torch.cat([gains, offsets], dim=0)
        field = self._sample(grid, H, W, x.device, x.dtype)  # (1, 6, H, W)
        g = field[:, 0:3].expand(B, 3, H, W)
        b = field[:, 3:6].expand(B, 3, H, W)
        return g * x + b

    def get_params(self) -> Dict[str, torch.Tensor]:
        return {
            "gains": clamp_param(self.control_grid[0:3], *GAIN_RANGE).detach(),
            "offsets": clamp_param(self.control_grid[3:6], *OFFSET_RANGE).detach(),
        }


class SpatialGamma(SpatialFilter):
    """Spatial gamma: per-pixel per-channel tone curve. ``I'_c = I_c^{γ_c}``."""

    def __init__(self, grid_size: int = 2, init_identity: bool = True) -> None:
        super().__init__(grid_size, n_field_channels=3, fill_value=1.0)
        # identity: gamma = 1 (already the fill value)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        grid = clamp_param(self.control_grid, *GAMMA_RANGE)
        field = self._sample(grid, H, W, x.device, x.dtype)  # (1, 3, H, W)
        base = torch.clamp(x, min=EPS)
        return torch.pow(base, field.expand(B, 3, H, W))

    def get_params(self) -> Dict[str, torch.Tensor]:
        return {"gamma": clamp_param(self.control_grid, *GAMMA_RANGE).detach()}
