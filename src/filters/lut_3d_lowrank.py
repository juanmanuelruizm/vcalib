"""Low-rank 3D LUT (F7) — generalization-friendly variant of the full 3D LUT.

LUT = identity + Σ_m w_m · B_m, where B_m are fixed smooth basis LUTs and w_m are
learned weights. Controls expressive capacity via M (the rank). The recommended
deployment form if the full LUT (F1) overfits the calibration pair.

Params: ``M`` (weights only; basis is fixed). M=16 → 16 params. ``M`` and ``size`` configurable.
Corrects: same as F1 (brightness + color-temp + non-linearity) but capacity-controlled.
Overfit risk: MED (controlled by M). reg_loss: L2 on weights (identity-anchoring).
"""

from __future__ import annotations

from typing import Dict, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Filter
from .lut_3d import _identity_lut


def _make_smooth_basis(size: int, seed: int) -> torch.Tensor:
    """Generate a smooth random basis LUT (low-frequency perturbation of identity)."""
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(3, size, size, size, generator=g) * 0.02
    # Smooth with 3D avg pool to keep only low frequencies
    noise_5d = noise.unsqueeze(0)  # (1, 3, N, N, N)
    k = min(3, size)
    smoothed = F.avg_pool3d(noise_5d, kernel_size=k, stride=1, padding=k // 2)
    return smoothed.squeeze(0)


class LUT3DLowRank(Filter):
    """Low-rank 3D LUT: ``LUT = identity + Σ w_m · B_m`` (fixed smooth basis, learned weights)."""

    def __init__(self, M: int = 16, size: int = 17, init_identity: bool = True) -> None:
        super().__init__()
        if size < 2:
            raise ValueError(f"size must be >= 2, got {size}")
        if M < 1:
            raise ValueError(f"M must be >= 1, got {M}")
        self.M = M
        self.size = size
        identity = _identity_lut(size)
        self.register_buffer("_identity", identity)
        basis = torch.stack([_make_smooth_basis(size, seed=1000 + m) for m in range(M)], dim=0)
        self.register_buffer("_basis", basis)  # (M, 3, N, N, N)
        self.weights = nn.Parameter(torch.zeros(M) if init_identity else torch.randn(M) * 0.01)

    def _lut(self) -> torch.Tensor:
        """Return the composed (3, N, N, N) LUT."""
        return cast(torch.Tensor, self._identity) + torch.einsum(
            "m,mcijk->cijk", self.weights, self._basis
        )

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        lut = self._lut().unsqueeze(0)  # (1, 3, N, N, N)
        grid = (2.0 * x - 1.0).permute(0, 2, 3, 1).reshape(1, B * H * W, 1, 1, 3)
        out = F.grid_sample(lut, grid, mode="bilinear", padding_mode="border", align_corners=True)
        return out.reshape(B, 3, H, W)

    def get_params(self) -> Dict[str, torch.Tensor]:
        return {"weights": self.weights.detach()}

    def reg_loss(self) -> torch.Tensor:
        return self.weights.pow(2).mean()
