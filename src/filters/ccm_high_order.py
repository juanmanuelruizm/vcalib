"""High-order / polynomial CCM (F3) — cross-channel coupling + non-linearity.

Extends the linear 3×3 CCM with polynomial features of RGB (degree ≤ 3) for
non-linear color correction. The degree-2 root-polynomial form (Finlayson) is
exposure-invariant; here we use standard polynomial features with the linear terms
initialized to identity and higher-order terms to zero.

Params: degree-2 → 3×9+3 = 30; degree-3 → 3×19+3 = 60. ``degree`` configurable.
Corrects: cross-channel coupling + mild non-linearity (structured cheaper alternative
to a full 3D LUT for color-temp + non-linearity).
reg_loss: L2 toward the identity block (identity-anchoring).
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .base import Filter, clamp_param

MATRIX_RANGE = (-2.0, 2.0)
OFFSET_RANGE = (-1.0, 1.0)


def _poly_features(x: torch.Tensor, degree: int) -> torch.Tensor:
    """Build polynomial features from (B, 3, H, W) → (B, K, H, W)."""
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    feats = [r, g, b]
    if degree >= 2:
        feats += [r * r, g * g, b * b, r * g, r * b, g * b]
    if degree >= 3:
        feats += [
            r * r * r,
            g * g * g,
            b * b * b,
            r * r * g,
            r * r * b,
            r * g * g,
            g * g * b,
            r * b * b,
            g * b * b,
            r * g * b,
        ]
    return torch.cat(feats, dim=1)


def _n_features(degree: int) -> int:
    if degree <= 1:
        return 3
    if degree == 2:
        return 9
    return 19


class HighOrderCCM(Filter):
    """Learnable polynomial CCM: ``I' = M @ poly(I) + b``."""

    def __init__(self, degree: int = 2, init_identity: bool = True) -> None:
        super().__init__()
        if degree < 1 or degree > 3:
            raise ValueError(f"degree must be 1..3, got {degree}")
        self.degree = degree
        K = _n_features(degree)
        if init_identity:
            m = torch.zeros(3, K)
            m[:, :3] = torch.eye(3)  # linear terms = identity
            b = torch.zeros(3)
        else:
            m = torch.zeros(3, K)
            b = torch.full((3,), 0.5 * (OFFSET_RANGE[0] + OFFSET_RANGE[1]))
        self.matrix = nn.Parameter(m)
        self.offset = nn.Parameter(b)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        feats = _poly_features(x, self.degree)  # (B, K, H, W)
        m = clamp_param(self.matrix, *MATRIX_RANGE)
        b = clamp_param(self.offset, *OFFSET_RANGE).view(1, 3, 1, 1)
        out = torch.einsum("ck,bkhw->bchw", m, feats) + b
        return out

    def get_params(self) -> Dict[str, torch.Tensor]:
        return {
            "matrix": clamp_param(self.matrix, *MATRIX_RANGE).detach(),
            "offset": clamp_param(self.offset, *OFFSET_RANGE).detach(),
        }

    def reg_loss(self) -> torch.Tensor:
        m = self.matrix
        id_block = torch.zeros_like(m)
        id_block[:, :3] = torch.eye(3, device=m.device, dtype=m.dtype)
        return (m - id_block).pow(2).mean()
