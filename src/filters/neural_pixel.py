from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .base import Filter


class NeuralPixelFilter(Filter):
    """Pixel-wise residual MLP: f(x) = clamp(x + MLP(x), 0, 1).

    Identity init: last linear layer zero-initialized so MLP outputs zeros at construction.
    Operates on each pixel independently — no spatial context.

    Args:
        hidden_dim: Width of each hidden layer.
        depth: Number of hidden layers (>= 1).
    """

    def __init__(self, hidden_dim: int = 32, depth: int = 2) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        self.hidden_dim = hidden_dim
        self.depth = depth

        layers: list[nn.Module] = []
        in_dim = 3
        for _ in range(depth):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True)])
            in_dim = hidden_dim

        out_layer = nn.Linear(in_dim, 3)
        nn.init.zeros_(out_layer.weight)
        nn.init.zeros_(out_layer.bias)
        layers.append(out_layer)

        self.mlp = nn.Sequential(*layers)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        B, _C, H, W = x.shape
        flat = x.permute(0, 2, 3, 1).reshape(-1, 3)  # (B*H*W, 3)
        mlp_out: torch.Tensor = self.mlp(flat)  # type: ignore[assignment]
        residual = mlp_out.reshape(B, H, W, 3).permute(0, 3, 1, 2)
        return x + residual  # base class clamps to [0, 1]

    def get_params(self) -> Dict[str, torch.Tensor]:
        return {name: param.detach() for name, param in self.named_parameters()}

    def reg_loss(self) -> torch.Tensor:
        params = list(self.parameters())
        if not params:
            return torch.tensor(0.0)
        squared_means = torch.stack([p.pow(2).mean() for p in params])
        return squared_means.mean()  # type: ignore
