"""Affine per-channel filter: I' = diag([a_R, a_G, a_B]) @ I + [b_R, b_G, b_B]."""

import torch
import torch.nn as nn


class Affine6Param(nn.Module):
    """
    Learnable affine transformation per RGB channel.

    Maps each channel independently: I'_c = a_c * I_c + b_c
    Covers: sensor gain, white balance, exposure offset.
    6 parameters total (3 gains + 3 offsets).
    """

    def __init__(self, init_identity: bool = True):
        super().__init__()
        # TODO: Implement initialization
        # - gains: 3 params, range [0.1, 2.0]
        # - offsets: 3 params, range [-1.0, 1.0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply affine transformation to input image."""
        # TODO: Implement forward pass
        # x: (B, 3, H, W) or (B, H, W, 3) in [0, 1]
        # return: transformed image, clamped to [0, 1]
        pass

    def get_params(self) -> dict:
        """Return current parameters."""
        # TODO: Return gains and offsets as dict
        pass
