"""Full 3x3 matrix + offset filter: I' = M @ I + b."""

import torch
import torch.nn as nn


class Matrix12Param(nn.Module):
    """
    Learnable 3x3 color correction matrix (CCM) + offset.

    Maps: I' = M @ I + b, where M is 3x3 and b is 3-dim offset.
    Covers: color mixing, white balance, cross-channel coupling, exposure.
    12 parameters total (9 matrix + 3 offset).
    """

    def __init__(self, init_identity: bool = True):
        super().__init__()
        # TODO: Implement initialization
        # - matrix: 3x3, initialized to identity if init_identity
        # - offset: 3-dim, initialized to zero if init_identity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply matrix transformation to input image."""
        # TODO: Implement forward pass
        # x: (B, 3, H, W) or (B, H, W, 3) in [0, 1]
        # Reshape x to apply matrix multiply
        # return: transformed image, clamped to [0, 1]
        pass

    def get_params(self) -> dict:
        """Return current parameters."""
        # TODO: Return matrix and offset as dict
        pass
