"""Contrast filter (1 param): I' = mu + c * (I - mu), mu = per-channel spatial mean.

Covers: contrast / haze changes under different illumination (e.g. foggy low-
contrast lighting). ``c=1`` is identity, ``c<1`` reduces contrast, ``c>1`` boosts it.
The mean ``mu`` is computed from the input image per-channel over spatial dims, so
the correction is adaptive to the image content.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .base import Filter, clamp_param

CONTRAST_RANGE = (0.0, 2.0)


class Contrast(Filter):
    """Learnable contrast around the per-channel image mean (1 param)."""

    def __init__(self, init_identity: bool = True) -> None:
        super().__init__()
        init = 1.0 if init_identity else 0.5 * (CONTRAST_RANGE[0] + CONTRAST_RANGE[1])
        self.contrast = nn.Parameter(torch.tensor(float(init)))

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        c = clamp_param(self.contrast, *CONTRAST_RANGE)
        mu = x.mean(dim=(2, 3), keepdim=True)
        return mu + c * (x - mu)

    def get_params(self) -> Dict[str, torch.Tensor]:
        return {"contrast": clamp_param(self.contrast, *CONTRAST_RANGE).detach()}
