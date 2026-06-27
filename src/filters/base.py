"""Base class for all parametric calibration filters.

Every filter operates on the **[0, 1] RGB tensor before ImageNet normalization**
(the filter insertion point, see ``src/utils/activations.py::to_unit_rgb``). The
contract:

- Input: ``(B, 3, H, W)`` or ``(3, H, W)`` float tensor in [0, 1] (NCHW).
- Output: same shape, clamped to [0, 1].
- Identity init: a freshly constructed filter is a no-op (``forward(x) == x``).
- Parameters are constrained to a physical range (clamped inside ``transform``);
  ``get_params()`` returns the in-range values.

Subclasses implement :meth:`transform` (operates on a 4-D tensor) and
:meth:`get_params`. The base class handles batch-dimension wrapping and the
final [0, 1] clamp.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


def clamp_param(p: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Clamp a parameter to [lo, hi]; differentiable (grad=1 inside, 0 outside)."""
    return torch.clamp(p, lo, hi)


class Filter(nn.Module):
    """Abstract parametric pixel-space filter operating on [0, 1] NCHW RGB."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        was_3d = x.dim() == 3
        if was_3d:
            if x.shape[0] != 3:
                raise ValueError(f"Expected (3, H, W) NCHW, got shape {tuple(x.shape)}")
            x = x.unsqueeze(0)
        if x.dim() != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected (B, 3, H, W) NCHW in [0, 1], got shape {tuple(x.shape)}")
        out = self.transform(x)
        out = torch.clamp(out, 0.0, 1.0)
        if was_3d:
            out = out.squeeze(0)
        return out

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the filter to a ``(B, 3, H, W)`` tensor; subclasses override this."""
        raise NotImplementedError

    @property
    def num_params(self) -> int:
        """Total number of learnable scalar parameters."""
        return int(sum(p.numel() for p in self.parameters()))

    def get_params(self) -> Dict[str, torch.Tensor]:
        """Return the constrained (in-range) parameters as a dict of tensors."""
        raise NotImplementedError

    def get_params_flat(self) -> torch.Tensor:
        """Return constrained parameters as a 1-D tensor (for logging / CSV)."""
        parts = [v.reshape(-1) for v in self.get_params().values()]
        return torch.cat(parts) if parts else torch.tensor([])
