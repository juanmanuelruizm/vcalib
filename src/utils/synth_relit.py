"""Deterministic programmatic re-lit pairs (stand-in validation data).

Until Phase 0 captures real A/B image pairs, the calibration loop's acceptance tests
(smoke + held-out generalization) need A/B pairs. This module produces them
deterministically from a real photo in ``data/raw/``:

- **A**     = original photo as a (3, H, W) [0, 1] tensor (the reference condition).
- **B_train** = A shifted by ``gamma=1.4 + per-channel gains=[1.10, 0.95, 0.85]``
  (warm + underexposed — both brightness and color-temperature shift).
- **B_val**   = A shifted by a *different magnitude* ``gamma=1.2 + gains=[1.05, 0.97, 0.90]``
  (same direction, milder — the held-out generalization probe).

These are a **proxy** for real A/B pairs; real-data generalization is re-verified after
Phase 0 capture. The shift direction (warm + underexposed) exercises both axes the
filter library targets.

All outputs are (3, H, W) float tensors in [0, 1] — the filter insertion point — ready
to feed directly into :func:`src.calibration.calibrate` (skipping ``to_unit_rgb``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, cast

import torch

from src.utils.activations import INPUT_SIZES, ImageLike, to_unit_rgb

B_TRAIN_GAMMA = 1.4
B_TRAIN_GAINS = (1.10, 0.95, 0.85)

B_VAL_GAMMA = 1.2
B_VAL_GAINS = (1.05, 0.97, 0.90)

EPS = 1e-6


def _is_unit_tensor(x: object) -> bool:
    """True if x is a (3, H, W) float tensor in [0, 1] — ready to use directly."""
    return (
        isinstance(x, torch.Tensor)
        and x.dim() == 3
        and x.shape[0] == 3
        and x.dtype.is_floating_point
    )


def _shift(unit: torch.Tensor, gamma: float, gains: Tuple[float, float, float]) -> torch.Tensor:
    """Apply per-channel gains + gamma to a (3, H, W) [0, 1] tensor; clamp to [0, 1]."""
    g = torch.tensor(gains, dtype=unit.dtype, device=unit.device).view(3, 1, 1)
    out = torch.clamp(unit, min=EPS) ** gamma * g
    return torch.clamp(out, 0.0, 1.0)


def make_relit_pair(
    image: ImageLike,
    input_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (A, B_train, B_val) as (3, H, W) [0, 1] tensors from a source image.

    Args:
        image: Path / PIL / array / (3,H,W) [0,1] tensor (used directly if already unit RGB).
        input_size: Resize target (default: nano's 384); ignored if ``image`` is already a
            (3, H, W) [0, 1] tensor.
    """
    if _is_unit_tensor(image):
        a = cast(torch.Tensor, image)
    else:
        size = input_size if input_size is not None else INPUT_SIZES["n"]
        a = to_unit_rgb(image, size)
    b_train = _shift(a, B_TRAIN_GAMMA, B_TRAIN_GAINS)
    b_val = _shift(a, B_VAL_GAMMA, B_VAL_GAINS)
    return a, b_train, b_val


def default_photo_path() -> Path:
    """Path to the first .jpg in ``data/raw`` (raises if none)."""
    for pat in ("*.jpg", "*.JPG", "*.png", "*.PNG"):
        hits = sorted(Path("data/raw").glob(pat))
        if hits:
            return hits[0]
    raise FileNotFoundError("No photos in data/raw/ (expected at least one A/B source).")
