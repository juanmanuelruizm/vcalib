"""Composite filter: chain an ordered list of filters.

Lets the grid / automatic search compose corrections that a single filter cannot
express, e.g. ``Affine6Param -> Gamma -> Saturation`` (linear CCM + non-linear tone +
color vividness, 6 + 3 + 1 = 10 params). Each sub-filter keeps its own identity init
and param ranges; the composite clamps to [0, 1] only once at the end (via the base
class), so intermediate values may briefly leave [0, 1] between stages.
"""

from __future__ import annotations

from typing import Dict, Iterable

import torch

from .base import Filter


class CompositeFilter(Filter):
    """Apply a sequence of filters in order (outputs chained, params summed)."""

    def __init__(self, filters: Iterable[Filter]) -> None:
        super().__init__()
        self.filters = torch.nn.ModuleList(list(filters))
        if not self.filters:
            raise ValueError("CompositeFilter requires at least one sub-filter")

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        for f in self.filters:
            x = f.transform(x)
        return x

    @property
    def num_params(self) -> int:
        return int(sum(f.num_params for f in self.filters))

    def reg_loss(self) -> torch.Tensor:
        """Sum of sub-filters' reg_loss() (propagates regularization through the chain)."""
        total = None
        for f in self.filters:
            r = f.reg_loss()
            total = r if total is None else total + r
        return total if total is not None else torch.zeros(())

    def get_params(self) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for i, f in enumerate(self.filters):
            for k, v in f.get_params().items():
                out[f"f{i}_{type(f).__name__}_{k}"] = v
        return out
