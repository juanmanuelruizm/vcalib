"""Parametric filter implementations for image calibration.

All filters operate on the [0, 1] NCHW RGB tensor before ImageNet normalization and
share the contract defined in :class:`src.filters.base.Filter` (identity init = no-op,
output clamped to [0, 1], params in a physical range). The :data:`FILTER_REGISTRY`
and :func:`get_filter` factory let the grid search and automatic layer-search loop
instantiate filters by name from ``configs/grid.yaml``.

Filter library (params):
- ``brightness_2param``  — global exposure (gain + offset)
- ``white_balance_3param`` — per-channel gains (color temperature)
- ``affine_6param`` — per-channel affine (gain + offset)  [flagship linear]
- ``saturation_1param`` — vividness toward luma
- ``contrast_1param`` — scaling around image mean
- ``gamma_3param`` — per-channel tone curve (non-linear)
- ``matrix_12param`` — 3x3 CCM + offset (cross-channel)  [flagship linear]
- ``spatial_brightness`` / ``spatial_whitebalance`` / ``spatial_affine`` / ``spatial_gamma``
  — bilinear K×K control-grid variants of the above (zone-dependent; params = K²·n_field)
- ``composite`` — ordered chain (variable params)
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Union, cast

from .affine_6param import Affine6Param
from .base import Filter
from .brightness_2param import Brightness
from .ccm_high_order import HighOrderCCM
from .chromatic_adaptation import ChromaticAdaptation
from .composite import CompositeFilter
from .contrast_1param import Contrast
from .gamma_3param import Gamma
from .local_tonemap import LocalTonemap
from .lut_3d import LUT3D
from .lut_3d_lowrank import LUT3DLowRank
from .matrix_12param import Matrix12Param
from .saturation_1param import Saturation
from .spatial import SpatialAffine, SpatialBrightness, SpatialGamma, SpatialWhiteBalance
from .spatial_tone_curve import SpatialToneCurve
from .tone_curve import ToneCurve
from .white_balance_3param import WhiteBalance

__all__ = [
    "Filter",
    "Brightness",
    "WhiteBalance",
    "Affine6Param",
    "Saturation",
    "Contrast",
    "Gamma",
    "Matrix12Param",
    "CompositeFilter",
    "SpatialBrightness",
    "SpatialWhiteBalance",
    "SpatialAffine",
    "SpatialGamma",
    "LUT3D",
    "ToneCurve",
    "HighOrderCCM",
    "ChromaticAdaptation",
    "SpatialToneCurve",
    "LocalTonemap",
    "LUT3DLowRank",
    "FILTER_REGISTRY",
    "get_filter",
    "make_composite",
]

FILTER_REGISTRY: Dict[str, Callable[[], Filter]] = {
    "brightness_2param": Brightness,
    "white_balance_3param": WhiteBalance,
    "affine_6param": Affine6Param,
    "saturation_1param": Saturation,
    "contrast_1param": Contrast,
    "gamma_3param": Gamma,
    "matrix_12param": Matrix12Param,
    "spatial_brightness": SpatialBrightness,
    "spatial_whitebalance": SpatialWhiteBalance,
    "spatial_affine": SpatialAffine,
    "spatial_gamma": SpatialGamma,
    "lut_3d": LUT3D,
    "tone_curve": ToneCurve,
    "ccm_high_order": HighOrderCCM,
    "chromatic_adaptation": ChromaticAdaptation,
    "spatial_tone_curve": SpatialToneCurve,
    "local_tonemap": LocalTonemap,
    "lut_3d_lowrank": LUT3DLowRank,
}


def get_filter(name: str) -> Filter:
    """Instantiate a filter by registry name (identity init)."""
    if name not in FILTER_REGISTRY:
        raise KeyError(f"Unknown filter {name!r}. Valid: {sorted(FILTER_REGISTRY)}")
    return FILTER_REGISTRY[name]()


def make_composite(names: Sequence[str]) -> CompositeFilter:
    """Build a CompositeFilter from an ordered list of registry names."""
    return CompositeFilter([get_filter(n) for n in names])


def build_filter(spec: Union[str, Sequence[str], Dict[str, object]]) -> Filter:
    """Flexible constructor for grid configs.

    - ``"affine_6param"`` -> single filter
    - ``["affine_6param", "gamma_3param"]`` -> composite (ordered chain)
    - ``{"type": "affine_6param", "params": 6}`` -> single (params ignored)
    - ``{"type": "spatial_affine", "grid_size": 3}`` -> spatial filter with K=3
    - ``{"composite": ["affine_6param", "gamma_3param"]}`` -> composite
    """
    if isinstance(spec, str):
        return get_filter(spec)
    if isinstance(spec, (list, tuple)):
        return make_composite(list(spec))
    if isinstance(spec, dict):
        if "composite" in spec:
            return make_composite(list(cast(Sequence[str], spec["composite"])))
        name = str(spec["type"])
        if name not in FILTER_REGISTRY:
            raise KeyError(f"Unknown filter {name!r}. Valid: {sorted(FILTER_REGISTRY)}")
        kwargs = {k: v for k, v in spec.items() if k not in ("type", "params")}
        cls = FILTER_REGISTRY[name]
        return cls(**kwargs)
    raise TypeError(f"Unsupported filter spec type: {type(spec)}")
