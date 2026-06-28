"""Tests for the spatial tone curve filter (F5)."""

from __future__ import annotations

import pytest
import torch

from src.filters import build_filter, get_filter
from src.filters.spatial_tone_curve import SpatialToneCurve
from tests._filter_helpers import (
    check_composite,
    check_differentiability,
    check_identity_on_real_photo,
    check_output_in_range,
    rand_image,
)


def test_identity_noop_synthetic():
    f = SpatialToneCurve(P=8, grid_size=3)
    x = rand_image(1)
    assert torch.allclose(f(x), x, atol=1e-5)


def test_identity_noop_real_photo():
    check_identity_on_real_photo(SpatialToneCurve(P=8, grid_size=3))


def test_output_in_range():
    f = SpatialToneCurve(P=8, grid_size=3, init_identity=False)
    check_output_in_range(f, rand_image(2))


def test_param_count():
    assert SpatialToneCurve(P=8, grid_size=3).num_params == 3 * 7 * 9  # 3*(P-1)*K²
    assert SpatialToneCurve(P=16, grid_size=4).num_params == 3 * 15 * 16


def test_differentiability():
    f = SpatialToneCurve(P=8, grid_size=3)
    check_differentiability(f, rand_image(3, 12, 12), min_nonzero=0.80)


def test_registry_and_build():
    assert isinstance(get_filter("spatial_tone_curve"), SpatialToneCurve)
    f = build_filter({"type": "spatial_tone_curve", "P": 16, "grid_size": 4})
    assert isinstance(f, SpatialToneCurve) and f.P == 16 and f.grid_size == 4


def test_composite():
    check_composite("spatial_tone_curve", "brightness_2param", 3 * 7 * 9 + 2)


def test_spatial_non_uniformity():
    """Different zones should produce different corrections when the grid is perturbed."""
    f = SpatialToneCurve(P=4, grid_size=2, init_identity=False)
    with torch.no_grad():
        f.control_grid.zero_()
        f.control_grid[0, 0, 0] = 2.0  # strong delta for R, top-left
    x = torch.full((1, 3, 16, 16), 0.5)
    out = f(x).detach()
    tl = float(out[0, 0, 0, 0])
    br = float(out[0, 0, 15, 15])
    assert abs(tl - br) > 0.05, "spatial tone curve not zone-dependent"


@pytest.mark.slow
def test_smoke_calibration_reduces_distance():
    from tests._filter_helpers import run_smoke_calibration

    f = SpatialToneCurve(P=8, grid_size=3)
    res = run_smoke_calibration(f, "spatial_tone_curve")
    assert res["train_reduction"] is not None and res["train_reduction"] > 0.0
    print(f"[spatial_tone_curve smoke] {res}")
