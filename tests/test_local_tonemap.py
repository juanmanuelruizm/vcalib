"""Tests for the local tone mapping (CLAHE-like) filter (F6)."""

from __future__ import annotations

import pytest
import torch

from src.filters import build_filter, get_filter
from src.filters.local_tonemap import LocalTonemap
from tests._filter_helpers import (
    check_composite,
    check_differentiability,
    check_identity_on_real_photo,
    check_output_in_range,
    rand_image,
)


def test_identity_noop_synthetic():
    f = LocalTonemap(grid_size=4)
    x = rand_image(1, 32, 32)
    assert torch.allclose(f(x), x, atol=1e-5)


def test_identity_noop_real_photo():
    check_identity_on_real_photo(LocalTonemap(grid_size=4))


def test_output_in_range():
    f = LocalTonemap(grid_size=4, init_identity=False)
    check_output_in_range(f, rand_image(2, 32, 32))


def test_param_count():
    assert LocalTonemap(grid_size=4).num_params == 16  # K² = 4²
    assert LocalTonemap(grid_size=3).num_params == 9


def test_differentiability():
    f = LocalTonemap(grid_size=4)
    check_differentiability(f, rand_image(3, 32, 32), min_nonzero=0.90)


def test_registry_and_build():
    assert isinstance(get_filter("local_tonemap"), LocalTonemap)
    f = build_filter({"type": "local_tonemap", "grid_size": 3})
    assert isinstance(f, LocalTonemap) and f.grid_size == 3


def test_composite():
    check_composite("local_tonemap", "brightness_2param", 16 + 2)


def test_local_contrast_effect():
    """With gain > 1, local contrast should increase (deviation from local mean amplified)."""
    f = LocalTonemap(grid_size=2, init_identity=False)
    with torch.no_grad():
        f.control_grid.fill_(1.5)  # gain = 1.5 everywhere
    x = torch.rand(1, 3, 32, 32)
    out = f(x).detach()
    mu = f._local_mean(x)
    orig_dev = (x - mu).abs().mean()
    out_dev = (out - mu).abs().mean()
    assert float(out_dev) > float(orig_dev), "gain > 1 should amplify local contrast"


@pytest.mark.slow
def test_smoke_calibration_reduces_distance():
    from tests._filter_helpers import run_smoke_calibration

    f = LocalTonemap(grid_size=4)
    res = run_smoke_calibration(f, "local_tonemap")
    assert res["train_reduction"] is not None and res["train_reduction"] > 0.0
    print(f"[local_tonemap smoke] {res}")
