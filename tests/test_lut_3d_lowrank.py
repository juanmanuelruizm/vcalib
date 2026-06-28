"""Tests for the low-rank 3D LUT filter (F7)."""

from __future__ import annotations

import pytest
import torch

from src.filters import build_filter, get_filter
from src.filters.lut_3d_lowrank import LUT3DLowRank
from tests._filter_helpers import (
    check_composite,
    check_differentiability,
    check_identity_on_real_photo,
    check_output_in_range,
    rand_image,
)


def test_identity_noop_synthetic():
    f = LUT3DLowRank(M=16, size=17)
    x = rand_image(1)
    assert torch.allclose(f(x), x, atol=1e-4)


def test_identity_noop_real_photo():
    check_identity_on_real_photo(LUT3DLowRank(M=16, size=17), atol=1e-3)


def test_output_in_range():
    f = LUT3DLowRank(M=16, size=17, init_identity=False)
    check_output_in_range(f, rand_image(2))


def test_param_count():
    assert LUT3DLowRank(M=16, size=17).num_params == 16  # only weights, basis is fixed
    assert LUT3DLowRank(M=32, size=9).num_params == 32


def test_differentiability():
    f = LUT3DLowRank(M=8, size=9)
    check_differentiability(f, rand_image(3, 8, 8), min_nonzero=0.9)


def test_registry_and_build():
    assert isinstance(get_filter("lut_3d_lowrank"), LUT3DLowRank)
    f = build_filter({"type": "lut_3d_lowrank", "M": 32, "size": 9})
    assert isinstance(f, LUT3DLowRank) and f.M == 32 and f.size == 9


def test_composite():
    check_composite("lut_3d_lowrank", "brightness_2param", 16 + 2)


def test_reg_loss_nonzero_after_perturbation():
    f = LUT3DLowRank(M=8, size=9)
    with torch.no_grad():
        f.weights[0] = 1.0
    assert float(f.reg_loss().detach()) > 0


@pytest.mark.slow
def test_smoke_calibration_reduces_distance():
    from tests._filter_helpers import run_smoke_calibration

    f = LUT3DLowRank(M=16, size=17)
    res = run_smoke_calibration(f, "lut_3d_lowrank")
    assert res["train_reduction"] is not None and res["train_reduction"] > 0.0
    print(f"[lut_3d_lowrank smoke] {res}")
