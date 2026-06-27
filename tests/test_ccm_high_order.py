"""Tests for the high-order polynomial CCM filter (F3)."""

from __future__ import annotations

import pytest
import torch

from src.filters import build_filter, get_filter
from src.filters.ccm_high_order import HighOrderCCM, _n_features
from tests._filter_helpers import (
    check_composite,
    check_differentiability,
    check_identity_on_real_photo,
    check_output_in_range,
    rand_image,
)


def test_identity_noop_synthetic():
    f = HighOrderCCM(degree=2)
    x = rand_image(1)
    assert torch.allclose(f(x), x, atol=1e-5)


def test_identity_noop_real_photo():
    check_identity_on_real_photo(HighOrderCCM(degree=2))


def test_output_in_range():
    f = HighOrderCCM(degree=2, init_identity=False)
    check_output_in_range(f, rand_image(2))


def test_param_count():
    assert HighOrderCCM(degree=1).num_params == 3 * 3 + 3
    assert HighOrderCCM(degree=2).num_params == 3 * 9 + 3
    assert HighOrderCCM(degree=3).num_params == 3 * 19 + 3


def test_n_features():
    assert _n_features(1) == 3
    assert _n_features(2) == 9
    assert _n_features(3) == 19


def test_degree_validation():
    with pytest.raises(ValueError, match="degree"):
        HighOrderCCM(degree=0)
    with pytest.raises(ValueError, match="degree"):
        HighOrderCCM(degree=4)


def test_differentiability():
    f = HighOrderCCM(degree=2)
    check_differentiability(f, rand_image(3), min_nonzero=0.95)


def test_registry_and_build():
    assert isinstance(get_filter("ccm_high_order"), HighOrderCCM)
    f = build_filter({"type": "ccm_high_order", "degree": 3})
    assert isinstance(f, HighOrderCCM) and f.degree == 3


def test_composite():
    check_composite("ccm_high_order", "gamma_3param", 3 * 9 + 3 + 3)


def test_reg_loss_nonzero_after_perturbation():
    f = HighOrderCCM(degree=2)
    with torch.no_grad():
        f.matrix[0, 3] = 1.0  # perturb a polynomial term
    assert float(f.reg_loss().detach()) > 0


@pytest.mark.slow
def test_smoke_calibration_reduces_distance():
    from tests._filter_helpers import run_smoke_calibration

    f = HighOrderCCM(degree=2)
    res = run_smoke_calibration(f, "ccm_high_order")
    assert res["train_reduction"] is not None and res["train_reduction"] > 0.0
    print(f"[ccm_high_order smoke] {res}")
