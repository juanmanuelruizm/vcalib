"""Tests for the pixel-wise residual MLP filter (neural_pixel)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.filters import build_filter, get_filter
from src.filters.neural_pixel import NeuralPixelFilter
from tests._filter_helpers import (
    check_composite,
    check_differentiability,
    check_identity_on_real_photo,
    check_output_in_range,
    rand_image,
)

# hidden_dim=32, depth=2: Linear(3,32)+Linear(32,32)+Linear(32,3) = 128+1056+99 = 1283
_DEFAULT_PARAMS = 1283
_RAW = sorted(set(list(Path("data/raw").glob("*.jpg")) + list(Path("data/raw").glob("*.JPG"))))


def test_identity_noop_synthetic():
    f = NeuralPixelFilter()
    x = rand_image(1)
    assert torch.allclose(f(x), x, atol=1e-5)


@pytest.mark.skipif(not _RAW, reason="no photos in data/raw")
def test_identity_noop_real_photo():
    check_identity_on_real_photo(NeuralPixelFilter())


def test_output_in_range():
    f = NeuralPixelFilter()
    with torch.no_grad():
        for p in f.parameters():
            p.add_(0.5 * torch.randn_like(p))
    check_output_in_range(f, rand_image(2))


def test_param_count_default():
    assert NeuralPixelFilter().num_params == _DEFAULT_PARAMS


def test_param_count_custom():
    # Linear(3,16)+Linear(16,3) = (3*16+16)+(16*3+3) = 64+51 = 115
    assert NeuralPixelFilter(hidden_dim=16, depth=1).num_params == 115


def test_depth_validation():
    with pytest.raises(ValueError, match="depth"):
        NeuralPixelFilter(depth=0)


def test_differentiability():
    f = NeuralPixelFilter()
    check_differentiability(f, rand_image(3), min_nonzero=0.80)


def test_registry_and_build():
    assert isinstance(get_filter("neural_pixel"), NeuralPixelFilter)
    f = build_filter({"type": "neural_pixel", "hidden_dim": 64, "depth": 3})
    assert isinstance(f, NeuralPixelFilter)
    assert f.hidden_dim == 64 and f.depth == 3


def test_composite():
    # neural_pixel(32,2)=1283 + brightness_2param=2 = 1285
    check_composite("neural_pixel", "brightness_2param", _DEFAULT_PARAMS + 2)


def test_reg_loss_nonnegative():
    # Hidden layers use Kaiming init so reg_loss > 0 at construction; only last layer is zeros.
    f = NeuralPixelFilter()
    assert float(f.reg_loss().detach()) >= 0.0


def test_reg_loss_nonzero_after_perturbation():
    f = NeuralPixelFilter()
    with torch.no_grad():
        for p in f.parameters():
            p.add_(1.0)
    assert float(f.reg_loss().detach()) > 0


@pytest.mark.slow
@pytest.mark.skipif(not _RAW, reason="no photos in data/raw")
def test_smoke_calibration_reduces_distance():
    from tests._filter_helpers import run_smoke_calibration

    f = NeuralPixelFilter(hidden_dim=32, depth=2)
    res = run_smoke_calibration(f, "neural_pixel", max_steps=60, lr=1e-2)
    assert res["train_reduction"] is not None and res["train_reduction"] > 0.0
    print(f"[neural_pixel smoke] {res}")
