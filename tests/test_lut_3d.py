"""Tests for the 3D LUT filter (F1). Covers the 7 acceptance criteria."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.filters import build_filter, get_filter, make_composite
from src.filters.lut_3d import LUT3D, _identity_lut

_RAW = sorted(set(list(Path("data/raw").glob("*.jpg")) + list(Path("data/raw").glob("*.JPG"))))


def test_identity_lut_shape():
    lut = _identity_lut(5)
    assert lut.shape == (3, 5, 5, 5)


def test_identity_init_noop_synthetic():
    f = LUT3D(size=9)
    f.eval()
    x = torch.rand(1, 3, 16, 16)
    out = f(x)
    assert torch.allclose(out, x, atol=1e-5)


@pytest.mark.skipif(not _RAW, reason="no photos in data/raw")
def test_identity_init_noop_real_photo():
    from src.utils.activations import to_unit_rgb

    img = to_unit_rgb(_RAW[0], input_size=128)
    f = LUT3D(size=9)
    out = f(img.unsqueeze(0))
    assert torch.allclose(out, img.unsqueeze(0), atol=1e-4)


def test_output_in_unit_range():
    f = LUT3D(size=5, init_identity=False)
    x = torch.rand(1, 3, 16, 16)
    out = f(x)
    assert out.shape == x.shape
    d = out.detach()
    assert float(d.min()) >= 0.0 and float(d.max()) <= 1.0


def test_param_count():
    assert LUT3D(size=9).num_params == 3 * 9**3
    assert LUT3D(size=17).num_params == 3 * 17**3
    assert LUT3D(size=4).num_params == 3 * 4**3


def test_size_minimum():
    with pytest.raises(ValueError, match="size"):
        LUT3D(size=1)


def test_differentiability_finite_nonzero_grads():
    f = LUT3D(size=5)
    with torch.no_grad():
        f.lut.add_(0.01 * torch.randn_like(f.lut))
    x = torch.rand(1, 3, 8, 8)
    out = f(x)
    loss = (out - x).pow(2).mean() + f.reg_loss()
    loss.backward()
    assert f.lut.grad is not None
    assert torch.isfinite(f.lut.grad).all()
    nonzero = (f.lut.grad.abs() > 0).float().mean()
    assert float(nonzero) > 0.80, f"only {nonzero:.1%} of LUT grads non-zero"


def test_reg_loss_nonzero_after_perturbation():
    f = LUT3D(size=5)
    with torch.no_grad():
        f.lut.add_(0.1)
    rl = f.reg_loss().detach()
    assert float(rl) > 0


def test_registry_get_filter():
    f = get_filter("lut_3d")
    assert isinstance(f, LUT3D)
    assert f.size == 9  # default


def test_build_filter_with_size():
    f = build_filter({"type": "lut_3d", "size": 7})
    assert isinstance(f, LUT3D)
    assert f.size == 7
    assert f.num_params == 3 * 343


def test_composite_chaining():
    comp = make_composite(["lut_3d", "brightness_2param"])
    assert comp.num_params == (3 * 9**3) + 2
    x = torch.rand(1, 3, 8, 8)
    out = comp(x)
    assert out.shape == x.shape
    # both identity → no-op
    assert torch.allclose(out, x, atol=1e-4)
    # reg_loss sums
    assert float(comp.reg_loss().detach()) >= 0


def test_shape_3d_and_4d():
    f = LUT3D(size=5)
    x4 = torch.rand(1, 3, 16, 16)
    assert f(x4).shape == x4.shape
    x3 = x4[0]
    assert f(x3).shape == x3.shape


@pytest.mark.skipif(not _RAW, reason="no photos in data/raw")
@pytest.mark.slow
def test_smoke_calibration_reduces_distance():
    from src.calibration import CalibrationConfig, calibrate, train_reduction
    from src.utils.activations import load_model
    from src.utils.synth_relit import make_relit_pair

    a, b_train, b_val = make_relit_pair(_RAW[0], input_size=384)
    model = load_model(size="n")
    layers = ["backbone.layer.0", "backbone.layer.1", "backbone.layer.2"]
    filt = LUT3D(size=9)
    cfg = CalibrationConfig(
        max_steps=40,
        early_stopping_patience=10,
        learning_rate=1e-2,
        reg_weight=0.01,
        log_every=0,
    )
    trained, result = calibrate(
        filt,
        a,
        b_train,
        layers,
        model=model,
        cfg=cfg,
        val_a_unit=a,
        val_b_unit=b_val,
    )
    tr = train_reduction(result)
    assert tr is not None and tr > 0.0, f"train reduction must be positive, got {tr}"
    print(f"[LUT3D smoke] train={tr:.3f} val={result.final_val_loss} steps={result.steps}")
