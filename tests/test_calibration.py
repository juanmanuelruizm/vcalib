"""Tests for src/calibration.py.

Unit tests (no model needed) cover activation_group_loss math and edge cases.
The smoke test for calibrate() requires rf-detr-nano.pth and is skipped otherwise.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.calibration import CalibrationResult, activation_group_loss, calibrate
from src.utils.layer_groups import LayerGroup

_WEIGHTS = Path("3rd_party/libreyolo/weights/rf-detr-nano.pth")
_SKIP_MODEL = pytest.mark.skipif(
    not _WEIGHTS.exists(), reason="rf-detr-nano weights not found"
)


def _group(*layers: str) -> LayerGroup:
    return LayerGroup(name="test_group", layers=tuple(layers))


# ---------------------------------------------------------------------------
# activation_group_loss — pure math, no model
# ---------------------------------------------------------------------------


def test_loss_zero_when_identical():
    a = torch.randn(4, 145, 384)
    group = _group("backbone.layer.0")
    acts_ref = {"backbone.layer.0": a.clone()}
    acts_cur = {"backbone.layer.0": a.clone().requires_grad_(True)}
    loss = activation_group_loss(acts_ref, acts_cur, group)
    assert float(loss.item()) < 1e-6


def test_loss_positive_when_different():
    group = _group("backbone.layer.0")
    acts_ref = {"backbone.layer.0": torch.ones(4, 145, 384)}
    acts_cur = {"backbone.layer.0": torch.zeros(4, 145, 384, requires_grad=True)}
    loss = activation_group_loss(acts_ref, acts_cur, group)
    assert float(loss.item()) > 0.0


def test_loss_mean_over_layers():
    """Loss equals the mean of per-layer relative L2 values."""
    ones = torch.ones(1, 10, 10)
    group = _group("backbone.layer.0", "backbone.layer.1", "backbone.layer.2")

    # layer 0: a=1s, b=0s → loss = norm(ones) / norm(ones) = 1.0
    a0 = ones.clone()
    b0 = torch.zeros_like(a0, requires_grad=True)

    # layer 1: a=2s, b=1s → loss = norm(ones) / norm(2*ones) = 0.5
    a1 = ones.clone() * 2.0
    b1 = ones.clone().detach().requires_grad_(True)

    # layer 2: identical → loss ≈ 0
    a2 = torch.randn(1, 10, 10)
    b2 = a2.clone().detach().requires_grad_(True)

    acts_ref = {"backbone.layer.0": a0, "backbone.layer.1": a1, "backbone.layer.2": a2}
    acts_cur = {"backbone.layer.0": b0, "backbone.layer.1": b1, "backbone.layer.2": b2}

    loss = activation_group_loss(acts_ref, acts_cur, group)
    expected = (1.0 + 0.5 + 0.0) / 3.0
    assert abs(float(loss.item()) - expected) < 1e-4


def test_loss_differentiable():
    group = _group("backbone.layer.0")
    a = torch.randn(2, 5, 5)
    b = torch.randn(2, 5, 5, requires_grad=True)
    acts_ref = {"backbone.layer.0": a}
    acts_cur = {"backbone.layer.0": b}
    loss = activation_group_loss(acts_ref, acts_cur, group)
    loss.backward()
    assert b.grad is not None
    assert torch.isfinite(b.grad).all()


def test_loss_skips_missing_layers():
    """A layer absent from acts_cur is silently ignored without raising."""
    group = _group("backbone.layer.0", "backbone.layer.1")
    a = torch.ones(2, 5, 5)
    b = torch.zeros(2, 5, 5, requires_grad=True)
    acts_ref = {"backbone.layer.0": a, "backbone.layer.1": a.clone()}
    acts_cur = {"backbone.layer.0": b}  # layer 1 absent
    loss = activation_group_loss(acts_ref, acts_cur, group)
    assert float(loss.item()) > 0.0
    loss.backward()
    assert b.grad is not None


# ---------------------------------------------------------------------------
# calibrate — requires model weights
# ---------------------------------------------------------------------------


@_SKIP_MODEL
def test_calibrate_reduces_loss():
    import numpy as np
    from PIL import Image

    from src.filters import get_filter
    from src.utils.activations import _synthetic_image

    group = _group("backbone.layer.0", "backbone.layer.1")

    img_ref_arr = _synthetic_image(seed=0)
    img_ref = Image.fromarray(img_ref_arr)
    img_tgt_arr = np.clip(img_ref_arr.astype(np.float32) * 0.6, 0, 255).astype(np.uint8)
    img_tgt = Image.fromarray(img_tgt_arr)

    # Measure initial loss (identity filter, 1 step)
    filt_init = get_filter("affine_6param")
    result_init = calibrate(
        filt_init, img_ref, img_tgt,
        group=group, model_size="n", device="cpu",
        lr=1e-3, max_steps=1, patience=100,
    )
    initial_loss = result_init.history[0]

    # Calibrate 10 steps
    filt = get_filter("affine_6param")
    result = calibrate(
        filt, img_ref, img_tgt,
        group=group, model_size="n", device="cpu",
        lr=1e-3, max_steps=10, patience=100,
    )

    assert result.best_loss < initial_loss, (
        f"Loss did not decrease: initial={initial_loss:.4f}, best={result.best_loss:.4f}"
    )
    assert result.steps == 10
    assert len(result.history) == 10
    assert result.wall_seconds > 0.0
    assert isinstance(result.filter, torch.nn.Module)
