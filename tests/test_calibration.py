"""Tests for the calibration loop (A4).

Pure tests for group_loss / reductions / overfit gate (no model), plus a slow
end-to-end smoke test that loads RF-DETR nano and verifies a Brightness filter
reduces activation distance on the stand-in re-lit pair.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.calibration import (
    CalibrationConfig,
    CalibrationResult,
    calibrate,
    group_loss,
    overfit_gate_ok,
    train_reduction,
    val_reduction,
)
from src.filters import Brightness

_RAW = sorted(set(list(Path("data/raw").glob("*.jpg")) + list(Path("data/raw").glob("*.JPG"))))


def test_group_loss_l2_rel_mean():
    a = {"l0": torch.ones(4, 4), "l1": torch.ones(4, 4)}
    b = {"l0": torch.zeros(4, 4), "l1": torch.ones(4, 4)}
    loss = group_loss(a, b, ["l0", "l1"], "l2_rel", "mean")
    # l0: ||1-0||/||1|| = 1.0; l1: 0.0 → mean = 0.5
    assert abs(float(loss) - 0.5) < 1e-6


def test_group_loss_sum_aggregation():
    a = {"l0": torch.ones(4, 4), "l1": torch.ones(4, 4)}
    b = {"l0": torch.zeros(4, 4), "l1": torch.ones(4, 4)}
    loss = group_loss(a, b, ["l0", "l1"], "l2_rel", "sum")
    assert abs(float(loss) - 1.0) < 1e-6


def test_group_loss_identity_is_zero():
    a = {"l0": torch.rand(2, 3)}
    loss = group_loss(a, a, ["l0"], "l2_rel", "mean")
    assert float(loss) < 1e-6


def test_group_loss_missing_layer_raises():
    with pytest.raises(KeyError, match="missing"):
        group_loss({"l0": torch.ones(4)}, {"l0": torch.ones(4)}, ["l1"], "l2_rel", "mean")


def test_group_loss_cosine():
    a = {"l0": torch.tensor([1.0, 0.0, 0.0])}
    b = {"l0": torch.tensor([0.0, 1.0, 0.0])}
    loss = group_loss(a, b, ["l0"], "cosine", "mean")
    assert abs(float(loss) - 1.0) < 1e-6  # orthogonal → 1 - 0 = 1


def test_group_loss_grad_flows():
    a = {"l0": torch.ones(4)}
    b_param = torch.ones(4) * 2
    b_param.requires_grad_(True)
    b = {"l0": b_param}
    loss = group_loss(a, b, ["l0"], "l2_rel", "mean")
    loss.backward()
    assert b_param.grad is not None
    assert float(b_param.grad.abs().sum()) > 0


def _make_result(train, val, b_train=1.0, b_val=1.0):
    r = CalibrationResult(
        filter_state={},
        steps=10,
        wall_clock_s=0.1,
        final_train_loss=train,
        final_val_loss=val,
        train_history=[],
        val_history=[],
        converged=True,
    )
    r.__dict__["baseline_train"] = b_train
    r.__dict__["baseline_val"] = b_val
    return r


def test_train_reduction():
    r = _make_result(0.4, None, b_train=1.0)
    assert abs(train_reduction(r) - 0.6) < 1e-6


def test_val_reduction_none_without_val():
    r = _make_result(0.4, None, b_train=1.0)
    assert val_reduction(r) is None


def test_overfit_gate_passes_when_val_keeps_up():
    r = _make_result(0.4, 0.5, b_train=1.0, b_val=1.0)  # train 60%, val 50% → ratio 0.83 ≥ 0.5
    assert overfit_gate_ok(r, min_ratio=0.5)


def test_overfit_gate_fails_when_val_lags():
    r = _make_result(0.1, 0.9, b_train=1.0, b_val=1.0)  # train 90%, val 10% → ratio 0.11 < 0.5
    assert not overfit_gate_ok(r, min_ratio=0.5)


def test_overfit_gate_no_val_passes():
    r = _make_result(0.4, None, b_train=1.0)
    assert overfit_gate_ok(r, min_ratio=0.5)


@pytest.mark.skipif(not _RAW, reason="no photos in data/raw")
@pytest.mark.slow
def test_smoke_calibration_reduces_distance():
    """End-to-end: Brightness filter reduces activation distance on the re-lit pair."""
    from src.utils.synth_relit import make_relit_pair

    from src.utils.activations import load_model

    a, b_train, b_val = make_relit_pair(_RAW[0], input_size=384)
    model = load_model(size="n")
    layers = ["backbone.layer.0", "backbone.layer.1", "backbone.layer.2"]

    filt = Brightness()
    cfg = CalibrationConfig(
        max_steps=30, early_stopping_patience=10, learning_rate=5e-3, log_every=0
    )
    trained, result = calibrate(
        filt, a, b_train, layers, model=model, cfg=cfg, val_a_unit=a, val_b_unit=b_val
    )

    tr = train_reduction(result)
    assert tr is not None and tr > 0.0, f"train reduction must be positive, got {tr}"
    print(
        f"[smoke] train reduction={tr:.3f}, val reduction={val_reduction(result)}, "
        f"steps={result.steps}, wall={result.wall_clock_s:.1f}s"
    )
