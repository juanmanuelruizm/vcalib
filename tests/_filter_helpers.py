"""Shared helpers for advanced filter acceptance tests.

Provides parametrized fixtures and helper functions so each test_<filter>.py can
exercise the 7 acceptance criteria with minimal boilerplate.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

_RAW = sorted(set(list(Path("data/raw").glob("*.jpg")) + list(Path("data/raw").glob("*.JPG"))))


def rand_image(seed: int = 0, H: int = 16, W: int = 16) -> torch.Tensor:
    return torch.rand(1, 3, H, W, generator=torch.Generator().manual_seed(seed))


@pytest.mark.skipif(not _RAW, reason="no photos in data/raw")
def check_identity_on_real_photo(f: torch.nn.Module, atol: float = 1e-4) -> None:
    from src.utils.activations import to_unit_rgb

    img = to_unit_rgb(_RAW[0], input_size=128)
    out = f(img.unsqueeze(0))
    assert torch.allclose(out, img.unsqueeze(0), atol=atol), (
        f"{type(f).__name__} not identity on real photo"
    )


def check_output_in_range(f: torch.nn.Module, x: torch.Tensor) -> None:
    out = f(x)
    assert out.shape == x.shape
    d = out.detach()
    assert float(d.min()) >= 0.0 and float(d.max()) <= 1.0


def check_differentiability(
    f: torch.nn.Module, x: torch.Tensor, min_nonzero: float = 0.80
) -> float:
    # Perturb params slightly so the data term (out - x) is non-zero → grads flow to all params.
    with torch.no_grad():
        for p in f.parameters():
            p.add_(0.01 * torch.randn_like(p))
    out = f(x)
    loss = (out - x).pow(2).mean() + f.reg_loss()
    loss.backward()
    grads = [p.grad for p in f.parameters() if p.grad is not None]
    assert all(torch.isfinite(g).all() for g in grads), "some grads not finite"
    all_grads = torch.cat([g.abs().reshape(-1) for g in grads])
    nonzero = float((all_grads > 0).float().mean())
    assert nonzero > min_nonzero, (
        f"only {nonzero:.1%} of grads non-zero (required > {min_nonzero:.0%})"
    )
    return nonzero


def check_composite(name: str, other_name: str, expected_params: int) -> None:
    from src.filters import make_composite

    comp = make_composite([name, other_name])
    assert comp.num_params == expected_params
    x = rand_image(5)
    out = comp(x)
    assert out.shape == x.shape
    assert float(comp.reg_loss().detach()) >= 0


def run_smoke_calibration(
    f: torch.nn.Module,
    filter_name: str,
    max_steps: int = 40,
    lr: float = 1e-2,
    reg_weight: float = 0.01,
) -> dict:
    """Run the calibration loop on the stand-in re-lit pair; return a results dict.

    Returns: {filter, train_reduction, val_reduction, val_train_ratio, steps, wall_s,
    overfit_gate_ok}
    """
    from src.calibration import (
        CalibrationConfig,
        calibrate,
        overfit_gate_ok,
        train_reduction,
        val_reduction,
    )
    from src.utils.activations import load_model
    from src.utils.synth_relit import make_relit_pair

    a, b_train, b_val = make_relit_pair(_RAW[0], input_size=384)
    model = load_model(size="n")
    layers = ["backbone.layer.0", "backbone.layer.1", "backbone.layer.2"]
    cfg = CalibrationConfig(
        max_steps=max_steps,
        early_stopping_patience=10,
        learning_rate=lr,
        reg_weight=reg_weight,
        log_every=0,
    )
    trained, result = calibrate(
        f,
        a,
        b_train,
        layers,
        model=model,
        cfg=cfg,
        val_a_unit=a,
        val_b_unit=b_val,
    )
    tr = train_reduction(result)
    vr = val_reduction(result)
    ratio = (vr / tr) if (tr and tr > 0 and vr is not None) else None
    gate_ok = overfit_gate_ok(result, min_ratio=0.5)
    return {
        "filter": filter_name,
        "train_reduction": tr,
        "val_reduction": vr,
        "val_train_ratio": ratio,
        "steps": result.steps,
        "wall_s": result.wall_clock_s,
        "overfit_gate_ok": gate_ok,
        "final_train_loss": result.final_train_loss,
        "final_val_loss": result.final_val_loss,
    }
