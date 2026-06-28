"""Calibration loop: optimize a filter's params so filtered-B activations match stored-A.

Core of the project, shared by the Phase 2 grid search and the deployed tool.

Pipeline per step:
  1. B [0,1] tensor  →  filter  →  filtered [0,1]  →  ImageNet normalize  →  model forward
  2. Grad-preserving forward hooks capture the layer-group activations of filtered-B.
  3. Group loss = aggregate(mean|sum) of per-layer normalized distance
     ``||a-b|| / ||a||`` between stored-A targets and filtered-B activations,
     + ``reg_weight * filter.reg_loss()``.
  4. ``loss.backward()`` → grads flow through the *frozen* model back to the filter params
     (the model params have ``requires_grad=False`` but the input carries grad via the
     filter); ``optimizer.step()``.

The model is frozen throughout; only the filter is trained. Early stopping on a held-out
val pair (or on train loss if no val pair is given).
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, cast

import torch
import torch.nn.functional as F

from src.utils.activations import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    _inner_model,
    _model_device,
    _to_tensor,
    load_model,
    resolve_layer_path,
)

EPS = 1e-6


def _make_nested_tensor(x: torch.Tensor) -> "Any":
    """Wrap a (B, 3, H, W) tensor as a LibreYOLO NestedTensor with a no-padding mask.

    Bypasses ``nested_tensor_from_tensor_list``, which does an inplace ``pad_img.copy_(img)``
    that fails when ``img`` requires grad (the filter's output) but ``pad_img`` (from
    ``torch.zeros``) does not. Our input is always a single square image — no padding
    needed — so the mask is all False.
    """
    from libreyolo.models.rfdetr.tensors import NestedTensor

    B, _, H, W = x.shape
    mask = torch.zeros((B, H, W), dtype=torch.bool, device=x.device)
    return NestedTensor(x, mask)


@dataclass
class CalibrationConfig:
    """Optimizer + loop hyperparameters (mirrors ``configs/grid.yaml::training``)."""

    optimizer: str = "adam"
    learning_rate: float = 1e-3
    max_steps: int = 100
    early_stopping_patience: int = 10
    reg_weight: float = 0.0
    metric: str = "l2_rel"  # "l2_rel" | "cosine"
    aggregation: str = "mean"  # "mean" | "sum"
    seed: int = 42
    log_every: int = 10


@dataclass
class CalibrationResult:
    """Output of :func:`calibrate`."""

    filter_state: Dict[str, torch.Tensor]
    steps: int
    wall_clock_s: float
    final_train_loss: float
    final_val_loss: Optional[float]
    train_history: List[float] = field(default_factory=list)
    val_history: List[float] = field(default_factory=list)
    converged: bool = False


def _normalize_batch(unit: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """ImageNet-normalize a (B, 3, H, W) [0,1] tensor → model input."""
    mean = torch.tensor(IMAGENET_MEAN, dtype=dtype, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=dtype, device=device).view(1, 3, 1, 1)
    return (unit - mean) / std


def _per_layer_distance(a: torch.Tensor, b: torch.Tensor, metric: str) -> torch.Tensor:
    """Normalized distance between two activation tensors (kept on the graph)."""
    if metric == "l2_rel":
        return cast(torch.Tensor, (a - b).flatten().norm() / a.flatten().norm().clamp(min=EPS))
    if metric == "cosine":
        return cast(
            torch.Tensor,
            1.0 - F.cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)),
        )
    raise ValueError(f"Unknown metric {metric!r}; expected 'l2_rel' or 'cosine'")


def group_loss(
    a_acts: Mapping[str, torch.Tensor],
    b_acts: Mapping[str, torch.Tensor],
    layer_names: Sequence[str],
    metric: str = "l2_rel",
    aggregation: str = "mean",
) -> torch.Tensor:
    """Aggregate per-layer normalized distance across the layer group."""
    dists: List[torch.Tensor] = []
    for name in layer_names:
        if name not in a_acts or name not in b_acts:
            raise KeyError(f"Layer {name!r} missing from activations (have {sorted(b_acts)})")
        dists.append(_per_layer_distance(a_acts[name], b_acts[name], metric))
    stacked = torch.stack(dists)
    return stacked.mean() if aggregation == "mean" else stacked.sum()


@contextmanager
def _grad_hooks(model: Any, layer_names: Sequence[str]):
    """Register forward hooks that capture activations **without detaching** (grad flows)."""
    inner = _inner_model(model)
    handles: List[Any] = []
    acts: Dict[str, torch.Tensor] = {}

    def _hook(name: str):
        def _fn(_mod: torch.nn.Module, _inp: Any, output: Any) -> None:
            t = _to_tensor(output)
            if t is not None:
                acts[name] = t  # keep on graph, on model device

        return _fn

    for name in layer_names:
        path = resolve_layer_path(name)
        mod = inner.get_submodule(path)
        handles.append(mod.register_forward_hook(_hook(name)))
    try:
        yield acts
    finally:
        for h in handles:
            h.remove()


def compute_reference_activations(
    model: Any,
    unit_image: torch.Tensor,
    layer_names: Sequence[str],
) -> Dict[str, torch.Tensor]:
    """Forward an A image (3,H,W [0,1]) through the frozen model; return detached targets.

    Runs WITHOUT ``torch.no_grad()``: the model params have ``requires_grad=False``, so no
    graph is built anyway, and avoiding ``no_grad`` prevents view/inplace conflicts with the
    DINOv2 backbone's inplace ops when the training pass later runs in grad mode.
    """
    dev = _model_device(model)
    dtype = next(model.model.parameters()).dtype
    x = _normalize_batch(unit_image.unsqueeze(0).to(dev, dtype), dev, dtype)
    with _grad_hooks(model, layer_names) as acts:
        model.model(_make_nested_tensor(x))
    return {k: v.detach().clone() for k, v in acts.items()}


def _forward_filtered(
    model: Any, filt: torch.nn.Module, b_unit: torch.Tensor, layer_names: Sequence[str]
) -> Dict[str, torch.Tensor]:
    """Apply filter to B [0,1], normalize, forward through model; return grad-attached acts."""
    dev = _model_device(model)
    dtype = next(model.model.parameters()).dtype
    b = b_unit.to(dev, dtype).unsqueeze(0)
    filtered = filt(b)  # (1, 3, H, W) [0,1], base clamps
    normed = _normalize_batch(filtered, dev, dtype)
    with _grad_hooks(model, layer_names) as acts:
        model.model(_make_nested_tensor(normed))
    return cast(Dict[str, torch.Tensor], acts)


def calibrate(
    filt: torch.nn.Module,
    a_unit: torch.Tensor,
    b_unit: torch.Tensor,
    layer_names: Sequence[str],
    model: Optional[Any] = None,
    size: str = "n",
    device: Optional[str] = None,
    cfg: Optional[CalibrationConfig] = None,
    val_a_unit: Optional[torch.Tensor] = None,
    val_b_unit: Optional[torch.Tensor] = None,
) -> Tuple[torch.nn.Module, CalibrationResult]:
    """Train ``filt`` so filtered-B activations match stored-A at ``layer_names``.

    Args:
        filt: Filter to train (moved to the model device; params set requires_grad).
        a_unit: A image as (3, H, W) [0, 1] tensor (reference condition).
        b_unit: B image as (3, H, W) [0, 1] tensor (target condition to correct).
        layer_names: Canonical layer names in the loss group (e.g. from ``LayerGroup.layers``).
        model: Pre-loaded ``LibreRFDETR`` wrapper (loaded if None).
        size / device: Used only if ``model`` is None.
        cfg: Hyperparameters (defaults to :class:`CalibrationConfig`).
        val_a_unit / val_b_unit: Held-out val pair (3, H, W [0,1]); enables val tracking +
            val-based early stopping. If None, early-stops on train loss.

    Returns:
        (trained filter, CalibrationResult). The filter is left in eval mode on the model device.
    """
    cfg = cfg or CalibrationConfig()
    torch.manual_seed(cfg.seed)
    model = model or load_model(size=size, device=device)
    dev = _model_device(model)

    filt = filt.to(dev)
    filt.train()
    for p in filt.parameters():
        p.requires_grad_(True)

    # Precompute A targets (detached) + val targets once.
    # NB: no torch.no_grad() — the model is frozen (params requires_grad=False), so no graph
    # is built. Using no_grad here conflicts with the DINOv2 backbone's inplace ops when the
    # training pass later runs in grad mode ("view created in no_grad modified in grad").
    a_acts = compute_reference_activations(model, a_unit, layer_names)
    val_a_acts = (
        compute_reference_activations(model, val_a_unit, layer_names)
        if val_a_unit is not None
        else None
    )

    opt = torch.optim.Adam(filt.parameters(), lr=cfg.learning_rate)

    best_val = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    patience_left = cfg.early_stopping_patience
    train_history: List[float] = []
    val_history: List[float] = []
    t0 = time.time()
    last_loss = float("inf")
    converged = False
    baseline_train: Optional[float] = None
    baseline_val: Optional[float] = None

    for step in range(1, cfg.max_steps + 1):
        opt.zero_grad(set_to_none=True)
        b_acts = _forward_filtered(model, filt, b_unit, layer_names)
        loss = group_loss(a_acts, b_acts, layer_names, cfg.metric, cfg.aggregation)
        if cfg.reg_weight > 0:
            loss = loss + cfg.reg_weight * cast(Any, filt).reg_loss()
        loss.backward()
        opt.step()
        last_loss = float(loss.detach())
        train_history.append(last_loss)
        if baseline_train is None:
            baseline_train = last_loss  # first step = pre-optimization baseline

        # Validation (detached, no optimizer step).
        v_loss: Optional[float] = None
        if val_a_acts is not None and val_b_unit is not None:
            vb_acts = _forward_filtered(model, filt, val_b_unit, layer_names)
            vl = group_loss(val_a_acts, vb_acts, layer_names, cfg.metric, cfg.aggregation)
            if cfg.reg_weight > 0:
                vl = vl + cfg.reg_weight * cast(Any, filt).reg_loss()
            v_loss = float(vl.detach())
            val_history.append(v_loss)
            if baseline_val is None:
                baseline_val = v_loss

        # Early stopping (val-based if val available, else train-based).
        monitor = v_loss if v_loss is not None else last_loss
        if monitor < best_val - 1e-7:
            best_val = monitor
            best_state = {k: v.detach().clone() for k, v in filt.state_dict().items()}
            patience_left = cfg.early_stopping_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                converged = True
                break

        if cfg.log_every and step % cfg.log_every == 0:
            tag = f"val={v_loss:.5f} " if v_loss is not None else ""
            print(f"[calibrate] step {step:3d} train={last_loss:.5f} {tag}patience={patience_left}")

    wall = time.time() - t0
    if best_state is not None:
        filt.load_state_dict(best_state)
    filt.eval()

    result = CalibrationResult(
        filter_state={k: v.detach().cpu().clone() for k, v in filt.state_dict().items()},
        steps=step,
        wall_clock_s=wall,
        final_train_loss=last_loss,
        final_val_loss=val_history[-1] if val_history else None,
        train_history=train_history,
        val_history=val_history,
        converged=converged,
    )
    # Stash baselines for reduction reporting (not part of the dataclass fields).
    result.__dict__["baseline_train"] = baseline_train  # type: ignore[attr-defined]
    result.__dict__["baseline_val"] = baseline_val  # type: ignore[attr-defined]
    return filt, result


def train_reduction(result: CalibrationResult) -> Optional[float]:
    """Fractional distance reduction on train: (baseline - final) / baseline. None if baseline=0."""
    b = result.__dict__.get("baseline_train", None)
    if b is None or b <= 0:
        return None
    return float((b - result.final_train_loss) / b)


def val_reduction(result: CalibrationResult) -> Optional[float]:
    """Fractional distance reduction on val. None if no val pair / baseline=0."""
    b = result.__dict__.get("baseline_val", None)
    if b is None or b <= 0 or result.final_val_loss is None:
        return None
    return float((b - result.final_val_loss) / b)


def overfit_gate_ok(result: CalibrationResult, min_ratio: float = 0.5) -> bool:
    """True if val reduction >= min_ratio * train reduction (the held-out generalization gate)."""
    tr, vr = train_reduction(result), val_reduction(result)
    if tr is None or vr is None:
        return True  # no val -> don't gate (caller decides)
    if tr <= 0:
        return vr > 0
    return vr / max(tr, 1e-9) >= min_ratio


def calibrate_multi(
    filt: torch.nn.Module,
    train_pairs: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    layer_names: Sequence[str],
    model: Any,
    cfg: Optional[CalibrationConfig] = None,
    test_pairs: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
) -> Tuple[torch.nn.Module, CalibrationResult]:
    """Train ``filt`` on MULTIPLE A/B pairs (cycling through them each step).

    Each step picks the next pair in round-robin order, applies the filter to that B,
    forwards through the model, and minimizes the group loss vs that pair's cached A
    activations. This prevents the filter from memorizing a single scene and forces it
    to learn the illumination shift itself -- which generalizes.

    Args:
        filt: Filter to train.
        train_pairs: List of (A, B) tuples, each (3, H, W) [0, 1] tensors.
        layer_names: Layer group for the loss.
        model: Pre-loaded LibreRFDETR wrapper.
        cfg: Hyperparameters.
        test_pairs: Optional held-out pairs for val-based early stopping.

    Returns:
        (trained filter, CalibrationResult). Same shape as :func:`calibrate`.
    """
    import random

    cfg = cfg or CalibrationConfig()
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    dev = _model_device(model)

    filt = filt.to(dev)
    filt.train()
    for p in filt.parameters():
        p.requires_grad_(True)

    # Precompute A activations for ALL train pairs (one forward each, model is frozen)
    print(f"  [calibrate_multi] precomputing A activations for {len(train_pairs)} train pairs...")
    a_acts_all: List[Dict[str, torch.Tensor]] = []
    for i, (a_unit, _) in enumerate(train_pairs):
        a_acts_all.append(compute_reference_activations(model, a_unit, layer_names))
        if (i + 1) % 20 == 0 or (i + 1) == len(train_pairs):
            print(f"    A activations: {i+1}/{len(train_pairs)}")

    # Precompute test A activations if test pairs given
    test_a_acts_all: List[Dict[str, torch.Tensor]] = []
    if test_pairs:
        print(f"  [calibrate_multi] precomputing A activations for {len(test_pairs)} test pairs...")
        for a_unit, _ in test_pairs:
            test_a_acts_all.append(compute_reference_activations(model, a_unit, layer_names))

    opt = torch.optim.Adam(filt.parameters(), lr=cfg.learning_rate)

    best_val = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    patience_left = cfg.early_stopping_patience
    train_history: List[float] = []
    val_history: List[float] = []
    t0 = time.time()
    last_loss = float("inf")
    converged = False
    baseline_train: Optional[float] = None
    baseline_val: Optional[float] = None
    n = len(train_pairs)
    order = list(range(n))
    rng.shuffle(order)

    for step in range(1, cfg.max_steps + 1):
        # Round-robin through pairs (shuffled order, reshuffled each epoch)
        pair_idx = order[(step - 1) % n]
        a_acts = a_acts_all[pair_idx]
        _, b_unit = train_pairs[pair_idx]

        opt.zero_grad(set_to_none=True)
        b_acts = _forward_filtered(model, filt, b_unit, layer_names)
        loss = group_loss(a_acts, b_acts, layer_names, cfg.metric, cfg.aggregation)
        if cfg.reg_weight > 0:
            loss = loss + cfg.reg_weight * cast(Any, filt).reg_loss()
        loss.backward()
        opt.step()
        last_loss = float(loss.detach())
        train_history.append(last_loss)
        if baseline_train is None:
            baseline_train = last_loss

        # Validation: average loss over ALL test pairs (not just one)
        v_loss: Optional[float] = None
        if test_pairs and test_a_acts_all:
            vlosses = []
            for ti, (_, tb_unit) in enumerate(test_pairs):
                vb_acts = _forward_filtered(model, filt, tb_unit, layer_names)
                vl = group_loss(test_a_acts_all[ti], vb_acts, layer_names, cfg.metric, cfg.aggregation)
                vlosses.append(float(vl.detach()))
            v_loss = sum(vlosses) / len(vlosses)
            val_history.append(v_loss)
            if baseline_val is None:
                baseline_val = v_loss

        # Early stopping
        monitor = v_loss if v_loss is not None else last_loss
        if monitor < best_val - 1e-7:
            best_val = monitor
            best_state = {k: v.detach().clone() for k, v in filt.state_dict().items()}
            patience_left = cfg.early_stopping_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                converged = True
                break

        # Reshuffle order each epoch
        if (step - 1) % n == n - 1:
            rng.shuffle(order)

        if cfg.log_every and step % cfg.log_every == 0:
            tag = f"val={v_loss:.5f} " if v_loss is not None else ""
            print(f"  [calibrate_multi] step {step:3d} pair={pair_idx} train={last_loss:.5f} {tag}patience={patience_left}")

    wall = time.time() - t0
    if best_state is not None:
        filt.load_state_dict(best_state)
    filt.eval()

    result = CalibrationResult(
        filter_state={k: v.detach().cpu().clone() for k, v in filt.state_dict().items()},
        steps=step,
        wall_clock_s=wall,
        final_train_loss=last_loss,
        final_val_loss=val_history[-1] if val_history else None,
        train_history=train_history,
        val_history=val_history,
        converged=converged,
    )
    result.__dict__["baseline_train"] = baseline_train
    result.__dict__["baseline_val"] = baseline_val
    return filt, result
