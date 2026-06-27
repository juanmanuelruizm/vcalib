"""Calibration loop: trains a filter to align activations from C' to C.

Given a reference image (condition C, matching model training) and a target image
(condition C', new illumination), this module optimises a Filter in-place so that
   acts(filter(img_C')) ≈ acts(img_C)
on the selected layer group, using per-layer relative L2 as the loss signal.

The frozen RF-DETR model is used purely as a feature extractor. Its parameters
never receive gradients; only the filter parameters are updated.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import torch
import torch.optim as optim

from .filters.base import Filter
from .utils.activations import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_SIZES,
    LAYER_PATHS,
    ActivationExtractor,
    _inner_model,
    _model_device,
    _to_tensor,
    load_model,
    to_unit_rgb,
)
from .utils.layer_groups import LayerGroup


def _normalize_batch(t: torch.Tensor) -> torch.Tensor:
    """ImageNet normalization for (B, 3, H, W); differentiable w.r.t. `t`."""
    mean = torch.tensor(IMAGENET_MEAN, dtype=t.dtype, device=t.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=t.dtype, device=t.device).view(1, 3, 1, 1)
    return (t - mean) / std


def _extract_ref_activations(
    libre: Any,
    image_ref: Any,
    layers: Iterable[str],
    dev: torch.device,
) -> Dict[str, torch.Tensor]:
    """Extract reference activations (detached) via standard no-grad inference."""
    with ActivationExtractor(libre, layers=layers) as ex:
        acts_cpu = ex.extract(image_ref)
    return {k: v.to(dev) for k, v in acts_cpu.items()}


def _forward_capturing(
    libre: Any,
    inner: torch.nn.Module,
    normalized: torch.Tensor,
    layers: Iterable[str],
) -> Dict[str, torch.Tensor]:
    """Forward pass capturing activations WITH grad (no detach, no no_grad).

    Unlike ActivationExtractor (which uses .detach().to('cpu')), this keeps
    tensors in the computation graph so loss.backward() reaches filter params.
    """
    acts: Dict[str, torch.Tensor] = {}
    handles = []

    for name in layers:
        path = LAYER_PATHS.get(name, name)
        try:
            module = inner.get_submodule(path)
        except AttributeError as exc:
            raise AttributeError(
                f"Layer {name!r} (path {path!r}) not found in model"
            ) from exc

        def _make_hook(n: str):
            def _hook(_mod: torch.nn.Module, _inp: Any, output: Any) -> None:
                t = _to_tensor(output)
                if t is not None:
                    acts[n] = t  # no .detach() — stays in computation graph

            return _hook

        handles.append(module.register_forward_hook(_make_hook(name)))

    try:
        libre.model(normalized)
    finally:
        for h in handles:
            h.remove()

    return acts


def activation_group_loss(
    acts_ref: Dict[str, torch.Tensor],
    acts_cur: Dict[str, torch.Tensor],
    group: LayerGroup,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean per-layer relative L2 over a layer group.

    loss_layer = ||acts_cur[l] - acts_ref[l]||_F / (||acts_ref[l]||_F + eps)
    group_loss = mean over layers in group

    acts_ref values must be detached (constant targets).
    acts_cur values must be in the computation graph (with grad path to filter).
    Layers absent from either dict are silently skipped.
    """
    per_layer: List[torch.Tensor] = []
    for name in group.layers:
        if name not in acts_ref or name not in acts_cur:
            continue
        a = acts_ref[name]
        b = acts_cur[name]
        a = a.to(b.device)
        per_layer.append((b - a).norm() / (a.norm() + eps))

    if not per_layer:
        # No layers captured — return differentiable zero so backward() still works
        dummy = next(iter(acts_cur.values())) if acts_cur else torch.tensor(0.0)
        return dummy.sum() * 0.0

    return torch.stack(per_layer).mean()


@dataclass
class CalibrationResult:
    filter: Filter
    best_loss: float
    steps: int
    wall_seconds: float
    history: List[float] = field(default_factory=list)


def calibrate(
    filt: Filter,
    image_ref: Any,
    image_tgt: Any,
    group: LayerGroup,
    model_size: str = "n",
    device: Optional[str] = None,
    lr: float = 1e-3,
    max_steps: int = 100,
    patience: int = 10,
    weights_dir: Optional[Union[str, Path]] = None,
) -> CalibrationResult:
    """Train `filt` in-place to minimise activation distance between
    filter(image_tgt) and image_ref on the given layer group.

    Args:
        filt:        Filter to train (modified in-place; best state is restored).
        image_ref:   Reference image in condition C (any ImageLike).
        image_tgt:   Target image in condition C' (any ImageLike).
        group:       LayerGroup defining which layers to use for loss.
        model_size:  RF-DETR size code ("n", "s", "m", "l").
        device:      "cpu", "cuda", or None for auto.
        lr:          Adam learning rate.
        max_steps:   Maximum optimisation steps.
        patience:    Early-stop patience (steps without improvement).
        weights_dir: Optional path to model weights directory.

    Returns:
        CalibrationResult with trained filter and training stats.
        The filter's state_dict reflects the best loss seen during training.
    """
    libre = load_model(size=model_size, device=device, weights_dir=weights_dir)
    inner = _inner_model(libre)
    dev = _model_device(libre)
    size_px = INPUT_SIZES[model_size]

    acts_ref = _extract_ref_activations(libre, image_ref, list(group.layers), dev)

    # Load target as [0,1] tensor; reused every step (filter changes, not image)
    b_raw = to_unit_rgb(image_tgt, size_px).unsqueeze(0).to(dev)  # (1,3,H,W)

    filt.train()
    filt.to(dev)
    optimizer = optim.Adam(filt.parameters(), lr=lr)

    best_loss = float("inf")
    best_state = copy.deepcopy(filt.state_dict())
    no_improve = 0
    history: List[float] = []

    t0 = time.perf_counter()

    for _ in range(max_steps):
        optimizer.zero_grad()

        filtered = filt(b_raw)                    # (1,3,H,W) in [0,1], with grad
        normalized = _normalize_batch(filtered)   # ImageNet norm, grad preserved

        acts_cur = _forward_capturing(libre, inner, normalized, group.layers)
        loss = activation_group_loss(acts_ref, acts_cur, group)

        if not torch.isfinite(loss):
            break

        val = float(loss.item())
        history.append(val)
        loss.backward()
        optimizer.step()

        if val < best_loss:
            best_loss = val
            best_state = copy.deepcopy(filt.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    filt.load_state_dict(best_state)
    filt.eval()

    return CalibrationResult(
        filter=filt,
        best_loss=best_loss,
        steps=len(history),
        wall_seconds=time.perf_counter() - t0,
        history=history,
    )
