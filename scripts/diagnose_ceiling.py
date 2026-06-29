#!/usr/bin/env python3
"""Diagnostic probes for the ~33 % activation-reduction ceiling.

Two decisive experiments that the existing pipeline never ran (see
``docs/diagnosis/2026-06-29-filter-performance-diagnosis.md``):

1. ``--mode oracle`` — replace the parametric filter with a *free per-pixel residual*
   (a learnable delta the size of the image; output = clamp(B + delta, 0, 1)) and run the
   same per-pair calibration. This is the upper bound any pixel-space filter could reach.
   Compared against a parametric filter on the SAME single pair and step budget, it
   separates two hypotheses:
       oracle >> parametric  → bottleneck is FILTER CAPACITY (a neural filter could help)
       oracle ~= parametric  → bottleneck is the model REPRESENTATION (no pixel filter helps)

2. ``--mode detection`` — close the loop the project never closed: measure how much a
   trained filter recovers the model's *detection output* (KL of logits + L1 of boxes vs A),
   not just activation distance. Reports the no-filter A↔B gap too: if the model barely
   degrades under the shift, the premise is weak for this model.

3. ``--mode noise-floor`` — irreducible activation distance between A and a *second image
   of the same scene* (same illumination, different geometry). The denominator of l2_rel
   includes this; 0.33 of the total is a larger fraction of what is actually recoverable.

This script reuses the vetted functions in ``src/calibration.py`` so the forward/loss path
is identical to training. It needs a real dataset + model (GPU recommended). Examples:

    uv run python scripts/diagnose_ceiling.py --mode oracle \
        --dataset data/datasets/level_1_vs_level_2 --n-pairs 6 --steps 300
    uv run python scripts/diagnose_ceiling.py --mode detection \
        --dataset data/datasets/level_1_vs_level_2 --checkpoint \
        results/experiments/runs/p2_stc_P16_g5_lv2/best.pt --filter spatial_tone_curve
    uv run python scripts/diagnose_ceiling.py --mode noise-floor \
        --dataset data/datasets/level_1_vs_level_2 --n-pairs 6
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.calibration import (  # noqa: E402
    CalibrationConfig,
    calibrate,
    compute_reference_activations,
    detection_output_loss,
    group_loss,
    _forward_filtered,
    _forward_filtered_full,
    _reference_with_detection,
)
from src.filters import build_filter  # noqa: E402
from src.filters.base import Filter  # noqa: E402
from src.utils.activations import _model_device, load_model, to_unit_rgb  # noqa: E402
from src.utils.layer_groups import expand_layer_spec  # noqa: E402


class FreePixelFilter(Filter):
    """Oracle filter: an unconstrained learnable per-pixel residual on the [0,1] image.

    Not deployable (one delta per fixed image size, fit to a single pair) — it exists only
    to measure the upper bound of activation reduction achievable by ANY pixel-space edit.
    """

    def __init__(self, size: int = 384) -> None:
        super().__init__()
        self.delta = torch.nn.Parameter(torch.zeros(1, 3, size, size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        single = x.dim() == 3
        if single:
            x = x.unsqueeze(0)
        out = torch.clamp(x + self.delta, 0.0, 1.0)
        return out.squeeze(0) if single else out


def _load_test_pairs(dataset: Path, n: int, size: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Load up to n (A, B) [0,1] tensor pairs from <dataset>/test/<scene>/level_*.jpg."""
    test_dir = dataset / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"No test/ dir under {dataset}")
    pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for scene in sorted(p for p in test_dir.iterdir() if p.is_dir()):
        imgs = {int(f.stem.split("_")[1]): f for f in scene.glob("level_*.jpg")}
        if 1 not in imgs:
            continue
        b_levels = sorted(l for l in imgs if l > 1)
        if not b_levels:
            continue
        a = to_unit_rgb(imgs[1], size)
        b = to_unit_rgb(imgs[b_levels[0]], size)
        pairs.append((a, b))
        if len(pairs) >= n:
            break
    if not pairs:
        raise FileNotFoundError(f"No usable pairs under {test_dir}")
    return pairs


def _resolve_layers(group: str) -> List[str]:
    """'backbone.projector' or a range like 'backbone.layer.8..11' or comma list."""
    out: List[str] = []
    for tok in group.split(","):
        out.extend(expand_layer_spec(tok.strip()))
    return out


def _per_pair_reduction(
    model, filt: torch.nn.Module, a, b, layers, cfg: CalibrationConfig
) -> float:
    """Train filt on this single (A,B) pair; return (baseline-final)/baseline activation gap."""
    _, res = calibrate(filt, a, b, layers, model=model, cfg=cfg)
    base = res.__dict__.get("baseline_train")
    if not base or base <= 0:
        return float("nan")
    return float((base - res.final_train_loss) / base)


def run_oracle(args) -> None:
    model = load_model(size=args.size, device=args.device)
    dev = _model_device(model)
    layers = _resolve_layers(args.group)
    pairs = _load_test_pairs(Path(args.dataset), args.n_pairs, args.input_size)
    cfg = CalibrationConfig(
        learning_rate=args.lr, max_steps=args.steps, early_stopping_patience=args.steps
    )

    print(f"[oracle] {len(pairs)} pairs · layers={layers} · steps={args.steps}")
    print(f"[oracle] parametric filter = {args.filter}")
    print(f"{'pair':>4}  {'parametric':>11}  {'oracle(free-px)':>15}  {'headroom':>9}")
    par_all, orc_all = [], []
    for i, (a, b) in enumerate(pairs):
        par = build_filter(args.filter).to(dev)
        par_r = _per_pair_reduction(model, par, a, b, layers, cfg)
        orc = FreePixelFilter(size=args.input_size).to(dev)
        orc_r = _per_pair_reduction(model, orc, a, b, layers, cfg)
        par_all.append(par_r)
        orc_all.append(orc_r)
        print(f"{i:>4}  {par_r:>11.4f}  {orc_r:>15.4f}  {orc_r - par_r:>9.4f}")

    pm, om = statistics.mean(par_all), statistics.mean(orc_all)
    print("-" * 48)
    print(f"{'mean':>4}  {pm:>11.4f}  {om:>15.4f}  {om - pm:>9.4f}")
    print()
    if om <= 0:
        print("[verdict] oracle ~0 → no recoverable signal at this layer/metric. Rethink target/model.")
    elif (om - pm) > 0.15 and om > pm * 1.4:
        print("[verdict] oracle >> parametric → bottleneck is FILTER CAPACITY. A neural filter is worth building.")
    else:
        print("[verdict] oracle ~= parametric → bottleneck is the MODEL REPRESENTATION. A neural filter will NOT help; change layer/metric or model.")


def run_detection(args) -> None:
    if not args.checkpoint:
        raise SystemExit("--mode detection requires --checkpoint (a trained filter .pt) and --filter")
    model = load_model(size=args.size, device=args.device)
    dev = _model_device(model)
    layers = _resolve_layers(args.group)
    pairs = _load_test_pairs(Path(args.dataset), args.n_pairs, args.input_size)

    filt = build_filter(args.filter).to(dev)
    state = torch.load(args.checkpoint, map_location=dev)
    state = state.get("filter_state", state) if isinstance(state, dict) else state
    filt.load_state_dict(state)
    filt.eval()

    print(f"[detection] {len(pairs)} pairs · checkpoint={args.checkpoint}")
    print(f"{'pair':>4}  {'gap A-B(no filt)':>16}  {'gap A-fB':>10}  {'det_recovery':>12}  {'act_recovery':>12}")
    rec_all, act_all, base_all = [], [], []
    for i, (a, b) in enumerate(pairs):
        a_acts, a_det = _reference_with_detection(model, a, layers)
        # no-filter baseline
        identity = build_filter("affine_6param").to(dev)  # identity init = no-op
        b_acts0, b_det0 = _forward_filtered_full(model, identity, b, layers)
        gap0 = float(detection_output_loss(a_det, b_det0).detach())
        act0 = float(group_loss(a_acts, b_acts0, layers).detach())
        # with trained filter
        with torch.no_grad():
            b_acts1, b_det1 = _forward_filtered_full(model, filt, b, layers)
        gap1 = float(detection_output_loss(a_det, b_det1).detach())
        act1 = float(group_loss(a_acts, b_acts1, layers).detach())
        det_rec = (gap0 - gap1) / gap0 if gap0 > 0 else float("nan")
        act_rec = (act0 - act1) / act0 if act0 > 0 else float("nan")
        rec_all.append(det_rec)
        act_all.append(act_rec)
        base_all.append(gap0)
        print(f"{i:>4}  {gap0:>16.4f}  {gap1:>10.4f}  {det_rec:>12.4f}  {act_rec:>12.4f}")

    print("-" * 70)
    print(
        f"{'mean':>4}  baseline_gap={statistics.mean(base_all):.4f}  "
        f"det_recovery={statistics.mean(rec_all):.4f}  act_recovery={statistics.mean(act_all):.4f}"
    )
    print()
    print("[read] baseline_gap small → model barely degrades under the shift (premise weak for this model).")
    print("[read] det_recovery >> act_recovery → 33% activations already recovers most detection.")
    print("[read] det_recovery ~ 0 with positive act_recovery → projector target is misaligned with detection.")


def run_noise_floor(args) -> None:
    """A vs a SECOND same-illumination image of the same scene = irreducible metric noise.

    Uses the augmented train dir where the same (scene, level_1) appears under different
    geometric augmentations (scene_XXX_aug0 vs scene_XXX_augN). Falls back to comparing
    distinct scenes' A images if no aug pairs are found (an upper, not a floor, reference).
    """
    model = load_model(size=args.size, device=args.device)
    layers = _resolve_layers(args.group)
    train_dir = Path(args.dataset) / "train"
    # group scene_<id>_aug<k> dirs by scene id
    groups = {}
    for d in sorted(p for p in train_dir.iterdir() if p.is_dir()):
        sid = d.name.rsplit("_aug", 1)[0]
        groups.setdefault(sid, []).append(d)
    floor_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for sid, dirs in groups.items():
        a1 = next((d / "level_1.jpg" for d in dirs if (d / "level_1.jpg").exists()), None)
        a2 = next(
            (d / "level_1.jpg" for d in dirs[1:] if (d / "level_1.jpg").exists()), None
        )
        if a1 and a2 and a1 != a2:
            floor_pairs.append(
                (to_unit_rgb(a1, args.input_size), to_unit_rgb(a2, args.input_size))
            )
        if len(floor_pairs) >= args.n_pairs:
            break
    if not floor_pairs:
        raise SystemExit("No same-scene augmentation pairs found; cannot estimate noise floor.")

    dists = []
    for a1, a2 in floor_pairs:
        acts1 = compute_reference_activations(model, a1, layers)
        acts2 = _forward_filtered(model, build_filter("affine_6param").to(_model_device(model)), a2, layers)
        with torch.no_grad():
            d = float(group_loss(acts1, acts2, layers).detach())
        dists.append(d)
    print(f"[noise-floor] {len(dists)} same-scene/same-light pairs")
    print(f"[noise-floor] mean irreducible l2_rel = {statistics.mean(dists):.4f}")
    print("[read] This is the floor baked into every A↔B distance. The recoverable fraction")
    print("       is computed over (gap − floor), so 0.33 of the total is more of the recoverable part.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["oracle", "detection", "noise-floor"])
    ap.add_argument("--dataset", required=True, help="data/datasets/level_1_vs_level_2")
    ap.add_argument("--group", default="backbone.projector", help="layer group (range/comma ok)")
    ap.add_argument("--filter", default="spatial_tone_curve", help="parametric filter type")
    ap.add_argument("--checkpoint", default=None, help="trained filter .pt (detection mode)")
    ap.add_argument("--n-pairs", type=int, default=6)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--input-size", type=int, default=384)
    ap.add_argument("--size", default="n", help="model size n/s/m/l")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if args.mode == "oracle":
        run_oracle(args)
    elif args.mode == "detection":
        run_detection(args)
    else:
        run_noise_floor(args)


if __name__ == "__main__":
    main()
