"""Train filters on the real A/B datasets using ALL train pairs.

.. deprecated::
    Superseded by ``run_configs.py`` + ``configs/experiments/*.yaml`` (config-driven
    experiment runner). This file is retained for backwards compatibility; new
    experiments should be defined as YAML configs and run via ``run_configs.py``.
    See ``docs/specs/2026-06-28-config-driven-experiments.md``.

Two independent experiments:
  - level_1_vs_level_2 (milder shift)
  - level_1_vs_level_3 (bigger shift)

For each experiment x filter x layer_group:
  1. Load ALL train pairs as (A, B) tensors
  2. Calibrate with calibrate_epochs (epoch-based loop)
  3. Evaluate on test pairs (mean + std reduction)

Outputs (all under results/experiments/):
  experiment_results_multi.csv         — final summary, one row per run
  runs/<exp>/<group>__<filter>/
    metrics.jsonl                      — one JSON line per epoch: epoch, train_loss, val_loss
    best.pt                            — filter state_dict at best val epoch
    epoch_NNNN.pt                      — periodic checkpoints every CHECKPOINT_EVERY epochs

Usage:
  uv run python run_experiments.py
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from src.calibration import (
    CalibrationConfig,
    calibrate_epochs,
    train_reduction,
    _forward_filtered,
    compute_reference_activations,
    group_loss,
)
from src.filters import build_filter, get_filter
from src.utils.activations import load_model, to_unit_rgb

DATASETS_ROOT = Path("data/datasets")
RESULTS_DIR = Path("results/experiments")
RUNS_DIR = RESULTS_DIR / "runs"

MAX_EPOCHS = 20
LR = 5e-3
REG_WEIGHT = 0.01
INPUT_SIZE = 384
CHECKPOINT_EVERY = 5  # save checkpoint every N epochs

FILTERS = [
    {"type": "brightness_2param"},
    {"type": "affine_6param"},
    {"type": "matrix_12param"},
    {"type": "gamma_3param"},
    {"type": "chromatic_adaptation"},
    {"type": "ccm_high_order"},
    {"type": "tone_curve", "P": 16},
    {"type": "lut_3d", "size": 9},
    {"type": "spatial_tone_curve", "P": 8, "grid_size": 3},
    {"type": "local_tonemap", "grid_size": 4},
]

LAYER_GROUPS = {
    "backbone.early": ["backbone.layer.0", "backbone.layer.1", "backbone.layer.2", "backbone.layer.3"],
    "backbone.all": [f"backbone.layer.{i}" for i in range(12)],
    "projector": ["backbone.projector"],
}


def filter_name(spec: Dict) -> str:
    name = spec["type"]
    extras = {k: v for k, v in spec.items() if k != "type"}
    if extras:
        return f"{name}({','.join(f'{k}={v}' for k, v in extras.items())})"
    return name


def discover_pairs(exp_dir: Path) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    """Return (train_pairs, test_pairs) as lists of (A_path, B_path)."""
    b_level = "level_2" if "level_2" in exp_dir.name else "level_3"

    def _scan(sub: str) -> List[Tuple[Path, Path]]:
        d = exp_dir / sub
        pairs = []
        for pair_dir in sorted(d.iterdir()):
            if not pair_dir.is_dir():
                continue
            a = pair_dir / "level_1.jpg"
            b = pair_dir / f"{b_level}.jpg"
            if a.exists() and b.exists():
                pairs.append((a, b))
        return pairs

    return _scan("train"), _scan("test")


def load_all_pairs(paths: List[Tuple[Path, Path]]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    pairs = []
    for a_path, b_path in paths:
        a = to_unit_rgb(a_path, INPUT_SIZE)
        b = to_unit_rgb(b_path, INPUT_SIZE)
        pairs.append((a, b))
    return pairs


def evaluate_on_test(
    filt: torch.nn.Module,
    model: Any,
    test_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    layer_names: List[str],
) -> Dict[str, Any]:
    import numpy as np

    per_pair = []
    for a_unit, b_unit in test_pairs:
        a_acts = compute_reference_activations(model, a_unit, layer_names)

        identity = get_filter("brightness_2param")
        identity.to(next(model.model.parameters()).device)
        b_acts_base = _forward_filtered(model, identity, b_unit, layer_names)
        base_loss = float(group_loss(a_acts, b_acts_base, layer_names))

        b_acts_filt = _forward_filtered(model, filt, b_unit, layer_names)
        filt_loss = float(group_loss(a_acts, b_acts_filt, layer_names))

        red = (base_loss - filt_loss) / base_loss if base_loss > 0 else 0.0
        per_pair.append({"baseline": base_loss, "filtered": filt_loss, "reduction": red})

    reductions = [p["reduction"] for p in per_pair]
    return {
        "per_pair": per_pair,
        "mean": float(np.mean(reductions)),
        "std": float(np.std(reductions)),
        "n": len(per_pair),
    }


def _make_epoch_callback(
    run_dir: Path,
    checkpoint_every: int,
) -> Callable[[int, float, Optional[float], torch.nn.Module], None]:
    """Build an on_epoch_end callback that writes metrics.jsonl and periodic checkpoints."""
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"

    def on_epoch_end(epoch: int, train_loss: float, val_loss: Optional[float], filt: torch.nn.Module) -> None:
        record: Dict[str, Any] = {"epoch": epoch, "train_loss": train_loss}
        if val_loss is not None:
            record["val_loss"] = val_loss
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        if epoch % checkpoint_every == 0:
            ckpt_path = run_dir / f"epoch_{epoch:04d}.pt"
            torch.save({k: v.cpu() for k, v in filt.state_dict().items()}, ckpt_path)

    return on_epoch_end


def run_experiment(exp_name: str, exp_dir: Path, model: Any) -> List[Dict]:
    train_paths, test_paths = discover_pairs(exp_dir)
    print(f"\n{'='*70}")
    print(f"Experiment: {exp_name}")
    print(f"  Train: {len(train_paths)} pairs | Test: {len(test_paths)} pairs")
    print(f"{'='*70}")

    if not train_paths or not test_paths:
        print("  [SKIP] No pairs found")
        return []

    print(f"  Loading {len(train_paths)} train pairs...")
    train_pairs = load_all_pairs(train_paths)
    print(f"  Loading {len(test_paths)} test pairs...")
    test_pairs = load_all_pairs(test_paths)

    results = []
    total = len(FILTERS) * len(LAYER_GROUPS)
    idx = 0

    for group_name, layer_names in LAYER_GROUPS.items():
        for spec in FILTERS:
            idx += 1
            fname = filter_name(spec)
            run_name = f"{group_name}__{fname}"
            run_dir = RUNS_DIR / exp_name / run_name
            print(f"\n  [{idx}/{total}] group={group_name} filter={fname}")

            filt = build_filter(spec)
            cfg = CalibrationConfig(
                max_epochs=MAX_EPOCHS,
                early_stopping_patience=5,
                learning_rate=LR,
                reg_weight=REG_WEIGHT,
                log_every=5,
                seed=42,
            )

            try:
                trained, result = calibrate_epochs(
                    filt, train_pairs, layer_names,
                    model=model, cfg=cfg,
                    test_pairs=test_pairs,
                    on_epoch_end=_make_epoch_callback(run_dir, CHECKPOINT_EVERY),
                )
            except Exception as exc:
                print(f"    [FAIL] {exc}")
                continue

            # Save best model
            best_path = run_dir / "best.pt"
            torch.save({k: v.cpu() for k, v in result.filter_state.items()}, best_path)

            test_eval = evaluate_on_test(trained, model, test_pairs, layer_names)
            train_red = train_reduction(result)

            row = {
                "experiment": exp_name,
                "group": group_name,
                "filter": fname,
                "train_reduction": f"{train_red:.4f}" if train_red else "0.0",
                "test_mean": f"{test_eval['mean']:.4f}",
                "test_std": f"{test_eval['std']:.4f}",
                "test_n": test_eval["n"],
                "steps": result.steps,
                "wall_s": f"{result.wall_clock_s:.1f}",
                "converged": result.converged,
                "per_pair_reductions": ";".join(f"{p['reduction']:.4f}" for p in test_eval["per_pair"]),
            }
            results.append(row)

            ratio = test_eval["mean"] / train_red if (train_red and train_red > 0) else 0
            print(
                f"    -> train_red={train_red:.3f} test_mean={test_eval['mean']:.3f}"
                f"+/-{test_eval['std']:.3f} ratio={ratio:.2f} steps={result.steps} {result.wall_clock_s:.1f}s"
            )

    return results


def print_full_table(results: List[Dict], exp_name: str) -> None:
    exp = [r for r in results if r["experiment"] == exp_name]
    if not exp:
        return
    print(f"\n{'='*90}")
    print(f"FULL RESULTS: {exp_name}")
    print(f"{'='*90}")
    print(f"{'group':<18} {'filter':<40} {'train':>7} {'test_mean':>10} {'test_std':>9} {'ratio':>7} {'steps':>6}")
    print("-" * 90)
    exp_sorted = sorted(exp, key=lambda r: float(r["test_mean"]), reverse=True)
    for r in exp_sorted:
        tr = float(r["train_reduction"])
        tm = float(r["test_mean"])
        ts = float(r["test_std"])
        ratio = tm / tr if tr > 0 else 0
        print(f"{r['group']:<18} {r['filter']:<40} {tr:>7.3f} {tm:>10.3f} {ts:>9.3f} {ratio:>7.2f} {r['steps']:>6}")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading RF-DETR nano model...")
    model = load_model(size="n")
    print("Model loaded.\n")

    all_results = []
    experiments = [
        ("level_1_vs_level_2", DATASETS_ROOT / "level_1_vs_level_2"),
        ("level_1_vs_level_3", DATASETS_ROOT / "level_1_vs_level_3"),
    ]

    for exp_name, exp_dir in experiments:
        if not exp_dir.exists():
            print(f"[SKIP] {exp_dir} not found")
            continue
        results = run_experiment(exp_name, exp_dir, model)
        all_results.extend(results)

    csv_path = RESULTS_DIR / "experiment_results_multi.csv"
    fieldnames = [
        "experiment", "group", "filter", "train_reduction",
        "test_mean", "test_std", "test_n", "steps", "wall_s", "converged", "per_pair_reductions",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)
    print(f"\nResults saved to {csv_path}")
    print(f"Per-epoch metrics in {RUNS_DIR}/<exp>/<group>__<filter>/metrics.jsonl")

    for exp_name, _ in experiments:
        print_full_table(all_results, exp_name)


if __name__ == "__main__":
    main()
