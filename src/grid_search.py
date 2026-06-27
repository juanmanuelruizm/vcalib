"""Phase 2: Grid search over layer groups × filters × loss functions.

Resolves the candidate layer groups (from ``configs/grid.yaml`` via
``layer_groups.py``) and the filter specs, then for each (group, filter) combination
trains a fresh filter via ``calibration.py`` on the dev-train pairs and measures the
distance reduction on the held-out dev-val pairs. Configs that fail the overfit gate
(val reduction < min_ratio × train reduction) are flagged. Results are logged to
``results/runs.csv`` with a config hash for reproducibility.

Usage:
    uv run python src/grid_search.py --config configs/grid.yaml --dataset-path data/raw/scenes_YYYYMMDD/
    uv run python src/grid_search.py --config configs/grid.yaml --subset 5   # 5 quick runs
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from src.calibration import (
    CalibrationConfig,
    calibrate,
    overfit_gate_ok,
    train_reduction,
    val_reduction,
)
from src.filters import build_filter
from src.utils.data_pairs import Dataset, discover_pairs, load_pair_tensors
from src.utils.layer_groups import load_grid_config, resolve_grid_groups


def _config_hash(cfg: Mapping[str, Any]) -> str:
    """Stable hash of the grid config for reproducibility logging."""
    raw = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.md5(raw).hexdigest()[:12]


def _filter_name(spec: Any) -> str:
    """Short human-readable name for a filter spec."""
    if isinstance(spec, str):
        return spec
    if isinstance(spec, (list, tuple)):
        return "composite[" + ",".join(spec) + "]"
    if isinstance(spec, dict):
        if "composite" in spec:
            return "composite[" + ",".join(spec["composite"]) + "]"
        name = str(spec.get("type", "?"))
        extras = {k: v for k, v in spec.items() if k not in ("type", "params")}
        if extras:
            return f"{name}({','.join(f'{k}={v}' for k, v in extras.items())})"
        return name
    return str(spec)


def run_grid(
    config: Mapping[str, Any],
    dataset: Dataset,
    subset: Optional[int] = None,
    size: str = "n",
    device: str = "auto",
    output_csv: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run the full grid sweep. Returns a list of run-result dicts (also written to CSV)."""
    from src.utils.activations import load_model

    groups = resolve_grid_groups(config)
    filter_specs = config.get("grid", {}).get("filters", [])
    loss_fns = config.get("grid", {}).get("loss_functions", ["l2"])
    train_cfg = config.get("training", {})
    val_cfg = config.get("validation", {})
    reg_weight = float(train_cfg.get("reg_weight", 0.0))
    gate_cfg = val_cfg.get("overfit_gate", {})
    gate_enabled = bool(gate_cfg.get("enabled", False))
    min_ratio = float(gate_cfg.get("min_val_recovery_ratio", 0.5))

    cfg_hash = _config_hash(config)
    model = load_model(size=size, device=device)

    # Build the full grid
    combos: List[tuple] = []
    for group in groups:
        for spec in filter_specs:
            for loss_fn in loss_fns:
                combos.append((group, spec, loss_fn))
    if subset and subset > 0:
        combos = combos[:subset]
    total = len(combos)
    print(
        f"[grid] {total} combos ({len(groups)} groups × {len(filter_specs)} filters × {len(loss_fns)} loss)"
    )
    print(f"[grid] reg_weight={reg_weight}, overfit_gate={gate_enabled} (min_ratio={min_ratio})")

    # Prepare CSV
    csv_path = (
        Path(output_csv)
        if output_csv
        else Path(config.get("output", {}).get("results_csv", "results/runs.csv"))
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "group_name",
        "filter",
        "loss",
        "n_layers",
        "final_train_loss",
        "final_val_loss",
        "train_reduction",
        "val_reduction",
        "val_train_ratio",
        "overfit_gate",
        "steps",
        "wall_s",
        "timestamp",
        "config_hash",
    ]

    results: List[Dict[str, Any]] = []
    t0 = time.time()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if f.tell() == 0:
            writer.writeheader()

        for idx, (group, spec, loss_fn) in enumerate(combos):
            run_id = f"run_{idx:04d}"
            fname = _filter_name(spec)
            print(f"\n[grid] {idx + 1}/{total}: group={group.name} filter={fname} loss={loss_fn}")

            # Build a fresh filter (identity init)
            filt = build_filter(spec)

            # Calibration config from YAML
            cfg = CalibrationConfig(
                learning_rate=float(train_cfg.get("learning_rate", 1e-3)),
                max_steps=int(train_cfg.get("max_steps", 100)),
                early_stopping_patience=int(train_cfg.get("early_stopping_patience", 10)),
                reg_weight=reg_weight,
                metric="l2_rel" if loss_fn == "l2" else "cosine",
                seed=int(val_cfg.get("random_seed", 42)),
                log_every=0,
            )

            # Use the first train pair for calibration, first val pair for val
            # (single-scene calibration, as per the spec)
            train_pairs = dataset.train_pairs
            val_pairs = dataset.val_pairs if dataset.val_pairs else train_pairs[:1]

            if not train_pairs:
                print("[grid] ⚠ no train pairs — skipping")
                continue

            a_unit, b_unit = load_pair_tensors(train_pairs[0], input_size=384)
            val_a, val_b = (None, None)
            if val_pairs:
                val_a, val_b = load_pair_tensors(val_pairs[0], input_size=384)

            try:
                trained, result = calibrate(
                    filt,
                    a_unit,
                    b_unit,
                    group.layers,
                    model=model,
                    cfg=cfg,
                    val_a_unit=val_a,
                    val_b_unit=val_b,
                )
            except Exception as exc:
                print(f"[grid] ⚠ calibration failed: {exc}")
                continue

            tr = train_reduction(result)
            vr = val_reduction(result)
            ratio = (vr / tr) if (tr and tr > 0 and vr is not None) else None
            gate_ok = overfit_gate_ok(result, min_ratio=min_ratio) if gate_enabled else None

            row = {
                "run_id": run_id,
                "group_name": group.name,
                "filter": fname,
                "loss": loss_fn,
                "n_layers": len(group.layers),
                "final_train_loss": f"{result.final_train_loss:.6f}",
                "final_val_loss": f"{result.final_val_loss:.6f}"
                if result.final_val_loss is not None
                else "",
                "train_reduction": f"{tr:.4f}" if tr is not None else "",
                "val_reduction": f"{vr:.4f}" if vr is not None else "",
                "val_train_ratio": f"{ratio:.4f}" if ratio is not None else "",
                "overfit_gate": str(gate_ok) if gate_ok is not None else "n/a",
                "steps": result.steps,
                "wall_s": f"{result.wall_clock_s:.1f}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "config_hash": cfg_hash,
            }
            writer.writerow(row)
            f.flush()
            results.append(row)
            gate_tag = "✓" if gate_ok else "✗" if gate_ok is False else "—"
            print(
                f"[grid] → train={tr:.3f} val={vr:.3f} ratio={ratio:.3f} gate={gate_tag} steps={result.steps}"
            )

    elapsed = time.time() - t0
    print(f"\n[grid] Done: {len(results)} runs in {elapsed:.1f}s → {csv_path}")

    # Rank by val_reduction (descending), filter gate-passed
    gate_passed = [r for r in results if r.get("overfit_gate") == "True"]
    print(f"[grid] {len(gate_passed)}/{len(results)} configs passed the overfit gate")
    if gate_passed:
        gate_passed.sort(key=lambda r: float(r.get("val_reduction", "0")), reverse=True)
        print("[grid] Top 3 by val_reduction (gate-passed):")
        for r in gate_passed[:3]:
            print(
                f"  {r['group_name']:28s} × {r['filter']:30s} val={r['val_reduction']} steps={r['steps']}"
            )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2: Grid search")
    parser.add_argument("--config", type=str, default="configs/grid.yaml")
    parser.add_argument("--dataset-path", type=str, default="data/raw/")
    parser.add_argument("--output", type=str, default=None, help="Override results CSV path")
    parser.add_argument(
        "--subset", type=int, default=None, help="Run only the first N combos (quick check)"
    )
    parser.add_argument("--val-split", type=float, default=None, help="Override val split fraction")
    parser.add_argument("--seed", type=int, default=None, help="Override split seed")
    parser.add_argument("--size", type=str, default="n")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    config = load_grid_config(args.config)
    val_split = (
        args.val_split
        if args.val_split is not None
        else float(config.get("validation", {}).get("split", 0.2))
    )
    seed = (
        args.seed
        if args.seed is not None
        else int(config.get("validation", {}).get("random_seed", 42))
    )

    dataset = discover_pairs(args.dataset_path, val_split=val_split, seed=seed)
    print(f"[grid] Dataset: {dataset}")
    if not dataset.pairs:
        print("[grid] ⚠ No pairs found. Capture the dev dataset first (see tasks/todo.md A7).")
        return

    run_grid(
        config,
        dataset,
        subset=args.subset,
        size=args.size,
        device=args.device,
        output_csv=args.output,
    )


if __name__ == "__main__":
    main()
