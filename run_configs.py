"""Config-driven experiment runner.

Reads one or more experiment YAML configs (each = one atomic training run), executes
them with the calibration loop, and aggregates every run into one CSV.

Usage::

    uv run python run_configs.py configs/experiments/
    uv run python run_configs.py configs/experiments/foo.yaml
    uv run python run_configs.py a.yaml b.yaml --dry-run
    uv run python run_configs.py configs/experiments/ --append --device cuda

See ``docs/specs/2026-06-28-config-driven-experiments.md`` for the schema and
``configs/experiments/`` for starter configs.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from src.calibration import (
    CalibrationConfig,
    calibrate_epochs,
    compute_reference_activations,
    _forward_filtered,
    group_loss,
    train_reduction,
)
from src.experiment_config import ExperimentConfig, load_configs, summarize
from src.filters import build_filter, get_filter
from src.utils.activations import load_model, to_unit_rgb


def discover_pairs(exp_dir: Path) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    """Return (train_pairs, test_pairs) as lists of (A_path, B_path).

    Auto-detects the B level from the directory name (``level_1_vs_level_N`` ->
    ``level_N``). Each pair directory must contain ``level_1.jpg`` and ``level_N.jpg``.
    """
    # Parse the B level from the directory name (``level_1_vs_level_N``).
    # Splitting on ``_`` cannot work (it would split ``level_3`` into ``level`` and
    # ``3``), so use a regex across the full name and pick the highest level != 1.
    import re

    levels = [int(n) for n in re.findall(r"level_(\d+)", exp_dir.name)]
    b_candidates = sorted((l for l in levels if l != 1), reverse=True)
    b_level = f"level_{b_candidates[0]}" if b_candidates else "level_2"

    def _scan(sub: str) -> List[Tuple[Path, Path]]:
        d = exp_dir / sub
        if not d.is_dir():
            return []
        pairs: List[Tuple[Path, Path]] = []
        for pair_dir in sorted(d.iterdir()):
            if not pair_dir.is_dir():
                continue
            a = pair_dir / "level_1.jpg"
            b = pair_dir / f"{b_level}.jpg"
            if a.exists() and b.exists():
                pairs.append((a, b))
        return pairs

    return _scan("train"), _scan("test")


def load_all_pairs(
    paths: List[Tuple[Path, Path]], input_size: int
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    return [(to_unit_rgb(a, input_size), to_unit_rgb(b, input_size)) for a, b in paths]


def evaluate_on_test(
    filt: torch.nn.Module,
    model: Any,
    test_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    layer_names: List[str],
) -> Dict[str, Any]:
    import numpy as np

    per_pair: List[Dict[str, float]] = []
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
        "mean": float(np.mean(reductions)) if reductions else 0.0,
        "std": float(np.std(reductions)) if reductions else 0.0,
        "n": len(per_pair),
    }


def make_epoch_callback(
    run_dir: Path, checkpoint_every: int
) -> Callable[[int, float, Optional[float], torch.nn.Module], None]:
    """Build an on_epoch_end callback writing metrics.jsonl + periodic checkpoints."""
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"

    def on_epoch_end(
        epoch: int, train_loss: float, val_loss: Optional[float], filt: torch.nn.Module
    ) -> None:
        record: Dict[str, Any] = {"epoch": epoch, "train_loss": train_loss}
        if val_loss is not None:
            record["val_loss"] = val_loss
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        if checkpoint_every > 0 and epoch % checkpoint_every == 0:
            ckpt_path = run_dir / f"epoch_{epoch:04d}.pt"
            torch.save({k: v.cpu() for k, v in filt.state_dict().items()}, ckpt_path)

    return on_epoch_end


CSV_FIELDS = [
    "config",
    "dataset",
    "group",
    "filter",
    "train_reduction",
    "test_mean",
    "test_std",
    "test_n",
    "steps",
    "wall_s",
    "converged",
    "per_pair_reductions",
]


def run_one(
    cfg: ExperimentConfig,
    model: Any,
) -> Dict[str, Any]:
    """Execute a single experiment config and return a CSV row dict."""
    train_paths, test_paths = discover_pairs(cfg.dataset)
    if not train_paths or not test_paths:
        raise RuntimeError(f"Config {cfg.name!r}: no train/test pairs under {cfg.dataset!s}")

    print(f"\n{'=' * 70}")
    print(
        f"Config: {cfg.name}  (dataset={cfg.dataset.name}, filter={cfg.filter_display}, group={cfg.group_name})"
    )
    print(f"  Train: {len(train_paths)} pairs | Test: {len(test_paths)} pairs")
    print(f"{'=' * 70}")

    train_pairs = load_all_pairs(train_paths, cfg.input_size)
    test_pairs = load_all_pairs(test_paths, cfg.input_size)

    run_dir = cfg.output_results_dir / "runs" / cfg.name

    filt = build_filter(cfg.filter_spec)
    cal_cfg = CalibrationConfig(
        max_epochs=cfg.training.max_epochs,
        early_stopping_patience=cfg.training.early_stopping_patience,
        learning_rate=cfg.training.learning_rate,
        reg_weight=cfg.training.reg_weight,
        seed=cfg.training.seed,
        log_every=5,
    )

    trained, result = calibrate_epochs(
        filt,
        train_pairs,
        cfg.group_layers,
        model=model,
        cfg=cal_cfg,
        test_pairs=test_pairs,
        on_epoch_end=make_epoch_callback(run_dir, cfg.training.checkpoint_every),
    )

    best_path = run_dir / "best.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in result.filter_state.items()}, best_path)

    test_eval = evaluate_on_test(trained, model, test_pairs, cfg.group_layers)
    tr = train_reduction(result)

    row = {
        "config": cfg.name,
        "dataset": cfg.dataset.name,
        "group": cfg.group_name,
        "filter": cfg.filter_display,
        "train_reduction": f"{tr:.4f}" if tr else "0.0",
        "test_mean": f"{test_eval['mean']:.4f}",
        "test_std": f"{test_eval['std']:.4f}",
        "test_n": test_eval["n"],
        "steps": result.steps,
        "wall_s": f"{result.wall_clock_s:.1f}",
        "converged": result.converged,
        "per_pair_reductions": ";".join(f"{p['reduction']:.4f}" for p in test_eval["per_pair"]),
    }

    ratio = test_eval["mean"] / tr if (tr and tr > 0) else 0.0
    print(
        f"  -> train_red={tr or 0.0:.3f} test_mean={test_eval['mean']:.3f}"
        f"+/-{test_eval['std']:.3f} ratio={ratio:.2f} steps={result.steps} {result.wall_clock_s:.1f}s"
    )
    return row


def write_csv(rows: List[Dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# --- Parallel worker infrastructure -------------------------------------------------
# Each worker subprocess loads its own RF-DETR model instance ONCE (in _worker_init),
# then processes every config it's handed via _worker_run_one. Threading is NOT viable
# because LibreYOLO's forward hooks register onto the module globally; concurrent calls
# on a shared model would clobber activations between threads. So we use processes.

_WORKER_DEVICE: Optional[str] = None
_WORKER_MODEL: Any = None


def _worker_init(device: Optional[str], worker_idx: int) -> None:
    """Pool initializer: load the model once per worker subprocess.

    ``worker_idx`` helps disambiguate processes in logs; set the multiprocessing
    "current process" name so error messages are attributable.
    """
    global _WORKER_DEVICE, _WORKER_MODEL
    _WORKER_DEVICE = device
    mp.current_process().name = f"worker-{worker_idx}"
    if _WORKER_MODEL is None:
        # Reduce OpenMP thread contention when N processes share the same physical CPU.
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        print(f"  [{mp.current_process().name}] loading RF-DETR nano...", flush=True)
        _WORKER_MODEL = load_model(size="n", device=device)
        print(f"  [{mp.current_process().name}] model loaded.", flush=True)


def _worker_run_one(cfg: ExperimentConfig) -> Dict[str, Any]:
    """Run one config inside a worker process; return the row dict (never raises)."""
    global _WORKER_MODEL
    assert _WORKER_MODEL is not None, "worker_run_one called before worker_init"
    try:
        return run_one(cfg, _WORKER_MODEL)
    except Exception as exc:
        print(
            f"  [{mp.current_process().name}] [FAIL] {cfg.name}: {exc}", file=sys.stderr, flush=True
        )
        return {
            "config": cfg.name,
            "dataset": cfg.dataset.name,
            "group": cfg.group_name,
            "filter": cfg.filter_display,
            "train_reduction": "0.0",
            "test_mean": "0.0",
            "test_std": "0.0",
            "test_n": 0,
            "steps": 0,
            "wall_s": "0.0",
            "converged": False,
            "per_pair_reductions": f"ERROR: {exc!s}",
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run config-driven experiments. Each YAML = one training run."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["configs/experiments/"],
        help="One or more YAML files or directories (default: configs/experiments/).",
    )
    parser.add_argument("--output", type=str, default=None, help="Override results dir.")
    parser.add_argument("--dry-run", action="store_true", help="Validate + print, no training.")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing CSV instead of overwriting.",
    )
    parser.add_argument("--device", type=str, default=None, help="cuda | cpu (auto if omitted).")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Parallel worker subprocesses (each loads its own model). "
            "Default 0 = sequential; >=1 enables multiprocessing. "
            "RTX 5070 Ti (16GB) -> 4 workers (~12GB peak) is a safe cap."
        ),
    )
    args = parser.parse_args()

    try:
        configs = load_configs(args.paths, check_dataset=not args.dry_run)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if not configs:
        print("No experiment configs found.", file=sys.stderr)
        return 1

    if args.output:
        out_dir = Path(args.output)
        configs = [
            type(c)(
                name=c.name,
                dataset=c.dataset,
                input_size=c.input_size,
                filter_spec=c.filter_spec,
                group_name=c.group_name,
                group_layers=c.group_layers,
                training=c.training,
                output_results_dir=out_dir,
                config_file=c.config_file,
            )
            for c in configs
        ]

    if args.dry_run:
        print(f"Found {len(configs)} config(s):")
        for i, c in enumerate(configs, 1):
            print(f"  [{i}] {summarize(c)}")
        print(f"\nTotal runs: {len(configs)}")
        print("Dry run - no training. Exiting.")
        return 0

    # Decide parallelism. If --workers is 0/None -> sequential. Cap to len(configs).
    workers = args.workers if args.workers and args.workers > 0 else 0
    workers = min(workers, len(configs))

    csv_path = configs[0].output_results_dir / "experiment_results.csv"
    existing_rows: List[Dict[str, Any]] = []
    if args.append and csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            existing_rows = [
                {k: v for k, v in dict(r).items() if k in CSV_FIELDS} for r in csv.DictReader(f)
            ]

    if workers <= 1:
        print("Loading RF-DETR nano model (sequential mode)...")
        model = load_model(size="n", device=args.device)
        print("Model loaded.\n")

        rows: List[Dict[str, Any]] = list(existing_rows)
        total = len(configs)
        for i, cfg in enumerate(configs, 1):
            print(f"\n[{i}/{total}]")
            try:
                row = run_one(cfg, model)
            except Exception as exc:
                print(f"  [FAIL] {cfg.name}: {exc}", file=sys.stderr)
                continue
            rows.append(row)
            write_csv(rows, csv_path)  # incremental flush (rewrites the file)
    else:
        print(f"Launching {workers} worker subprocesses (parallel mode)...")
        # Use "spawn" reliably on Windows (Python's default on win32 is already spawn;
        # set explicitly for cross-platform reproducibility).
        ctx = mp.get_context("spawn")
        with ctx.Pool(  # type: ignore[assignment]
            processes=workers,
            initializer=_worker_init,
            initargs=(args.device, 0),
        ) as pool:
            # chunksize=1 keeps the work evenly spread (configs vary in runtime a lot).
            worker_rows = pool.map(_worker_run_one, configs, chunksize=1)
        rows = [*existing_rows, *worker_rows]
        write_csv(rows, csv_path)

    print(f"\nResults saved to {csv_path}")
    if rows:
        valid_rows = [
            r
            for r in rows
            if "config" in r and not str(r.get("per_pair_reductions", "")).startswith("ERROR")
        ]
        if valid_rows:
            print_results_table(valid_rows)
        n_fail = sum(1 for r in rows if str(r.get("per_pair_reductions", "")).startswith("ERROR"))
        if n_fail:
            print(f"\n[!] {n_fail} run(s) failed — see stderr above.")
    return 0


def print_results_table(rows: List[Dict[str, Any]]) -> None:
    print(f"\n{'=' * 100}")
    print("FULL RESULTS (sorted by test_mean desc)")
    print(f"{'=' * 100}")
    print(
        f"{'config':<40} {'group':<16} {'filter':<28} {'train':>7} "
        f"{'test_mean':>10} {'test_std':>9} {'ratio':>7} {'steps':>6}"
    )
    print("-" * 100)
    rows_sorted = sorted(rows, key=lambda r: float(r["test_mean"]), reverse=True)
    for r in rows_sorted:
        tr = float(r["train_reduction"])
        tm = float(r["test_mean"])
        ts = float(r["test_std"])
        ratio = tm / tr if tr > 0 else 0.0
        print(
            f"{r['config']:<40} {r['group']:<16} {r['filter']:<28} {tr:>7.3f} "
            f"{tm:>10.3f} {ts:>9.3f} {ratio:>7.2f} {r['steps']:>6}"
        )


if __name__ == "__main__":
    sys.exit(main())
