"""Generate sweep-2 experiment YAML configs.

Focused sweep based on sweep-1 findings:
- Only the 4 best injection groups (projector, proj+decoder, backbone.late+proj, backbone.mid)
- New filters: lut_3d_lowrank, and three composite filter chains
- Better hyperparameters: lower LR, more epochs/patience, cosine scheduler

Usage:
    python3 scripts/generate_sweep2_configs.py
    python3 scripts/generate_sweep2_configs.py --out configs/experiments/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import yaml

DATASETS: List[Dict[str, str]] = [
    {"name": "level2", "path": "data/datasets/level_1_vs_level_2"},
    {"name": "level3", "path": "data/datasets/level_1_vs_level_3"},
]

FILTERS: List[Dict[str, Any]] = [
    # Already-implemented filter not tested in sweep 1
    {"type": "lut_3d_lowrank", "M": 32, "display": "lut3d_lowrank"},
    # Composite chains: linear + non-linear correction
    {"composite": ["affine_6param", "gamma_3param"], "display": "composite_affine_gamma"},
    {"composite": ["chromatic_adaptation", "tone_curve"], "display": "composite_chromatic_tone"},
    {"composite": ["matrix_12param", "tone_curve"], "display": "composite_matrix_tone"},
]

# Only the 4 best-performing groups from sweep 1
LAYER_GROUPS: List[Dict[str, Any]] = [
    {"name": "projector", "layers": ["backbone.projector"], "display": "projector"},
    {
        "name": "proj+decoder",
        "layers": ["backbone.projector", "decoder.layer.0", "decoder.layer.1"],
        "display": "proj_decoder",
    },
    {
        "name": "backbone.late+proj",
        "layers": [f"backbone.layer.{i}" for i in range(8, 12)] + ["backbone.projector"],
        "display": "backbone_late_proj",
    },
    {
        "name": "backbone.mid",
        "layers": [f"backbone.layer.{i}" for i in range(4, 8)],
        "display": "backbone_mid",
    },
]

TRAINING: Dict[str, Any] = {
    "max_epochs": 80,
    "learning_rate": 0.001,
    "reg_weight": 0.005,
    "early_stopping_patience": 15,
    "seed": 42,
    "checkpoint_every": 10,
    "scheduler": "cosine",
    "scheduler_kwargs": {"T_max": 80, "eta_min": 1e-6},
}


def build_config(
    dataset: Dict[str, str],
    filt: Dict[str, Any],
    group: Dict[str, Any],
) -> Dict[str, Any]:
    if "composite" in filt:
        filter_spec: Dict[str, Any] = {"composite": filt["composite"]}
    else:
        filter_spec = {k: v for k, v in filt.items() if k != "display"}
    return {
        "name": f"{dataset['name']}_{filt['display']}_{group['display']}_v2",
        "dataset": dataset["path"],
        "input_size": 384,
        "filter": filter_spec,
        "layer_group": {"name": group["name"], "layers": group["layers"]},
        "training": dict(TRAINING),
        "output": {"results_dir": "results/experiments"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sweep-2 experiment YAML configs.")
    parser.add_argument("--out", type=str, default="configs/experiments/")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for dataset in DATASETS:
        for filt in FILTERS:
            for group in LAYER_GROUPS:
                cfg = build_config(dataset, filt, group)
                path = out_dir / f"{cfg['name']}.yaml"
                with open(path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
                print(f"  {cfg['name']}")
                total += 1

    print(f"\nGenerated {total} configs in {out_dir}")
    print(f"  Filters: {len(FILTERS)}")
    print(f"  Groups:  {len(LAYER_GROUPS)}")
    print(f"  Datasets: {len(DATASETS)}")


if __name__ == "__main__":
    main()
