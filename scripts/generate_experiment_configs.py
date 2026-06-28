"""Generate starter experiment YAML configs.

Emits one file per (dataset x filter x layer_group) combo, replicating the sweep
that was hardcoded in ``run_experiments.py`` on the ``epochs`` branch. Idempotent:
re-running overwrites the files in ``configs/experiments/`` with the same content.

Usage:
    uv run python scripts/generate_experiment_configs.py
    uv run python scripts/generate_experiment_configs.py --out configs/experiments/
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
    {"type": "brightness_2param", "display": "brightness"},
    {"type": "affine_6param", "display": "affine"},
    {"type": "matrix_12param", "display": "matrix"},
    {"type": "gamma_3param", "display": "gamma"},
    {"type": "chromatic_adaptation", "display": "chromatic"},
    {"type": "ccm_high_order", "display": "ccm"},
    {"type": "tone_curve", "P": 16, "display": "tone_curve"},
    {"type": "lut_3d", "size": 9, "display": "lut3d"},
    {"type": "spatial_tone_curve", "P": 8, "grid_size": 3, "display": "spatial_tone"},
    {"type": "local_tonemap", "grid_size": 4, "display": "local_tonemap"},
]

LAYER_GROUPS: List[Dict[str, Any]] = [
    {
        "name": "backbone.early",
        "layers": ["backbone.layer.0", "backbone.layer.1", "backbone.layer.2", "backbone.layer.3"],
        "display": "backbone_early",
    },
    {
        "name": "backbone.mid",
        "layers": [f"backbone.layer.{i}" for i in range(4, 8)],
        "display": "backbone_mid",
    },
    {
        "name": "backbone.late",
        "layers": [f"backbone.layer.{i}" for i in range(8, 12)],
        "display": "backbone_late",
    },
    {
        "name": "backbone.all",
        "layers": [f"backbone.layer.{i}" for i in range(12)],
        "display": "backbone_all",
    },
    {"name": "projector", "layers": ["backbone.projector"], "display": "projector"},
    {
        "name": "backbone.early+proj",
        "layers": [
            "backbone.layer.0", "backbone.layer.1", "backbone.layer.2", "backbone.layer.3",
            "backbone.projector",
        ],
        "display": "backbone_early_proj",
    },
    {
        "name": "backbone.mid+proj",
        "layers": [f"backbone.layer.{i}" for i in range(4, 8)] + ["backbone.projector"],
        "display": "backbone_mid_proj",
    },
    {
        "name": "backbone.late+proj",
        "layers": [f"backbone.layer.{i}" for i in range(8, 12)] + ["backbone.projector"],
        "display": "backbone_late_proj",
    },
    {
        "name": "backbone.all+proj",
        "layers": [f"backbone.layer.{i}" for i in range(12)] + ["backbone.projector"],
        "display": "backbone_all_proj",
    },
    {
        "name": "proj+decoder",
        "layers": ["backbone.projector", "decoder.layer.0", "decoder.layer.1"],
        "display": "proj_decoder",
    },
]

TRAINING = {
    "max_epochs": 50,
    "learning_rate": 0.005,
    "reg_weight": 0.01,
    "early_stopping_patience": 10,
    "seed": 42,
    "checkpoint_every": 10,
}


def build_config(
    dataset: Dict[str, str],
    filt: Dict[str, Any],
    group: Dict[str, Any],
) -> Dict[str, Any]:
    filter_spec = {k: v for k, v in filt.items() if k != "display"}
    return {
        "name": f"{dataset['name']}_{filt['display']}_{group['display']}",
        "dataset": dataset["path"],
        "input_size": 384,
        "filter": filter_spec,
        "layer_group": {"name": group["name"], "layers": group["layers"]},
        "training": dict(TRAINING),
        "output": {"results_dir": "results/experiments"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate starter experiment YAML configs.")
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
                total += 1

    print(f"Generated {total} configs in {out_dir}")


if __name__ == "__main__":
    main()
