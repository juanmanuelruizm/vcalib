"""Generate Phase-2 experiment YAML configs.

Phase-2 targets the projector layer (and a few extensions) with:
  Family A: filter capacity sweep (spatial_tone_curve, lut_3d, tone_curve)
  Family B: training hyperparameter sweep (lr, reg_weight) for top-2 filters
  Family C: layer combination sweep (projector+decoder, late-backbone+projector)

All output files go to configs/experiments/phase2/ with family prefix
(a_*, b_*, c_*) so run_configs.py can glob them per-family.

Usage::

    uv run python scripts/generate_phase2_configs.py
    uv run python scripts/generate_phase2_configs.py --force   # overwrite existing
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "configs" / "experiments" / "phase2"

DATASETS = {
    "lv2": "data/datasets/level_1_vs_level_2",
    "lv3": "data/datasets/level_1_vs_level_3",
}

DEFAULTS = dict(
    input_size=384,
    training=dict(
        max_epochs=50,
        learning_rate=0.005,
        reg_weight=0.01,
        early_stopping_patience=10,
        seed=42,
        checkpoint_every=10,
    ),
    output=dict(results_dir="results/experiments"),
)


def make_config(name: str, dataset_key: str, filter_cfg: dict,
                layers: list, training_overrides: dict | None = None) -> dict:
    tr = dict(DEFAULTS["training"])
    if training_overrides:
        tr.update(training_overrides)
    return {
        "name": name,
        "dataset": DATASETS[dataset_key],
        "input_size": DEFAULTS["input_size"],
        "filter": filter_cfg,
        "layer_group": {
            "name": "phase2_group",
            "layers": layers,
        },
        "training": tr,
        "output": dict(DEFAULTS["output"]),
    }


def write(path: Path, cfg: dict, force: bool) -> bool:
    if path.exists() and not force:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return True


def generate_all(force: bool) -> int:
    written = 0

    # ------------------------------------------------------------------
    # Family A — filter capacity, projector layer
    # ------------------------------------------------------------------
    proj_layers = ["backbone.projector"]

    # A1: spatial_tone_curve  P x grid_size, skip already-run (8, 3)
    for P in [8, 12, 16]:
        for G in [3, 4, 5]:
            if P == 8 and G == 3:
                continue  # already in Phase-1
            for ds in ["lv2", "lv3"]:
                name = f"p2_stc_P{P}_g{G}_{ds}"
                cfg = make_config(
                    name, ds,
                    {"type": "spatial_tone_curve", "P": P, "grid_size": G},
                    proj_layers,
                )
                written += write(OUT_DIR / f"a_{name}.yaml", cfg, force)

    # A2: lut_3d  size in {13, 17}  (size=9 already in Phase-1)
    for size in [13, 17]:
        for ds in ["lv2", "lv3"]:
            name = f"p2_lut3d_s{size}_{ds}"
            cfg = make_config(
                name, ds,
                {"type": "lut_3d", "size": size},
                proj_layers,
            )
            written += write(OUT_DIR / f"a_{name}.yaml", cfg, force)

    # A3: tone_curve  P in {16, 24, 32}  max_epochs=100 (P=16 didn't converge at 50)
    for P in [16, 24, 32]:
        for ds in ["lv2", "lv3"]:
            name = f"p2_tone_P{P}_{ds}"
            cfg = make_config(
                name, ds,
                {"type": "tone_curve", "P": P},
                proj_layers,
                training_overrides={"max_epochs": 100},
            )
            written += write(OUT_DIR / f"a_{name}.yaml", cfg, force)

    # ------------------------------------------------------------------
    # Family B — training hyperparameters, top-2 filters, projector
    # ------------------------------------------------------------------
    top2_filters = [
        ("stc", {"type": "spatial_tone_curve", "P": 8, "grid_size": 3}),
        ("lut3d", {"type": "lut_3d", "size": 9}),
    ]

    # B1: learning rate
    for lr in [0.001, 0.01]:
        lr_tag = "lr0001" if lr == 0.001 else "lr001"
        for fkey, fcfg in top2_filters:
            for ds in ["lv2", "lv3"]:
                name = f"p2_{fkey}_{lr_tag}_{ds}"
                cfg = make_config(
                    name, ds, fcfg, proj_layers,
                    training_overrides={"learning_rate": lr},
                )
                written += write(OUT_DIR / f"b_{name}.yaml", cfg, force)

    # B2: regularisation weight
    for reg in [0.001, 0.05]:
        reg_tag = "reg0001" if reg == 0.001 else "reg005"
        for fkey, fcfg in top2_filters:
            for ds in ["lv2", "lv3"]:
                name = f"p2_{fkey}_{reg_tag}_{ds}"
                cfg = make_config(
                    name, ds, fcfg, proj_layers,
                    training_overrides={"reg_weight": reg},
                )
                written += write(OUT_DIR / f"b_{name}.yaml", cfg, force)

    # ------------------------------------------------------------------
    # Family C — layer combinations, top-3 filters
    # ------------------------------------------------------------------
    top3_filters = [
        ("stc", {"type": "spatial_tone_curve", "P": 8, "grid_size": 3}),
        ("lut3d", {"type": "lut_3d", "size": 9}),
        ("ccm", {"type": "ccm_high_order"}),
    ]

    # C1: projector + decoder.layer.0
    for fkey, fcfg in top3_filters:
        for ds in ["lv2", "lv3"]:
            name = f"p2_{fkey}_proj_dec_{ds}"
            cfg = make_config(
                name, ds, fcfg,
                ["backbone.projector", "decoder.layer.0"],
            )
            written += write(OUT_DIR / f"c_{name}.yaml", cfg, force)

    # C2: backbone.layer.11 + projector
    for fkey, fcfg in top3_filters:
        for ds in ["lv2", "lv3"]:
            name = f"p2_{fkey}_b11_proj_{ds}"
            cfg = make_config(
                name, ds, fcfg,
                ["backbone.layer.11", "backbone.projector"],
            )
            written += write(OUT_DIR / f"c_{name}.yaml", cfg, force)

    # C3: backbone.layer.10..11 + projector
    for fkey, fcfg in top3_filters:
        for ds in ["lv2", "lv3"]:
            name = f"p2_{fkey}_b1011_proj_{ds}"
            cfg = make_config(
                name, ds, fcfg,
                ["backbone.layer.10", "backbone.layer.11", "backbone.projector"],
            )
            written += write(OUT_DIR / f"c_{name}.yaml", cfg, force)

    # C4: decoder.layer.0 alone
    for fkey, fcfg in top3_filters:
        for ds in ["lv2", "lv3"]:
            name = f"p2_{fkey}_dec_{ds}"
            cfg = make_config(
                name, ds, fcfg,
                ["decoder.layer.0"],
            )
            written += write(OUT_DIR / f"c_{name}.yaml", cfg, force)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase-2 experiment configs")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    written = generate_all(args.force)
    total = len(list(OUT_DIR.glob("*.yaml")))
    print(f"Written: {written} new files")
    print(f"Total in {OUT_DIR.relative_to(ROOT)}: {total} configs")


if __name__ == "__main__":
    main()
