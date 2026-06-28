# Spec: Config-Driven Experiment Runner

**Status:** Approved
**Date:** 2026-06-28
**Owner:** Carlos
**Branch:** `configs` (from `epochs`)

## Motivation

`run_experiments.py` hardcodes `FILTERS`, `LAYER_GROUPS`, and all hyperparameters at the top of the script. Every change to the sweep requires editing Python. We want to **decouple configuration from execution**: one YAML per atomic training run, a folder of YAMLs describing the full sweep, and a single runner script that globs them and aggregates results.

## Granularity

**One YAML file = one atomic run** (1 filter × 1 layer group × 1 dataset). Running the full current sweep = globbing 60 YAMLs (10 filters × 3 groups × 2 datasets). The runner accepts any number of YAML paths or directories and aggregates all runs into a single CSV.

## YAML Schema

```yaml
# configs/experiments/level2_lut3d_projector.yaml
name: level2_lut3d_projector          # required, unique; used in CSV + run dir
dataset: data/datasets/level_1_vs_level_2   # required, relative to repo root
input_size: 384                        # optional, default 384
filter:                                # exactly one filter spec (dict)
  type: lut_3d
  size: 9
layer_group:                           # exactly one group (name -> list of layer names)
  name: projector
  layers: [backbone.projector]
training:
  max_epochs: 20
  learning_rate: 0.005
  reg_weight: 0.01
  early_stopping_patience: 5
  seed: 42
  checkpoint_every: 5                  # epochs; 0 disables periodic ckpts
output:
  results_dir: results/experiments     # optional, default results/experiments
```

### Layer range syntax
`backbone.layer.0..3` expands to `backbone.layer.{0,1,2,3}` via `src/utils/layer_groups.py::expand_layer_spec`. Allowed in `layer_group.layers`.

### Filter spec
Same dict format as `configs/grid.yaml::grid.filters` and `src/filters::build_filter`:
- `type` (required) — must be in `FILTER_REGISTRY`.
- Extra kwargs (e.g. `size`, `P`, `grid_size`, `degree`, `M`) passed through to the filter constructor.

## Config Loader — `src/experiment_config.py`

### `ExperimentConfig` dataclass
Fields: `name`, `dataset` (Path), `input_size` (int), `filter` (dict), `layer_group` (`LayerGroup`-like: `name` + `layers: List[str]`), `training` (sub-dataclass with `max_epochs`, `learning_rate`, `reg_weight`, `early_stopping_patience`, `seed`, `checkpoint_every`), `output_results_dir` (Path).

### Functions
- `load_config(path: Path) -> ExperimentConfig` — reads + validates a single YAML.
- `load_configs(paths) -> List[ExperimentConfig]` — accepts a file, a directory (globs `*.yaml` non-recursively), or an iterable of paths. Returns a flat list sorted by (filename stem). Deduplicates on `name` (raises `ValueError` if two files share a `name`).

### Validation
- `name` non-empty, unique across the loaded set.
- `dataset` path exists.
- `filter.type` in `FILTER_REGISTRY`.
- Each layer in `layer_group.layers` (after range expansion) in `LAYER_PATHS`.
- `training` fields positive; `max_epochs >= 1`; `checkpoint_every >= 0`.
- On error, `ValueError(msg, config_file=...)` names the offending config file.

## Runner — `run_configs.py`

### CLI
```
uv run python run_configs.py PATH... [--output results/experiments] [--dry-run] [--device cuda|cpu] [--append]
```
- `PATH...` (required, variadic): one or more files or directories. A directory globs `*.yaml` non-recursively. Default if omitted: `configs/experiments/`.
- `--output`: override `output.results_dir` from all configs.
- `--dry-run`: parse + validate all configs, print the resolved run list, exit 0 without loading the model.
- `--append`: append to the existing CSV instead of overwriting.
- `--device`: passed to `load_model` (default: auto).

### Execution flow
1. `load_configs(paths)` → flat list of `ExperimentConfig`.
2. Load RF-DETR nano once; reuse for all runs.
3. For each `ExperimentConfig`:
   a. `discover_pairs(cfg.dataset)` → train/test pairs (reuse the `run_experiments.py::discover_pairs` + `load_all_pairs` logic, lifted into a small helper or inlined).
   b. `build_filter(cfg.filter)` → one filter instance.
   c. `CalibrationConfig(max_epochs=cfg.training.max_epochs, learning_rate=..., reg_weight=..., early_stopping_patience=..., seed=...)`.
   d. `calibrate_epochs(filt, train_pairs, layer_group.layers, model, cfg=..., test_pairs=test_pairs, on_epoch_end=callback)`.
   e. `on_epoch_end` callback writes `metrics.jsonl` + periodic checkpoints, same as `run_experiments.py::_make_epoch_callback`.
   f. `evaluate_on_test(trained, model, test_pairs, layer_group.layers)` (lifted from `run_experiments.py`).
   g. Append one CSV row tagged with `cfg.name`, `group`, `filter`, train_reduction, test_mean/std, per_pair, steps, wall_s, converged.
4. Write aggregated CSV to `{output}/experiment_results.csv` (overwrite unless `--append`).
5. Run artifacts under `{output}/runs/{cfg.name}/{metrics.jsonl, best.pt, epoch_NNNN.pt}`.

Since each YAML = one run, the run dir is simply `{output}/runs/{cfg.name}/` (no `<group>__<filter>` nesting needed).

### Dry-run output
```
Found 3 configs:
  [1] level2_lut3d_projector   dataset=data/datasets/level_1_vs_level_2  filter=lut_3d(size=9)  group=projector  epochs=20
  [2] level3_ccm_backbone_early dataset=data/datasets/level_1_vs_level_3  filter=ccm_high_order  group=backbone.early  epochs=20
  [3] ...
Total runs: 3
Dry run — no training. Exiting.
```

## Starter Configs

Generate all 60 YAMLs reproducing today's sweep (10 filters × 3 groups × 2 datasets), so the new system is a drop-in replacement for `run_experiments.py`. Plus 2 focused configs as schema demos:

- 60 sweep YAMLs in `configs/experiments/` named `{dataset}_{filter}_{group}.yaml` (e.g. `level2_lut3d_projector.yaml`, `level3_ccm_high_order_backbone.early.yaml`).
- A generator script `scripts/generate_experiment_configs.py` that emits the 60 YAMLs (idempotent; safe to re-run) so future sweep changes regenerate the set.

Filter specs and layer groups match `run_experiments.py` exactly:

**Filters (10):**
`brightness_2param`, `affine_6param`, `matrix_12param`, `gamma_3param`, `chromatic_adaptation`, `ccm_high_order`, `tone_curve` (P=16), `lut_3d` (size=9), `spatial_tone_curve` (P=8, grid_size=3), `local_tonemap` (grid_size=4).

**Layer groups (3):**
- `backbone.early`: `backbone.layer.0..3`
- `backbone.all`: `backbone.layer.0..11`
- `projector`: `backbone.projector`

**Datasets (2):** `data/datasets/level_1_vs_level_2`, `data/datasets/level_1_vs_level_3`

**Training (shared):** `max_epochs=20`, `learning_rate=0.005`, `reg_weight=0.01`, `early_stopping_patience=5`, `seed=42`, `checkpoint_every=5`.

## `run_experiments.py`

Keep the file. Add a deprecation note at the top: "Superseded by `run_configs.py` + `configs/experiments/`. Retained for backwards compatibility." No code change.

## Tests — `tests/test_experiment_config.py`

- Parse + validate a minimal valid config (round-trip into `ExperimentConfig`).
- Reject: missing `name`, nonexistent `dataset`, unknown filter `type`, unknown layer name, bad type in `training` block, negative `max_epochs`.
- `load_configs` on a directory returns sorted list of all YAMLs.
- `load_configs` on a single file returns one-element list.
- Duplicate `name` across two files raises `ValueError`.
- Expansion of `backbone.layer.0..3` resolves to 4 layer names.
- Slow subprocess test (marked `slow`): `run_configs.py configs/experiments/ --dry-run` exits 0 on the starter YAMLs without loading the model.

## Out of Scope

- No changes to `src/calibration.py`, `src/filters/`, `src/utils/activations.py`, `src/utils/layer_groups.py`, `configs/grid.yaml`, dataset scripts, or `scripts/augment_dataset*.py`.
- No `--resume`/checkpoint replay (artifacts stay on disk; loading them is a future task).
- `src/grid_search.py` keeps its own config path — not touched.

## Verification

- `uv run python run_configs.py configs/experiments/ --dry-run` prints 62 configs (60 sweep + 2 focused) and exits 0.
- `uv run python run_configs.py configs/experiments/level2_lut3d_projector.yaml` trains one run end-to-end, writes `results/experiments/runs/level2_lut3d_projector/{metrics.jsonl,best.pt}`, and appends one row to the CSV.
- `uv run pytest tests/test_experiment_config.py -v` passes.
- `uv run ruff check src/experiment_config.py run_configs.py tests/test_experiment_config.py` and `uv run mypy src/experiment_config.py` clean.

## Deliverables

1. `configs/experiments/*.yaml` (60 sweep + 2 focused = 62 files)
2. `scripts/generate_experiment_configs.py`
3. `src/experiment_config.py`
4. `run_configs.py`
5. `tests/test_experiment_config.py`
6. Deprecation note in `run_experiments.py`
7. Updated `tasks/todo.md`