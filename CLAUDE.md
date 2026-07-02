# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Active experiment — branch `finetune-rfdetr-ref`:** fine-tune RF-DETR on the reference
> condition + train a **pair-free, GT-supervised** filter (no reference-A needed), validated on
> cooktop + ExDark. Full runbook and **fresh-clone prerequisites** (submodule init, copy the
> gitignored `data/raw/`) in [`docs/finetune-rfdetr-ref.md`](docs/finetune-rfdetr-ref.md).
> Run with `./scripts/run_finetune_matrix.sh cuda`. Read that runbook before starting work on
> this branch — Claude's per-machine memory does not travel across machines.

## What This Project Is

A research framework for training a **learnable differentiable preprocessing filter** that corrects illumination shifts in RF-DETR (nano) object detection — recovering detection performance without retraining the model. The filter is a pixel-space transformation applied before ImageNet normalization; the model stays frozen throughout.

## Commands

```bash
uv sync                                          # install deps (creates .venv)
uv run pytest tests/ -v                          # full test suite
uv run pytest tests/test_calibration.py -v       # single test file
uv run pytest -m "not slow" -v                   # skip model-loading tests
uv run python run_configs.py configs/experiments/ --dry-run   # validate YAMLs
uv run python run_configs.py configs/experiments/ --append --workers 4  # run sweep
uv run python scripts/generate_phase2_configs.py # regenerate phase-2 YAMLs
uv run python scripts/generate_visual_report.py  # rebuild results/visual_report.html
uv run ruff check src/
uv run mypy src/
```

## Architecture

### Data flow

```
Raw image (level_1.jpg = A, level_2/3.jpg = B)
  └─ to_unit_rgb(image, 384)  →  [0,1] NCHW tensor
       └─ filter(tensor)       →  filtered [0,1] NCHW          ← only the filter trains
            └─ normalize()     →  ImageNet-normalized
                 └─ model forward  →  activations + pred_logits/pred_boxes
```

### Model internals

LibreYOLO wraps RF-DETR nano. The internal model is `libre.model` (an `LWDETR`). Activations are captured with `register_forward_hook` — LibreYOLO exposes no `output_hidden_states`. Canonical layer names defined in `src/utils/activations.py::LAYER_PATHS`:

- `backbone.layer.0..11` — 12 DINOv2 ViT-S/14 encoder blocks
- `backbone.projector` — multi-scale projection head (best calibration signal; see below)
- `decoder.layer.0..1` — RT-DETR transformer decoder

**Why projector dominates:** The projector compresses rich ViT features into task-ready detector features, discarding photometric detail. It's the narrowest task-specific bottleneck — photometric filters can actually close the feature gap there. Backbone internals are too sensitive to scene content that the filter can't touch. Decoder layers are too abstract.

### Filter contract (`src/filters/base.py::Filter`)

Every filter: input `(B,3,H,W)` or `(3,H,W)` float in [0,1] → output same shape clamped to [0,1]. Identity init = no-op. Registered in `FILTER_REGISTRY` (`src/filters/__init__.py`). Instantiate with `build_filter({"type": "lut_3d", "size": 9})`.

High-capacity filters (`lut_3d`, `spatial_tone_curve`) override `reg_loss()` with TV + identity-anchoring; the calibration loop adds `reg_weight * filter.reg_loss()` to the group loss.

### Calibration loop (`src/calibration.py`)

Three entry points:
- `calibrate()` — single pair, single forward per step
- `calibrate_multi()` — list of pairs, cycles through them
- `calibrate_epochs()` — epoch-based, early stopping, checkpoint callbacks; used by the experiment runner

The loss is `group_loss(a_acts, b_acts, layer_names)` = mean of `||a-b||/||a||` (L2-rel) or cosine distance across all layers in the group. The reference activations of A are computed once with `compute_reference_activations()` and cached; only filtered-B runs the grad-enabled forward.

**DINOv2 inplace ops:** do NOT wrap the reference A forward in `torch.no_grad()` — DINOv2's inplace ops conflict with it. The workaround is already in `_forward_filtered` and `compute_reference_activations`.

### Experiment runner (`run_configs.py`)

One YAML per atomic run (1 filter × 1 layer group × 1 dataset). `discover_pairs()` auto-detects the B level from the dataset directory name (`level_1_vs_level_N` → looks for `level_N.jpg`). Results append to `results/experiments/experiment_results.csv`. Checkpoints go to `results/experiments/runs/<name>/best.pt` (gitignored).

YAML schema:
```yaml
name: string
dataset: data/datasets/level_1_vs_level_2   # must have train/ and test/ subdirs
input_size: 384
filter:
  type: spatial_tone_curve    # key in FILTER_REGISTRY
  P: 8
  grid_size: 3
layer_group:
  name: projector
  layers: [backbone.projector]  # supports range syntax: backbone.layer.0..3
training:
  max_epochs: 50
  learning_rate: 0.005
  reg_weight: 0.01
  early_stopping_patience: 10
  seed: 42
  checkpoint_every: 10
output:
  results_dir: results/experiments
```

### Dataset layout

```
data/datasets/level_1_vs_level_2/
  train/scene_001_aug0/{level_1.jpg, level_2.jpg}
  test/scene_007  →  ../../augmented/test/scene_007_2   # symlink
data/augmented/train/   # real images, gitignored
data/augmented/test/    # real images, gitignored
```

`data/datasets/` is gitignored. The `test/` symlinks must be recreated locally if lost — see `scripts/create_datasets.py` for the pattern (`scene_XXX → ../../../augmented/test/scene_XXX_N`).

### Phase-2 experiment configs (`configs/experiments/phase2/`)

Generated by `scripts/generate_phase2_configs.py`. Prefix `p2_`, family prefix in filename (`a_*`, `b_*`, `c_*`) for per-family batch runs. Run with `--append` so results survive partial runs. Check exit condition after each family:
```python
import csv
rows = list(csv.DictReader(open("results/experiments/experiment_results.csv")))
p2 = [float(r["test_mean"]) for r in rows if r["config"].startswith("p2_") and not r["per_pair_reductions"].startswith("ERROR")]
print(f"best: {max(p2):.4f}" if p2 else "none")
```

### Key results (Phase 1 + 2)

- **Best config:** `p2_stc_P16_g5_lv2` — `spatial_tone_curve(P=16, grid_size=5)` + `backbone.projector` → `test_mean = 0.3338` (33% feature gap closure)
- `test_mean` = fractional reduction in projector-layer feature distance: `(dist_before − dist_after) / dist_before`. Higher = better.
- The filter is applied to **B** (the shifted target image), not A.
- Decoder layers as calibration target are near-useless (test_mean ≈ 0.01–0.09).

## Gotchas

- **`git-filter-repo` removes the remote.** After running it, re-add with `git remote add origin <url>` then force-push.
- **`results/` is not blanket-gitignored** (changed from original). Specific patterns: `results/**/*.pt` and `results/*.log` are ignored; CSV, JSONL, and `results/visual_report.html` are tracked.
- **`session-ses_*.md` files** are gitignored (private session transcripts).
- **`data/datasets/level_1_vs_level_2/test/`** symlinks are not committed and must be recreated after a fresh clone (`scripts/create_datasets.py` or manual `ln -s`).
- **Workers:** `--workers 4` uses multiprocessing spawn; each worker loads its own model copy (~2GB VRAM each on CUDA).
