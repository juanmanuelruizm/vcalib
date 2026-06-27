# vcalib Task Checklist

> Reframed as a **deployable on-edge calibration tool**. See `docs/specs/2026-06-27-experimental-plan.md`.
> Locked decisions: aligned pairs · A precomputed offline · filter edits pixels (activations = loss signal) ·
> layer GROUP chosen by sweep (loss = mean of per-layer normalized L2 over the group; encoded in `layer_groups` + `layer_group_search`) · proxy metric (agreement with A's predictions) · model loaded via **LibreYOLO** (`LibreRFDETR`, internal `LWDETR`); activations via **forward hooks** (no `output_hidden_states`).
> Two datasets: diverse **dev dataset** (Phases 1–2) vs single fixed **calibration scene** (runtime).
> Carlos chose real capture (no synthetic data) → Phase 1–3 *runs* gated on captured data; *code* built now.

## Phase A — Unblocked Now (no dataset needed)

- [x] **A1. Dependency setup + model/activation verification**
  - `uv sync` (torch, torchvision, transformers ≥5.1, numpy, pillow, pyyaml, tqdm)
  - Model loader = **LibreYOLO** (`3rd_party/libreyolo`, `LibreRFDETR(size="n")`); rf-detr-nano weights downloaded to `3rd_party/libreyolo/weights/rf-detr-nano.pth`
  - Activation access = **PyTorch `register_forward_hook`** on `libre.model.model` submodules (LibreYOLO does not expose `output_hidden_states`)
  - Real layer names: `backbone.layer.0..11` (12 DINOv2 blocks), `backbone.projector`, `decoder.layer.0..1` → updated `configs/grid.yaml`
  - Filter position: pre-normalization [0,1] RGB (confirmed via LibreYOLO's `preprocess_numpy`)

- [x] **A2. Activation helper (`src/utils/activations.py`)**
  - `load_model()` — frozen `rf-detr-nano` via LibreYOLO
  - `extract_activations(image) -> dict[str, Tensor]` via forward hooks (`ActivationExtractor`)
  - `cache_activations()` / `load_cached_activations()` → `data/processed/activations_cache/`
  - Verified: 15 layers captured with correct shapes; `ruff`/`mypy` clean; 12 unit tests pass

- [x] **A3. Filters** — full library built upfront (scope expanded so the grid sweeps filter types too)
  - [x] `src/filters/base.py` — `Filter` base (4D wrap, clamp01, `num_params`, `get_params`); `clamp_param` helper
  - [x] `brightness_2param`, `white_balance_3param`, `saturation_1param`, `contrast_1param`, `gamma_3param` — exposure, WB, vividness, contrast, non-linear tone
  - [x] `affine_6param` (gains `[0.1,2.0]`, offsets `[-1,1]`, identity init) + `matrix_12param` (`I'=M·I+b`, M=identity)
  - [x] `composite.py` — `CompositeFilter` chains ordered filters (variable params)
  - [x] `__init__.py` — `FILTER_REGISTRY` + `get_filter(name)` / `build_filter(spec)` factory for the grid/automatic search
  - Verified: identity=no-op on real photos, params in range, 59 filter tests pass

- [ ] **A4. Calibration loop (`src/calibration.py`, new)**
  - Input: filter, stored A-targets, B image, **layer group** (from `layer_groups.py`), loss+aggregation cfg, optimizer cfg
  - Group loss = mean of per-layer normalized L2 (`||a-b||/||a||`) across the group's layers
  - Adam lr 1e-3, max 100 steps, early-stop patience 10; seed for reproducibility
  - Output: trained filter + stats (steps, wall-clock, final loss)
  - Shared by grid search **and** the deployed tool

- [ ] **A5. Unit tests (`tests/test_filters.py`)**
  - Identity init = no-op (every filter, incl. on real photos in `data/raw/`); params stay in range
  - Registry `get_filter`/`build_filter`/`make_composite`; composite chaining & param counts
  - 59 filter tests passing; smoke: loop reduces activation distance on one image with a programmatic light tweak (smoke only, not a dataset) — **pending A4**

- [x] **A6. Rewrite spec + this todo to match the tool reframing** (done)

- [ ] **A7. Finalize capture protocol (Phase 0 prep — the real bottleneck)**
  - **Dev dataset:** 20–30 scenes × 3–5 illumination levels = 60–150 images; tripod; document light temp/angle/distance; held-out val split
  - **Calibration scene:** one fixed re-shootable reference (grey-card / fixed object set)
  - Save layout: `data/raw/scenes_YYYYMMDD/scene_001_level_1.jpg`, …; `data/raw/calibration_scene/`
  - Decide illumination shift type (brightness vs color-temp vs both) and reference condition A

## Phase 1 — Diagnostics (after dev dataset captured)

- [ ] **Diagnostic sweep (`src/diagnostics.py`)** — forward A & B; per-layer L2(norm)+cosine per illumination level; aggregate mean±std over scenes → `results/phase1_diagnostics.json`
- [ ] **Plots** — heatmap (layer × level) + line plots with error bars → `results/phase1_plots/`
- [ ] **Early-bailout** — flat distance everywhere → report "no signal", STOP
- [ ] **Phase 1 report (`docs/phase1_report.md`)** — layer rankings + candidate layer(s) for Phase 2

## Phase 2 — Grid Search (after Phase 1)

- [x] **Layer-group encoding (`configs/grid.yaml` + `src/utils/layer_groups.py`)** — sweep is over GROUPS of layers (mean of per-layer normalized L2), not single layers. Explicit baselines + auto-generated DINOv2 windows; `resolve_grid_groups()` + `--dump` verified (35 groups). Loss aggregation strategy documented.
- [ ] **Grid config (`configs/grid.yaml`)** — layer groups (resolved via `layer_groups.py`) × {affine_6param, matrix_12param} × {l2}; adam, lr 1e-3, 100 steps, patience 10; val split 0.2, seed 42
- [ ] **Grid executor (`src/grid_search.py`)** — train via `calibration.py` on dev-train; measure distance reduction on dev-val; append `results/runs.csv` `[run_id, group_name, filter, loss, final_train_loss, val_distance, steps_to_converge, timestamp, config_hash]`; checkpoint `results/checkpoints/`
- [ ] **Automatic layer-search loop** — iterate `resolve_grid_groups()` candidates, rank by val_distance (asc), steps (tie-break); pick top 3 for Phase 3

## Phase 3 — Benchmark (after Phase 2)

- [ ] **Benchmark harness (`src/benchmark.py`)** — top 3: model on B with/without filter; **proxy** vs A's predictions (box IoU + class agreement); calibration cost → `results/runs_phase3.csv`
- [ ] **Rank** by proxy recovery %; winner = highest recovery with ≤100 steps
- [ ] **Benchmark report (`docs/phase3_report.md`)** — top configs, recommended deploy choice, calibration cost, residual failure modes

## Phase D — Deployable Tool

- [ ] **`src/deploy_calibrate.py` (new)** — store A-targets of calibration scene → re-shoot under B → run `calibration.py` → freeze filter (<1MB) → ready for inference
- [ ] **Update CLAUDE.md** — new gotchas, real layer names, per-phase resource costs

---

### Current Status
- A1 + A2 done: environment set up, rf-detr-nano downloaded via LibreYOLO, activation helper built and verified. Phase A continues with filters (A3) + calibration loop (A4).
### Next Action
- A3: implement `affine_6param` and `matrix_12param` filters (identity init, clamp [0,1]).

**Last Updated:** 2026-06-27 · **Owner:** Carlos
