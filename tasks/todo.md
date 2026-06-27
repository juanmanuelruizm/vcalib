# vcalib Task Checklist

> Reframed as a **deployable on-edge calibration tool**. See `docs/specs/2026-06-27-experimental-plan.md`.
> Locked decisions: aligned pairs · A precomputed offline · filter edits pixels (activations = loss signal) ·
> layer chosen by sweep · proxy metric (agreement with A's predictions) · HF transformers `rf-detr-nano`.
> Two datasets: diverse **dev dataset** (Phases 1–2) vs single fixed **calibration scene** (runtime).
> Carlos chose real capture (no synthetic data) → Phase 1–3 *runs* gated on captured data; *code* built now.

## Phase A — Unblocked Now (no dataset needed)

- [ ] **A1. Dependency setup + model/activation verification**
  - `uv sync` (torch, torchvision, transformers ≥5.1, numpy, pillow, pyyaml, tqdm)
  - Confirm load: `uv run python -c "from transformers import RTDetr... ; print('OK')"` (use the real RF-DETR class name)
  - `uv run python src/utils/activations.py --test` → print **real layer names + shapes** from `output_hidden_states=True`
  - Decide filter position: pre-normalization RGB (assumed) vs processor's normalized tensor — inspect the HF image processor
  - Output: documented layer-name map; replace placeholders in `configs/grid.yaml`

- [ ] **A2. Activation helper (`src/utils/activations.py`)**
  - `load_model()` — frozen `rf-detr-nano`
  - `extract_activations(image) -> dict[str, Tensor]` via `output_hidden_states=True`
  - `cache_activations()` / `load_cached_activations()` → `data/processed/activations_cache/`

- [ ] **A3. Filters**
  - [ ] `src/filters/affine_6param.py` — gains `[0.1,2.0]`, offsets `[-1,1]`, identity init, `forward→clamp[0,1]`, `get_params()`
  - [ ] `src/filters/matrix_12param.py` — `I' = M·I + b`, M init = identity, same pixel-space contract

- [ ] **A4. Calibration loop (`src/calibration.py`, new)**
  - Input: filter, stored A-targets, B image, layer, loss fn, optimizer cfg
  - Adam lr 1e-3, max 100 steps, early-stop patience 10; seed for reproducibility
  - Output: trained filter + stats (steps, wall-clock, final loss)
  - Shared by grid search **and** the deployed tool

- [ ] **A5. Unit tests (`tests/test_filters.py`)**
  - Identity init = no-op; params stay in range
  - Smoke: loop reduces activation distance on one image with a programmatic light tweak (smoke only, not a dataset)
  - `uv run pytest tests/ -v` green; `uv run mypy src/` + `uv run ruff check src/` clean

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

- [ ] **Grid config (`configs/grid.yaml`)** — real layer names (from A1) × {affine_6param, matrix_12param} × {l2}; adam, lr 1e-3, 100 steps, patience 10; val split 0.2, seed 42
- [ ] **Grid executor (`src/grid_search.py`)** — train via `calibration.py` on dev-train; measure distance reduction on dev-val; append `results/runs.csv` `[run_id, layer, filter, loss, final_train_loss, val_distance, steps_to_converge, timestamp, config_hash]`; checkpoint `results/checkpoints/`
- [ ] **Rank** by val_distance (asc), steps (tie-break); pick top 3 for Phase 3

## Phase 3 — Benchmark (after Phase 2)

- [ ] **Benchmark harness (`src/benchmark.py`)** — top 3: model on B with/without filter; **proxy** vs A's predictions (box IoU + class agreement); calibration cost → `results/runs_phase3.csv`
- [ ] **Rank** by proxy recovery %; winner = highest recovery with ≤100 steps
- [ ] **Benchmark report (`docs/phase3_report.md`)** — top configs, recommended deploy choice, calibration cost, residual failure modes

## Phase D — Deployable Tool

- [ ] **`src/deploy_calibrate.py` (new)** — store A-targets of calibration scene → re-shoot under B → run `calibration.py` → freeze filter (<1MB) → ready for inference
- [ ] **Update CLAUDE.md** — new gotchas, real layer names, per-phase resource costs

---

### Current Status
- Spec reframed; plan approved. Phase A in progress.
### Next Action
- A1: `uv sync` + verify `rf-detr-nano` loads and dump real activation layer names/shapes.

**Last Updated:** 2026-06-27 · **Owner:** Carlos
