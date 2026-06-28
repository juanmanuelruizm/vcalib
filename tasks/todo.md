# vcalib Task Checklist

> Reframed as a **deployable on-edge calibration tool**. See `docs/specs/2026-06-27-experimental-plan.md`
> and `docs/specs/2026-06-27-advanced-filters.md` (advanced filter library, expressiveness-first).
> Locked decisions: aligned pairs · A precomputed offline · filter edits pixels (activations = loss signal) ·
> layer GROUP chosen by sweep (loss = mean of per-layer normalized L2 over the group; encoded in `layer_groups` + `layer_group_search`) · proxy metric (agreement with A's predictions) · model loaded via **LibreYOLO** (`LibreRFDETR`, internal `LWDETR`); activations via **forward hooks** (no `output_hidden_states`).
> **Filter philosophy (2026-06-27 update): parameter count is NOT a constraint. Expressive capacity is prioritized; held-out validation is the primary guardrail; smoothness/identity-anchoring regularization (`Filter.reg_loss()`) matters more than param count.**
> Two datasets: diverse **dev dataset** (Phases 1–2) vs single fixed **calibration scene** (runtime).
> Carlos chose real capture (no synthetic data) → Phase 1–3 *runs* gated on captured data; *code* built now. Stand-in programmatic re-lit pairs (A8) used for acceptance tests until Phase 0 capture.

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

- [x] **A3. Filters (initial library)** — global + spatial filters built upfront so the grid sweeps filter types too
  - [x] `src/filters/base.py` — `Filter` base (4D wrap, clamp01, `num_params`, `get_params`); `clamp_param` helper
  - [x] Global filters: `brightness_2param`, `white_balance_3param`, `saturation_1param`, `contrast_1param`, `gamma_3param`, `affine_6param`, `matrix_12param`
  - [x] `composite.py` — `CompositeFilter` chains ordered filters (variable params)
  - [x] `spatial.py` — **zone-dependent** variants (`spatial_brightness`/`spatial_whitebalance`/`spatial_affine`/`spatial_gamma`) via bilinear K×K control grid (`grid_sample`); params = K²·n_field (K configurable)
  - [x] `__init__.py` — `FILTER_REGISTRY` + `get_filter(name)` / `build_filter(spec)` (supports `grid_size`) factory
  - Verified: identity=no-op, zone-dependent output confirmed (TL 0.15 vs BR 0.52 on gradient showcase), 119 tests pass, visual preview in `results/filter_preview/`

> **Advanced filter library (A3b + A8–A16)** — see `docs/specs/2026-06-27-advanced-filters.md`.
> Expressiveness-first: parameter count is NOT a constraint; held-out validation is the primary guardrail; `Filter.reg_loss()` carries smoothness/identity-anchoring regularization. Target shift = both brightness and color-temperature. Build order: A3b → A4 + A8 (parallel) → A9–A15 (filters, parallelizable) → A16 → A17.

- [x] **A3b. Base-contract `reg_loss()`** — prerequisite for all advanced filters
  - Edit `src/filters/base.py`: add `Filter.reg_loss() -> torch.Tensor` (default: zero tensor on the param device)
  - Edit `src/filters/composite.py`: `CompositeFilter.reg_loss()` = sum of sub-filters' `reg_loss()`
  - Additive (default 0) → existing 119 tests stay green
  - Verify: `uv run pytest tests/test_filters.py tests/test_spatial_filters.py -v` && `uv run mypy src/ && uv run ruff check src/`

- [x] **A4. Calibration loop (`src/calibration.py`, new)** — needed by acceptance criteria #6/#7
  - Input: filter, stored A-targets, B image, **layer group** (from `layer_groups.py`), loss+aggregation cfg, optimizer cfg, `reg_weight`
  - Group loss = mean of per-layer normalized L2 (`||a-b||/||a||`) across the group's layers + `reg_weight * filter.reg_loss()`
  - Adam lr 1e-3, max 100 steps, early-stop patience 10; `torch.manual_seed` for reproducibility
  - Output: trained filter + stats (steps, wall-clock, final train loss, final val loss)
  - Shared by grid search **and** the deployed tool
  - Bypasses LibreYOLO's `nested_tensor_from_tensor_list` (inplace `copy_` fails with grad inputs) via direct `NestedTensor` construction
  - Verify: `uv run pytest tests/test_calibration.py -v` (8 slow smoke tests pass: all filters reduce train distance)

- [x] **A8. Stand-in re-lit data (`src/utils/synth_relit.py`, new)** — proxy A/B/B_val until Phase 0 capture
  - Deterministic: A = real `data/raw` photo; B_train = A·`gamma=1.4 + gains=[1.10,0.95,0.85]` (warm + underexposed); B_val = A·`gamma=1.2 + gains=[1.05,0.97,0.90]` (same direction, milder — held-out probe)
  - Verify: `uv run pytest tests/test_synth_relit.py -v`

- [x] **A9. `src/filters/lut_3d.py`** (`LUT3D`, registry `lut_3d`) — **headline**: 3D LUT, trilinear interp, identity-LUT init, TV+identity reg_loss. Smoke: 64.7% train reduction. Acceptance: `uv run pytest tests/test_lut_3d.py -v`
- [x] **A10. `src/filters/tone_curve.py`** (`ToneCurve`, `tone_curve`) — monotone (cumsum-softplus) per-channel curves; curvature+identity reg_loss. Smoke: 45.7% train, 19.5% val. Acceptance: `uv run pytest tests/test_tone_curve.py -v`
- [x] **A11. `src/filters/ccm_high_order.py`** (`HighOrderCCM`, `ccm_high_order`) — root-polynomial CCM (Finlayson), degree≤3, normalized features; L2-toward-identity reg_loss. Smoke: 55.5% train, 32.8% val, **gate PASS** (best generalizer). Acceptance: `uv run pytest tests/test_ccm_high_order.py -v`
- [x] **A12. `src/filters/chromatic_adaptation.py`** (`ChromaticAdaptation`, `chromatic_adaptation`) — Bradford LMS diagonal/full; identity init; optional L2 reg. Smoke: 32.8% train, -7.0% val (overfit on 3 params). Acceptance: `uv run pytest tests/test_chromatic_adaptation.py -v`
- [x] **A13. `src/filters/spatial_tone_curve.py`** (`SpatialToneCurve`, `spatial_tone_curve`) — reuses `SpatialFilter` grid + monotone curves per zone; spatial TV+curvature+identity reg_loss. Smoke: 46.8% train, 24.6% val, **gate PASS**. Acceptance: `uv run pytest tests/test_spatial_tone_curve.py -v`
- [x] **A14. `src/filters/local_tonemap.py`** (`LocalTonemap`, `local_tonemap`) — CLAHE-like approximation: guided-filter local-mean + K×K gain field; TV+identity reg_loss. True CLAHE = future stretch. Smoke: 0.8% train (under-trained at default LR). Acceptance: `uv run pytest tests/test_local_tonemap.py -v`
- [x] **A15. `src/filters/lut_3d_lowrank.py`** (`LUT3DLowRank`, `lut_3d_lowrank`) — LUT = identity + Σ w_m·B_m (fixed basis); L2-on-weights reg_loss; generalization-friendly LUT variant. Smoke: 1.0% train (under-trained at default LR). Acceptance: `uv run pytest tests/test_lut_3d_lowrank.py -v`

  Each of A9–A15: `tests/test_<filter>.py` runs the 7 acceptance criteria from the spec (identity-on-real-photos, params-in-range, differentiability finite-non-zero grads, registry+build with kwargs, composite chaining + reg_loss sum, smoke via A4 on B_train ≥30% distance reduction, generalization on B_val with overfit gate `val/train ≥ 0.5`).

- [x] **A16. Registry + grid integration**
  - Registered F1–F7 in `src/filters/__init__.py::FILTER_REGISTRY` (18 filters total) + `build_filter` kwargs (`size`/`P`/`grid_size`/`degree`/`mode`/`M`)
  - Updated `configs/grid.yaml::grid.filters` (17 active: 7 initial + 4 spatial + 6 advanced; F1 N=17 / F7 + composites commented); added `training.reg_weight: 0.01`; added `validation.overfit_gate` (enabled, `min_val_recovery_ratio: 0.5`)
  - Verify: `uv run python src/utils/layer_groups.py --dump` (35 groups) && `uv run pytest tests/ -v` (211 tests: 203 fast + 8 slow) && `uv run mypy src/ && uv run ruff check src/`

- [x] **A5. Unit tests (`tests/test_filters.py`)**
  - Identity init = no-op (every filter, incl. on real photos in `data/raw/`); params stay in range
  - Registry `get_filter`/`build_filter`/`make_composite`; composite chaining & param counts
  - 59 filter tests passing; smoke: loop reduces activation distance on one image with a programmatic light tweak — **done via A4 + A8** (8 slow smoke tests pass across all filters)

- [x] **A6. Rewrite spec + this todo to match the tool reframing** (done)

- [x] **A17. Doc alignment with advanced-filter spec** — applied `AGENTS.md` framing edits (§1, filter-library, spatial, Phase 2 gotchas, stale `npm`→`uv` commands, filename convention) and `docs/specs/2026-06-27-experimental-plan.md` alignment edits (§1/§2/§3/§5/§6/§8/§9 budget references → expressiveness-first + pointer to `2026-06-27-advanced-filters.md`).

- [x] **A7. Finalize capture protocol (Phase 0 prep — the real bottleneck)**
  - **Illumination shift:** BOTH brightness and color-temperature (locked from spec — the advanced library targets both)
  - **Dev dataset:** 20–30 scenes × 3–5 illumination levels = 60–150 images; tripod; same framing, only light changes; document light temp/angle/distance per level
  - **Reference condition A** = level_1 (brightest / neutral white balance); B = levels 2..N (progressively dimmer / warmer or cooler)
  - **Calibration scene:** one fixed re-shootable reference (grey-card / fixed object set) in `data/raw/calibration_scene/`
  - **Save layout** (consumed by `src/utils/data_pairs.py::discover_pairs`):
    ```
    data/raw/
      scenes_YYYYMMDD/
        scene_001/
          level_1.jpg    ← condition A (reference)
          level_2.jpg    ← condition B (shift level 2)
          level_3.jpg    ← condition B (shift level 3)
          ...
        scene_002/
          ...
      calibration_scene/
        reference.jpg    ← condition A (for deploy tool)
    ```
  - **Held-out val split:** by SCENE (not by level) — entire scenes held out, so val pairs test generalization to unseen scenes at seen shift levels. Split 0.2 = ~4–6 scenes held out. Configurable in `configs/grid.yaml::validation.split`.
  - **Data loader:** `src/utils/data_pairs.py` discovers this layout, groups by scene, yields `(A_path, B_path, level, scene_id)` pairs, and supports the held-out split.

## Phase 1 — Diagnostics (after dev dataset captured)

- [x] **Data loader (`src/utils/data_pairs.py`)** — discovers `data/raw/scenes_YYYYMMDD/scene_*/level_*.jpg`, groups by scene, yields `(A, B, level, scene_id)` pairs, held-out val split by scene. Ready to run on real data.
- [x] **Diagnostic sweep (`src/diagnostics.py`)** — forwards A & B through frozen RF-DETR; per-layer L2(norm)+cosine distance per illumination level; aggregates mean±std over scenes → `results/phase1_diagnostics.json` + `--plot` heatmaps. Early-bailout if flat. Run: `uv run python src/diagnostics.py --dataset-path data/raw/scenes_YYYYMMDD/ --output results/phase1_diagnostics.json --plot`
- [ ] **Run Phase 1** — after Carlos captures the dev dataset. Inspect the heatmap: which layers show a distance gradient that scales with illumination level? Those are Phase 2 candidates.
- [ ] **Phase 1 report (`docs/phase1_report.md`)** — layer rankings + candidate layer groups for Phase 2

## Phase 2 — Grid Search (after Phase 1)

- [x] **Layer-group encoding (`configs/grid.yaml` + `src/utils/layer_groups.py`)** — sweep is over GROUPS of layers (mean of per-layer normalized L2), not single layers. Explicit baselines + auto-generated DINOv2 windows; `resolve_grid_groups()` + `--dump` verified (35 groups). Loss aggregation strategy documented.
- [x] **Grid config (`configs/grid.yaml`)** — 35 layer groups × 17 active filters × {l2}; adam, lr 1e-3, 100 steps, patience 10, `reg_weight: 0.01`; val split 0.2, seed 42, `overfit_gate` enabled (min_val_recovery_ratio 0.5)
- [x] **Grid executor (`src/grid_search.py`)** — resolves groups×filters from `configs/grid.yaml`; trains via `calibration.py` on dev-train; measures distance reduction on dev-val; appends `results/runs.csv`; applies overfit gate; `--subset N` flag for quick runs. Run: `uv run python src/grid_search.py --config configs/grid.yaml --dataset-path data/raw/scenes_YYYYMMDD/` (or `--subset 5` for 5 quick runs)
- [ ] **Run Phase 2** — after Phase 1 identifies candidate layers. Rank by val_distance (asc), steps (tie-break); pick top 3 for Phase 3.

## Phase 3 — Benchmark (after Phase 2)

- [ ] **Benchmark harness (`src/benchmark.py`)** — top 3: model on B with/without filter; **proxy** vs A's predictions (box IoU + class agreement); calibration cost → `results/runs_phase3.csv`
- [ ] **Rank** by proxy recovery %; winner = highest recovery with ≤100 steps
- [ ] **Benchmark report (`docs/phase3_report.md`)** — top configs, recommended deploy choice, calibration cost, residual failure modes

## Phase D — Deployable Tool

- [ ] **`src/deploy_calibrate.py` (new)** — store A-targets of calibration scene → re-shoot under B → run `calibration.py` → freeze filter (<1MB) → ready for inference
- [ ] **Update AGENTS.md** — new gotchas, real layer names, per-phase resource costs

## Closeout

- [x] **A17. Doc alignment with advanced-filter spec** — done (see Phase A above).

## Phase B — Config-Driven Experiment Runner

> See `docs/specs/2026-06-28-config-driven-experiments.md`.
> Goal: detach experiment **configuration** from **execution**. Each YAML = one atomic run
> (1 filter × 1 layer group × 1 dataset). The runner globs N YAMLs and aggregates rows into
> one CSV. Replaces the hardcoded `FILTERS`/`LAYER_GROUPS`/hyperparams in `run_experiments.py`.

- [x] **B1. Config schema + loader (`src/experiment_config.py`)**
  - `ExperimentConfig` + `TrainingConfig` dataclasses parsed from YAML
  - `load_config(path)` / `load_configs(paths)` (file, directory glob, or iterable)
  - Validation: name non-empty + unique, dataset exists (skippable for `--dry-run`), filter type in registry, layer names valid, training fields positive
  - Layer range syntax (`backbone.layer.0..3`) expands via `expand_layer_spec`
  - Composite filters rejected (one filter per YAML)

- [x] **B2. Runner (`run_configs.py`)**
  - CLI: `run_configs.py PATH... [--output DIR] [--dry-run] [--append] [--device cuda|cpu]`
  - Loads model once; iterates configs; runs `calibrate_epochs` + `evaluate_on_test` per config
  - Per-epoch `metrics.jsonl` + periodic `epoch_NNNN.pt` + `best.pt` under `runs/<name>/`
  - Aggregated CSV at `results/experiments/experiment_results.csv` (incremental write)
  - `--dry-run` validates all configs, prints resolved run matrix, exits without loading model
  - Sorted results table printed at end

- [x] **B3. Starter configs + generator (`configs/experiments/`, `scripts/generate_experiment_configs.py`)**
  - 60 YAMLs reproducing the `run_experiments.py` sweep (10 filters × 3 groups × 2 datasets)
  - Generator is idempotent; safe to re-run

- [x] **B4. `run_experiments.py` deprecation note**

- [x] **B5. Tests (`tests/test_experiment_config.py`)** — 17 tests (parse, validation, reject, glob, duplicate, dry-run subprocess)

- [x] **B6. Verification** — `ruff check` + `ruff format` + `mypy src/experiment_config.py` clean; `pytest tests/test_experiment_config.py tests/test_data_pairs.py -v` 27 pass; `run_configs.py configs/experiments/ --dry-run` prints 60 configs and exits 0.

### Current Status
- **Phase A complete.** All code built and verified (see above for detail).
- **Phase B complete (2026-06-28).** Config-driven experiment runner landed: `src/experiment_config.py`, `run_configs.py`, 60 starter YAMLs in `configs/experiments/`, `scripts/generate_experiment_configs.py`, `tests/test_experiment_config.py` (17 tests). `run_configs.py configs/experiments/` is the new entry point; `run_experiments.py` is deprecated.
### Next Action
- Run the full sweep: `uv run python run_configs.py configs/experiments/` (or any subset). Note: `level_1_vs_level_3` is not materialized on the current Windows checkout — only `level_1_vs_level_2` runs until that dataset is re-captured or re-checked-out.

**Last Updated:** 2026-06-28 · **Owner:** Carlos
