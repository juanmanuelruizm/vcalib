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

- [ ] **A3b. Base-contract `reg_loss()`** — prerequisite for all advanced filters
  - Edit `src/filters/base.py`: add `Filter.reg_loss() -> torch.Tensor` (default: zero tensor on the param device)
  - Edit `src/filters/composite.py`: `CompositeFilter.reg_loss()` = sum of sub-filters' `reg_loss()`
  - Additive (default 0) → existing 119 tests stay green
  - Verify: `uv run pytest tests/test_filters.py tests/test_spatial_filters.py -v` && `uv run mypy src/ && uv run ruff check src/`

- [ ] **A4. Calibration loop (`src/calibration.py`, new)** — needed by acceptance criteria #6/#7
  - Input: filter, stored A-targets, B image, **layer group** (from `layer_groups.py`), loss+aggregation cfg, optimizer cfg, `reg_weight`
  - Group loss = mean of per-layer normalized L2 (`||a-b||/||a||`) across the group's layers + `reg_weight * filter.reg_loss()`
  - Adam lr 1e-3, max 100 steps, early-stop patience 10; `torch.manual_seed` for reproducibility
  - Output: trained filter + stats (steps, wall-clock, final train loss, final val loss)
  - Shared by grid search **and** the deployed tool
  - Verify: `uv run pytest tests/test_calibration.py -v`

- [ ] **A8. Stand-in re-lit data (`src/utils/synth_relit.py`, new)** — proxy A/B/B_val until Phase 0 capture
  - Deterministic: A = real `data/raw` photo; B_train = A·`gamma=1.4 + gains=[1.10,0.95,0.85]` (warm + underexposed); B_val = A·`gamma=1.2 + gains=[1.05,0.97,0.90]` (same direction, milder — held-out probe)
  - Verify: `uv run pytest tests/test_synth_relit.py -v`

- [ ] **A9. `src/filters/lut_3d.py`** (`LUT3D`, registry `lut_3d`) — **headline**: 3D LUT, trilinear interp, identity-LUT init, TV+identity reg_loss. Acceptance: `uv run pytest tests/test_lut_3d.py -v`
- [ ] **A10. `src/filters/tone_curve.py`** (`ToneCurve`, `tone_curve`) — monotone (cumsum-softplus) per-channel curves; curvature+identity reg_loss. Acceptance: `uv run pytest tests/test_tone_curve.py -v`
- [ ] **A11. `src/filters/ccm_high_order.py`** (`HighOrderCCM`, `ccm_high_order`) — root-polynomial CCM (Finlayson), degree≤3, normalized features; L2-toward-identity reg_loss. Acceptance: `uv run pytest tests/test_ccm_high_order.py -v`
- [ ] **A12. `src/filters/chromatic_adaptation.py`** (`ChromaticAdaptation`, `chromatic_adaptation`) — Bradford LMS diagonal/full; identity init; optional L2 reg. Acceptance: `uv run pytest tests/test_chromatic_adaptation.py -v`
- [ ] **A13. `src/filters/spatial_tone_curve.py`** (`SpatialToneCurve`, `spatial_tone_curve`) — reuses `SpatialFilter` grid + monotone curves per zone; spatial TV+curvature+identity reg_loss. Acceptance: `uv run pytest tests/test_spatial_tone_curve.py -v`
- [ ] **A14. `src/filters/local_tonemap.py`** (`LocalTonemap`, `local_tonemap`) — CLAHE-like approximation: guided-filter local-mean + K×K gain field; TV+identity reg_loss. True CLAHE = future stretch. Acceptance: `uv run pytest tests/test_local_tonemap.py -v`
- [ ] **A15. `src/filters/lut_3d_lowrank.py`** (`LUT3DLowRank`, `lut_3d_lowrank`) — LUT = identity + Σ w_m·B_m (fixed basis); L2-on-weights reg_loss; generalization-friendly LUT variant. Acceptance: `uv run pytest tests/test_lut_3d_lowrank.py -v`

  Each of A9–A15: `tests/test_<filter>.py` runs the 7 acceptance criteria from the spec (identity-on-real-photos, params-in-range, differentiability finite-non-zero grads, registry+build with kwargs, composite chaining + reg_loss sum, smoke via A4 on B_train ≥30% distance reduction, generalization on B_val with overfit gate `val/train ≥ 0.5`).

- [ ] **A16. Registry + grid integration**
  - Register F1–F7 in `src/filters/__init__.py::FILTER_REGISTRY` + `build_filter` kwargs (`size`/`P`/`grid_size`/`degree`/`mode`/`M`)
  - Update `configs/grid.yaml::grid.filters` (F4/F3/F2/F1(N=9)/F5/F6 active; F1(N=17)/F7 + composites commented); add `training.reg_weight: 0.01`; add `validation.overfit_gate` (enabled, `min_val_recovery_ratio: 0.5`)
  - Verify: `uv run python src/utils/layer_groups.py --dump` && `uv run pytest tests/ -v` && `uv run mypy src/ && uv run ruff check src/ && uv run ruff format --check src/ tests/`

- [ ] **A5. Unit tests (`tests/test_filters.py`)**
  - Identity init = no-op (every filter, incl. on real photos in `data/raw/`); params stay in range
  - Registry `get_filter`/`build_filter`/`make_composite`; composite chaining & param counts
  - 59 filter tests passing; smoke: loop reduces activation distance on one image with a programmatic light tweak (smoke only, not a dataset) — **done via A4 + A8**

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
- [ ] **Grid config (`configs/grid.yaml`)** — layer groups (resolved via `layer_groups.py`) × full filter library (initial A3 + advanced F1–F7 from A9–A15) × {l2}; adam, lr 1e-3, 100 steps, patience 10, `reg_weight: 0.01`; val split 0.2, seed 42, `overfit_gate` enabled (min_val_recovery_ratio 0.5)
- [ ] **Grid executor (`src/grid_search.py`)** — train via `calibration.py` on dev-train; measure distance reduction on dev-val; append `results/runs.csv` `[run_id, group_name, filter, loss, final_train_loss, val_distance, steps_to_converge, timestamp, config_hash]`; checkpoint `results/checkpoints/`; apply overfit gate (drop configs with `val_reduction/train_reduction < min_val_recovery_ratio`); add `--subset N` flag for quick runs
- [ ] **Automatic layer-search loop** — iterate `resolve_grid_groups()` × filter candidates, rank by val_distance (asc), steps (tie-break); pick top 3 for Phase 3

## Phase 3 — Benchmark (after Phase 2)

- [ ] **Benchmark harness (`src/benchmark.py`)** — top 3: model on B with/without filter; **proxy** vs A's predictions (box IoU + class agreement); calibration cost → `results/runs_phase3.csv`
- [ ] **Rank** by proxy recovery %; winner = highest recovery with ≤100 steps
- [ ] **Benchmark report (`docs/phase3_report.md`)** — top configs, recommended deploy choice, calibration cost, residual failure modes

## Phase D — Deployable Tool

- [ ] **`src/deploy_calibrate.py` (new)** — store A-targets of calibration scene → re-shoot under B → run `calibration.py` → freeze filter (<1MB) → ready for inference
- [ ] **Update AGENTS.md** — new gotchas, real layer names, per-phase resource costs

## Closeout

- [ ] **A17. Doc alignment with advanced-filter spec** — apply the `AGENTS.md` framing edits (§1, filter-library, spatial, Phase 2 gotchas, stale `npm`→`uv` commands, filename convention) and the `docs/specs/2026-06-27-experimental-plan.md` alignment edits (§1/§2/§3/§5/§6/§8/§9 budget references → expressiveness-first + pointer to `2026-06-27-advanced-filters.md`); mark A3b/A4/A8–A16 done here as they complete.

---

### Current Status
- A1 + A2 + A3 done: environment set up, rf-detr-nano downloaded via LibreYOLO, activation helper + initial filter library (global + spatial) built and verified (119 tests pass). Advanced-filter spec approved (`docs/specs/2026-06-27-advanced-filters.md`): expressiveness-first, `reg_loss()` base change, F1–F7 roadmap, stand-in re-lit data, held-out-val overfit gate.
### Next Action
- A3b: add `Filter.reg_loss()` (base.py + composite.py) — prerequisite for all advanced filters. Then A4 (calibration loop) + A8 (stand-in re-lit data) in parallel.

**Last Updated:** 2026-06-27 · **Owner:** Carlos
