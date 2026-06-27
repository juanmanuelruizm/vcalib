# Project: vcalib — Calibration Filter for RF-DETR

## What Is This

A research framework to develop and validate a learnable **differentiable** preprocessing filter that corrects illumination shifts in RF-DETR real-time object detection. Goal: recover mAP performance when deployed under different lighting without retraining the model. **Parameter count is not a constraint; the primary guardrail is held-out validation generalization.** Calibration runs on edge in seconds, with no model retraining. See `docs/specs/2026-06-27-advanced-filters.md` for the expressiveness-first filter philosophy.

## Stack

- **Language:** Python 3.10+
- **Model loader:** LibreYOLO (git submodule at `3rd_party/libreyolo`, v1.2.0.dev0) — `from libreyolo import LibreRFDETR; LibreRFDETR(size="n")`. Pulls `transformers>=5.1` for the DINOv2 backbone only. RF-DETR nano weights auto-download to `3rd_party/libreyolo/weights/rf-detr-nano.pth`.
- **Activation access:** LibreYOLO does **not** expose `output_hidden_states` through its wrapper. The internal model is `libre.model.model` (an `LWDETR`); activations are captured with **PyTorch `register_forward_hook`** on submodules. Canonical layer names (see `src/utils/activations.py::LAYER_PATHS`): `backbone.layer.0..11` (12 DINOv2 ViT blocks), `backbone.projector` (multi-scale projector = encoder memory), `decoder.layer.0..1`.
- **Filter insertion point:** the [0,1] RGB tensor **before** ImageNet mean/std normalization (confirmed by inspecting LibreYOLO's `preprocess_numpy`).
- **Filter library:** `src/filters/` — all filters operate on [0,1] NCHW RGB, identity init = no-op, output clamped to [0,1], params in a physical range, end-to-end differentiable. `FILTER_REGISTRY` + `get_filter(name)` / `build_filter(spec)` instantiate by name for the grid. **Initial library (A3):** `brightness_2param`, `white_balance_3param`, `affine_6param`, `saturation_1param`, `contrast_1param`, `gamma_3param`, `matrix_12param`, `CompositeFilter` (chain). **Advanced library (A9–A15, see `docs/specs/2026-06-27-advanced-filters.md`):** 3D LUT (`lut_3d`), per-channel monotone tone curves (`tone_curve`), high-order/root-polynomial CCM (`ccm_high_order`), LMS/Bradford chromatic adaptation (`chromatic_adaptation`), large-K spatial tone curve (`spatial_tone_curve`), CLAHE-like local tone mapping (`local_tonemap`), low-rank 3D LUT (`lut_3d_lowrank`). The library spans low- to high-capacity filters; **filter type is a grid axis, not an escalation ladder.**
- **Filter regularization:** `Filter.reg_loss() -> Tensor` (default 0; high-capacity filters override with TV/smoothness + identity-anchoring). The calibration loop adds `reg_weight * filter.reg_loss()` to the group loss (`configs/grid.yaml::training.reg_weight`). This matters more than param count for high-capacity filters.
- **Spatial filters:** `src/filters/spatial.py` — zone-dependent variants (`spatial_brightness`, `spatial_whitebalance`, `spatial_affine`, `spatial_gamma`) via a bilinear K×K control grid upsampled with `grid_sample`. Params = `K²·n_field` (K=2→8/12/12/24; K=3→18/27/27/54). For non-uniform illumination (vignetting/directional light/shadow). K controls spatial precision (informational); larger K raises overfitting risk on a single calibration scene, gated by held-out val.
- **Package manager:** uv (fast, reproducible)
- **Compute:** GPU optional (CUDA if available; CPU fallback for diagnostics)
- **Dev tools:** pytest, mypy, ruff (linting + formatting)

## Commands (Essential)

```bash
# Install dependencies (creates .venv, installs from pyproject.toml)
uv sync

# Run the full test suite (all phases on a small subset)
uv run pytest tests/ -v

# Phase 1 only: diagnostic sweep (which layers have signal?)
uv run python src/diagnostics.py --dataset-path data/raw/ --output results/phase1_diagnostics.json

# Phase 2 only: grid sweep (which filter/layer combo is best?)
uv run python src/grid_search.py --config configs/grid.yaml

# Phase 3 only: benchmark & report (mAP recovery, convergence, cost)
uv run python src/benchmark.py --results-dir results/

# Type check & lint (pre-commit hook runs this)
uv run mypy src/             # type errors
uv run ruff check src/       # style + logic
uv run ruff format src/      # auto-format

# Interactive debugging (Phase 1 with plots)
uv run python src/diagnostics.py --dataset-path data/raw/ --debug --plot
```

## Conventions

- **Naming:** `kebab-case` for files, `snake_case` for Python functions/vars, `PascalCase` for classes
- **Filter names:** initial library uses explicit parameter count (`affine_6param.py`, `matrix_12param.py`). Advanced filters (A9–A15) use descriptive names without a count (`lut_3d.py`, `tone_curve.py`, `ccm_high_order.py`, …) since param count is no longer fixed (varies with `size`/`P`/`grid_size`/`degree`/`M`). Existing count-suffixed names are kept (no rename).
- **Commits:** Conventional Commits (`feat: add X`, `fix: resolve Y`, `refactor: Z`); one commit per completed phase/task
- **Config files:** YAML in `configs/`; versioned, never hardcoded hyperparams
- **Results:** Always save to `results/runs.csv` with timestamp + config hash for reproducibility

## Workflow (Mandatory)

1. **For tasks ≥ 3 steps or architecture decisions:**
   - Enter plan mode (`Shift+Tab` in Claude Code, or `/plan`)
   - Have Claude propose the plan; **review it carefully**; approve before code
   - Write the plan in `docs/specs/YYYYMMDD-[feature].md` and `tasks/todo.md`

2. **Write code against the plan** — don't explore mid-implementation

3. **Test before committing:**
   - Diagnostic phase: `uv run pytest tests/ -v`
   - Grid phase: `uv run python src/grid_search.py --config configs/grid.yaml --subset 5` (5 quick runs, not all 35 groups × ~15 filters)
   - Full verification only when claiming done

4. **Commit per completed task** (in `tasks/todo.md`), never per-file

5. **If tests fail:** pause, go back to plan mode, understand why, re-plan, then code again. Don't iterate blind.

## Gotchas (Lessons Learned / To Avoid)

### Dataset & Capture
- **More illumination levels beats more scenes.** 3–5 intensity steps (uniform progression between A and B) catch non-linearity; 2 extremes are too sparse.
- **Pair images **exactly**: same camera, tripod, same framing, only light changes. Even 1° of rotation breaks alignment.
- **Don't skip ground truth.** Without bounding box labels on condition B, you can't compute true mAP; proxies (IoU agreement with A) are weak signals.

### Phase 1 (Diagnostics)
- **Early bailout if no signal.** If distance curve is flat across all layers (model already invariant), stop. Spend time on a harder shift or different camera/object type.
- **Distinguish backbone vs. encoder vs. decoder.** Early backbone layers are best for photometric correction; decoder layers are too abstract and harder to optimize.

### Phase 2 (Grid Search)
- **Loss is computed over GROUPS of layers, not a single layer.** The group loss = mean of per-layer normalized L2 (`||a-b||/||a||`) across all layers in the group (see `configs/grid.yaml::loss`). Groups are encoded declaratively — explicit baselines (`layer_groups`) + auto-generated contiguous DINOv2 windows (`layer_group_search`) — and resolved by `src/utils/layer_groups.py::resolve_grid_groups()`. Run `uv run python src/utils/layer_groups.py --dump` to see the resolved candidate groups; this is what the automatic layer-search loop iterates.
- **Held-out validation is the primary guardrail, not parameter count.** Calibration fits against a single reference scene, so high-capacity filters (3D LUT, large-K spatial) can perfectly fit the calibration A/B pair and generalize badly to deployment B. Always rank configs by val distance; the `overfit_gate` (`configs/grid.yaml::validation`) drops configs whose val recovery is too small a fraction of their train recovery. Smoothness/identity-anchoring regularization (`Filter.reg_loss()`, weighted by `training.reg_weight`) matters more than param count.
- **Loss function rarely matters.** L2 vs. cosine usually <5% difference; don't waste grid dimension on this unless Phase 1 shows extreme outliers.

### Phase 3 (Benchmark)
- **Calibration cost is offline.** The filter adds <1ms inference; the cost is finding optimal params (10–100 gradient steps, ~100ms). Must happen in deployment before detection runs.
- **GPU memory tight.** Even RF-DETR nano with activation hooks uses ~2GB; batch=1 if constrained. Pre-compute and cache activations if doing multiple runs.
- **Reproducibility via seeding.** Fix `torch.manual_seed()` and dataset shuffle seed; results must be bitwise identical across reruns for comparison.

### General
- **Don't commit without running tests.** If you say "tests pass," run them and show output; don't assume.
- **Phase 0 (dataset capture) is the bottleneck.** Everything else is compute; getting good diverse image pairs is the hard part. Invest time here.
- **Prefer specs → plans → code over code → specs.** Once code is written, specs are written to justify code. Reverse the order; code comes last.

## Project-Specific Rules

- **Activation caching:** RF-DETR forward passes are slow; cache backbone/encoder/decoder outputs to `results/activations_cache/` and reuse across Phase 2 runs.
- **Config versioning:** Every grid run logs its config; `results/runs.csv` has a `config_hash` column. Reproducibility via config+seed, not manual re-runs.
- **Phases are blockers:** Don't start Phase 2 until Phase 1 diagnostics are written. Don't start Phase 3 until Phase 2 grid is complete. Phases order matters.

## References

- **LibreYOLO (model loader):** https://github.com/LibreYOLO/libreyolo — `LibreRFDETR` wraps RF-DETR with DINOv2 backbone; internal `LWDETR` at `libre.model.model`.
- **RF-DETR docs (HF transformers, backbone only):** https://huggingface.co/docs/transformers/model_doc/rf_detr
- **Color correction theory (ISP CCM):** See `docs/` for references on white balance, sensor gain, cross-channel crosstalk
- **Phase plan details:** See `docs/specs/2026-06-27-experimental-plan.md` (frozen spec before implementation)

---

**Project Owner:** Carlos  
**Last Updated:** 2026-06-27  
**Status:** Planning phase — specs + task list written; code not yet started
