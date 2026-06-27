# Project: vcalib — Calibration Filter for RF-DETR

## What Is This

A research framework to develop and validate a learnable parametric filter (6–12 parameters) that corrects illumination shifts in RF-DETR real-time object detection. Goal: recover mAP performance when deployed under different lighting without retraining the model. Constraint: small enough to calibrate on edge in seconds.

## Stack

- **Language:** Python 3.10+
- **Framework:** PyTorch + HuggingFace transformers (v5.1+, RF-DETR)
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
- **Filter names:** `affine_6param.py`, `matrix_12param.py` (explicit parameter count)
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
   - Diagnostic phase: `npm test`
   - Grid phase: `npm run benchmark --subset 5` (5 quick runs, not all 18)
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
- **Overfitting risk.** 20–30 scenes × 1 filter config = you're fitting to the specific pairs. Always hold out a validation subset before calling a config "optimal."
- **Don't add parameters you don't need.** If 6 params (affine) plateau at 80% mAP recovery, jumping to 12 (matrix) likely won't help much; investigate residuals instead.
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

- **RF-DETR docs:** https://huggingface.co/docs/transformers/model_doc/rf_detr
- **Hooks into HuggingFace models:** https://huggingface.co/docs/transformers/en/internal/modeling_outputs
- **Color correction theory (ISP CCM):** See `docs/` for references on white balance, sensor gain, cross-channel crosstalk
- **Phase plan details:** See `docs/specs/2026-06-27-experimental-plan.md` (frozen spec before implementation)

---

**Project Owner:** Carlos  
**Last Updated:** 2026-06-27  
**Status:** Planning phase — specs + task list written; code not yet started
