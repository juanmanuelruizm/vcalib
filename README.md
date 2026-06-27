# vcalib — Calibration Filter for RF-DETR Under Illumination Shifts

*Lightweight learnable filter to recover RF-DETR detection performance under real-world illumination changes. Deployed on edge with 6–12 parameters.*

---

## Quick Start

```bash
# Setup (once)
uv sync              # install dependencies from pyproject.toml

# Daily workflow
uv run pytest        # run full test suite
uv run python src/diagnostics.py --dataset-path data/raw/  # Phase 1
uv run python src/benchmark.py --results-dir results/      # Phase 3
```

**For the first time:** read `CLAUDE.md` (project rules + gotchas), then check `tasks/todo.md` (what's being worked on).

---

## Navigation Map

| What | Where | Why |
|------|-------|-----|
| **Project rules & gotchas** | `CLAUDE.md` (root) | Claude loads this in every session; curated over time as we learn what works |
| **Feature specs** | `docs/specs/2026-06-27-*.md` | High-level design before code; frozen checkpoint for each feature |
| **Task checklist** | `tasks/todo.md` | What's done, what's in progress, what's blocked |
| **Diagnostic code** | `src/diagnostics.py` | Phase 1: understand which layers carry illumination signal |
| **Grid executor** | `src/grid_search.py` | Phase 2: train filters across the parameter sweep |
| **Benchmark runner** | `src/benchmark.py` | Phase 3: automated evaluation; produces `results/runs.csv` |
| **Parametric filters** | `src/filters/` | 6-param affine, 12-param matrix, optional curve filters |
| **Test data** | `data/raw/` | A/B image pairs (same scene, different illumination) |
| **Results** | `results/` | Logs, plots, trained filter checkpoints per run |

---

## The Problem

RF-DETR achieves real-time object detection but **fails when camera illumination differs from training**. Standard solutions (retrain, domain adaptation, hand-tuned preprocessing) are slow or non-portable. 

**Our approach:** a **learnable parametric filter** (6–12 parameters) that aligns image representations at an intermediate RF-DETR layer, recovering detection performance at inference time without retraining the detector.

---

## The Solution in Three Phases

### Phase 1: Diagnostic — Where Is the Signal?
*Status: Not started*  
*Expected cost: ~5 min CPU*

- Forward A/B image pairs through frozen RF-DETR model
- Measure per-layer distance (L2, cosine) between activations
- Identify which layers carry illumination-related information
- Output: heatmap (layer × illumination level) → determines candidates for Phase 2

**Runnable soon:** `uv run python src/diagnostics.py --dataset-path data/raw/`

### Phase 2: Grid Search — Optimal Filter
*Status: Not started*  
*Expected cost: ~1–2 hours GPU*

Sweep in priority order (cheap → expensive):
1. **Capa/layer** (determined by Phase 1): early backbone vs. encoder output vs. decoder
2. **Filter complexity**: 6 params (affine per-channel) → 12 params (matrix 3×3)
3. **Loss function**: L2 norm vs. cosine (usually minimal impact)

Best configuration: minimal parameters that recover mAP.

**Runnable soon:** `uv run python src/grid_search.py --config configs/grid.yaml`

### Phase 3: Benchmark — Verify Downstream Effect
*Status: Not started*  
*Expected cost: ~10 min per run (parallelizable)*

For each Phase 2 result:
- Measure loss convergence (training metric)
- Measure distance reduction (sanity check on optimization)
- **Measure mAP recovery** (the real criterion)
- Count detections gained/lost vs. reference
- Log CPU cost + wall-clock calibration time

Output: `results/runs.csv` ranked by mAP recovery.

**Runnable soon:** `uv run python src/benchmark.py --results-dir results/`

---

## Project Rules

**Read `CLAUDE.md` for the canonical project rules, conventions, and known gotchas.** It's versioned in git and loaded in every Claude Code session.

Key excerpts:

- **No code changes without running tests first.** Test command: `npm test`
- **Prefer plans over exploratory code.** Use plan mode (`Shift+Tab`) for architectural decisions; spec must be written before implementation.
- **Commits by phase.** One commit per completed phase/task, not one per file.
- **Verification before claiming success.** If you say "tests pass", run them and paste the output.

---

## Development Workflow

This project uses the **4-phase workflow** from Claude Code best practices:

```
Explore → Plan → Implement → Commit
```

**For non-trivial work:**
1. Enter plan mode (`Shift+Tab`)
2. Explore code + existing specs
3. Get approval on the plan before writing code
4. Implement against plan, commit per step
5. Run verification (tests, benchmark)
6. Update `CLAUDE.md` if you discover new gotchas

**For trivial changes** (1-liner, no architecture decision): skip the plan, but still commit and test.

---

## Verification

**The most important practice: give Claude a way to verify its own work.**

- **Tests:** `uv run pytest` runs unit + integration tests. CI gate.
- **Benchmarks:** `uv run python src/benchmark.py --results-dir results/` runs the full Phase 1–3 pipeline on a test subset, producing results tables.
- **Downstream detection:** `uv run python src/benchmark.py --plot` generates visualizations of mAP recovery.

**Rule:** If you claim something works, run the verification command and show the output. Avoid "it should work" — measure instead.

---

## Stack & Commands

| What | Command |
|------|---------|
| **Install** | `uv sync` |
| **Dev/interactive** | `uv run python src/diagnostics.py --debug` (Phase 1) |
| **Full pipeline** | `uv run pytest` (all tests) |
| **Grid sweep** | `uv run python src/grid_search.py --config configs/grid.yaml` |
| **Benchmark report** | `uv run python src/benchmark.py --results-dir results/` |
| **Type check** | `uv run mypy src/` |
| **Lint** | `uv run ruff check src/` |
| **Format** | `uv run ruff format src/` |

**Language:** Python 3.10+  
**Package manager:** uv (fast, deterministic)  
**GPU:** Optional (CUDA if available, CPU fallback)  
**Main dependency:** `transformers` (RF-DETR, HuggingFace v5.1+)

---

## Key Files to Know

- **`CLAUDE.md`** — Project rules, conventions, command reference, known gotchas. **Start here.**
- **`docs/specs/`** — Specs frozen at decision time; referenced in PRs.
- **`tasks/todo.md`** — What's done/in-progress; checkmarks as we go.
- **`configs/grid.yaml`** — Phase 2 sweep parameters (layers, filter types, loss functions).
- **`src/filters/affine_6param.py`**, **`src/filters/matrix_12param.py`** — Filter implementations.
- **`results/runs.csv`** — Output of Phase 3; ranked by mAP recovery.

---

## Architecture Decision: Parametric Filters

Two filter options under investigation:

| Filter | Params | Covers | Notes |
|--------|--------|--------|-------|
| **Affine per-channel** | 6 | Exposure, color temperature (per-channel gains), offsets | Physics-correct: matches sensor gain + white balance |
| **Full matrix 3×3** | 12 | Above + cross-channel coupling (sensor crosstalk, metamerism) | Standard ISP color correction matrix (CCM) |
| **Curves (future)** | 12–24 | Non-linear response if needed | Only if Phase 2 shows systematic non-linear residuals |

**Rationale:** Start with 6 params (minimal, robust); if mAP plateaus, escalate to 12 (full matrix). Higher-order models only if diagnostics show non-linearity.

---

## Known Gotchas

- **Dataset balance:** More illumination levels (3–5, not just 2 extremes) → cleaner Phase 1 diagnostics.
- **Layer selection:** Don't blindly pick all layers. Phase 1 heatmap guides choice; training only on layers with real signal.
- **Overfitting to pair:** Training on 20–30 scene pairs risks overfitting; always validate on held-out test subset.
- **GPU memory:** Even nano RF-DETR with activation hooks uses ~2GB; batch size = 1 if constrained.
- **Inference time:** Filter itself adds <1ms; the cost is model forward pass (unchanged). Calibration (inference mode finding optimal params) costs 10–100 gradient steps; must happen offline.

---

## Roadmap (Phases)

| Phase | Status | Owner | Deadline | Notes |
|-------|--------|-------|----------|-------|
| **Phase 0** | Not started | — | TBD | Dataset capture (A/B pairs + intensity levels) |
| **Phase 1** | Not started | — | TBD | Diagnostic sweep (distance heatmap per layer) |
| **Phase 2** | Not started | — | TBD | Grid search (6 + 12 param filters) |
| **Phase 3** | Not started | — | TBD | Benchmark & report (ranked mAP recovery) |
| **Deployment** | Future | — | — | Edge calibration binary + model integration |

---

## Contributing

1. **Check `CLAUDE.md`** for project rules before starting.
2. **Check `tasks/todo.md`** to see what's in-progress or blocked.
3. **Use plan mode** (`Shift+Tab`) for architectural decisions.
4. **Write code against a plan**, not exploratory.
5. **Run tests/benchmark before claiming done.**
6. **Commit per phase**, with descriptive messages.
7. **Update `CLAUDE.md`** if you discover new gotchas or patterns.

---

## References

- **RF-DETR**: HuggingFace transformers (v5.1+) — [`RfDetrModel`, `RfDetrForObjectDetection`](https://huggingface.co/docs/transformers/model_doc/rf_detr)
- **Color science**: ISP CCM (3×3 matrix), white balance via per-channel gain — standard in camera signal processing
- **Related work**: 
  - Exposure (Hu et al.) — RL for tonal curves
  - White-Box Photo Post-Processing — parametric filter taxonomy
  - DeepLPF — learned per-layer filtering

---

## Project Info

- **Owner**: Carlos
- **Created**: 2026-06-27
- **Status**: Planning phase (specs & tasks finalized, code not yet started)
- **Last updated**: 2026-06-27
- **Next action**: Finalize dataset capture protocol → start Phase 1 diagnostics

---

**For setup help or clarifications on workflow:** see `CLAUDE.md` or run `/help` in Claude Code.
