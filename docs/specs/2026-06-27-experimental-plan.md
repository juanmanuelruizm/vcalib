# Spec: vcalib — Illumination-Calibration Filter for RF-DETR nano

**Status:** Approved (reframed as deployable tool)
**Date:** 2026-06-27
**Owner:** Carlos

---

## 1. Objective

Build a deployable **on-edge calibration tool**: a tiny learnable preprocessing filter
(6–12 parameters) that edits the *input image* so that a frozen RF-DETR nano model produces
the same *intermediate activations* under new lighting (condition B) as it did under the
reference lighting it was validated on (condition A). The **distance between activations** is
the training signal — the filter only touches pixels, never the model. Calibration must run on
edge in seconds, with no model retraining.

## 2. How It Works (the core idea)

1. Capture aligned image pairs: same camera/tripod/framing/objects; **only illumination changes** (A → B).
2. Forward both through frozen RF-DETR nano; read intermediate activations at a chosen layer.
3. Insert a small pixel-space filter before the model. Optimize its 6–12 params (gradient
   descent) so that the **filtered-B** activations match the **stored-A** activations.
4. Because activations align, the model's *outputs* on filtered-B approximate those on A —
   recovering detection quality without retraining.

The activations are the **loss signal only**; the filter edits the RGB image.

## 3. Locked Design Decisions

1. **Aligned pairs.** Position-for-position comparable activations require identical framing; only light differs.
2. **Reference A precomputed offline.** Store A's target activations once; at deploy time only B is seen and pulled toward stored A.
3. **Filter edits pixels, not activations.** Per-channel affine (6 params) or 3×3 matrix + offset (12 params), operating on the 0–1 RGB tensor before the model's normalization.
4. **Layer chosen by the diagnostic sweep**, not assumed up front.
5. **Goal = deployable on-edge tool** (calibrate in seconds), not just a research report.
6. **Metric = proxy.** Agreement with A's predictions (box IoU + class agreement). No true mAP / ground-truth labeling.
7. **Model = HuggingFace transformers RF-DETR** (`rf-detr-nano`, merged into transformers 2026-05-07; `output_hidden_states=True` exposes hidden states). Roboflow `rfdetr` is the fallback. Verify activation access as task one.

## 4. Two Distinct Datasets (important)

- **Development dataset** (Phases 1–2): multiple scenes × 3–5 illumination levels, with a
  held-out validation split. Used to *decide* which layer + filter generalize. Diversity matters.
- **Calibration scene** (runtime): a single fixed, re-shootable reference scene (grey-card /
  fixed object set). The deployed tool stores A-activations of this scene, re-shoots it under
  new light B, runs N gradient steps, freezes the filter, then runs normal detection.

> **Data status:** Carlos chose to **wait for real capture** (no synthetic dataset). Therefore
> the immediately buildable deliverables are: model/activation verification, filter +
> calibration-loop implementation with unit tests (single-image smoke test only), and a
> finalized capture protocol. Phase 1–3 *runs* are gated on the captured dataset; their *code*
> is built now.

## 5. Scope

### In Scope
- Pixel-space parametric filters: affine per-channel (6 params), full matrix 3×3 + offset (12 params).
- Reusable calibration loop (`src/calibration.py`) shared by grid search and the deployed tool.
- Diagnostic sweep to find which layer carries illumination signal.
- Grid search to pick layer + filter + loss, validated on a held-out split.
- Proxy benchmark (agreement with A's predictions) + calibration-cost measurement.
- A deployable calibration CLI (`src/deploy_calibrate.py`).

### Out of Scope
- Retraining / fine-tuning RF-DETR (frozen throughout).
- Domain adaptation (adversarial, unsupervised).
- Other sensors (multi-spectral, thermal).
- Non-parametric filters (LUTs, curves) — optional future tier if parametric saturate.
- True mAP evaluation / ground-truth labeling (using proxy instead).
- Synthetic illumination dataset (Carlos opted for real capture).

## 6. Architecture: Components

| Component | File | Role |
|-----------|------|------|
| Activation helper | `src/utils/activations.py` | Load frozen `rf-detr-nano`; `extract_activations(image)` → named layer dict via `output_hidden_states=True`; cache/load to `data/processed/activations_cache/`. **Task one: dump real layer names/shapes.** |
| Affine filter | `src/filters/affine_6param.py` | `Affine6Param(nn.Module)`: gains `a_c∈[0.1,2.0]`, offsets `b_c∈[-1,1]`, identity init, `forward(x)→clamp[0,1]`, `get_params()`. |
| Matrix filter | `src/filters/matrix_12param.py` | `Matrix12Param(nn.Module)`: `I' = M·I + b`, M init = identity. Same pixel-space contract. |
| Calibration loop | `src/calibration.py` (new) | Core loop shared by grid search + deployed tool: given filter, stored A-targets, B image, layer, loss, optimizer cfg → gradient steps minimizing activation distance, early stopping → trained filter + convergence stats (steps, wall-clock, final loss). |
| Phase 1 diagnostics | `src/diagnostics.py` | Per-layer L2(normalized)+cosine distance per illumination level, aggregated mean±std over scenes → `results/phase1_diagnostics.json` + heatmap/line plots. Early-bailout if flat. |
| Phase 2 grid | `src/grid_search.py` | Sweep layer × filter × loss from `configs/grid.yaml`; train via `calibration.py` on dev-train; measure distance reduction on dev-val; log `results/runs.csv` (+ `config_hash`, timestamp); checkpoint top configs. |
| Phase 3 benchmark | `src/benchmark.py` | Top configs: model on B with/without filter; proxy metric vs A's predictions (IoU + class agreement); calibration cost → `results/runs_phase3.csv` + `docs/phase3_report.md`. |
| Deploy CLI | `src/deploy_calibrate.py` (new) | Store A-targets of calibration scene → re-shoot under B → run `calibration.py` → freeze filter (<1MB) → ready for inference. |

### Loss & training
- Distance at chosen layer between filtered-B and stored-A activations: normalized L2 (primary),
  cosine (secondary, deferred unless Phase 1 shows extremes).
- Adam, lr 1e-3, max 100 steps, early-stop patience 10.
- Seed `torch.manual_seed` + split seed for bitwise-reproducible reruns.

### Parametric filter design

| Filter | Params | Physics | Caveats |
|--------|--------|---------|---------|
| **Affine per-channel** `I'_c = a_c·I_c + b_c` | 6 | Sensor gain + per-channel offset (white balance + exposure) | No cross-channel mixing / non-linearity |
| **Matrix 3×3 + offset** `I' = M·I + b` | 12 | ISP color-correction matrix (CCM); cross-channel coupling | Still linear; no gamma / clipping |
| **Curves (optional)** spline/channel | 12–24 | Non-linear tone response | Overfitting risk; only if needed |

Start at 6 params; escalate to 12 only if Phase 2 shows systematic residuals.

## 7. Phased Execution (phases are blockers)

**Phase A — Unblocked now (no dataset needed)**
1. `uv sync`; verify `rf-detr-nano` loads; dump real layer names/shapes; replace placeholder names in `configs/grid.yaml` + activations helper.
2. Implement both filters + `src/calibration.py`; unit tests (identity = no-op, params in range, single-image smoke test that the loop reduces activation distance on a programmatically re-lit image).
3. (This rewrite.) Spec + `tasks/todo.md` reflect the tool reframing.
4. Finalize the capture protocol (fixed calibration scene + dev dataset).

**Phase 1 — Diagnostics** (after dev dataset captured) → pick candidate layer(s).
**Phase 2 — Grid search** → pick filter + layer, validated on held-out split.
**Phase 3 — Benchmark** → proxy recovery + calibration cost; recommend deploy config.
**Phase D — Deployable tool** → `src/deploy_calibrate.py` runtime calibration story.

## 8. Evaluation Criteria

### Must Have
1. **Phase 1:** ≥1 layer shows a clear distance gradient that scales with illumination shift.
2. **Phase 2:** ≥1 config converges in <100 steps with ≥50% distance reduction on the held-out val set.
3. **Phase 3:** top config recovers ≥70% of the proxy-agreement gap (filtered-B vs A predictions).
4. **Calibration time:** <1 s on edge CPU (or <100 ms on GPU), <100 steps.

### Nice to Have
- Which layers matter (early backbone photometric vs decoder semantic).
- Generalization to unseen illumination levels (train on levels 2–4, test 1 & 5).
- Overfitting signature: train loss vs val distance gap.

### Acceptable Negative Results
- No Phase 1 signal → model already invariant; redesign the shift or accept robustness.
- 12-param barely beats 6-param → use affine (Occam's razor).
- Proxy recovery <50% → shift too extreme or needs non-linear correction; escalate to curves/LUT tier.

## 9. Decision Points & Contingencies

| Scenario | Action |
|----------|--------|
| Phase 1 shows no signal | **STOP**; revisit shift / camera / dataset |
| Signal in decoder only | Proceed cautiously; consider matrix-12 early; non-parametric may be needed |
| Phase 2 baseline doesn't converge | Debug loss + LR; if still failing, shift may not be linearly separable |
| Phase 2 overfits (train→0, val↑) | Lower LR / add L2 reg / early stop |
| Proxy recovery <50% | Inspect residuals: photometric (color cast) vs geometric (blur/focus) |
| Calibration >1 s on edge | Reduce steps/batch; pre-calibrate offline if needed |

## 10. Open Items to Confirm During Execution

- Exact real layer names from RF-DETR nano (placeholders `backbone_3` / `encoder_output` in `configs/grid.yaml` will change).
- Whether the filter sits strictly pre-normalization (assumed) vs on the processor's normalized tensor — decided once the HF image-processor pipeline is inspected in task one.

## 11. References

- **RF-DETR (HF transformers):** https://huggingface.co/docs/transformers/main/model_doc/rf_detr
- **rf-detr-nano checkpoint:** https://huggingface.co/stevenbucaille/rf-detr-nano
- **Roboflow rf-detr (fallback):** https://github.com/roboflow/rf-detr
- **ISP / CCM color correction:** standard 3×3 white-balance matrix — this spec learns it.

---

**Author:** Carlos · **Approved:** 2026-06-27 · **Next step:** Phase A tasks (model/activation verification + filters), then Phase 0 capture.
