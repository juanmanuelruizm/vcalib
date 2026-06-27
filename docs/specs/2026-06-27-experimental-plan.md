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
4. **Layer chosen by the diagnostic sweep, evaluated over GROUPS of layers.** The loss aggregates a per-layer distance across all layers in a group, not at a single layer. Groups are encoded declaratively (explicit baselines + auto-generated contiguous-window candidates) in `configs/grid.yaml` and resolved by `src/utils/layer_groups.py`, so an automatic layer-search loop can iterate candidate groups.
5. **Goal = deployable on-edge tool** (calibrate in seconds), not just a research report.
6. **Metric = proxy.** Agreement with A's predictions (box IoU + class agreement). No true mAP / ground-truth labeling.
7. **Model loader = LibreYOLO** (git submodule at `3rd_party/libreyolo`, v1.2.0.dev0): `from libreyolo import LibreRFDETR; LibreRFDETR(size="n")`. The wrapper exposes a native `LWDETR` port (not HuggingFace `RfDetrForObjectDetection`); it pulls `transformers>=5.1` only for the DINOv2 backbone. **Activation access is via PyTorch `register_forward_hook`** on submodules of `libre.model.model` — `output_hidden_states` is not reachable through the wrapper. Verified layer names: `backbone.layer.0..11` (12 DINOv2 ViT blocks), `backbone.projector` (multi-scale projector = encoder memory), `decoder.layer.0..1` (nano has 2 decoder layers). See `src/utils/activations.py::LAYER_PATHS`.

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
- **Full pixel-space parametric filter library** (built upfront so the grid/automatic search can sweep filter types too, not just layer groups): brightness (2), white-balance (3), affine per-channel (6), saturation (1), contrast (1), gamma per-channel (3), matrix 3×3 + offset (12), and ordered **composites** that chain corrections a single filter cannot express (linear + non-linear + color).
- Reusable calibration loop (`src/calibration.py`) shared by grid search and the deployed tool.
- Diagnostic sweep to find which layers carry illumination signal.
- Grid search to pick layer group + filter + loss, validated on a held-out split.
- Proxy benchmark (agreement with A's predictions) + calibration-cost measurement.
- A deployable calibration CLI (`src/deploy_calibrate.py`).

### Out of Scope
- Retraining / fine-tuning RF-DETR (frozen throughout).
- Domain adaptation (adversarial, unsupervised).
- Other sensors (multi-spectral, thermal).
- Non-parametric filters (LUTs, free-form splines) — the gamma/contrast/saturation filters cover the main non-linearities; full LUTs remain a future tier if parametric composites saturate.
- True mAP evaluation / ground-truth labeling (using proxy instead).
- Synthetic illumination dataset (Carlos opted for real capture).

## 6. Architecture: Components

| Component | File | Role |
|-----------|------|------|
| Activation helper | `src/utils/activations.py` | Load frozen RF-DETR nano via LibreYOLO; `extract_activations(image)` → named layer dict via **PyTorch forward hooks** on `libre.model.model` submodules (LibreYOLO does not expose `output_hidden_states`); cache/load to `data/processed/activations_cache/`. Real layer names: `backbone.layer.0..11`, `backbone.projector`, `decoder.layer.0..1`. |
| Filter library | `src/filters/` | Parametric pixel-space filters on [0,1] NCHW RGB, identity init = no-op, output clamped to [0,1]. `Filter` base + 7 concrete filters (brightness/white-balance/affine/saturation/contrast/gamma/matrix) + `CompositeFilter` chain. `FILTER_REGISTRY` + `get_filter(name)` / `build_filter(spec)` factory consumed by the grid. |
| Affine filter | `src/filters/affine_6param.py` | `Affine6Param`: gains `a_c∈[0.1,2.0]`, offsets `b_c∈[-1,1]`, identity init. Flagship linear. |
| Matrix filter | `src/filters/matrix_12param.py` | `Matrix12Param`: `I' = M·I + b`, M init = identity, M∈[-2,2]. Flagship linear w/ cross-channel coupling. |
| Other filters | `src/filters/{brightness,white_balance,saturation,contrast,gamma}_*.py` | brightness(2), white-balance(3), saturation(1, toward luma), contrast(1, around image mean), gamma(3, per-channel tone curve). |
| Calibration loop | `src/calibration.py` (new) | Core loop shared by grid search + deployed tool: given filter, stored A-targets, B image, **layer group**, loss+aggregation cfg, optimizer cfg → gradient steps minimizing the **group loss** (mean of per-layer normalized distances), early stopping → trained filter + convergence stats. |
| Layer-group helper | `src/utils/layer_groups.py` | Encodes the group-based sweep: `LayerGroup` dataclass, range-syntax expansion (`"backbone.layer.0..3"`), explicit-group loader + auto-search generator (contiguous DINOv2 windows, `+proj`/`+dec`), `resolve_grid_groups()` union. Drives the automatic layer-search loop. |
| Phase 1 diagnostics | `src/diagnostics.py` | Per-layer L2(normalized)+cosine distance per illumination level, aggregated mean±std over scenes → `results/phase1_diagnostics.json` + heatmap/line plots. Early-bailout if flat. (Diagnostics stay per-layer to guide group design.) |
| Phase 2 grid | `src/grid_search.py` | Sweep **layer group** × filter × loss from `configs/grid.yaml` (groups resolved via `layer_groups.py`); train via `calibration.py` on dev-train; measure distance reduction on dev-val; log `results/runs.csv` (`group_name`, filter, loss, ...); checkpoint top configs. |
| Phase 3 benchmark | `src/benchmark.py` | Top configs: model on B with/without filter; proxy metric vs A's predictions (IoU + class agreement); calibration cost → `results/runs_phase3.csv` + `docs/phase3_report.md`. |
| Deploy CLI | `src/deploy_calibrate.py` (new) | Store A-targets of calibration scene → re-shoot under B → run `calibration.py` → freeze filter (<1MB) → ready for inference. |

### Loss & training
- **Group loss:** the loss is computed over a **group of layers**, not a single layer. For each layer in the group, a per-layer normalized distance is computed — `l2_rel = ||a-b||_2 / ||a||_2` (primary) — so layers of heterogeneous shape (backbone `(4,145,384)`, projector `(1,256,24,24)`, decoder `(1,300,256)`) contribute on a comparable scale. The group loss aggregates them via `mean` (default) or `sum` (see `configs/grid.yaml::loss`).
- **Layer groups** come from two sources, unioned (explicit baselines win on collisions): (1) hand-curated named groups in `layer_groups` (e.g. `backbone.early` = blocks 0..3, `encoder+decoder`); (2) auto-generated candidates in `layer_group_search` (contiguous DINOv2 windows of configurable sizes/stride, optionally `+proj`/`+dec`). Resolve with `src/utils/layer_groups.py::resolve_grid_groups()`. This is the encoding the automatic layer-search loop consumes.
- Adam, lr 1e-3, max 100 steps, early-stop patience 10.
- Seed `torch.manual_seed` + split seed for bitwise-reproducible reruns.

### Parametric filter design

The full library is built upfront (scope expansion: the grid sweeps filter types too).
All operate on the [0,1] RGB tensor before normalization, identity init = no-op.

| Filter | Params | Physics | Caveats |
|--------|--------|---------|---------|
| **Brightness** `I' = a·I + b` (global) | 2 | Exposure / illumination intensity | No per-channel / cross-channel |
| **White balance** `I'_c = a_c·I_c` | 3 | Color-temperature / white-balance gains | No offset (black-level) |
| **Affine per-channel** `I'_c = a_c·I_c + b_c` | 6 | Sensor gain + per-channel offset (WB + exposure) | No cross-channel mixing / non-linearity |
| **Saturation** `I' = L + s·(I−L)` (luma) | 1 | Color vividness / desaturation | Single global scalar |
| **Contrast** `I' = μ + c·(I−μ)` (image mean) | 1 | Contrast / haze | Adaptive to image mean |
| **Gamma** `I'_c = I_c^{γ_c}` (per-channel) | 3 | Non-linear tone response | First non-linear tier |
| **Matrix 3×3 + offset** `I' = M·I + b` | 12 | ISP CCM; cross-channel coupling | Still linear; no gamma/clipping |
| **Composite** ordered chain | Σ | Combine linear + non-linear + color | Overfitting risk at high param count |

Start with the cheap low-param filters (brightness / white-balance / affine) and the
flagship linear (matrix); escalate to gamma + composites only if single filters plateau.
`configs/grid.yaml::grid.filters` lists the active sweep; composites are commented out
by default to keep the grid tractable (35 groups × N filters × 1 loss).

## 7. Phased Execution (phases are blockers)

**Phase A — Unblocked now (no dataset needed)**
1. `uv sync`; verify `rf-detr-nano` loads; dump real layer names/shapes; replace placeholder names in `configs/grid.yaml` + activations helper.
2. Implement the full filter library (`src/filters/`) + `src/calibration.py`; unit tests (identity = no-op, params in range, single-image smoke test that the loop reduces activation distance on a programmatically re-lit image).
3. (This rewrite.) Spec + `tasks/todo.md` reflect the tool reframing.
4. Finalize the capture protocol (fixed calibration scene + dev dataset).

**Phase 1 — Diagnostics** (after dev dataset captured) → per-layer distance map to inform group design.
**Phase 2 — Grid search** → sweep **layer group** × filter × loss; auto-search loop ranks candidate groups; pick filter + group, validated on held-out split.
**Phase 3 — Benchmark** → proxy recovery + calibration cost; recommend deploy config.
**Phase D — Deployable tool** → `src/deploy_calibrate.py` runtime calibration story.

## 8. Evaluation Criteria

### Must Have
1. **Phase 1:** ≥1 layer shows a clear distance gradient that scales with illumination shift.
2. **Phase 2:** ≥1 **layer group** converges in <100 steps with ≥50% distance reduction on the held-out val set.
3. **Phase 3:** top config recovers ≥70% of the proxy-agreement gap (filtered-B vs A predictions).
4. **Calibration time:** <1 s on edge CPU (or <100 ms on GPU), <100 steps.

### Nice to Have
- Which layer groups matter (early-backbone photometric windows vs projector+decoder semantic combos); the auto-search ranks contiguous DINOv2 windows by val distance.
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

- ~~Exact real layer names from RF-DETR nano~~ — **resolved (A1):** `backbone.layer.0..11`, `backbone.projector`, `decoder.layer.0..1`; placeholders in `configs/grid.yaml` replaced (now encoded as layer groups, not single layers).
- ~~Whether the filter sits strictly pre-normalization vs on the processor's normalized tensor~~ — **resolved (A1):** filter operates on the [0,1] RGB tensor before ImageNet mean/std normalization (confirmed via LibreYOLO's `preprocess_numpy`).
- **Layer-group sweep encoding** — the loss is computed over **groups of layers** (mean of per-layer normalized L2). Encoded in `configs/grid.yaml` (`layer_groups` + `layer_group_search`) and resolved by `src/utils/layer_groups.py`, enabling an automatic layer-search loop.

## 11. References

- **RF-DETR (HF transformers):** https://huggingface.co/docs/transformers/main/model_doc/rf_detr
- **rf-detr-nano checkpoint:** https://huggingface.co/stevenbucaille/rf-detr-nano
- **Roboflow rf-detr (fallback):** https://github.com/roboflow/rf-detr
- **ISP / CCM color correction:** standard 3×3 white-balance matrix — this spec learns it.

---

**Author:** Carlos · **Approved:** 2026-06-27 · **Next step:** Phase A tasks (model/activation verification + filters), then Phase 0 capture.
