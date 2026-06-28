# Spec: Advanced Filter Library — Expressiveness-First Reframing

**Status:** Approved (supersedes the filter-design subsection of `2026-06-27-experimental-plan.md`)
**Date:** 2026-06-27
**Owner:** Carlos

---

## 1. Philosophy Change

The previous framing treated **parameter count as a hard design constraint** ("start at
6–12 params, escalate only on plateau"). That is dropped.

- **Expressive capacity is now the priority.** High-parameter filters — full
  differentiable 3D LUTs, multi-point per-channel tone curves, large-K spatial / local
  tone mapping, high-order / root-polynomial CCMs, chromatic-adaptation transforms —
  are **first-class default options, not last-resort escalations**. Parameter count is
  **informational, not a limit**.
- **The real pressure is generalization, not budget.** Calibration fits against a
  **single reference scene** (the calibration A/B pair). An expressive filter can
  perfectly fit that one pair and generalize badly to the real deployment scene B.
  **Held-out validation is the primary guardrail.** Configs are ranked by val distance,
  and a configurable overfit gate drops configs whose val recovery is too small a
  fraction of their train recovery.
- **Smoothness / identity-anchoring regularization matters more than param count.**
  For high-capacity filters (LUT, curves, large-K spatial), a per-filter `reg_loss()`
  term (total-variation smoothness + identity-distance penalty) is the structural
  counterweight to overfitting, summed into the group loss with a config weight.

### Target illumination shift
**Both brightness and color temperature.** This justifies the full library: the 3D LUT
is the headline (it models channel coupling + non-linearity in one mapping); high-order
CCM and chromatic adaptation cover the color-temp axis; tone curves and local tone
mapping cover the brightness axis.

---

## 2. Filter Contract — Unchanged, Plus `reg_loss()`

All filters (existing and new) satisfy the contract in `src/filters/base.py`:

- End-to-end differentiable: finite, non-zero grads on all params via `loss.backward()`.
- Operate on NCHW RGB in [0,1] **before** ImageNet normalization.
- Identity init = exact no-op (`forward(x) == x`, even on real photos in `data/raw/`).
- Output clamped to [0,1] (by the base `forward`).
- Physically-ranged, well-conditioned params; `clamp_param` used where it does not
  create dead-zone gradients (high-capacity filters avoid `clamp_param` on their core
  params — see per-filter grad risk).
- Inherits `Filter`; exposes `num_params`, `get_params()`, `get_params_flat()`.
- Registered in `FILTER_REGISTRY`; buildable via `get_filter(name)` / `build_filter(spec)`.
- Chainable in `CompositeFilter`.

### Base-contract change (additive, prerequisite for all new filters)
Add to `src/filters/base.py::Filter`:
```python
def reg_loss(self) -> torch.Tensor:  # default: zero (low-capacity filters)
    return torch.zeros((), device=next(self.parameters()).device)
```
`CompositeFilter.reg_loss()` = sum of sub-filters' `reg_loss()`. The calibration loop
adds `reg_weight * filter.reg_loss()` to the group loss. `reg_weight` lives in
`configs/grid.yaml::training` (default `0.0`). High-capacity filters override
`reg_loss()` with smoothness + identity-anchoring terms. The change is additive
(default 0), so the existing 119 tests stay green.

---

## 3. Filter Roadmap (F1–F7)

Priority by expressive capacity. Param counts are **informational, not limits**.
"Overfit risk" = on a single calibration scene (the real pressure). "Grad risk" =
gradient-conditioning risk.

| ID | File | Class | Registry | Params (ex.) | Shift corrected | Overfit | Grad | reg_loss |
|----|------|-------|----------|--------------|-----------------|---------|------|----------|
| **F1** | `src/filters/lut_3d.py` | `LUT3D` | `lut_3d` | 3·N³ (N=9→2,187; N=17→14,739; N=33→107,811) | brightness + color-temp **simultaneously** (channel coupling + non-linearity) — **headline** | HIGH | LOW-MED | TV on LUT grid + ‖LUT−LUT_id‖² |
| **F2** | `src/filters/tone_curve.py` | `ToneCurve` | `tone_curve` | 3·P (P=16→48; P=32→96) | per-channel non-linear tone (brightness/exposure non-linearity) | MED | LOW | 2nd-difference (curvature) + identity-dist |
| **F3** | `src/filters/ccm_high_order.py` | `HighOrderCCM` | `ccm_high_order` | deg-2→3×9+3=30; deg-3→3×19+3=60 | cross-channel coupling + mild non-linearity (exposure-invariant, Finlayson) | MED | MED (normalize poly features) | L2 toward identity block |
| **F4** | `src/filters/chromatic_adaptation.py` | `ChromaticAdaptation` | `chromatic_adaptation` | 3 (diagonal LMS) or 9 (full) | color-temp / illuminant (physically grounded cone-response) | LOW | LOW | 0 (or optional L2 toward id) |
| **F5** | `src/filters/spatial_tone_curve.py` | `SpatialToneCurve` | `spatial_tone_curve` | 3·P·K² (P=8,K=3→216; P=16,K=4→768) | zone-dependent non-linear tone (uneven lighting + per-zone exposure response) | HIGH (large K·P) | LOW (grid_sample + monotone cumsum) | spatial TV + curve curvature + identity-dist |
| **F6** | `src/filters/local_tonemap.py` | `LocalTonemap` | `local_tonemap` | K² (+optional 3·P·K²) | local contrast / spatially-varying exposure (CLAHE intent, guided-filter approx) | MED-HIGH | LOW-MED (box-conv grads stable) | TV on gain field + identity-anchoring |
| **F7** | `src/filters/lut_3d_lowrank.py` | `LUT3DLowRank` | `lut_3d_lowrank` | M weights (8–32) + fixed basis | same as F1, low-rank (LUT = identity + Σ w_m·B_m); generalization-friendly variant | MED (controlled by M) | LOW | L2 on weights |

### Mechanism notes
- **F1 3D LUT**: `(3,N,N,N)` vertex grid; trilinear interp via `F.grid_sample` on RGB
  coords; identity init = identity LUT (vertices on the `r=g=b` diagonal map to
  themselves). No `clamp_param` on vertices (the base final clamp handles [0,1]);
  trilinear weights give clean grads to the 8 surrounding vertices.
- **F2 ToneCurve**: per-channel control points; **monotone by construction** via
  cumulative-sum of `softplus(deltas)` → normalized; linear interp; identity init =
  linear ramp (equal deltas). `softplus`+`cumsum` keep grads non-zero everywhere.
- **F3 HighOrderCCM**: root-polynomial features of RGB (degree ≤3); learned `3×K`
  matrix + offset; identity init = only linear terms active (identity block). Inputs
  normalized for conditioning.
- **F4 ChromaticAdaptation**: RGB→LMS (fixed Bradford matrix), learnable diagonal (or
  full) adaptation in LMS, LMS→RGB. Identity init = identity adaptation.
- **F5 SpatialToneCurve**: reuses `SpatialFilter`'s bilinear K×K grid + the monotone
  curve parameterization from F2, per channel, per zone.
- **F6 LocalTonemap**: guided-filter-style local mean subtraction
  `I' = μ_local + g·(I − μ_local)` with a learnable K×K gain field `g` (box-conv-based
  guided filter, fast on 384²); optional chained spatial curve. **True differentiable
  CLAHE (soft sliding-window histograms) is explicitly future stretch, not in this spec.**
- **F7 LUT3DLowRank**: `LUT = LUT_identity + Σ_m w_m · B_m` with fixed smooth basis
  LUTs `B_m` and learned weights `w_m`. The recommended deployment form if the full LUT
  (F1) overfits.

### Recommended composite combos (no new code — `CompositeFilter`)
- `ccm_high_order` → `tone_curve` (structured non-linear CCM + per-channel curve;
  cheaper than 3D LUT)
- `chromatic_adaptation` → `lut_3d` (size 9) (physics WB + residual LUT)
- `spatial_affine` → `local_tonemap` (spatial color + local contrast)

---

## 4. Stand-in Validation Data (until Phase 0 capture)

`src/calibration.py` (A4) and the acceptance criteria #6/#7 need A/B pairs + a held-out
val pair. `data/raw/` currently has 3 standalone photos (no A/B pairs; Phase 0 capture
gated). Until real pairs are captured, use **deterministic programmatic re-lit pairs**
from a real photo, in `src/utils/synth_relit.py`:

- **A** = original photo (from `data/raw/`).
- **B_train** = A shifted by `gamma=1.4 + per-channel gains=[1.10, 0.95, 0.85]`
  (warm + underexposed — both brightness and color-temp shift, matching the target).
- **B_val** = A shifted by a **different magnitude** `gamma=1.2 + gains=[1.05, 0.97, 0.90]`
  (same direction, milder — the held-out generalization probe).

These are a **proxy** for real A/B pairs; real-data generalization is re-verified after
Phase 0 capture. The shift direction (warm + underexposed) is chosen to exercise both
axes the library targets.

---

## 5. Uniform Acceptance Criteria (per filter F1–F7)

Each filter gets `tests/test_<filter>.py` exercising all of:

1. **Identity init = no-op on real photos**: for each `data/raw/*.jpg`,
   `assert torch.allclose(f(img), img, atol=1e-5)`.
2. **Params in range**: `get_params()` values within declared physical ranges; finite.
3. **Differentiability**: `loss = (f(x)-x).pow(2).mean() + f.reg_loss()`; `loss.backward()`;
   assert every `p in f.parameters()` has finite `p.grad` and `p.grad.abs().sum() > 0`
   (for large LUTs, require ≥99% of params non-zero; report the fraction).
4. **Registry + build**: `get_filter(name)` and `build_filter(spec)` (with kwargs
   `size`/`P`/`grid_size`/`degree`/`mode`/`M`) return the correct type; `num_params`
   matches the formula.
5. **Composite chaining**: filter chains in `CompositeFilter` with a global filter;
   composite `num_params` and `reg_loss()` correctly sum.
6. **Smoke (calibration loop)**: on the B_train stand-in pair, the A4 loop with this
   filter reduces activation distance vs unfiltered B by ≥30% (configurable threshold).
7. **Generalization (held-out val)**: improvement holds on B_val — val distance
   reduction > 0 (no negative transfer), and `val_reduction / train_reduction ≥ 0.5`
   (overfit gate).

Criteria #6/#7 depend on A4 (`src/calibration.py`) + A8 (`src/utils/synth_relit.py`).

---

## 6. Grid Integration

Add to `configs/grid.yaml::grid.filters` (active, swept against all 35 layer groups):

```yaml
    # Advanced / high-capacity (expressiveness-first; gated by held-out val)
    - type: "chromatic_adaptation"           # F4: low-risk color-temp baseline
    - type: "ccm_high_order"                 # F3: degree-2 (30 params)
    - type: "tone_curve"                     # F2: P=16 (48 params)
    - type: "lut_3d"                         # F1: N=9 (2187 params) — headline
      size: 9
    # - type: "lut_3d"                       # F1: N=17 (14739 params) — heavier
    #   size: 17
    - type: "spatial_tone_curve"             # F5: P=8, K=3 (216 params)
      P: 8
      grid_size: 3
    - type: "local_tonemap"                  # F6: K=4 local contrast
      grid_size: 4
    # - type: "lut_3d_lowrank"               # F7: M=16 — generalization-friendly LUT variant
    #   M: 16
    # Composite combos (structured cheaper alternatives to the full LUT)
    # - composite: ["ccm_high_order", "tone_curve"]
    # - composite: ["chromatic_adaptation", "lut_3d"]   # pass size via per-spec kwargs
```

Add to `configs/grid.yaml::training`:
```yaml
  reg_weight: 0.01          # weight on filter.reg_loss() in the group loss (0 = off)
```

Add to `configs/grid.yaml::validation`:
```yaml
  overfit_gate:
    enabled: true
    min_val_recovery_ratio: 0.5   # val reduction must be ≥50% of train reduction to advance
```

The grid executor (`src/grid_search.py`, Phase 2) ranks by `val_distance` and, when
`overfit_gate.enabled`, drops configs where `val_reduction / train_reduction <
min_val_recovery_ratio`. This is the **primary guardrail** that lets high-capacity
filters compete without being selected on overfitting alone. The gate is config-driven.

### Cost note
A 35-group × ~15-filter × 1-loss grid with the 3D LUT at N=17 is expensive (each run =
≤100 fwd/bwd of RF-DETR nano with hooks). **N=9 is the default active; N=17 commented.**
The activation cache (`data/processed/activations_cache/`) and a `--subset` flag on the
grid executor mitigate this.

---

## 7. Naming Convention Change

`AGENTS.md` previously said "filter names: `affine_6param.py`, `matrix_12param.py`
(explicit parameter count)". New filters do **not** carry a param count in the filename
(param count is no longer fixed: `lut_3d` with `size=N`, `tone_curve` with `P`, etc.).
New convention: descriptive names (`lut_3d.py`, `tone_curve.py`, `ccm_high_order.py`,
`chromatic_adaptation.py`, `spatial_tone_curve.py`, `local_tonemap.py`,
`lut_3d_lowrank.py`). Existing count-suffixed names are kept (no rename). This note is
mirrored into `AGENTS.md`.

---

## 8. Task Ordering (summary; full list in `tasks/todo.md`)

1. **A3b** — base-contract `reg_loss()` (`base.py` + `composite.py`).
2. **A4 + A8** (parallel) — calibration loop (`src/calibration.py`) + stand-in re-lit
   data (`src/utils/synth_relit.py`).
3. **A9–A15** — filters F1–F7 (any order, parallelizable), each with `test_<filter>.py`
   running the 7 acceptance criteria.
4. **A16** — registry + `configs/grid.yaml` integration.
5. **A17** — update `AGENTS.md` + experimental-plan spec; mark tasks done.

---

**Author:** Carlos · **Approved:** 2026-06-27 · **Supersedes:** filter-design subsection of `2026-06-27-experimental-plan.md` (§6 "Parametric filter design" and the budget references in §1/§2/§3/§5/§8/§9).
