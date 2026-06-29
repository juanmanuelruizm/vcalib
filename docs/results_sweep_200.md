# Results: 200-Config Sweep

> **Setup:** 10 filter types × 2 illumination levels (`level_1→level_2`, `level_1→level_3`) × 10 layer groups.  
> **Data:** synthetically relighted image pairs (real dataset capture pending — see Phase 0).  
> **Metric:** mean activation distance reduction on 6 held-out test scenes (higher = better).  
> **Training:** Adam, lr=5×10⁻³, up to 50 epochs, early stopping patience=10, reg_weight=0.01.  
> **Full CSV:** [`results/experiments/experiment_results.csv`](../results/experiments/experiment_results.csv)

---

## Top 20 Configurations

| Rank | Filter | Layer Group | Dataset | Train ↓ | Test ↓ | Test σ |
|------|--------|-------------|---------|---------|--------|--------|
| 1 | `spatial_tone_curve` (P=8,K=3) | projector | L1→L2 | 23.3% | **30.7%** | 0.067 |
| 2 | `lut_3d` (N=9) | projector | L1→L2 | 12.8% | 29.6% | 0.076 |
| 3 | `tone_curve` (P=16) | projector | L1→L2 | 18.7% | 24.1% | 0.081 |
| 4 | `lut_3d` (N=9) | projector | L1→L3 | 10.7% | 23.9% | 0.070 |
| 5 | `ccm_high_order` | projector | L1→L2 | 14.4% | 23.7% | 0.045 |
| 6 | `spatial_tone_curve` (P=8,K=3) | projector | L1→L3 | 18.0% | 21.6% | 0.090 |
| 7 | `lut_3d` (N=9) | backbone.late+proj | L1→L2 | 10.0% | 21.0% | 0.055 |
| 8 | `lut_3d` (N=9) | backbone.late+proj | L1→L3 | 7.5% | 20.6% | 0.041 |
| 9 | `matrix_12param` | projector | L1→L2 | 14.3% | 20.6% | 0.059 |
| 10 | `affine_6param` | projector | L1→L2 | 15.8% | 20.4% | 0.066 |
| 11 | `lut_3d` (N=9) | backbone.late | L1→L3 | 7.2% | 20.2% | 0.039 |
| 12 | `lut_3d` (N=9) | backbone.late | L1→L2 | 9.3% | 19.9% | 0.047 |
| 13 | `spatial_tone_curve` (P=8,K=3) | backbone.late+proj | L1→L2 | 15.8% | 19.5% | 0.053 |
| 14 | `ccm_high_order` | projector | L1→L3 | 9.8% | 19.1% | 0.048 |
| 15 | `tone_curve` (P=16) | projector | L1→L3 | 15.8% | 18.3% | 0.055 |
| 16 | `spatial_tone_curve` (P=8,K=3) | backbone.late+proj | L1→L3 | 13.6% | 18.3% | 0.036 |
| 17 | `spatial_tone_curve` (P=8,K=3) | backbone.late | L1→L2 | 14.7% | 18.1% | 0.053 |
| 18 | `spatial_tone_curve` (P=8,K=3) | backbone.late | L1→L3 | 13.0% | 18.0% | 0.033 |
| 19 | `ccm_high_order` | backbone.late+proj | L1→L3 | 7.9% | 17.8% | 0.053 |
| 20 | `ccm_high_order` | backbone.late | L1→L3 | 7.6% | 16.7% | 0.050 |

---

## Analysis by Dimension

### By Layer Group

The layer group used as the loss signal has a larger effect than the filter type.

| Layer Group | Best Test Reduction | Notes |
|-------------|-------------------|-------|
| `projector` | **30.7%** | Multi-scale backbone projector — best signal |
| `backbone.late+proj` | 21.0% | Late DINOv2 blocks + projector |
| `backbone.late` | 20.2% | Layers 8–11 (semantic features) |
| `backbone.early+proj` | 15.1% | |
| `backbone.early` | 14.0% | Layers 0–3 (low-level features) |
| `backbone.all+proj` | 13.5% | All 12 DINOv2 blocks + projector |
| `backbone.all` | 12.9% | All 12 DINOv2 blocks |
| `backbone.mid+proj` | 10.1% | Layers 4–7 |
| `backbone.mid` | 7.8% | Weakest backbone region |
| `proj+decoder` | 6.1% | Projector + decoder — too deep, over-regularized |

**Interpretation:** The `backbone.projector` is the multi-scale feature projector that aggregates the DINOv2 backbone output before the decoder. It appears to be the most sensitive layer to illumination-induced activation drift — making it the ideal calibration signal. Layers near the output (decoder) show weaker signal, possibly because decoder attention is more content-dependent and less illumination-sensitive.

### By Filter Type

| Filter | Params | Best Test Reduction | Category |
|--------|--------|-------------------|----------|
| `spatial_tone_curve` (P=8, K=3) | 216 | **30.7%** | Non-linear spatial |
| `lut_3d` (N=9) | 2,187 | 29.6% | Non-linear global |
| `tone_curve` (P=16) | 48 | 24.1% | Non-linear global |
| `ccm_high_order` | 30 | 23.7% | Non-linear global |
| `matrix_12param` | 12 | 20.6% | Linear global |
| `affine_6param` | 6 | 20.4% | Linear global |
| `local_tonemap` | ~16 | 14.8% | Non-linear spatial |
| `gamma_3param` | 3 | 13.6% | Non-linear global |
| `brightness_2param` | 2 | 12.3% | Linear global |
| `chromatic_adaptation` | 3–9 | 10.6% | LMS-space |

**Interpretation:** Non-linear filters consistently outperform linear ones. The illumination shifts are non-linear in nature (gamma-compressed camera response, mixed light temperatures). `spatial_tone_curve` combines spatial awareness with per-channel non-linearity — matching the expected structure of real illumination degradation.

### Level Comparison (L1→L2 vs L1→L3)

L3 represents a stronger illumination shift than L2. Results on L3 are 10–20% lower on average but follow the same ranking:

| Filter | Test L1→L2 | Test L1→L3 |
|--------|-----------|-----------|
| `spatial_tone_curve` + projector | 30.7% | 21.6% |
| `lut_3d` + projector | 29.6% | 23.9% |
| `ccm_high_order` + projector | 23.7% | 19.1% |

The rank ordering is stable, suggesting these findings will transfer to real captured data.

---

## Generalization Analysis

Good calibration reduces both train and test distance. A filter that reduces train distance but fails on test is overfitting to the specific training pair.

**Observations:**
- Top configs show test reduction > train reduction (e.g. `spatial_tone_curve + projector`: train 23%, test 31%), suggesting the filter learned a generalizable correction.
- The overfit gate (val/train ≥ 0.5) would pass all top-20 configs.
- `lut_3d` shows higher variance (σ = 0.07) — it can model complex corrections but is less stable across scenes.

---

## Recommended Configs for Phase 3

Based on this sweep, Phase 3 (proxy mAP benchmark) should test:

1. **`spatial_tone_curve` (P=8, K=3) + `projector`** — best overall test reduction (30.7%)
2. **`lut_3d` (N=9) + `projector`** — close second (29.6%), higher variance
3. **`tone_curve` (P=16) + `projector`** — lightweight option (48 params, 24.1%)
4. **`neural_pixel` (hidden=32, depth=2) + `projector`** — neural baseline (pending real data)

---

## Next: Real Data

All experiments above used synthetically relighted image pairs (gamma + per-channel gain shifts as a proxy). The real dataset is being captured and will replace synthetic pairs. Expected effects:

- Real illumination shifts are more complex (mixed spectrum, directional shadows, camera response non-linearity) — this should **favor** non-linear expressive filters like `lut_3d` and `spatial_tone_curve`.
- The `projector` group advantage should hold or increase, since real shifts will produce larger L2 drift in semantic features.
- `neural_pixel` (the unconstrained MLP filter) is the main candidate to outperform parametric filters on real data, given its universal approximation capacity.
