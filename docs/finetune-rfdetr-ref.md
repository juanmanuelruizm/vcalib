# Runbook — branch `finetune-rfdetr-ref` (domain-adapted detector + pair-free filter)

> Portable handoff so a Claude Code session (or you) on another machine has the full
> context. Claude's local memory and `~/.claude/plans/` do **not** travel across
> machines — this file (committed) is the source of truth.

## What this branch does & why

The earlier benchmark used **off-the-shelf RF-DETR** on objects outside COCO, so the
reference arm A was a weak ceiling, and the filter could only be **trained with paired
A** (even `loss_mode: detection` uses A's outputs as the target — `src/calibration.py`
`detection_output_loss`). In reality the reference illumination is not reproducible, so
pairs don't exist. This branch:

1. **Fine-tune** RF-DETR on the reference (nominal) condition → an honest, domain-adapted
   frozen ceiling A'.
2. **Train the filter WITHOUT pairs**, supervised by real GT detection loss:
   `filter(shifted) → frozen RF-DETR → its own SetCriterion vs GT boxes → backprop to the
   filter`. No reference image A.
3. Validate on **cooktop** (SAM3 boxes) and **ExDark** (real human boxes, unpaired).

This can cut against the filter thesis (an adapted model may be robust enough that the
filter adds little) — that falsifiability is the point.

## Fresh-clone prerequisites (IMPORTANT — `git clone` alone is not enough)

- **`3rd_party/libreyolo` is a git submodule** → clone with `--recurse-submodules` or run
  `git submodule update --init --recursive`.
- **`data/raw/` (the 90 cooktop JPEGs) is gitignored** → copy it from the Mac
  (`rsync -av data/raw/ user@host:~/vcalib/data/raw/`, ~33 MB). Do **not** copy
  `data/coco/` — the matrix runner regenerates it from `data/raw` + the committed GT.
- `rf-detr-nano.pth` (366 MB) auto-downloads on first model load (needs internet); else
  copy `3rd_party/libreyolo/weights/rf-detr-nano.pth`.
- The SAM3 GT (`data/labels/instances_sam3.json`) **is** committed — nothing to do.

```bash
git clone --recurse-submodules <url> vcalib && cd vcalib
git checkout finetune-rfdetr-ref
git submodule update --init --recursive
# from the Mac: rsync -av data/raw/ user@gpu:~/vcalib/data/raw/
uv sync
./scripts/run_finetune_matrix.sh cuda        # regenerates data/coco, runs the matrix
```

Results (benchmark CSVs) land in `results/ft_bench/`. **ExDark** (the real, unpaired
validation set) — the GitHub repo ships only metadata; the images (1.5 GB) and bbGt GT
(5 MB) live on Google Drive, so `git clone` alone is **not** enough:

```bash
git clone --depth 1 https://github.com/cs-chan/Exclusively-Dark-Image-Dataset data/exdark_raw
cd data/exdark_raw
uv run --with gdown gdown 1BHmPgu8EsHoFDDkMGLVoXIlCth2dW6Yx -O images.zip  # 1.5 GB -> ExDark/
uv run --with gdown gdown 1P3iO3UYn7KoBi5jiUkogJq96N6maZS1i -O gt.zip      # 5 MB  -> ExDark_Annno/
unzip -q images.zip && unzip -q gt.zip && rm -rf __MACOSX && find . -name '._*' -delete
cd ../.. && uv run python scripts/exdark_to_coco.py \
    --exdark-root data/exdark_raw --out data/coco/exdark --multiclass
```

The converter auto-detects the canonical layout (`ExDark/`, `ExDark_Annno/`,
`Groundtruth/imageclasslist.txt`) and emits `bright` (reference, 5,071 imgs) +
`dark_{train,val,test}` (903/606/782). Delete the zips after unzip.

## Scripts (all built & validated end-to-end on Mac, except where noted)

- `scripts/make_detection_coco.py` — SAM3 → single-class COCO, strict split (test = held-out
  7,11,15,20,23,27; val carved; `--levels 1` reference / `2,3` shifted).
- `scripts/augment_illumination.py` — offline box-preserving brightness/contrast/gamma/
  colour-temp jitter (`--preset moderate|aggressive`), train split only.
- `scripts/finetune_rfdetr.py` — wrapper over `LibreRFDETR.train` (`--freeze backbone`, `--lora`,
  EMA, `--patience`). Produces `results/finetune/<name>/weights/best.pt`.
- `scripts/train_filter_detloss.py` — **pair-free filter trainer** (the core): detector-GT
  loss via `build_criterion_and_postprocess`, `--model-checkpoint` for the fine-tuned model.
  **Combined mode** (`--act-weight λ --ref-split bright_train`): adds an unpaired CORAL
  activation term on top of the GT-detection loss (see "Combined pair-free" section below).
- `scripts/benchmark_detection.py --model-checkpoint <ckpt>` — A'/B/filter(B) on held-out test.
- `scripts/run_finetune_matrix.sh [cuda]` — full cooktop matrix {head-only, LoRA} ×
  {no-aug, aggressive} + off-the-shelf baseline.
- `scripts/exdark_to_coco.py` — validated on the real Drive data (7,362 imgs w/ GT): auto-detects
  the `ExDark/`+`ExDark_Annno/`+`imageclasslist.txt` layout (override via `--images-root/--gt-root/--meta`);
  single-class by default or 12 classes with `--multiclass`.

## Gotchas to honor when interpreting results

- In the pure realistic (unpaired) framing there is **no A ceiling** → report **absolute AP
  gain** from the filter; cooktop still has A' (fine-tuned on I1) for classic recovery.
- **Overfit**: only ~20 cooktop train scenes → head-only/LoRA + aggressive aug + early stop.
- A model over-adapted to reference light may be **more** brittle to the shift — a valid outcome.
- **ExDark**: heterogeneous resolution + `to_unit_rgb`'s square resize distorts aspect ratio
  per image (consider letterbox); watch small objects at 384².
- **3-way split discipline**: test = final only; val = early-stop only; never leak test.
- Fine-tune is single-class "object" (head → 2 logits) to match the GT and class-agnostic eval.

## Results — ExDark (unpaired, 12-class) · 2026-07-03

Full matrix run via `run_exdark_matrix.sh` (phase 2 fine-tune + phase 3 filter/eval).
A' fine-tune on `bright`: 60 epochs, best mAP50-95 = 0.5919 @ epoch 49
(`results/finetune/exdark_bright/weights/best.pt`). Pair-free filters trained on
`dark_train` (val `dark_val`, GT detection loss), evaluated on `dark_test`.

**Headline: the pair-free filter did NOT help — it slightly degraded AP in BOTH arms.**
This is the falsifiable outcome the branch was designed to test; here it fires clearly.

`dark_test`, B vs filter(B):

| arm (label-map) | metric | B | filter(B) | Δ |
|---|---|---|---|---|
| **baseline** (`exdark_coco`, off-the-shelf) | AP | 0.3718 | 0.3102 | **−0.0616 (−16.6%)** |
| | AP50 | 0.6928 | 0.5955 | −0.0973 |
| **adapted** (`none`, A' on bright) | AP | 0.5528 | 0.5410 | **−0.0118 (−2.1%)** |
| | AP50 | 0.8241 | 0.8038 | −0.0203 |

CSVs: `results/ft_bench/exdark_{baseline,adapted}.csv` (gitignored — not in repo).

Supporting signals:
- Filter training barely moved val loss (baseline best 10.099 from ~10.13; adapted best
  9.315 from ~9.33) and early-stopped fast (adapted @ epoch 6, baseline @ epoch 26) → the
  filter converged to a near-identity that, on test, distorts just enough to cost AP.
- Degradation is consistent across AP/AP50/AP75/AR100 in both arms.

**Caveats when reading this:**
- AP is **not comparable across arms** — baseline maps preds to COCO-91, adapted scores the
  12 native ExDark classes (`label-map none`). Only the within-arm B vs filter(B) delta is clean.
  Do not read "0.55 vs 0.37" as an adaptation gain; it is confounded by the label mapping.
- Next step (now done — see below): contrast with **cooktop** (has A pairs + classic recovery)
  and diagnose *why* the filter learns no useful signal. The coupling study answers it: on
  **paired** cooktop the filter recovers most of the detection gap, so filter capacity is not
  the bottleneck — the ExDark null is about the **unpaired / pair-free** supervision, not the filter.

## Results — activation↔detection coupling (cooktop, paired) · 2026-07-03

Run: `uv run python scripts/run_coupling_study.py --device cuda --epochs 50 --eval-every 2`
(STC `P=16, g=5` on `backbone.projector`, 41 train pairs). Every 2 epochs it records TWO
decoupled read-outs on the 6 held-out GT-backed test scenes: **feature-gap closure** at the
projector (the classic `test_mean`) and **AP/AP50** of filter(B) vs GT (COCOeval). Reference
lines: A(ceiling) AP **0.5435**, B(floor) AP **0.3766** → gap **0.1669**.

**Headline: closing the feature gap and improving detection are decoupled — and past a point,
anti-correlated.** Optimizing the activation metric overshoots into a region that *costs* AP.

| loss_mode | best AP (epoch) | gap recovered | feat_closure @ best | AP at max feat_closure (0.33) | max AP50 |
|---|---|---|---|---|---|
| `activation` | 0.4951 (ep 10) | **71%** | 0.257 | **0.4505** (ep 50) ↓ | 0.6590 |
| `detection` | 0.5172 (ep 20) | **84%** | 0.185 | — (plateaus ~0.19) | 0.7647 |
| `combined` | 0.5287 (ep 42) | **91%** | 0.191 | — (plateaus ~0.19) | 0.7652 |

Supporting signals:
- **Activation mode overshoots.** `feat_closure` climbs monotonically 0.17→**0.333** (our
  known headline `test_mean`), but AP peaks early at ep 10 (0.495) and then **decays to 0.451**
  while the feature gap keeps closing; AP50 falls 0.659→0.552. The pure-feature optimum is *not*
  the detection optimum — the headline `test_mean=0.3338` config maximizes the wrong thing.
- **Detection / combined win with LESS feature closure.** Both plateau at `feat_closure ≈ 0.19`
  (vs 0.33) yet reach far higher AP (0.52–0.53) and AP50 (~0.76), and are **stable** — no tail
  decay. `combined` recovers **91%** of the A→B AP gap.
- **So the filter works when the supervision sees detection.** On paired cooktop the capacity is
  ample; the ExDark degradation above is a property of the unpaired/pair-free signal, not the filter.

Practical takeaway: for detection recovery use `loss_mode=combined` (or `detection`) and
early-stop on **AP**, not on the feature metric. Raw trajectories:
`results/coupling/{activation,detection,combined}.jsonl` (CSV `coupling.csv` is gitignored).

## Combined pair-free filter — det-GT + unpaired CORAL activations · 2026-07-04

The coupling study showed the winning recipe on **paired** cooktop is `combined` = activations
+ detections. We want to keep that "both signals" idea in the **pair-free** (no per-scene A)
setting where the plain detection-only filter *degraded* AP on ExDark. The blocker: the paired
`combined` activation term is `group_loss(A_acts, filter(B)_acts)` — it needs A of the same
scene, which does not exist here.

**Design (implemented in `scripts/train_filter_detloss.py`):** replace the per-scene activation
target with **unpaired, domain-level activation alignment**. We still never touch a per-scene A —
we only use the reference-condition *domain* (`bright_train`), which is observable:

1. Precompute, once, the projector-feature statistics (channel **mean** `μ_ref` + **covariance**
   `Σ_ref`, C=256) over the whole reference/bright domain through the frozen model. Streamed +
   cached at `data/coco/exdark/ref_moments/<split>__<layer>__n<N>.pt` (gitignored).
2. Per step: `filter(dark) → projector acts`; add a **CORAL** term pulling the filtered image's
   `(μ, Σ)` toward `(μ_ref, Σ_ref)`. Total loss = `det_GT_loss + λ·(mean_term + cov_term)`.
3. Model selection early-stops on **`val_det`** (the real detection objective), *not* the CORAL
   term — directly honoring the coupling-study lesson that selecting on the feature metric
   overshoots AP.

**CORAL scaling gotcha (important):** classic Deep-CORAL scales the covariance by `1/(4d²)`.
At d=256 that is `1/262144`, which crushes `cov_term` to ~1e-6 vs a `mean_term` ~0.25 — i.e. it
silently degenerates to mean-matching. We use **raw Frobenius** so `‖Σ−Σ_ref‖²_F` (~0.76) stays
comparable to (in fact larger than) the mean term, and the covariance signal actually
contributes. `metrics.jsonl` logs `train/val_act_mean` and `train/val_act_cov` separately.

Run (adds `baseline_coral` + `adapted_coral` arms alongside the detection-only ones):
```bash
ACT_WEIGHT=1.0 REF_SPLIT=bright_train ./scripts/run_exdark_matrix.sh cuda
# or a single arm directly:
uv run python scripts/train_filter_detloss.py --data data/coco/exdark \
    --train-split dark_train --val-split dark_val --label-map exdark_coco \
    --act-weight 1.0 --ref-split bright_train --act-layer backbone.projector \
    --out results/experiments/runs/exdark_pairfree_baseline_coral --device cuda
```
`--act-weight 0` (the default) keeps the original detection-only behaviour byte-for-byte.
Smoke-validated end-to-end on CUDA (ref moments + cache, CORAL train/val, early-stop, benchmark
plumbing).

### Result — adapted arm, det-GT + CORAL vs det-only (ExDark `dark_test`) · 2026-07-04

Ran only the `adapted_coral` arm over the existing A' (`results/finetune/exdark_bright/weights/best.pt`):
`λ=1.0`, `--ref-split bright_train` (moments through A'), `backbone.projector`, 50 ep / patience 10.

**Headline: the unpaired CORAL activation term did NOT recover AP — it degraded it *more* than
detection-only.** The falsifiable question fires negative.

| pair-free supervision | filter(B) AP | AP50 | Δ vs B (0.5528) |
|---|---|---|---|
| detection-only (recorded) | 0.5410 | 0.8038 | −0.0118 (−2.1%) |
| **det + CORAL (this run)** | **0.5332** | 0.7953 | **−0.0196 (−3.5%)** |

CSV: `results/ft_bench/exdark_adapted_coral.csv`. Why it fails (training diagnostics):
- Converged/early-stopped fast: `best val_det = 9.333 @ epoch 4` — *worse* than det-only (~9.315
  @ ep 6), i.e. the CORAL term pulls the filter off the pure-detection optimum.
- The CORAL terms barely move (val `act_cov` 1.26→1.22, `act_mean` ~0.92 flat): a tone-curve
  filter can hardly close the domain-level statistic gap, and the little it closes costs detection.

**Interpretation:** reinforces the coupling-study conclusion — the bottleneck is the *unpaired*
supervision itself. Domain-level (mean+cov) activation alignment is too weak a signal (it washes
out per-scene content) and conflicts with per-image detection fitting. Paired `combined` worked
only because A was a *same-scene* reference; there is no unpaired substitute at the projector here.
Not falsified only for this design point (λ=1, projector, mean+cov) — a λ sweep would harden the
null, but the flat CORAL trajectory suggests it is not a weighting artifact.
