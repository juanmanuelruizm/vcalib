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
- `scripts/benchmark_detection.py --model-checkpoint <ckpt>` — A'/B/filter(B) on held-out test.
- `scripts/run_finetune_matrix.sh [cuda]` — full cooktop matrix {head-only, LoRA} ×
  {no-aug, aggressive} + off-the-shelf baseline.
- `scripts/exdark_to_coco.py` — validated on the real Drive data (7,362 imgs w/ GT): auto-detects
  the `ExDark/`+`ExDark_Annno/`+`imageclasslist.txt` layout (override via `--images-root/--gt-root/--meta`);
  single-class by default or 12 classes with `--multiclass`.

## Extending to other LibreYOLO families (branch `finetune-all-models`)

`src/utils/activations.py::load_model()` / `src/calibration.py` now take a
`family` param (default `"rfdetr"`, unchanged behavior) so the same
fine-tune → pair-free-filter → benchmark recipe can target other LibreYOLO
model families. Per-family fine-tune scripts:

- `scripts/finetune_rtdetrv4.py` (`family="rtdetrv4"`) — DETR-lineage, shares
  RF-DETR's `{pred_logits, pred_boxes}` output contract, no `NestedTensor`
  needed. `--freeze` works (generic integer groups, not named). `--lora` is
  **not** supported (only RF-DETR's trainer sets `supports_lora=True`).
- `scripts/finetune_yolo9.py` (`family="yolo9"`) — anchor-free `DDetect` head,
  needs its own output adapter: `detection_output_loss_yolo9` in
  `src/calibration.py` (BCE on per-anchor sigmoid class scores + L1 on decoded
  boxes, anchor-aligned since anchors are deterministic per input resolution)
  and a `family="yolo9"` branch in `scripts/benchmark_detection.py::detect()`.
  `--freeze` supports **named** groups (e.g. `backbone.conv0`) here, unlike
  RT-DETRv4/D-FINE.
- `scripts/finetune_fomo.py` (`family="fomo"`) — **separate, lower-confidence
  track**: LibreFOMO is a point-localizer (no AP metric applies), has no
  redistributable pretrained weights (must pass `--model-path` or train from
  random init), and its own `.train()` requires `allow_experimental=True` and
  is flagged unstable upstream. Its own pair-free loss adapter is
  `detection_output_loss_fomo` (per-cell KL over its dense heatmap output — no
  box term, since FOMO has no box head). Full validation still needs a
  point-distance/recall-at-radius eval metric, not yet implemented (there is no
  `benchmark_detection.py` equivalent for this family — AP doesn't apply).

Layer discovery for all non-`rfdetr` families uses each model's own
`get_available_layer_names()` (a stable API every LibreYOLO family
implements) rather than vcalib's hand-picked `LAYER_PATHS` dotted-path dict,
which stays RF-DETR-only.

**Checkpoint disk hygiene — do this after every family's run, physically, on
whichever machine ran it:** fine-tuned/filter checkpoints
(`results/finetune/<name>/weights/*.pt`, `results/experiments/runs/<name>/best.pt`)
are working files, not the artifact of record — the committed YAML config +
seed + this recipe is what's reproducible, not the binary. Once a family's
results row is verified in the CSV/this doc, delete its local checkpoints:

```bash
rm -rf results/finetune/*<family>* results/experiments/runs/*<family>*
```

`results/finetune/`, `results/experiments/runs/`, and `results/**/*.pt` are
already gitignored, but that only stops future commits — it does nothing about
what's already on disk from runs just completed. This is a real, deliberate
`rm` step per family, not something to leave to `.gitignore`.

## Results — RT-DETRv4 / YOLOv9 fine-tune + REAL pair-free filter · 2026-07-04

Full pipeline, both steps, both families: (1) fine-tune on I1 → frozen A'
ceiling, (2) train a **real, GT-supervised, pair-free** filter (no reference
image, detector's own GT loss) against the frozen A', (3) benchmark A'/B/filter(B)
on the 6 held-out SAM3-labeled test scenes. Unlike the identity-filter smoke
check earlier in this session (superseded — see git history of this file if
needed), `filter(B)` below is a genuinely trained filter.

**New code this required** (the earlier version of this doc flagged this as a
gap; now closed):
- `scripts/train_filter_detloss_rtdetrv4.py` — builds `HungarianMatcher` +
  `DFINECriterion` with the exact hyperparameters D-FINE's own trainer uses
  (`3rd_party/libreyolo/libreyolo/models/dfine/trainer.py::on_setup`), since
  `LibreRTDETRv4`/`LibreDFINE` expose no `build_criterion_and_postprocess()`
  wrapper the way RF-DETR does.
- `scripts/train_filter_detloss_yolo9.py` — YOLOv9 needs no separate criterion
  object: `LibreYOLO9Model.forward(x, targets=...)` in training mode routes to
  its own `YOLO9Loss` (Task-Aligned-Assignment) internally and returns
  `{"total_loss": ..., ...}` directly. Needed a cxcywh→xyxy target-format
  conversion (`cxcywh_to_yolo9_targets`) since vcalib's shared `load_split()`
  (from `train_filter_detloss.py`) produces cxcywh-normalized boxes.
- **Both** needed `src/calibration.py::train_mode_except_norm` — a new
  context manager. `DFINECriterion` *requires* `aux_outputs` in the model
  output, which the decoder only emits when `self.training=True`
  (`dfine/decoder.py:898`); YOLOv9's internal loss path is similarly gated on
  training mode. But the model must stay frozen (`requires_grad=False`
  already holds) — the risk is BatchNorm's running_mean/running_var buffers,
  which update on every train-mode forward pass **regardless of
  requires_grad**. `train_mode_except_norm` sets `.train()` on the whole
  module but immediately forces every `BatchNorm{1,2,3}d` submodule back to
  `.eval()`, so training-mode code paths activate without drifting frozen BN
  statistics.

**Exact commands to regenerate everything below:**

```bash
uv run python scripts/make_detection_coco.py --levels 1 --out data/coco/cooktop_ref
uv run python scripts/make_detection_coco.py --levels 2 --out data/coco/cooktop_shift_lv2
uv run python scripts/make_detection_coco.py --levels 3 --out data/coco/cooktop_shift_lv3

# step 1: fine-tune ceilings
uv run python scripts/finetune_rtdetrv4.py --data data/coco/cooktop_ref/data.yaml \
    --project results/finetune --name cooktop_rtdetrv4_s_full \
    --size s --epochs 60 --batch 4 --device cuda --seed 42
uv run python scripts/finetune_yolo9.py --data data/coco/cooktop_ref/data.yaml \
    --project results/finetune --name cooktop_yolo9_t_full \
    --size t --epochs 60 --batch 4 --device cuda --seed 42

# step 2: pair-free filter, per family per shifted level (lv=2,3)
uv run python scripts/train_filter_detloss_rtdetrv4.py \
    --data data/coco/cooktop_shift_lv<lv> \
    --model-checkpoint results/finetune/cooktop_rtdetrv4_s_full/weights/best.pt \
    --size s --input-size 640 --out results/experiments/runs/pairfree_rtdetrv4_lv<lv> \
    --epochs 50 --patience 10 --device cuda --seed 42
uv run python scripts/train_filter_detloss_yolo9.py \
    --data data/coco/cooktop_shift_lv<lv> \
    --model-checkpoint results/finetune/cooktop_yolo9_t_full/weights/best.pt \
    --size t --input-size 640 --out results/experiments/runs/pairfree_yolo9_lv<lv> \
    --epochs 50 --patience 10 --device cuda --seed 42

# step 3: benchmark
uv run python scripts/benchmark_detection.py --level <lv> --family <family> --size <size> \
    --input-size 640 --model-checkpoint <finetune-ckpt> \
    --checkpoint results/experiments/runs/pairfree_<family>_lv<lv>/best.pt --device cuda
```

**IMPORTANT reproducibility caveat found this run**: re-running the *identical*
RT-DETRv4 fine-tune command (same seed=42, same code, same 20-image dataset)
produced a **different** result across two runs in this session — first run
best mAP50-95=0.8403 @ epoch 50, second run best mAP50-95=0.6915 @ epoch 20.
YOLOv9's fine-tune was bit-identical across both runs (0.4976 @ epoch 50,
matched exactly). So `--seed` does **not** guarantee reproducibility for
RT-DETRv4 on this box — likely non-deterministic CUDA ops in its deformable-
attention/transformer path (cuDNN algorithm selection, non-deterministic
atomics in backward, or similar), not a code bug introduced here. **Practical
implication**: for RT-DETRv4, treat "the config + seed" as reproducing *a*
result in the same distribution, not a bit-exact one — re-run and average
if a precise number matters. The table below uses the **second** (regenerated)
fine-tune checkpoint, since that's the one the filters were actually trained
against.

**Fine-tune ceiling metrics** (val split, 4 images — sanity check, not reliable
on its own):

| family | size | best epoch | val mAP50-95 | val mAP50 | wall time |
|---|---|---|---|---|---|
| RT-DETRv4 | s | 20/60 | 0.6915 | 0.6943 | 1m40s |
| YOLOv9 | t | 50/60 | 0.4976 | 0.8000 | 0m54s |

**Pair-free filter training** (`spatial_tone_curve(P=16, grid_size=5)`,
`reg_weight=0.01`, `lr=0.005`, early-stop patience 10):

| family | level | best val loss | best epoch | early-stopped @ |
|---|---|---|---|---|
| RT-DETRv4 | 2 | 19.7339 | 4 | 14 |
| RT-DETRv4 | 3 | 16.9124 | 1 | 11 |
| YOLOv9 | 2 | 5.1892 | 25 | 35 |
| YOLOv9 | 3 | 6.2277 | 20 | 30 |

**A'/B/filter(B) benchmark** (6 held-out SAM3-labeled test scenes):

| family | level | A' (AP) | B (AP) | filter(B) (AP) | A'→B gap | recovery |
|---|---|---|---|---|---|---|
| RT-DETRv4 | 2 | 0.5284 | 0.5394 | 0.5784 | −0.0110 (B>A') | n/a — filter still **+0.039 AP** over B |
| RT-DETRv4 | 3 | 0.5284 | 0.4095 | 0.4060 | +0.1189 | **−2.9%** (slightly worse than B) |
| YOLOv9 | 2 | 0.2263 | 0.0939 | 0.2607 | +0.1324 | **+125.9%** (filter(B) exceeds A'!) |
| YOLOv9 | 3 | 0.2263 | 0.1240 | 0.1733 | +0.1023 | **+48.2%** |

**Reading this table — real, if noisy, signal:**
- **YOLOv9 shows real recovery.** On level 2 the trained filter doesn't just
  close the A'→B gap, it *exceeds the fine-tuned ceiling itself*
  (filter(B)=0.2607 > A'=0.2263) — meaning the filter is doing something
  beyond simply undoing the illumination shift; it may be exploiting an
  interaction between `spatial_tone_curve`'s contrast/tone adjustments and
  YOLOv9's specific training distribution. Level 3 recovers a solid 48%.
  Given YOLOv9's low absolute AP ceiling (0.23) on this tiny dataset, treat the
  125.9% figure as encouraging but noisy, not a load-bearing number — it would
  need more test scenes / repeated seeds to trust as a precise percentage.
- **RT-DETRv4 shows no reliable recovery** — level 3's small negative "recovery"
  (−2.9%) and level 2's undefined case (B already exceeds A', so "gap" isn't
  meaningful) both suggest RT-DETRv4-s here is either already fairly robust to
  this cooktop shift, or the filter/criterion interaction isn't working well
  for this architecture yet. Given the fine-tune reproducibility issue above,
  this A' ceiling itself is noisier than YOLOv9's — don't over-read a single run.
- Neither family had enough of a controlled comparison (single seed, single
  fine-tune run, tiny 20-image train set) to be a confident final answer —
  this is a first real data point, not a validated conclusion. A next step
  would be repeating each cell across 2-3 seeds to see if YOLOv9's positive
  signal and RT-DETRv4's null both hold up.

**Checkpoint disk hygiene applied**: `results/finetune/cooktop_rtdetrv4_s_full/`
(1.3GB), `results/finetune/cooktop_yolo9_t_full/` (197MB), and the four
`results/experiments/runs/pairfree_{rtdetrv4,yolo9}_lv{2,3}/` filter checkpoints
(16KB each) were deleted from local disk after every number in this section was
recorded here. Regenerate via the exact commands above if needed again — note
the RT-DETRv4 non-determinism caveat means a regenerated RT-DETRv4 ceiling may
not exactly match 0.6915/0.6943 even with the same seed.

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
