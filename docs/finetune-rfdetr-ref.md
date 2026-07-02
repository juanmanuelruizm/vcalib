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

Results (benchmark CSVs) land in `results/ft_bench/`. ExDark (optional):
`git clone https://github.com/cs-chan/Exclusively-Dark-Image-Dataset data/exdark_raw`
then `uv run python scripts/exdark_to_coco.py --exdark-root data/exdark_raw --out data/coco/exdark`.

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
- `scripts/exdark_to_coco.py` — **UNTESTED until ExDark is downloaded**; validate the bbGt/
  `imageclasslist.txt` parsing after download.

## Gotchas to honor when interpreting results

- In the pure realistic (unpaired) framing there is **no A ceiling** → report **absolute AP
  gain** from the filter; cooktop still has A' (fine-tuned on I1) for classic recovery.
- **Overfit**: only ~20 cooktop train scenes → head-only/LoRA + aggressive aug + early stop.
- A model over-adapted to reference light may be **more** brittle to the shift — a valid outcome.
- **ExDark**: heterogeneous resolution + `to_unit_rgb`'s square resize distorts aspect ratio
  per image (consider letterbox); watch small objects at 384².
- **3-way split discipline**: test = final only; val = early-stop only; never leak test.
- Fine-tune is single-class "object" (head → 2 logits) to match the GT and class-agnostic eval.
