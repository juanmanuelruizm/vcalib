#!/usr/bin/env bash
# Full cooktop matrix for the finetune-rfdetr-ref experiment. Meant for the CUDA box.
#
#   ./scripts/run_finetune_matrix.sh [DEVICE]        # DEVICE defaults to cuda
#   FT_EPOCHS=60 FILT_EPOCHS=50 NAUG=5 ./scripts/run_finetune_matrix.sh cuda
#
# Pipeline per cell:
#   fine-tune RF-DETR on the reference (I1) -> frozen adapted model M
#   train the PAIR-FREE filter on shifted I2/I3 (GT detection loss, no A)
#   benchmark A'(=M on I1) / B / filter(B) on the 6 held-out test scenes
#
# Matrix: {head-only-freeze, LoRA} x {no-aug, aggressive-aug}, + off-the-shelf baseline.
set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${1:-cuda}"
FT_EPOCHS="${FT_EPOCHS:-60}"
FILT_EPOCHS="${FILT_EPOCHS:-50}"
NAUG="${NAUG:-5}"
PY="uv run python"
RUNS="results/experiments/runs"
BENCH="results/ft_bench"
mkdir -p "$BENCH"

echo "== 1. datasets =="
$PY scripts/make_detection_coco.py --levels 1   --out data/coco/cooktop_ref
$PY scripts/make_detection_coco.py --levels 2   --out data/coco/cooktop_shift_lv2
$PY scripts/make_detection_coco.py --levels 3   --out data/coco/cooktop_shift_lv3
$PY scripts/augment_illumination.py --data data/coco/cooktop_ref --preset aggressive --n-aug "$NAUG" --out data/coco/cooktop_ref_augA

# fine-tune one cell: name  data.yaml  extra-args...
finetune () {  # $1=name  $2=data  $3...=extra
  local name="$1" data="$2"; shift 2
  echo "== fine-tune $name =="
  $PY scripts/finetune_rfdetr.py --data "$data" --out "results/finetune/$name" \
      --epochs "$FT_EPOCHS" --device "$DEVICE" "$@"
}

finetune ref_headonly_noaug data/coco/cooktop_ref/data.yaml       --freeze backbone
finetune ref_headonly_augA  data/coco/cooktop_ref_augA/data.yaml  --freeze backbone
finetune ref_lora_noaug     data/coco/cooktop_ref/data.yaml       --freeze none --lora
finetune ref_lora_augA      data/coco/cooktop_ref_augA/data.yaml  --freeze none --lora

# per model: pair-free filter on each shifted level, then benchmark A'/B/filter(B)
pairfree_and_bench () {  # $1=tag  $2=model-ckpt-or-empty(off-the-shelf)
  local tag="$1" ckpt="${2:-}"
  local mc=(); [ -n "$ckpt" ] && mc=(--model-checkpoint "$ckpt")
  for lv in 2 3; do
    local filt="$RUNS/pairfree_${tag}_lv${lv}"
    echo "== pair-free filter $tag lv$lv =="
    $PY scripts/train_filter_detloss.py --data "data/coco/cooktop_shift_lv${lv}" \
        "${mc[@]}" --out "$filt" --epochs "$FILT_EPOCHS" --device "$DEVICE"
    echo "== benchmark $tag lv$lv =="
    $PY scripts/benchmark_detection.py --level "$lv" "${mc[@]}" \
        --checkpoint "$filt/best.pt" --scenes test --device "$DEVICE" \
        --out "$BENCH/${tag}_lv${lv}.csv"
  done
}

pairfree_and_bench offtheshelf                 ""                                              # baseline
pairfree_and_bench ref_headonly_noaug results/finetune/ref_headonly_noaug/weights/best.pt
pairfree_and_bench ref_headonly_augA  results/finetune/ref_headonly_augA/weights/best.pt
pairfree_and_bench ref_lora_noaug     results/finetune/ref_lora_noaug/weights/best.pt
pairfree_and_bench ref_lora_augA      results/finetune/ref_lora_augA/weights/best.pt

echo "== done. benchmark CSVs in $BENCH/ =="
ls -1 "$BENCH"
