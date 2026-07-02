#!/usr/bin/env bash
# Full ExDark matrix for the finetune-rfdetr-ref experiment (unpaired, 12-class).
#
# Two frozen-detector arms, each evaluated no-filter (B) vs pair-free filter:
#   * baseline : off-the-shelf COCO RF-DETR       (labels mapped to COCO-91)
#   * adapted  : A' fine-tuned on the bright (reference) ExDark conditions
#
# Pipeline: build COCO -> fine-tune A' on bright -> pair-free filter on dark_train
#           (per arm) -> eval absolute AP on dark_test (B vs filter(B)).
#
#   ./scripts/run_exdark_matrix.sh [DEVICE]           # DEVICE defaults to cuda
#   FT_EPOCHS=60 FILT_EPOCHS=50 ./scripts/run_exdark_matrix.sh cuda
#
# Prereq: data/exdark_raw populated from the Drive downloads (see
#         docs/finetune-rfdetr-ref.md). data/coco/exdark is regenerated here.
set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${1:-cuda}"
FT_EPOCHS="${FT_EPOCHS:-60}"
FILT_EPOCHS="${FILT_EPOCHS:-50}"
PY="uv run python"
RUNS="results/experiments/runs"
BENCH="results/ft_bench"
DATA="data/coco/exdark"
mkdir -p "$BENCH"

echo "== 1. build ExDark COCO (12-class) =="
$PY scripts/exdark_to_coco.py --exdark-root data/exdark_raw --out "$DATA" --multiclass

echo "== 2. fine-tune A' on the bright (reference) conditions =="
$PY scripts/finetune_rfdetr.py --data "$DATA/data_bright.yaml" \
    --out results/finetune/exdark_bright --freeze backbone \
    --epochs "$FT_EPOCHS" --device "$DEVICE"
ADAPTED="results/finetune/exdark_bright/weights/best.pt"

# train the pair-free filter on dark_train, then eval absolute AP on dark_test.
#   $1=tag  $2=label-map  $3=model-checkpoint (empty -> off-the-shelf)
filter_and_bench () {
  local tag="$1" lmap="$2" ckpt="${3:-}"
  local mc=(); [ -n "$ckpt" ] && mc=(--model-checkpoint "$ckpt")
  local filt="$RUNS/exdark_pairfree_${tag}"
  echo "== pair-free filter ($tag) =="
  $PY scripts/train_filter_detloss.py --data "$DATA" \
      --train-split dark_train --val-split dark_val --label-map "$lmap" \
      "${mc[@]}" --out "$filt" --epochs "$FILT_EPOCHS" --device "$DEVICE"
  echo "== eval ($tag) on dark_test =="
  $PY scripts/benchmark_exdark.py --data "$DATA" --split dark_test \
      --checkpoint "$filt/best.pt" --label-map "$lmap" "${mc[@]}" \
      --device "$DEVICE" --out "$BENCH/exdark_${tag}.csv"
}

filter_and_bench baseline exdark_coco ""          # off-the-shelf arm
filter_and_bench adapted  none        "$ADAPTED"  # A' (fine-tuned on bright) arm

echo "== done. benchmark CSVs in $BENCH/ =="
ls -1 "$BENCH"/exdark_*.csv
