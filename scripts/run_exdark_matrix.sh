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
# Pair-free CORAL activation term (unpaired domain alignment to the reference/bright
# condition). ACT_WEIGHT=0 -> detection-only (original behaviour). REF_SPLIT is the
# reference domain whose projector statistics the filter aligns to.
ACT_WEIGHT="${ACT_WEIGHT:-1.0}"
REF_SPLIT="${REF_SPLIT:-bright_train}"
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
#   $1=tag  $2=label-map  $3=model-checkpoint (empty -> off-the-shelf)  $4=coral (0/1)
filter_and_bench () {
  local tag="$1" lmap="$2" ckpt="${3:-}" coral="${4:-0}"
  local mc=(); [ -n "$ckpt" ] && mc=(--model-checkpoint "$ckpt")
  local extra=()
  [ "$coral" = "1" ] && extra=(--act-weight "$ACT_WEIGHT" --ref-split "$REF_SPLIT")
  local filt="$RUNS/exdark_pairfree_${tag}"
  echo "== pair-free filter ($tag) =="
  $PY scripts/train_filter_detloss.py --data "$DATA" \
      --train-split dark_train --val-split dark_val --label-map "$lmap" \
      "${mc[@]}" "${extra[@]}" --out "$filt" --epochs "$FILT_EPOCHS" --device "$DEVICE"
  echo "== eval ($tag) on dark_test =="
  $PY scripts/benchmark_exdark.py --data "$DATA" --split dark_test \
      --checkpoint "$filt/best.pt" --label-map "$lmap" "${mc[@]}" \
      --device "$DEVICE" --out "$BENCH/exdark_${tag}.csv"
}

# detection-only arms (original) ...
filter_and_bench baseline exdark_coco ""          0   # off-the-shelf arm
filter_and_bench adapted  none        "$ADAPTED"  0   # A' (fine-tuned on bright) arm
# ... vs combined det-GT + CORAL arms (unpaired activation alignment).
# Skip only when explicitly disabled (ACT_WEIGHT=0 or 0.0).
case "$ACT_WEIGHT" in
  0|0.0|0.00) echo "== CORAL arms skipped (ACT_WEIGHT=$ACT_WEIGHT) ==" ;;
  *)
    filter_and_bench baseline_coral exdark_coco ""          1
    filter_and_bench adapted_coral  none        "$ADAPTED"  1
    ;;
esac

echo "== done. benchmark CSVs in $BENCH/ =="
ls -1 "$BENCH"/exdark_*.csv
