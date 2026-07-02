#!/usr/bin/env python3
"""Fine-tune RF-DETR-nano on a reference/nominal COCO dataset (domain adaptation).

Thin wrapper over LibreYOLO's native trainer (``LibreRFDETR.train``) with the
regularisation knobs that matter for a tiny fine-tune set: layer freezing or LoRA,
EMA, and early stopping. Produces a domain-adapted, single-class detector whose
``weights/best.pt`` is then used (frozen) as the reference detector for both the
pair-free filter trainer and the benchmark.

  --freeze backbone   head-only fine-tune (freeze the DINOv2 backbone)
  --lora              parameter-efficient LoRA adapters instead of full head

Usage:
  uv run python scripts/finetune_rfdetr.py \
      --data data/coco/cooktop_ref/data.yaml \
      --out results/finetune/cooktop_ref_headonly \
      --freeze backbone --epochs 60 --device cuda
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: F401  (ensures MPS fallback env is respected before libreyolo import)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to data.yaml (make_detection_coco.py)")
    ap.add_argument("--out", required=True, help="output dir (checkpoints under <out>/weights)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--freeze", default="backbone",
                    help="int (first N groups), module-name (e.g. 'backbone'), or 'none'")
    ap.add_argument("--lora", action="store_true", help="LoRA adapters instead of full fine-tune")
    ap.add_argument("--patience", type=int, default=15, help="early-stopping patience (0=off)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from src.utils.activations import DEFAULT_WEIGHTS_DIR
    from libreyolo import LibreRFDETR

    torch.manual_seed(args.seed)
    weights = str(Path(DEFAULT_WEIGHTS_DIR) / "rf-detr-nano.pth")
    m = LibreRFDETR(model_path=weights, size="n", device=args.device)

    freeze = None
    if args.freeze and args.freeze.lower() != "none":
        freeze = int(args.freeze) if args.freeze.isdigit() else args.freeze

    print(f"fine-tune | data={args.data} freeze={freeze} lora={args.lora} "
          f"epochs={args.epochs} bs={args.batch_size} lr={args.lr} device={args.device}")
    res = m.train(
        data=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.out,
        freeze=freeze,
        lora=args.lora,
        patience=args.patience,
        ema=True,
        seed=args.seed,
        device=args.device,
    )
    print(f"\nbest mAP50-95={res.get('best_mAP50_95')} mAP50={res.get('best_mAP50')} "
          f"@epoch {res.get('best_epoch')}")
    print(f"  checkpoint: {res.get('best_checkpoint')}")


if __name__ == "__main__":
    main()
