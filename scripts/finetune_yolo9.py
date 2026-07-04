#!/usr/bin/env python3
"""Fine-tune YOLOv9 on a reference/nominal COCO dataset (domain adaptation).

Sibling of ``finetune_rfdetr.py`` for the ``yolo9`` family (see
``src/utils/activations.py``'s ``family`` param). Thin wrapper over LibreYOLO's
native trainer (``LibreYOLO9.train``).

``freeze`` IS supported here with **named** freeze groups (YOLOv9's trainer
overrides ``get_freeze_groups()`` with e.g. ``"backbone.conv0"``,
``"neck.elan_up1"``, ``"head"`` — see
``3rd_party/libreyolo/libreyolo/models/yolo9/trainer.py``), unlike D-FINE/
RT-DETRv4 which only get generic integer groups. ``lora`` is NOT supported
(only RF-DETR's trainer sets ``supports_lora = True``).

Note: YOLOv9 is an anchor-free CNN detector (``DDetect`` head), not a DETR
query-based one — its raw ``forward()`` output is NOT the
``{pred_logits, pred_boxes}`` dict that ``src/calibration.py``'s
``detection_output_loss`` and ``scripts/benchmark_detection.py`` assume. A
pair-free-filter run against YOLOv9 needs the YOLO9-specific output adapter
(see the TODO in ``src/calibration.py`` / ``scripts/benchmark_detection.py``)
before ``train_filter_detloss.py``/``benchmark_detection.py`` can be reused
as-is; this script only covers the fine-tune step.

Usage:
  uv run python scripts/finetune_yolo9.py \
      --data data/coco/cooktop_ref/data.yaml \
      --project results/finetune --name cooktop_yolo9 \
      --size t --epochs 60 --device cuda
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
    ap.add_argument("--project", required=True, help="root dir for training runs")
    ap.add_argument("--name", required=True, help="experiment name (run dir = <project>/<name>)")
    ap.add_argument("--size", default="t", help="YOLOv9 size: t, s, m, c")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--lr0", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=15, help="early-stopping patience (0=off)")
    ap.add_argument("--freeze", default="none",
                    help="int (first N named groups), module-name selector "
                    "(e.g. 'backbone.conv0'), or 'none'")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    freeze = None
    if args.freeze and args.freeze.lower() != "none":
        freeze = int(args.freeze) if args.freeze.isdigit() else args.freeze

    from src.utils.activations import DEFAULT_WEIGHTS_DIR
    from libreyolo import LibreYOLO9
    from libreyolo.utils.download import download_weights

    torch.manual_seed(args.seed)
    weights = str(Path(DEFAULT_WEIGHTS_DIR) / f"LibreYOLO9{args.size}.pt")
    download_weights(weights, args.size)
    m = LibreYOLO9(model_path=weights, size=args.size, device=args.device)

    print(f"fine-tune | family=yolo9 data={args.data} size={args.size} freeze={freeze} "
          f"epochs={args.epochs} batch={args.batch} lr0={args.lr0} device={args.device}")
    res = m.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr0,
        project=args.project,
        name=args.name,
        patience=args.patience,
        freeze=freeze,
        seed=args.seed,
        device=args.device,
    )
    print(f"\nbest mAP50-95={res.get('best_mAP50_95')} mAP50={res.get('best_mAP50')} "
          f"@epoch {res.get('best_epoch')}")
    print(f"  checkpoint: {res.get('best_checkpoint')}")


if __name__ == "__main__":
    main()
