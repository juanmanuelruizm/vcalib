#!/usr/bin/env python3
"""Fine-tune RT-DETRv4 on a reference/nominal COCO dataset (domain adaptation).

Sibling of ``finetune_rfdetr.py`` for the ``rtdetrv4`` family (see
``src/utils/activations.py``'s ``family`` param). Thin wrapper over LibreYOLO's
native trainer (``LibreRTDETRv4.train``).

``freeze`` is generically supported here (a ``TrainConfig`` field every
``BaseTrainer`` subclass consumes via ``**kwargs``), but D-FINE/RT-DETRv4 don't
override ``get_freeze_groups()`` the way YOLOv9 does, so freeze groups are
generic integer indices, not named selectors like ``"backbone"``. ``lora`` is
NOT supported here: only RF-DETR's trainer sets ``supports_lora = True``
(`3rd_party/libreyolo/libreyolo/models/rfdetr/trainer.py:78`); passing
``lora=True`` for this family raises upstream. Kwarg names also differ from
RF-DETR's own script (``batch``/``imgsz``/``lr0``/``project``/``name``/``amp``
vs. RF-DETR's ``batch_size``/``lr``/``output_dir``/``ema``) — this mirrors
LibreYOLO's own inconsistency rather than papering over it.

Usage:
  uv run python scripts/finetune_rtdetrv4.py \
      --data data/coco/cooktop_ref/data.yaml \
      --project results/finetune --name cooktop_rtdetrv4 \
      --size s --epochs 60 --device cuda
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
    ap.add_argument("--size", default="s", help="RT-DETRv4 size: s, m, l, x")
    ap.add_argument("--epochs", type=int, default=58)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--lr0", type=float, default=5e-4)
    ap.add_argument("--patience", type=int, default=50, help="early-stopping patience (0=off)")
    ap.add_argument("--freeze", default="none",
                    help="int (first N generic freeze groups) or 'none'; no named "
                    "selectors like RF-DETR's 'backbone' (D-FINE/RTDETRv4 don't "
                    "override get_freeze_groups())")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    freeze = None
    if args.freeze and args.freeze.lower() != "none":
        freeze = int(args.freeze) if args.freeze.isdigit() else args.freeze

    from src.utils.activations import DEFAULT_WEIGHTS_DIR
    from libreyolo import LibreRTDETRv4
    from libreyolo.utils.download import download_weights

    torch.manual_seed(args.seed)
    weights = str(Path(DEFAULT_WEIGHTS_DIR) / f"LibreRTDETRv4{args.size}.pt")
    download_weights(weights, args.size)  # LibreDFINE._load_weights doesn't auto-download
    m = LibreRTDETRv4(model_path=weights, size=args.size, device=args.device)

    print(f"fine-tune | family=rtdetrv4 data={args.data} size={args.size} "
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
