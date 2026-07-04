#!/usr/bin/env python3
"""Fine-tune (or train from scratch) LibreFOMO on a reference/nominal dataset.

Separate, lower-confidence track from ``finetune_rtdetrv4.py``/``finetune_yolo9.py``:

- LibreFOMO is a **point-localizer**, not a box detector (``SUPPORTED_TASKS =
  ("point",)``) — it has no AP metric; downstream eval needs a point-distance /
  recall-at-radius metric instead (see ``detection_output_loss_fomo`` in
  ``src/calibration.py`` for the analogous pair-free-loss adapter, which
  compares per-cell heatmaps, not boxes).
- No pretrained weights are redistributed upstream (``LibreFOMO.get_download_url``
  always returns ``None`` — "not redistributed or auto-downloaded"). Pass
  ``--model-path`` to fine-tune from a checkpoint you already have; omit it to
  train from random initialization instead (documented here, not silently
  assumed, since it changes what "ceiling" A' means for this family).
- LibreFOMO's own ``.train()`` requires ``allow_experimental=True`` and warns
  that training support is experimental/unstable upstream — this script passes
  that flag explicitly and surfaces the warning rather than suppressing it.

Usage:
  uv run python scripts/finetune_fomo.py \
      --data data/coco/cooktop_ref/data.yaml \
      --project results/finetune --name cooktop_fomo \
      --size s --epochs 60 --device cuda
  # or, to fine-tune from an existing checkpoint:
  uv run python scripts/finetune_fomo.py --model-path path/to/checkpoint.pt ...
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
    ap.add_argument("--size", default="s", help="LibreFOMO size: s, m, l (no nano)")
    ap.add_argument("--model-path", default=None,
                    help="checkpoint to fine-tune from; omit to train from random init "
                    "(no pretrained weights are redistributed for this family)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr0", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=15, help="early-stopping patience (0=off)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from libreyolo import LibreFOMO

    torch.manual_seed(args.seed)
    if args.model_path is not None and not Path(args.model_path).exists():
        raise FileNotFoundError(f"--model-path does not exist: {args.model_path}")

    print(
        "[finetune_fomo] LibreFOMO training is EXPERIMENTAL upstream — treat results "
        "as exploratory, not a validated baseline."
    )
    m = LibreFOMO(model_path=args.model_path, size=args.size, device=args.device)

    print(f"fine-tune | family=fomo data={args.data} size={args.size} "
          f"from_scratch={args.model_path is None} epochs={args.epochs} "
          f"batch={args.batch} lr0={args.lr0} device={args.device}")
    res = m.train(
        data=args.data,
        allow_experimental=True,
        epochs=args.epochs,
        batch=args.batch,
        lr0=args.lr0,
        project=args.project,
        name=args.name,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
    )
    print(f"\ntrain result keys: {list(res.keys())}")


if __name__ == "__main__":
    main()
