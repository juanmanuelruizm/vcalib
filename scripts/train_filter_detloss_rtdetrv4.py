#!/usr/bin/env python3
"""Train a preprocessing filter WITHOUT pairs, supervised by RT-DETRv4's own GT loss.

Sibling of ``train_filter_detloss.py`` for ``family="rtdetrv4"``. RF-DETR's version
calls ``model.model.build_criterion_and_postprocess()`` — an RF-DETR-only method
(``3rd_party/libreyolo/libreyolo/models/rfdetr/nn.py:737``). RT-DETRv4 (sharing
D-FINE's architecture) has no such wrapper method, but the underlying pieces exist:
``HungarianMatcher`` + ``DFINECriterion``
(``3rd_party/libreyolo/libreyolo/models/dfine/{matcher,loss}.py``), built here with
the exact hyperparameters D-FINE's own trainer uses
(``3rd_party/libreyolo/libreyolo/models/dfine/trainer.py::on_setup``).

Two things this needs that RF-DETR's simpler path didn't:
  1. ``DFINECriterion`` REQUIRES ``aux_outputs`` in the model output (raises
     otherwise) — these are only emitted when the underlying decoder module has
     ``self.training=True`` (``3rd_party/libreyolo/libreyolo/models/dfine/decoder.py:898``).
     So the forward pass must run with the model in train() mode.
  2. The model must stay FROZEN throughout (no weight updates) — but BatchNorm's
     running_mean/running_var buffers update on every train-mode forward pass
     regardless of ``requires_grad``. ``src/calibration.py::train_mode_except_norm``
     pins BatchNorm submodules to eval while everything else is in train() mode, so
     ``self.training`` reads True (aux_outputs get emitted) without drifting BN stats.

  filter(shifted) -> frozen RT-DETRv4 (train-mode-except-BN, +targets for aux_outputs)
      -> DFINECriterion(outputs, GT boxes) -> backprop to filter

Usage:
  uv run python scripts/train_filter_detloss_rtdetrv4.py \
      --data data/coco/cooktop_shift_lv2 \
      --model-checkpoint results/finetune/cooktop_rtdetrv4_s_full/weights/best.pt \
      --out results/experiments/runs/pairfree_rtdetrv4_lv2
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from PIL import ImageFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.filters import build_filter
from src.utils.activations import load_model, to_unit_rgb, normalize
from src.calibration import train_mode_except_norm
from scripts.train_filter_detloss import load_split

ImageFile.LOAD_TRUNCATED_IMAGES = True


def build_dfine_criterion(num_classes: int):
    """Same matcher/criterion hyperparameters as D-FINE's own trainer
    (``3rd_party/libreyolo/libreyolo/models/dfine/trainer.py::on_setup``)."""
    from libreyolo.models.dfine.loss import DFINECriterion
    from libreyolo.models.dfine.matcher import HungarianMatcher

    matcher = HungarianMatcher(
        weight_dict={"cost_class": 2.0, "cost_bbox": 5.0, "cost_giou": 2.0},
        use_focal_loss=True,
        alpha=0.25,
        gamma=2.0,
    )
    return DFINECriterion(
        matcher=matcher,
        weight_dict={
            "loss_vfl": 1.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
            "loss_fgl": 0.15,
            "loss_ddf": 1.5,
        },
        losses=["vfl", "boxes", "local"],
        alpha=0.75,
        gamma=2.0,
        num_classes=num_classes,
        reg_max=32,
    )


def det_loss(model, crit, filt, img_dir, file_name, boxes, labels, size, dev):
    unit = to_unit_rgb(img_dir / file_name, size).to(dev)
    fb = filt(unit.unsqueeze(0)).clamp(0, 1)
    normed = normalize(fb[0]).unsqueeze(0)
    targets = [{"labels": labels.to(dev), "boxes": boxes.to(dev)}]
    with train_mode_except_norm(model.model):
        outputs = model.model(normed, targets=targets)
    losses = crit(outputs, targets)
    return sum(losses.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="COCO dataset dir (make_detection_coco.py)")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="val")
    ap.add_argument("--model-checkpoint", default=None, help="fine-tuned RT-DETRv4 checkpoint; default off-the-shelf")
    ap.add_argument("--size", default="s", help="RT-DETRv4 size: s, m, l, x")
    ap.add_argument("--filter-type", default="spatial_tone_curve")
    ap.add_argument("--P", type=int, default=16)
    ap.add_argument("--grid-size", type=int, default=5)
    ap.add_argument("--out", required=True, help="filter checkpoint dir")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--reg-weight", type=float, default=0.01)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--input-size", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    dev = args.device
    data_dir = Path(args.data)
    train_dir = data_dir / "images" / args.train_split
    val_dir = data_dir / "images" / args.val_split

    model = load_model(size=args.size, device=dev, model_path=args.model_checkpoint, family="rtdetrv4")
    crit = build_dfine_criterion(model.nb_classes).to(dev)

    spec = {"type": args.filter_type, "P": args.P, "grid_size": args.grid_size}
    filt = build_filter(spec).to(dev).train()
    opt = torch.optim.Adam(filt.parameters(), lr=args.lr)

    train = load_split(data_dir, args.train_split)
    val = load_split(data_dir, args.val_split)

    print(f"filter {spec} | model={'off-the-shelf' if not args.model_checkpoint else args.model_checkpoint} (rtdetrv4/{args.size})")
    print(f"train={len(train)} val={len(val)} | detector-GT loss (pair-free, DFINECriterion)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics_f = open(out / "metrics.jsonl", "w")
    best_val, best_epoch, since = float("inf"), -1, 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        filt.train()
        random.shuffle(train)
        tr_loss = 0.0
        for fn, W, H, boxes, labels in train:
            if len(boxes) == 0:
                continue
            opt.zero_grad()
            loss = det_loss(model, crit, filt, train_dir, fn, boxes, labels, args.input_size, dev)
            loss = loss + args.reg_weight * filt.reg_loss()
            loss.backward()
            opt.step()
            tr_loss += float(loss.detach())
        tr_loss /= max(1, len(train))

        filt.eval()
        vl = 0.0
        for fn, W, H, boxes, labels in val:
            if len(boxes) == 0:
                continue
            vloss = det_loss(model, crit, filt, val_dir, fn, boxes, labels, args.input_size, dev)
            vl += float(vloss.detach())
        vl /= max(1, len(val))

        metrics_f.write(json.dumps({"epoch": epoch, "train_loss": tr_loss, "val_loss": vl}) + "\n")
        metrics_f.flush()
        flag = ""
        if vl < best_val:
            best_val, best_epoch, since = vl, epoch, 0
            torch.save(filt.state_dict(), out / "best.pt")
            flag = "  *best"
        else:
            since += 1
        print(f"[{epoch:3}/{args.epochs}] train={tr_loss:.4f} val={vl:.4f}{flag}")
        if args.patience and since >= args.patience:
            print(f"early stop (no val improvement for {args.patience} epochs)")
            break

    metrics_f.close()
    print(f"\nDONE in {time.time()-t0:.0f}s | best val={best_val:.4f} @epoch {best_epoch}")
    print(f"  wrote {out/'best.pt'}")


if __name__ == "__main__":
    main()
