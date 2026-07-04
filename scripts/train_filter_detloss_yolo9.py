#!/usr/bin/env python3
"""Train a preprocessing filter WITHOUT pairs, supervised by YOLOv9's own GT loss.

Sibling of ``train_filter_detloss.py`` for ``family="yolo9"``. Unlike RF-DETR/
RT-DETRv4's DETR-style Hungarian-matcher criterion, YOLOv9's own head computes its
GT loss internally: ``LibreYOLO9Model.forward(x, targets=...)`` in training mode
routes to ``DDetect.forward``'s Task-Aligned-Assignment ``YOLO9Loss``
(``3rd_party/libreyolo/libreyolo/models/yolo9/loss.py``) and returns a single
``loss_dict`` (with ``"total_loss"`` plus per-component breakdown) directly — no
separate criterion object to build.

Same BatchNorm-safety concern as the RT-DETRv4 sibling applies here too (YOLOv9's
CNN backbone/neck are full of BatchNorm2d): the forward must run with
``self.training=True`` for the internal loss path to activate, but BN running
stats must not drift, hence ``src/calibration.py::train_mode_except_norm``.

Target format YOLOv9 expects (``DDetect.forward``'s docstring): ``(B, max_targets, 5)``
with ``[class_id, x1, y1, x2, y2]``, all normalized to [0, 1] — vcalib's
``load_split`` (shared with ``train_filter_detloss.py``) gives cxcywh-normalized
boxes, converted to xyxy here.

  filter(shifted) -> frozen YOLOv9 (train-mode-except-BN, +targets)
      -> YOLO9Loss(outputs, GT boxes) -> backprop to filter

Usage:
  uv run python scripts/train_filter_detloss_yolo9.py \
      --data data/coco/cooktop_shift_lv2 \
      --model-checkpoint results/finetune/cooktop_yolo9_t_full/weights/best.pt \
      --out results/experiments/runs/pairfree_yolo9_lv2
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


def cxcywh_to_yolo9_targets(boxes: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """(N, 4) cxcywh[0,1] + (N,) labels -> (1, N, 5) [class, x1, y1, x2, y2] normalized."""
    cx, cy, w, h = boxes.unbind(-1)
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy + h / 2
    xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
    row = torch.cat([labels.unsqueeze(-1).float(), xyxy], dim=-1)
    return row.unsqueeze(0)


def det_loss(model, filt, img_dir, file_name, boxes, labels, size, dev):
    unit = to_unit_rgb(img_dir / file_name, size).to(dev)
    fb = filt(unit.unsqueeze(0)).clamp(0, 1)
    normed = normalize(fb[0]).unsqueeze(0)
    targets = cxcywh_to_yolo9_targets(boxes, labels).to(dev)
    with train_mode_except_norm(model.model):
        loss_dict = model.model(normed, targets=targets)
    return loss_dict["total_loss"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="COCO dataset dir (make_detection_coco.py)")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="val")
    ap.add_argument("--model-checkpoint", default=None, help="fine-tuned YOLOv9 checkpoint; default off-the-shelf")
    ap.add_argument("--size", default="t", help="YOLOv9 size: t, s, m, c")
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

    model = load_model(size=args.size, device=dev, model_path=args.model_checkpoint, family="yolo9")

    spec = {"type": args.filter_type, "P": args.P, "grid_size": args.grid_size}
    filt = build_filter(spec).to(dev).train()
    opt = torch.optim.Adam(filt.parameters(), lr=args.lr)

    train = load_split(data_dir, args.train_split)
    val = load_split(data_dir, args.val_split)

    print(f"filter {spec} | model={'off-the-shelf' if not args.model_checkpoint else args.model_checkpoint} (yolo9/{args.size})")
    print(f"train={len(train)} val={len(val)} | detector-GT loss (pair-free, YOLO9Loss)")

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
            loss = det_loss(model, filt, train_dir, fn, boxes, labels, args.input_size, dev)
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
            vloss = det_loss(model, filt, val_dir, fn, boxes, labels, args.input_size, dev)
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
