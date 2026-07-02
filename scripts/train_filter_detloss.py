#!/usr/bin/env python3
"""Train a preprocessing filter WITHOUT pairs, supervised by detection GT.

The realistic deployment setting: the reference illumination is not reproducible,
so you cannot collect (reference, shifted) *pairs* — you can only label the new
shifted images. This trains the filter to make a **frozen** (optionally
domain-adapted) RF-DETR detect the labeled objects in shifted images, using the
detector's own DETR ``SetCriterion`` (Hungarian matcher + focal/L1/GIoU) against
the real GT boxes. No reference image A is used.

  filter(shifted) -> frozen RF-DETR -> SetCriterion(outputs, GT boxes) -> backprop to filter

Data is a COCO dataset produced by ``scripts/make_detection_coco.py`` (a shifted
level, e.g. ``--levels 2``). Evaluate later with ``scripts/benchmark_detection.py``.

Usage:
  uv run python scripts/train_filter_detloss.py \
      --data data/coco/cooktop_shift_lv2 \
      --model-checkpoint results/finetune/cooktop_ref_headonly/weights/best.pt \
      --out results/experiments/runs/pairfree_stc_lv2

  # ExDark (unpaired) off-the-shelf baseline arm -> remap labels to COCO-91:
  uv run python scripts/train_filter_detloss.py --data data/coco/exdark \
      --train-split dark_train --val-split dark_val --label-map exdark_coco \
      --out results/experiments/runs/exdark_pairfree_offtheshelf
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
from src.calibration import _make_nested_tensor

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ExDark contiguous class (by name) -> RF-DETR raw logit column (COCO 91-id space).
# Only for the off-the-shelf baseline arm, whose criterion is 91-class; the
# fine-tuned A' head is native 12-class, so it needs no remap. Derived from
# libreyolo's _COCO91_TO_COCO80 + the model's COCO-80 names (verified).
EXDARK_TO_COCO91 = {
    "Bicycle": 2, "Boat": 9, "Bottle": 44, "Bus": 6, "Car": 3, "Cat": 17,
    "Chair": 62, "Cup": 47, "Dog": 18, "Motorbike": 4, "People": 1, "Table": 67,
}


def load_split(data_dir: Path, split: str):
    """Return [(file_name, W, H, targets)] where targets = {labels, boxes cxcywh[0,1]}."""
    coco = json.load(open(data_dir / "annotations" / f"instances_{split}.json"))
    cat2idx = {c["id"]: i for i, c in enumerate(sorted(coco["categories"], key=lambda c: c["id"]))}
    anns = {}
    for a in coco["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    items = []
    for im in coco["images"]:
        W, H = im["width"], im["height"]
        boxes, labels = [], []
        for a in anns.get(im["id"], []):
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([(x + w / 2) / W, (y + h / 2) / H, w / W, h / H])
            labels.append(cat2idx[a["category_id"]])
        items.append((im["file_name"], W, H,
                      torch.tensor(boxes, dtype=torch.float32),
                      torch.tensor(labels, dtype=torch.long)))
    return items


def det_loss(model, crit, filt, img_dir, file_name, boxes, labels, size, dev):
    unit = to_unit_rgb(img_dir / file_name, size).to(dev)
    fb = filt(unit.unsqueeze(0)).clamp(0, 1)
    normed = normalize(fb[0]).unsqueeze(0)
    outputs = model.model(_make_nested_tensor(normed))
    targets = [{"labels": labels.to(dev), "boxes": boxes.to(dev)}]
    ld = crit(outputs, targets)
    wd = crit.weight_dict
    return sum(ld[k] * wd[k] for k in ld if k in wd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="COCO dataset dir (make_detection_coco.py / exdark_to_coco.py)")
    ap.add_argument("--train-split", default="train", help="training split name (e.g. dark_train for ExDark)")
    ap.add_argument("--val-split", default="val", help="validation split name (e.g. dark_val for ExDark)")
    ap.add_argument("--label-map", choices=["none", "exdark_coco"], default="none",
                    help="'exdark_coco' remaps ExDark labels to RF-DETR's COCO-91 logit columns "
                         "(off-the-shelf baseline arm); 'none' keeps native dataset labels (A')")
    ap.add_argument("--model-checkpoint", default=None, help="fine-tuned detector; default off-the-shelf")
    ap.add_argument("--filter-type", default="spatial_tone_curve")
    ap.add_argument("--P", type=int, default=16)
    ap.add_argument("--grid-size", type=int, default=5)
    ap.add_argument("--out", required=True, help="filter checkpoint dir")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--reg-weight", type=float, default=0.01)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--input-size", type=int, default=384)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    dev = args.device
    data_dir = Path(args.data)
    train_dir = data_dir / "images" / args.train_split
    val_dir = data_dir / "images" / args.val_split

    model = load_model(size="n", device=dev, model_path=args.model_checkpoint)
    crit = model.model.build_criterion_and_postprocess()[0]
    crit.to(dev)

    spec = {"type": args.filter_type, "P": args.P, "grid_size": args.grid_size}
    filt = build_filter(spec).to(dev).train()
    opt = torch.optim.Adam(filt.parameters(), lr=args.lr)

    train = load_split(data_dir, args.train_split)
    val = load_split(data_dir, args.val_split)

    # optional label remap: the off-the-shelf baseline arm needs ExDark labels in
    # RF-DETR's COCO-91 logit space (A' is native 12-class -> remap stays None).
    remap = None
    if args.label_map == "exdark_coco":
        cats = sorted(
            json.load(open(data_dir / "annotations" / f"instances_{args.train_split}.json"))["categories"],
            key=lambda c: c["id"])
        remap = torch.tensor([EXDARK_TO_COCO91[c["name"]] for c in cats], dtype=torch.long)
        print(f"label-map exdark_coco: contiguous->COCO91 = {remap.tolist()}")

    print(f"filter {spec} | model={'off-the-shelf' if not args.model_checkpoint else args.model_checkpoint}")
    print(f"train={len(train)} val={len(val)} | splits={args.train_split}/{args.val_split} | detector-GT loss (pair-free)")

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
            lbl = labels if remap is None else remap[labels]
            loss = det_loss(model, crit, filt, train_dir, fn, boxes, lbl, args.input_size, dev)
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
            lbl = labels if remap is None else remap[labels]
            vloss = det_loss(model, crit, filt, val_dir, fn, boxes, lbl, args.input_size, dev)
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
