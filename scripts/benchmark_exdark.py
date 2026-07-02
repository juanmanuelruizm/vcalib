#!/usr/bin/env python3
"""Unpaired multiclass detection eval for ExDark: absolute AP of B vs filter(B).

ExDark has no reference (bright) frame paired to a given dark image, so there is
no A ceiling per image. We therefore report the **absolute** COCO AP of a frozen
detector on the raw dark images (B) and on the filter-corrected dark images
(filter(B)), plus the gain filter(B) - B. Scoring is standard multiclass COCOeval
against the split's GT.

Two detector arms (mirrors the trainer's --label-map):
  * off-the-shelf COCO RF-DETR  -> --label-map exdark_coco: predictions are read
    from the 12 COCO-91 logit columns that correspond to ExDark's classes.
  * fine-tuned 12-class A'       -> --label-map none: native columns 0..nc-1.

Usage:
  uv run python scripts/benchmark_exdark.py --data data/coco/exdark --split dark_test \
      --checkpoint results/experiments/runs/exdark_pairfree_offtheshelf/best.pt \
      --label-map exdark_coco --device cuda \
      --out results/ft_bench/exdark_offtheshelf.csv
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from PIL import ImageFile
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.filters import build_filter
from src.utils.activations import load_model, normalize, to_unit_rgb

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Mirror of train_filter_detloss.py::EXDARK_TO_COCO91 (ExDark class name ->
# RF-DETR raw logit column in COCO-91 id space). COCO category ids are fixed, so
# this table is stable; keep the two in sync.
EXDARK_TO_COCO91 = {
    "Bicycle": 2, "Boat": 9, "Bottle": 44, "Bus": 6, "Car": 3, "Cat": 17,
    "Chair": 62, "Cup": 47, "Dog": 18, "Motorbike": 4, "People": 1, "Table": 67,
}


def cxcywh_to_xywh_pixels(boxes: torch.Tensor, W: int, H: int):
    cx, cy, w, h = boxes.unbind(-1)
    x0 = ((cx - w / 2) * W).clamp(0, W)
    y0 = ((cy - h / 2) * H).clamp(0, H)
    x1 = ((cx + w / 2) * W).clamp(0, W)
    y1 = ((cy + h / 2) * H).clamp(0, H)
    return torch.stack([x0, y0, x1 - x0, y1 - y0], dim=-1)


@torch.no_grad()
def detect(model, unit, W, H, image_id, cols, cat_ids, max_det, device):
    """DETR-style top-k over (query x class) detections for one frame.

    ``cols`` selects the logit columns to score (the 12 ExDark-relevant ones);
    ``cat_ids`` gives the parallel COCO category_id to emit for each column.
    """
    normed = normalize(unit).unsqueeze(0).to(device)
    out = model.model(normed)
    logits = out["pred_logits"][0]          # (Q, C)
    boxes = out["pred_boxes"][0]            # (Q, 4) cxcywh normalized
    sub = logits[:, cols].sigmoid()        # (Q, K)
    K = sub.shape[1]
    flat = sub.reshape(-1)                  # (Q*K,)
    k = min(max_det, flat.shape[0])
    topv, topi = flat.topk(k)
    q = (topi // K)
    c = (topi % K)
    xywh = cxcywh_to_xywh_pixels(boxes[q].float().cpu(), W, H)
    res = []
    for i in range(k):
        b = [round(float(v), 2) for v in xywh[i]]
        if b[2] <= 1 or b[3] <= 1:
            continue
        res.append({"image_id": image_id, "category_id": int(cat_ids[int(c[i])]),
                    "bbox": b, "score": float(topv[i])})
    return res


def evaluate(gt: COCO, preds, img_ids, cat_ids):
    if not preds:
        return None
    dt = gt.loadRes(preds)
    ev = COCOeval(gt, dt, "bbox")
    ev.params.imgIds = list(img_ids)
    ev.params.catIds = list(cat_ids)
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return ev.stats  # [AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100, ...]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="COCO dataset dir (exdark_to_coco.py)")
    ap.add_argument("--split", default="dark_test", help="split to evaluate")
    ap.add_argument("--checkpoint", required=True, help="filter checkpoint (best.pt)")
    ap.add_argument("--model-checkpoint", default=None,
                    help="fine-tuned detector; default = off-the-shelf COCO nano")
    ap.add_argument("--label-map", choices=["none", "exdark_coco"], default="none",
                    help="'exdark_coco' for the off-the-shelf arm; 'none' for native A'")
    ap.add_argument("--filter-type", default="spatial_tone_curve")
    ap.add_argument("--P", type=int, default=16)
    ap.add_argument("--grid-size", type=int, default=5)
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--input-size", type=int, default=384)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/ft_bench/exdark.csv")
    args = ap.parse_args()

    data_dir = Path(args.data)
    img_dir = data_dir / "images" / args.split
    gt_path = data_dir / "annotations" / f"instances_{args.split}.json"
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(gt_path))
    cats = sorted(gt.dataset["categories"], key=lambda c: c["id"])
    cat_ids = [c["id"] for c in cats]
    if args.label_map == "exdark_coco":
        cols = [EXDARK_TO_COCO91[c["name"]] for c in cats]
    else:
        cols = list(range(len(cats)))
    dims = {im["id"]: (im["width"], im["height"]) for im in gt.dataset["images"]}

    model = load_model(size="n", device=args.device, model_path=args.model_checkpoint)
    spec = {"type": args.filter_type, "P": args.P, "grid_size": args.grid_size}
    filt = build_filter(spec).to(args.device).eval()
    filt.load_state_dict(torch.load(args.checkpoint, map_location=args.device, weights_only=False))
    print(f"model={'off-the-shelf' if not args.model_checkpoint else args.model_checkpoint} | "
          f"label-map={args.label_map} | cols={cols}")
    print(f"filter {spec} <- {args.checkpoint}")
    print(f"split={args.split} | {len(gt.dataset['images'])} imgs, {len(cat_ids)} classes\n")

    preds = {"B": [], "filterB": []}
    ids = []
    for im in gt.dataset["images"]:
        image_id, fn = im["id"], im["file_name"]
        W, H = dims[image_id]
        b_unit = to_unit_rgb(img_dir / fn, args.input_size)
        with torch.no_grad():
            fb_unit = filt(b_unit.unsqueeze(0).to(args.device))[0].clamp(0, 1).cpu()
        preds["B"] += detect(model, b_unit, W, H, image_id, cols, cat_ids, args.max_det, args.device)
        preds["filterB"] += detect(model, fb_unit, W, H, image_id, cols, cat_ids, args.max_det, args.device)
        ids.append(image_id)

    stats = {k: evaluate(gt, preds[k], ids, cat_ids) for k in preds}

    def row(k):
        s = stats[k]
        return dict(arm=k, AP=s[0], AP50=s[1], AP75=s[2], AR100=s[8]) if s is not None \
            else dict(arm=k, AP=0.0, AP50=0.0, AP75=0.0, AR100=0.0)

    B, F = row("B"), row("filterB")
    print(f"{'arm':10} {'AP@[.5:.95]':>12} {'AP50':>8} {'AP75':>8} {'AR100':>8}")
    for r in (B, F):
        print(f"{r['arm']:10} {r['AP']:12.4f} {r['AP50']:8.4f} {r['AP75']:8.4f} {r['AR100']:8.4f}")
    gain = F["AP"] - B["AP"]
    print(f"\nfilter(B) - B (absolute AP gain): {gain:+.4f}")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "AP", "AP50", "AP75", "AR100", "split", "label_map", "checkpoint"])
        for r in (B, F):
            w.writerow([r["arm"], f"{r['AP']:.4f}", f"{r['AP50']:.4f}", f"{r['AP75']:.4f}",
                        f"{r['AR100']:.4f}", args.split, args.label_map, args.checkpoint])
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
