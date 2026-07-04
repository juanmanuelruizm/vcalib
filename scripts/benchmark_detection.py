"""Phase 3: real detection mAP recovery against SAM3 pseudo-ground-truth.

Measures whether the calibration filter recovers RF-DETR *detection* quality under
an illumination shift, using class-agnostic box AP against the SAM3-labeled GT
(``data/labels/instances_sam3.json``) instead of the model's own predictions.

For each scene we run frozen RF-DETR three ways and score each against the SAM3 GT
of the matching frame:
  A        : reference frame I1              (upper bound)
  B        : shifted frame I{level}          (degraded)
  filter(B): filter-corrected shifted frame  (recovered)

Recovery = (AP_filterB - AP_B) / (AP_A - AP_B): fraction of the A->B detection gap
the filter closes. Class-agnostic (single category "object") because the captured
objects are largely outside the COCO label space.

Usage:
  uv run python scripts/benchmark_detection.py --level 2                 # test scenes
  uv run python scripts/benchmark_detection.py --level 2 --scenes all    # all 30
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
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

TEST_SCENES = [7, 11, 15, 20, 23, 27]
DEFAULT_CKPT = "results/experiments/runs/real_stc_P16_g5_lv2/best.pt"
DEFAULT_SPEC = {"type": "spatial_tone_curve", "P": 16, "grid_size": 5}


def cxcywh_to_xywh_pixels(boxes: torch.Tensor, W: int, H: int):
    cx, cy, w, h = boxes.unbind(-1)
    x0 = ((cx - w / 2) * W).clamp(0, W)
    y0 = ((cy - h / 2) * H).clamp(0, H)
    x1 = ((cx + w / 2) * W).clamp(0, W)
    y1 = ((cy + h / 2) * H).clamp(0, H)
    return torch.stack([x0, y0, x1 - x0, y1 - y0], dim=-1)


def xyxy_pixels_to_xywh(boxes: torch.Tensor, W: int, H: int, imgsz: int):
    """Rescale YOLOv9's decoded xyxy boxes (pixel scale of the square ``imgsz`` input)
    to the original frame's (W, H) and convert to COCO xywh."""
    sx, sy = W / imgsz, H / imgsz
    x0, y0, x1, y1 = boxes.unbind(-1)
    x0, x1 = (x0 * sx).clamp(0, W), (x1 * sx).clamp(0, W)
    y0, y1 = (y0 * sy).clamp(0, H), (y1 * sy).clamp(0, H)
    return torch.stack([x0, y0, x1 - x0, y1 - y0], dim=-1)


@torch.no_grad()
def detect(libre, unit: torch.Tensor, W: int, H: int, image_id: int, max_det: int, device, family: str = "rfdetr"):
    """Class-agnostic detections as COCO results dicts for one frame.

    ``family="yolo9"`` uses YOLOv9's dense per-anchor ``DDetect`` output
    (``{"predictions": (1, 4+nc, N)}``, decoded xyxy boxes + per-class sigmoid
    scores) instead of RF-DETR/DEIM/D-FINE/RT-DETRv4's query-based
    ``{pred_logits, pred_boxes}``. See ``detection_output_loss_yolo9`` in
    ``src/calibration.py`` for the analogous pair-free-loss adapter.
    """
    normed = normalize(unit).unsqueeze(0).to(device)
    out = libre.model(normed)

    if family == "yolo9":
        pred = out["predictions"][0]        # (4+nc, N)
        boxes_xyxy, cls = pred[:4].T, pred[4:]  # (N, 4), (nc, N)
        scores = cls.amax(dim=0)                # per-anchor best-class score (already sigmoid)
        k = min(max_det, scores.shape[0])
        topv, topi = scores.topk(k)
        imgsz = unit.shape[-1]
        xywh = xyxy_pixels_to_xywh(boxes_xyxy[topi].float().cpu(), W, H, imgsz)
    else:
        logits = out["pred_logits"][0]          # (Q, C)
        boxes = out["pred_boxes"][0]            # (Q, 4) cxcywh normalized
        scores = logits.sigmoid().amax(dim=-1)  # per-query objectness
        k = min(max_det, scores.shape[0])
        topv, topi = scores.topk(k)
        xywh = cxcywh_to_xywh_pixels(boxes[topi].float().cpu(), W, H)

    res = []
    for i in range(k):
        b = [round(float(v), 2) for v in xywh[i]]
        if b[2] <= 1 or b[3] <= 1:
            continue
        res.append({"image_id": image_id, "category_id": 1, "bbox": b, "score": float(topv[i])})
    return res


def evaluate(gt: COCO, preds, img_ids):
    if not preds:
        return None
    dt = gt.loadRes(preds)
    ev = COCOeval(gt, dt, "bbox")
    ev.params.imgIds = list(img_ids)
    ev.params.catIds = [1]
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return ev.stats  # [AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100, ...]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="data/labels/instances_sam3.json")
    ap.add_argument("--level", type=int, default=2, choices=[2, 3])
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--filter-config", default=None, help="experiment YAML to read filter spec from")
    ap.add_argument("--scenes", default="test", help="'test', 'all', or comma list e.g. 7,11,20")
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--input-size", type=int, default=384)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model-checkpoint", default=None,
                    help="fine-tuned checkpoint (weights/best.pt); default = off-the-shelf pretrained")
    ap.add_argument("--family", default="rfdetr",
                    help="model family key (rfdetr, rtdetrv4, yolo9, fomo); see src/utils/activations.py")
    ap.add_argument("--size", default="n", help="model size code (family-specific)")
    ap.add_argument("--out", default="results/detection_benchmark.csv")
    args = ap.parse_args()

    spec = DEFAULT_SPEC
    if args.filter_config:
        import yaml
        spec = yaml.safe_load(open(args.filter_config))["filter"]

    gt = COCO(args.gt)
    name2id = {im["file_name"]: im["id"] for im in gt.dataset["images"]}
    dims = {im["id"]: (im["width"], im["height"]) for im in gt.dataset["images"]}

    if args.scenes == "test":
        scenes = TEST_SCENES
    elif args.scenes == "all":
        scenes = sorted({int(im["scene"].split("_")[-1]) for im in gt.dataset["images"]})
    else:
        scenes = [int(s) for s in args.scenes.split(",")]

    raw = Path(args.gt).parent.parent / "raw"
    libre = load_model(size=args.size, device=args.device, model_path=args.model_checkpoint, family=args.family)
    filt = build_filter(spec).to(args.device).eval()
    filt.load_state_dict(torch.load(args.checkpoint, map_location=args.device, weights_only=False))
    print(f"filter {spec} <- {args.checkpoint}")
    print(f"level={args.level} | scenes={args.scenes} ({len(scenes)}) | GT={args.gt}\n")

    preds = {"A": [], "B": [], "filterB": []}
    ids = {"A": [], "B": [], "filterB": []}
    missing = []
    for sc in scenes:
        a_name, b_name = f"IMG_{sc}_I1.jpg", f"IMG_{sc}_I{args.level}.jpg"
        if a_name not in name2id or b_name not in name2id:
            missing.append(sc); continue
        a_id, b_id = name2id[a_name], name2id[b_name]
        a_unit = to_unit_rgb(raw / a_name, args.input_size)
        b_unit = to_unit_rgb(raw / b_name, args.input_size)
        with torch.no_grad():
            fb_unit = filt(b_unit.unsqueeze(0).to(args.device))[0].clamp(0, 1).cpu()
        preds["A"] += detect(libre, a_unit, *dims[a_id], a_id, args.max_det, args.device, args.family)
        preds["B"] += detect(libre, b_unit, *dims[b_id], b_id, args.max_det, args.device, args.family)
        preds["filterB"] += detect(libre, fb_unit, *dims[b_id], b_id, args.max_det, args.device, args.family)
        ids["A"].append(a_id); ids["B"].append(b_id); ids["filterB"].append(b_id)

    stats = {k: evaluate(gt, preds[k], ids[k]) for k in preds}
    if missing:
        print(f"skipped scenes without GT for level {args.level}: {missing}\n")

    def row(k):
        s = stats[k]
        return dict(arm=k, AP=s[0], AP50=s[1], AP75=s[2], AR100=s[8]) if s is not None else dict(arm=k, AP=0, AP50=0, AP75=0, AR100=0)

    A, B, F = row("A"), row("B"), row("filterB")
    print(f"{'arm':10} {'AP@[.5:.95]':>12} {'AP50':>8} {'AP75':>8} {'AR100':>8}")
    for r in (A, B, F):
        print(f"{r['arm']:10} {r['AP']:12.4f} {r['AP50']:8.4f} {r['AP75']:8.4f} {r['AR100']:8.4f}")

    gap = A["AP"] - B["AP"]
    dfb = F["AP"] - B["AP"]
    rec = (dfb / gap) if gap > 1e-9 else float("nan")
    print(f"\nA->B detection gap (AP): {gap:+.4f}")
    print(f"filter(B) - B (AP):      {dfb:+.4f}")
    print(f"recovery (fraction of gap closed): {rec*100:.1f}%" if gap > 1e-9 else "recovery: n/a (no A>B gap)")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "AP", "AP50", "AP75", "AR100", "level", "scenes", "checkpoint"])
        for r in (A, B, F):
            w.writerow([r["arm"], f"{r['AP']:.4f}", f"{r['AP50']:.4f}", f"{r['AP75']:.4f}",
                        f"{r['AR100']:.4f}", args.level, args.scenes, args.checkpoint])
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
