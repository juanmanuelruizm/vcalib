"""Auto-label the raw capture frames with SAM3 concept segmentation.

Produces class-agnostic (single category "object") COCO-format bbox + mask
ground truth for the foreground object(s) sitting on the cooktop, so RF-DETR
detection recovery can be measured with real mAP instead of the model's own
self-referential predictions.

The object identity changes per scene (moka pot, sneaker, book, jar, ...), so a
single fixed text prompt cannot work. Each scene declares its concept noun(s) in
configs/autolabel/prompts.yaml; the 3 illumination levels reuse the scene list.

Selection per frame:
  1. run SAM3 for every scene prompt (vision features encoded once, reused)
  2. pool instances, drop the static background region (dish rack on the right)
  3. drop glossy-cooktop reflections (mirror instance below a kept object)
  4. NMS to dedupe an object caught by two synonymous prompts
  5. keep all foreground objects (default) or only the most salient (--top1)

Usage:
  uv run python scripts/autolabel_sam3.py                         # all raw frames
  uv run python scripts/autolabel_sam3.py --scenes IMG_10,IMG_20  # subset
  uv run python scripts/autolabel_sam3.py --top1                  # one box/frame
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import cv2
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFile
from pycocotools import mask as mask_utils

ImageFile.LOAD_TRUNCATED_IMAGES = True  # one raw jpg is slightly truncated

REPO = "vil-uob/sam3-litetext-s0"
LEVELS = ("I1", "I2", "I3")
# The dish rack + background plates/glass live on the right edge; the target
# object is always left-of-center to center. Anything centred past this x is
# background clutter, never the subject.
BG_RIGHT_X = 0.75
MIN_AREA_FRAC = 0.0015  # drop specks smaller than this fraction of the frame


def scene_of(stem: str) -> str:
    return stem.split("_I")[0]


def normalize_spec(entry):
    """A scene entry is either a list of text prompts or a dict with
    optional `text` (list) and `boxes` (list of xyxy exemplar boxes)."""
    if isinstance(entry, dict):
        return {"text": list(entry.get("text", [])), "boxes": list(entry.get("boxes", []))}
    if isinstance(entry, str):                  # a bare scalar is a single prompt
        return {"text": [entry], "boxes": []}
    return {"text": list(entry), "boxes": []}


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes (dark screen regions inside a box-prompted object)."""
    mm = mask.astype(np.uint8)
    h, w = mm.shape
    # Pad a 1px background border so the flood seed at the corner is guaranteed
    # to be background even when the object touches an image edge/corner.
    flood = np.zeros((h + 2, w + 2), np.uint8)
    flood[1:-1, 1:-1] = mm
    ff_mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 1)   # fill exterior background from corner
    holes = (flood == 0)[1:-1, 1:-1]            # unreachable interior pixels are holes
    return (mm.astype(bool) | holes)


def load_model(device: str):
    from transformers import AutoModel, AutoProcessor

    model = AutoModel.from_pretrained(REPO, dtype=torch.float32).to(device).eval()
    processor = AutoProcessor.from_pretrained(REPO)
    return model, processor


@torch.no_grad()
def detect(model, processor, image: Image.Image, spec, device, threshold):
    """Return pooled instances [{box(xyxy px), score, mask(bool HxW), noun}].

    `spec` = {"text": [noun, ...], "boxes": [[x0,y0,x1,y1], ...]}. Text prompts
    use concept segmentation; box prompts use SAM3 box exemplars and keep the
    instance whose predicted box best matches the exemplar.
    """
    W, H = image.size
    img_inputs = processor(images=image, return_tensors="pt").to(device)
    vision = None
    if hasattr(model, "get_vision_features"):
        try:
            vision = model.get_vision_features(pixel_values=img_inputs.pixel_values)
        except Exception:
            vision = None

    pooled = []
    for noun in spec["text"]:
        if vision is not None:
            text_inputs = processor(text=noun, return_tensors="pt").to(device)
            outputs = model(vision_embeds=vision, **text_inputs)
        else:
            inp = processor(images=image, text=noun, return_tensors="pt").to(device)
            outputs = model(**inp)
        res = processor.post_process_instance_segmentation(
            outputs, threshold=threshold, mask_threshold=0.5, target_sizes=[[H, W]],
        )[0]
        for i in range(len(res["scores"])):
            pooled.append({"box": [float(v) for v in res["boxes"][i]],
                           "score": float(res["scores"][i]),
                           "mask": res["masks"][i].detach().cpu().numpy().astype(bool),
                           "noun": noun})

    for box in spec["boxes"]:
        inp = processor(images=image, input_boxes=[[box]], input_boxes_labels=[[1]],
                        return_tensors="pt").to(device)
        outputs = model(**inp)
        res = processor.post_process_instance_segmentation(
            outputs, threshold=0.0, mask_threshold=0.5, target_sizes=[[H, W]],
        )[0]
        if len(res["scores"]) == 0:
            continue
        cand = [[float(v) for v in res["boxes"][i]] for i in range(len(res["scores"]))]
        j = int(np.argmax([_iou(box, cb) for cb in cand]))
        if _iou(box, cand[j]) < 0.2:          # exemplar found nothing matching
            continue
        m = fill_holes(res["masks"][j].detach().cpu().numpy().astype(bool))
        pooled.append({"box": cand[j], "score": float(res["scores"][j]),
                       "mask": m, "noun": "[box]"})
    return pooled


def _xoverlap(a, b) -> float:
    lo, hi = max(a[0], b[0]), min(a[2], b[2])
    inter = max(0.0, hi - lo)
    wmin = min(a[2] - a[0], b[2] - b[0]) or 1.0
    return inter / wmin


def _iou(a, b) -> float:
    lo_x, lo_y = max(a[0], b[0]), max(a[1], b[1])
    hi_x, hi_y = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, hi_x - lo_x) * max(0.0, hi_y - lo_y)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter or 1.0
    return inter / union


def select(pooled, W, H, top1: bool):
    """Filter background + reflections, NMS-dedupe, return kept instances."""
    frame_area = W * H
    cand = []
    for c in pooled:
        x0, y0, x1, y1 = c["box"]
        cx = (x0 + x1) / 2 / W
        area = (x1 - x0) * (y1 - y0)
        if cx > BG_RIGHT_X:                 # dish-rack / background clutter
            continue
        if area < MIN_AREA_FRAC * frame_area:
            continue
        cand.append(c)

    cand.sort(key=lambda c: c["score"], reverse=True)
    kept = []
    for c in cand:
        cb = c["box"]
        c_cy = (cb[1] + cb[3]) / 2
        drop = False
        for k in kept:
            kb = k["box"]
            k_cy = (kb[1] + kb[3]) / 2
            if _iou(cb, kb) > 0.6:          # same object via synonym prompt
                drop = True
                break
            # reflection: mostly below a kept object, sharing its x-span
            if _xoverlap(cb, kb) > 0.5 and cb[1] >= k_cy and c_cy > k_cy:
                drop = True
                break
        if not drop:
            kept.append(c)
    if top1 and kept:
        kept = [max(kept, key=lambda c: c["score"])]
    return kept


def to_annotation(mask: np.ndarray, ann_id: int, image_id: int, score: float, noun: str):
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    x, y, w, h = [float(v) for v in mask_utils.toBbox(rle)]
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": 1,
        "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
        "area": float(mask_utils.area(rle)),
        "segmentation": rle,
        "iscrowd": 0,
        "score": round(score, 4),   # provenance: SAM3 confidence (non-standard)
        "prompt": noun,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--config", default="configs/autolabel/prompts.yaml")
    ap.add_argument("--out", default="data/labels/instances_sam3.json")
    ap.add_argument("--qa-dir", default="data/labels/qa")
    ap.add_argument("--scenes", default="", help="comma list e.g. IMG_10,IMG_20")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--top1", action="store_true")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    default_prompts = cfg.get("default_prompts", ["object"])
    threshold = args.threshold if args.threshold is not None else cfg.get("threshold", 0.3)
    scene_prompts = cfg.get("scenes", {})

    scene_filter = set(s.strip() for s in args.scenes.split(",") if s.strip())
    paths = sorted(
        glob.glob(os.path.join(args.raw_dir, "*.jpg")),
        key=lambda p: (int(re.sub(r"\D", "", scene_of(Path(p).stem))),
                       Path(p).stem.split("_I")[-1]),
    )
    if scene_filter:
        paths = [p for p in paths if scene_of(Path(p).stem) in scene_filter]

    print(f"device={args.device} | frames={len(paths)} | threshold={threshold} | top1={args.top1}")
    model, processor = load_model(args.device)
    os.makedirs(args.qa_dir, exist_ok=True)

    images, annotations, rows = [], [], []
    ann_id = 1
    t0 = time.time()
    for image_id, path in enumerate(paths, 1):
        stem = Path(path).stem
        scene = scene_of(stem)
        spec = normalize_spec(scene_prompts.get(scene, default_prompts))
        image = Image.open(path).convert("RGB")
        W, H = image.size
        pooled = detect(model, processor, image, spec, args.device, threshold)
        kept = select(pooled, W, H, args.top1)

        images.append({"id": image_id, "file_name": os.path.basename(path),
                       "width": W, "height": H, "scene": scene,
                       "level": stem.split("_I")[-1]})
        for c in kept:
            annotations.append(to_annotation(c["mask"], ann_id, image_id, c["score"], c["noun"]))
            ann_id += 1

        scores = ",".join(f"{c['score']:.2f}" for c in kept) or "-"
        prompt_desc = "|".join(spec["text"]) + ("|+box" * len(spec["boxes"]))
        rows.append({"file": stem, "n": len(kept), "prompts": prompt_desc.strip("|"), "scores": scores})
        flag = "  <-- CHECK" if (not kept or max((c["score"] for c in kept), default=0) < 0.5) else ""
        print(f"[{image_id:3}/{len(paths)}] {stem:12} n={len(kept)} scores={scores}{flag}")

        if stem.endswith("_I1"):   # one QA overlay per scene
            _save_overlay(image, kept, os.path.join(args.qa_dir, f"{stem}.jpg"))

    coco = {
        "info": {"description": "SAM3 auto-labeled foreground objects (class-agnostic)",
                 "model": REPO, "top1": args.top1, "threshold": threshold},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "object"}],
    }
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    json.dump(coco, open(args.out, "w"))
    with open(os.path.splitext(args.out)[0] + "_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "n", "prompts", "scores"])
        w.writeheader(); w.writerows(rows)

    empties = [r["file"] for r in rows if r["n"] == 0]
    print(f"\nDONE in {time.time()-t0:.0f}s | images={len(images)} annotations={len(annotations)}")
    print(f"  wrote {args.out}")
    print(f"  QA overlays: {args.qa_dir}")
    if empties:
        print(f"  {len(empties)} frames with NO detection: {empties}")


def _save_overlay(image, kept, out_path):
    ov = np.asarray(image).copy()
    palette = [(0, 255, 0), (255, 140, 0), (0, 180, 255), (255, 0, 200)]
    for j, c in enumerate(kept):
        col = np.array(palette[j % len(palette)])
        ov[c["mask"]] = (0.45 * col + 0.55 * ov[c["mask"]]).astype(np.uint8)
    im = Image.fromarray(ov)
    dr = ImageDraw.Draw(im)
    for j, c in enumerate(kept):
        col = tuple(palette[j % len(palette)])
        b = c["box"]
        dr.rectangle(b, outline=col, width=6)
        dr.text((b[0] + 4, max(0, b[1] - 16)), f"{c['noun']} {c['score']:.2f}", fill=col)
    im.save(out_path, quality=85)


if __name__ == "__main__":
    main()
