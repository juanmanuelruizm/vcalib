#!/usr/bin/env python3
"""Convert the ExDark low-light detection dataset to COCO, split by illumination.

ExDark (https://github.com/cs-chan/Exclusively-Dark-Image-Dataset) is a REAL,
human-labelled low-light detection set: 7,363 images, 12 classes, **bounding
boxes** (no masks), across **10 lighting conditions** from very-dark to twilight.
Because it is not paired, we use it exactly for the realistic experiment: fine-tune
the detector on the *brighter* conditions (the reference/nominal domain) and train
+ evaluate the pair-free filter on the *darker* conditions (the shift).

Expected layout of ``--exdark-root`` (the cloned repo):
  Dataset/<Class>/<img>            images, one folder per class
  Groundtruth/<Class>/<img>.txt    bbGt v3: header line, then '<cls> l t w h ...'
  imageclasslist.txt               'Name Class Lighting In/Out Split' (Lighting 1..10)

Lighting index (ExDark): 1 Low 2 Ambient 3 Object 4 Single 5 Weak 6 Strong
7 Screen 8 Window 9 Shadow 10 Twilight.

Emits (single-class 'object' by default, or 12 classes with --multiclass):
  annotations/instances_bright.json                 reference fine-tune set
  annotations/instances_dark_{train,val,test}.json  shift set (official split flag)
  images/<split>/...                                relative symlinks
  data_dark.yaml                                    for the dark (shift) filter/eval

NOTE: untested until ExDark is downloaded; validate on the GPU box.

Usage:
  uv run python scripts/exdark_to_coco.py --exdark-root data/exdark_raw --out data/coco/exdark
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

CLASSES = ["Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat", "Chair", "Cup",
           "Dog", "Motorbike", "People", "Table"]
LIGHTING = ["Low", "Ambient", "Object", "Single", "Weak", "Strong",
            "Screen", "Window", "Shadow", "Twilight"]  # index+1 in imageclasslist
# The relatively brighter conditions we treat as the reference domain.
DEFAULT_BRIGHT = {"Ambient", "Twilight", "Strong", "Object"}


def parse_meta(root: Path) -> dict:
    """image_name -> {'lighting': str, 'split': int(1 train/2 val/3 test)}."""
    meta = {}
    for i, line in enumerate(open(root / "imageclasslist.txt")):
        parts = line.split()
        if i == 0 and not parts[1].isdigit():   # header row
            continue
        if len(parts) < 5:
            continue
        name, _cls, light, _io, split = parts[:5]
        meta[name.lower()] = {"lighting": LIGHTING[int(light) - 1], "split": int(split)}
    return meta


def parse_gt(txt_path: Path) -> list:
    """bbGt v3 -> [(class_name, l, t, w, h)]."""
    out = []
    for line in open(txt_path):
        p = line.split()
        if not p or p[0].startswith("%") or not p[0][0].isalpha():
            continue
        try:
            l, t, w, h = (float(v) for v in p[1:5])
        except (ValueError, IndexError):
            continue
        out.append((p[0], l, t, w, h))
    return out


def main() -> None:
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    ap = argparse.ArgumentParser()
    ap.add_argument("--exdark-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bright", default=",".join(sorted(DEFAULT_BRIGHT)),
                    help="comma list of lighting conditions treated as reference/bright")
    ap.add_argument("--multiclass", action="store_true",
                    help="keep the 12 classes (default: collapse to single 'object')")
    args = ap.parse_args()

    root, out = Path(args.exdark_root), Path(args.out)
    bright = {b.strip() for b in args.bright.split(",")}
    meta = parse_meta(root)
    if args.multiclass:
        cat_id = {c: i + 1 for i, c in enumerate(CLASSES)}
        categories = [{"id": i + 1, "name": c} for i, c in enumerate(CLASSES)]
    else:
        cat_id = {c: 1 for c in CLASSES}
        categories = [{"id": 1, "name": "object"}]

    # group images -> which output split (bright / dark_{train,val,test})
    groups = {"bright": [], "dark_train": [], "dark_val": [], "dark_test": []}
    gt_root = root / "Groundtruth"
    for cls in CLASSES:
        for img_path in sorted((root / "Dataset" / cls).glob("*")):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            m = meta.get(img_path.name.lower())
            if m is None:
                continue
            gt = gt_root / cls / (img_path.name + ".txt")
            if not gt.exists():
                continue
            if m["lighting"] in bright:
                key = "bright"
            else:
                key = {1: "dark_train", 2: "dark_val", 3: "dark_test"}[m["split"]]
            groups[key].append((cls, img_path, gt))

    (out / "annotations").mkdir(parents=True, exist_ok=True)
    for key, items in groups.items():
        images, anns = [], []
        iid, aid = 1, 1
        split_dir = out / "images" / key
        split_dir.mkdir(parents=True, exist_ok=True)
        for cls, img_path, gt in items:
            with Image.open(img_path) as im:
                W, H = im.size
            fn = f"{cls}__{img_path.name}"
            link = split_dir / fn
            if not (link.exists() or link.is_symlink()):
                link.symlink_to(os.path.relpath(img_path.resolve(), start=split_dir.resolve()))
            images.append({"id": iid, "file_name": fn, "width": W, "height": H,
                           "lighting": meta[img_path.name.lower()]["lighting"]})
            for cname, l, t, w, h in parse_gt(gt):
                if w <= 0 or h <= 0:
                    continue
                anns.append({"id": aid, "image_id": iid, "category_id": cat_id.get(cname, 1),
                             "bbox": [l, t, w, h], "area": w * h, "iscrowd": 0})
                aid += 1
            iid += 1
        json.dump({"images": images, "annotations": anns, "categories": categories},
                  open(out / "annotations" / f"instances_{key}.json", "w"))
        print(f"  {key:11}: {len(images):4} imgs, {len(anns):4} anns")

    nc = len(categories)
    (out / "data_dark.yaml").write_text(
        f"path: {out.resolve()}\n"
        "train: images/dark_train\n"
        "val: images/dark_val\n"
        "test: images/dark_test\n"
        "annotations:\n"
        "  train: annotations/instances_dark_train.json\n"
        "  val: annotations/instances_dark_val.json\n"
        "  test: annotations/instances_dark_test.json\n"
        f"names:\n" + "".join(f"  {i}: {c['name']}\n" for i, c in enumerate(categories)) +
        f"nc: {nc}\n"
    )
    print(f"bright(reference)={sorted(bright)} | multiclass={args.multiclass} -> {out}")


if __name__ == "__main__":
    main()
