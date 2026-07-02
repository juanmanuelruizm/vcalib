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
  annotations/instances_bright_{train,val}.json     reference fine-tune set (per-class split)
  annotations/instances_dark_{train,val,test}.json  shift set (official split flag)
  images/<split>/...                                relative symlinks
  data_bright.yaml / data_dark.yaml                 fine-tune A' / filter+eval

NOTE: untested until ExDark is downloaded; validate on the GPU box.

Usage:
  uv run python scripts/exdark_to_coco.py --exdark-root data/exdark_raw --out data/coco/exdark
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

CLASSES = ["Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat", "Chair", "Cup",
           "Dog", "Motorbike", "People", "Table"]
LIGHTING = ["Low", "Ambient", "Object", "Single", "Weak", "Strong",
            "Screen", "Window", "Shadow", "Twilight"]  # index+1 in imageclasslist
# The relatively brighter conditions we treat as the reference domain.
DEFAULT_BRIGHT = {"Ambient", "Twilight", "Strong", "Object"}


def parse_meta(meta_path: Path) -> dict:
    """image_name -> {'lighting': str, 'split': int(1 train/2 val/3 test)}."""
    meta = {}
    for i, line in enumerate(open(meta_path)):
        parts = line.split()
        if i == 0 and (len(parts) < 2 or not parts[1].isdigit()):   # header row
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


def _resolve_dir(root: Path, override, candidates: list, classes: list) -> Path:
    """Pick the images/GT root: an explicit override, else the first candidate
    that actually contains the per-class subfolders. Disambiguates the GitHub
    stub dirs (``Dataset/``, ``Groundtruth/`` from the repo, which hold only
    READMEs) from the real Drive-extracted data (``ExDark/``, ``ExDark_Annno/``)."""
    if override:
        return Path(override)
    for name in candidates:
        d = root / name
        if d.is_dir() and any((d / c).is_dir() for c in classes):
            return d
    raise FileNotFoundError(
        f"none of {candidates} under {root} contains class subfolders "
        f"(e.g. {classes[0]}/); extract the ExDark Drive downloads first "
        f"or pass an explicit --images-root/--gt-root"
    )


def _resolve_meta(root: Path, override, gt_root: Path) -> Path:
    if override:
        return Path(override)
    for cand in (root / "imageclasslist.txt",
                 root / "Groundtruth" / "imageclasslist.txt",
                 gt_root / "imageclasslist.txt"):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"imageclasslist.txt not found under {root}")


def main() -> None:
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    ap = argparse.ArgumentParser()
    ap.add_argument("--exdark-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--images-root", default=None,
                    help="override images dir (default: auto-detect Dataset/ or ExDark/)")
    ap.add_argument("--gt-root", default=None,
                    help="override bbGt dir (default: auto-detect Groundtruth/ or ExDark_Annno/)")
    ap.add_argument("--meta", default=None,
                    help="override imageclasslist.txt path (default: auto-detect)")
    ap.add_argument("--bright", default=",".join(sorted(DEFAULT_BRIGHT)),
                    help="comma list of lighting conditions treated as reference/bright")
    ap.add_argument("--multiclass", action="store_true",
                    help="keep the 12 classes (default: collapse to single 'object')")
    ap.add_argument("--bright-val-frac", type=float, default=0.15,
                    help="fraction of the bright (reference) set held out per class "
                         "for fine-tune early-stopping")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root, out = Path(args.exdark_root), Path(args.out)
    bright = {b.strip() for b in args.bright.split(",")}
    images_root = _resolve_dir(root, args.images_root, ["Dataset", "ExDark"], CLASSES)
    gt_root = _resolve_dir(root, args.gt_root, ["Groundtruth", "ExDark_Annno"], CLASSES)
    meta_path = _resolve_meta(root, args.meta, gt_root)
    print(f"images_root={images_root} | gt_root={gt_root} | meta={meta_path}")
    meta = parse_meta(meta_path)
    if args.multiclass:
        cat_id = {c: i + 1 for i, c in enumerate(CLASSES)}
        categories = [{"id": i + 1, "name": c} for i, c in enumerate(CLASSES)]
    else:
        cat_id = {c: 1 for c in CLASSES}
        categories = [{"id": 1, "name": "object"}]

    # group images -> output split. The bright (reference) set is split per-class
    # into train/val for the reference fine-tune; the dark (shift) set uses
    # ExDark's official train/val/test flag from imageclasslist.txt.
    groups = {"bright_train": [], "bright_val": [],
              "dark_train": [], "dark_val": [], "dark_test": []}
    bright_items = []
    for cls in CLASSES:
        for img_path in sorted((images_root / cls).glob("*")):
            if img_path.name.startswith("._"):
                continue
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            m = meta.get(img_path.name.lower())
            if m is None:
                continue
            gt = gt_root / cls / (img_path.name + ".txt")
            if not gt.exists():
                continue
            if m["lighting"] in bright:
                bright_items.append((cls, img_path, gt))
            else:
                key = {1: "dark_train", 2: "dark_val", 3: "dark_test"}[m["split"]]
                groups[key].append((cls, img_path, gt))

    # deterministic per-class train/val split of the bright reference set
    rng = random.Random(args.seed)
    by_cls = defaultdict(list)
    for item in bright_items:
        by_cls[item[0]].append(item)
    for items in by_cls.values():
        rng.shuffle(items)
        n_val = round(len(items) * args.bright_val_frac) if len(items) > 1 else 0
        groups["bright_val"].extend(items[:n_val])
        groups["bright_train"].extend(items[n_val:])

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
    names_block = ("names:\n"
                   + "".join(f"  {i}: {c['name']}\n" for i, c in enumerate(categories))
                   + f"nc: {nc}\n")

    def _yaml(prefix: str) -> str:
        # bright has no separate test split -> point test at its val.
        test = "dark_test" if prefix == "dark" else "bright_val"
        return (
            f"path: {out.resolve()}\n"
            f"train: images/{prefix}_train\n"
            f"val: images/{prefix}_val\n"
            f"test: images/{test}\n"
            "annotations:\n"
            f"  train: annotations/instances_{prefix}_train.json\n"
            f"  val: annotations/instances_{prefix}_val.json\n"
            f"  test: annotations/instances_{test}.json\n"
            + names_block
        )

    (out / "data_bright.yaml").write_text(_yaml("bright"))
    (out / "data_dark.yaml").write_text(_yaml("dark"))
    print(f"bright(reference)={sorted(bright)} | multiclass={args.multiclass} -> {out}")
    print("  wrote data_bright.yaml (fine-tune A') + data_dark.yaml (filter/eval)")


if __name__ == "__main__":
    main()
