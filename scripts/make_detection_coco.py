#!/usr/bin/env python3
"""Build a single-class COCO dataset from the SAM3 labels for RF-DETR fine-tuning.

Emits the layout LibreYOLO's ``load_data_config`` expects (a ``data.yaml`` with a
``path`` root, ``train/val/test`` image dirs, and a native-COCO ``annotations:``
block), with a **strict scene split** so held-out test scenes never leak into
training:

  test  = the 6 canonical held-out scenes (7,11,15,20,23,27), final eval only
  val   = a few scenes carved from the train pool, for early stopping
  train = the remaining scenes

Levels select which illumination frames go in: ``--levels 1`` for the reference
(nominal) fine-tune, ``--levels 2,3`` for the shifted frames used later to train
and evaluate the pair-free filter. Images are relative symlinks to ``data/raw``.

Usage:
  uv run python scripts/make_detection_coco.py --levels 1 --out data/coco/cooktop_ref
  uv run python scripts/make_detection_coco.py --levels 2,3 --out data/coco/cooktop_shift
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

# discover_pairs(seed=42, val_split=0.2) holds out exactly these scenes as the test set.
TEST_SCENES = [7, 11, 15, 20, 23, 27]


def scene_num(img: dict) -> int:
    return int(img["scene"].split("_")[-1])


def _coco(images: list, anns_by_img: dict) -> dict:
    """Re-index a subset of images + their annotations into a standalone COCO dict."""
    out_images, out_anns = [], []
    next_img_id, next_ann_id = 1, 1
    for im in images:
        new_id = next_img_id
        next_img_id += 1
        out_images.append({"id": new_id, "file_name": im["file_name"],
                           "width": im["width"], "height": im["height"]})
        for a in anns_by_img.get(im["id"], []):
            out_anns.append({**a, "id": next_ann_id, "image_id": new_id})
            next_ann_id += 1
    return {
        "images": out_images,
        "annotations": out_anns,
        "categories": [{"id": 1, "name": "object"}],
    }


def _link_images(images: list, raw: Path, split_dir: Path) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    for im in images:
        link = split_dir / im["file_name"]
        if link.is_symlink() or link.exists():
            continue
        target = (raw / im["file_name"]).resolve()
        link.symlink_to(os.path.relpath(target, start=split_dir.resolve()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/labels/instances_sam3.json")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--levels", default="1", help="comma list of illumination levels, e.g. '1' or '2,3'")
    ap.add_argument("--out", required=True, help="output dataset root, e.g. data/coco/cooktop_ref")
    ap.add_argument("--val-scenes", type=int, default=4, help="scenes carved from train for early stopping")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    levels = {s.strip() for s in args.levels.split(",") if s.strip()}
    gt = json.load(open(args.labels))
    anns_by_img = defaultdict(list)
    for a in gt["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    imgs = [im for im in gt["images"] if im["level"] in levels]
    by_split = {"test": [], "val": [], "train": []}

    train_scene_ids = sorted({scene_num(im) for im in imgs if scene_num(im) not in TEST_SCENES})
    rng = random.Random(args.seed)
    rng.shuffle(train_scene_ids)
    val_scene_ids = set(train_scene_ids[: args.val_scenes])

    for im in imgs:
        s = scene_num(im)
        if s in TEST_SCENES:
            by_split["test"].append(im)
        elif s in val_scene_ids:
            by_split["val"].append(im)
        else:
            by_split["train"].append(im)

    out = Path(args.out)
    raw = Path(args.raw)
    (out / "annotations").mkdir(parents=True, exist_ok=True)
    for split, images in by_split.items():
        coco = _coco(images, anns_by_img)
        json.dump(coco, open(out / "annotations" / f"instances_{split}.json", "w"))
        _link_images(images, raw, out / "images" / split)
        print(f"  {split:5}: {len(images):3} imgs, {len(coco['annotations']):3} anns")

    data_yaml = out / "data.yaml"
    with open(data_yaml, "w") as f:
        f.write(
            f"path: {out.resolve()}\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n"
            "annotations:\n"
            "  train: annotations/instances_train.json\n"
            "  val: annotations/instances_val.json\n"
            "  test: annotations/instances_test.json\n"
            "names:\n"
            "  0: object\n"
            "nc: 1\n"
        )
    print(f"levels={sorted(levels)} -> wrote {data_yaml}")


if __name__ == "__main__":
    main()
