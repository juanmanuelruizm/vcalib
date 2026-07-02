#!/usr/bin/env python3
"""Offline, box-preserving illumination augmentation for a COCO dataset.

The RF-DETR trainer does not expose tunable photometric augmentation
(``hsv_prob``/``mosaic``/... are in ``UNSUPPORTED_TRAIN_PARAMS``), so to teach the
detector illumination robustness we pre-render jittered copies on disk. All
transforms are **photometric only** (brightness / contrast / gamma / colour
temperature), so the SAM3/GT boxes are unchanged and copied verbatim.

Takes a COCO dataset built by ``scripts/make_detection_coco.py`` and writes a new
dataset where **train** = original + ``--n-aug`` jittered copies per image, while
**val/test are copied unchanged** (never augment the evaluation splits).

Presets:
  moderate   — mild jitter, roughly within the I1->I2 range
  aggressive — wide jitter incl. darkening, spanning and exceeding I1->I3

Usage:
  uv run python scripts/augment_illumination.py \
      --data data/coco/cooktop_ref --preset aggressive --n-aug 5 \
      --out data/coco/cooktop_ref_augA
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

PRESETS = {
    #            brightness     contrast      gamma        colour-temp (+-)
    "moderate":   dict(bright=(0.80, 1.20), contrast=(0.85, 1.15), gamma=(0.80, 1.25), temp=0.08),
    "aggressive": dict(bright=(0.45, 1.55), contrast=(0.65, 1.40), gamma=(0.55, 1.90), temp=0.18),
}


def jitter(img: Image.Image, rng: random.Random, p: dict) -> Image.Image:
    x = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    x = x * rng.uniform(*p["bright"])                       # brightness
    c = rng.uniform(*p["contrast"])
    x = (x - 0.5) * c + 0.5                                 # contrast about mid-grey
    x = np.clip(x, 0.0, 1.0) ** rng.uniform(*p["gamma"])    # gamma
    t = rng.uniform(-p["temp"], p["temp"])                  # colour temperature
    x[..., 0] *= (1.0 + t)                                  # warm R up / cool R down
    x[..., 2] *= (1.0 - t)                                  # B opposite
    return Image.fromarray((np.clip(x, 0.0, 1.0) * 255).astype(np.uint8))


def _copy_split_symlinked(src_dir: Path, dst_dir: Path, images: list, raw: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for im in images:
        link = dst_dir / im["file_name"]
        if link.exists() or link.is_symlink():
            continue
        target = (raw / im["file_name"]).resolve()
        link.symlink_to(os.path.relpath(target, start=dst_dir.resolve()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="input COCO dataset dir")
    ap.add_argument("--out", required=True, help="output augmented COCO dataset dir")
    ap.add_argument("--preset", choices=list(PRESETS), default="aggressive")
    ap.add_argument("--n-aug", type=int, default=5, help="jittered copies per train image")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    p = PRESETS[args.preset]
    rng = random.Random(args.seed)
    src, out, raw = Path(args.data), Path(args.out), Path(args.raw)
    (out / "annotations").mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        coco = json.load(open(src / "annotations" / f"instances_{split}.json"))
        anns = {}
        for a in coco["annotations"]:
            anns.setdefault(a["image_id"], []).append(a)

        if split != "train":
            _copy_split_symlinked(src / "images" / split, out / "images" / split,
                                  coco["images"], raw)
            json.dump(coco, open(out / "annotations" / f"instances_{split}.json", "w"))
            print(f"  {split:5}: {len(coco['images'])} imgs (copied, no aug)")
            continue

        img_dir = out / "images" / "train"
        img_dir.mkdir(parents=True, exist_ok=True)
        out_imgs, out_anns = [], []
        next_img, next_ann = 1, 1
        for im in coco["images"]:
            src_img = Image.open(raw / im["file_name"]).convert("RGB")
            for k in range(args.n_aug + 1):                    # copy 0 = original
                stem = Path(im["file_name"]).stem
                fn = f"{stem}.jpg" if k == 0 else f"{stem}_aug{k}.jpg"
                pic = src_img if k == 0 else jitter(src_img, rng, p)
                pic.save(img_dir / fn, quality=92)
                out_imgs.append({"id": next_img, "file_name": fn,
                                 "width": im["width"], "height": im["height"]})
                for a in anns.get(im["id"], []):
                    out_anns.append({**a, "id": next_ann, "image_id": next_img})
                    next_ann += 1
                next_img += 1
        json.dump({"images": out_imgs, "annotations": out_anns,
                   "categories": coco["categories"]},
                  open(out / "annotations" / "instances_train.json", "w"))
        print(f"  train: {len(out_imgs)} imgs ({args.n_aug}x aug), {len(out_anns)} anns")

    shutil.copy(src / "data.yaml", out / "data.yaml")
    # repoint path to the new root
    txt = (out / "data.yaml").read_text().splitlines()
    txt = [f"path: {out.resolve()}" if l.startswith("path:") else l for l in txt]
    (out / "data.yaml").write_text("\n".join(txt) + "\n")
    print(f"preset={args.preset} n_aug={args.n_aug} -> {out}")


if __name__ == "__main__":
    main()
