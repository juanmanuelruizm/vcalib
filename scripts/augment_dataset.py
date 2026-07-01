#!/usr/bin/env python3
"""Apply geometry-only augmentation to the captured A/B dataset.

Stage 2 of data pipeline. Reads raw dataset (scenes_YYYYMMDD/scene_XXX/level_Y.jpg),
splits 80/20 by scene, applies identical geometric transforms to A/B image pairs.

PIPELINE:
  Raw images (data/raw/scenes_YYYYMMDD/)
    → discover_pairs() discovers A/B relationships
    → 80/20 scene split (train: 24 scenes, test: 6 scenes)
    → Train: 1 original + N augmented copies per pair
    → Test: 1 original (unchanged)

AUGMENTATIONS (applied identically to both A and B):
  - Rotation: ±5° (black corners eliminated by zoom)
  - Zoom: 1.2× to 1.35× (covers rotation artifacts)
  - Horizontal flip: 50% probability
  - Crop offset: ±12% for composition variation

OUTPUT:
  data/augmented/train/scene_001_2_aug0/level_1.jpg  ← A original
  data/augmented/train/scene_001_2_aug0/level_2.jpg  ← B original
  data/augmented/train/scene_001_2_aug1/level_1.jpg  ← A augmented
  data/augmented/train/scene_001_2_aug1/level_2.jpg  ← B augmented (same transforms)
  ...
  data/augmented/test/scene_007_2/level_1.jpg        ← A from held-out scene
  data/augmented/test/scene_007_2/level_2.jpg        ← B from held-out scene

KEY: A and B always receive identical transforms → illumination relationship preserved

Usage:
    uv run python scripts/augment_dataset.py \\
        --raw  data/raw \\
        --out  data/augmented \\
        --n-aug 5 \\
        --seed  42
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True  # some raw captures are truncated to 256KB

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.data_pairs import ImagePair, discover_pairs


# ---------------------------------------------------------------------------
# Augmentation primitives
# ---------------------------------------------------------------------------

@dataclass
class AugParams:
    hflip: bool
    angle: float    # degrees, clockwise positive
    scale: float    # >1.0: zoom-in factor (1/scale of image area is kept)
    crop_dx: float  # horizontal crop-center offset as fraction of crop_width  [-0.15, 0.15]
    crop_dy: float  # vertical   crop-center offset as fraction of crop_height [-0.15, 0.15]


def _sample_aug(
    rng: random.Random,
    max_angle: float = 5.0,
    min_scale: float = 1.20,
    max_scale: float = 1.35,
    max_offset: float = 0.12,
) -> AugParams:
    return AugParams(
        hflip=rng.random() < 0.5,
        angle=rng.uniform(-max_angle, max_angle),
        scale=rng.uniform(min_scale, max_scale),
        crop_dx=rng.uniform(-max_offset, max_offset),
        crop_dy=rng.uniform(-max_offset, max_offset),
    )


def _apply_aug(img: Image.Image, params: AugParams) -> Image.Image:
    """Apply AugParams to a PIL Image, returning a new image of the same size."""
    w, h = img.size

    if params.hflip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    if abs(params.angle) > 0.01:
        # Rotate with black fill; the following zoom-crop eliminates those corners.
        # min_scale=1.2 is calibrated to cover ±5° rotation on ≤2:1 aspect ratios.
        img = img.rotate(-params.angle, resample=Image.BICUBIC, expand=False, fillcolor=(0, 0, 0))

    # Zoom-in: take a (crop_w × crop_h) sub-region, then resize to original (w × h).
    crop_w = max(1, int(w / params.scale))
    crop_h = max(1, int(h / params.scale))

    # Center of the crop, shifted by (crop_dx, crop_dy) — clamped to stay inside.
    cx = w / 2 + params.crop_dx * crop_w
    cy = h / 2 + params.crop_dy * crop_h
    left = int(max(0, min(w - crop_w, cx - crop_w / 2)))
    top  = int(max(0, min(h - crop_h, cy - crop_h / 2)))

    img = img.crop((left, top, left + crop_w, top + crop_h))
    img = img.resize((w, h), Image.BICUBIC)
    return img


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _save_pair(
    a: Image.Image,
    b: Image.Image,
    b_level: int,
    scene_dir: Path,
    jpeg_quality: int = 95,
) -> None:
    scene_dir.mkdir(parents=True, exist_ok=True)
    a.save(scene_dir / "level_1.jpg", quality=jpeg_quality, subsampling=0)
    b.save(scene_dir / f"level_{b_level}.jpg", quality=jpeg_quality, subsampling=0)


def _open_pair(pair: ImagePair) -> tuple[Image.Image, Image.Image]:
    return Image.open(pair.a_path).convert("RGB"), Image.open(pair.b_path).convert("RGB")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def augment_dataset(
    raw: Path,
    out: Path,
    n_aug: int,
    val_split: float,
    seed: int,
    jpeg_quality: int,
    dry_run: bool,
) -> None:
    dataset = discover_pairs(raw, val_split=val_split, seed=seed)

    print(f"Dataset : {dataset}")
    print(f"  train pairs : {len(dataset.train_pairs)}")
    print(f"  test  pairs : {len(dataset.val_pairs)}")
    print(f"  n_aug       : {n_aug}  (total train = {len(dataset.train_pairs) * (n_aug + 1)})")
    print(f"  output root : {out}")
    print()

    if dry_run:
        print("[dry-run] No files written.")
        return

    rng = random.Random(seed)
    train_out = out / "train"
    test_out  = out / "test"

    # --- train: original (aug0) + n_aug augmented copies ---
    for pair in dataset.train_pairs:
        a_img, b_img = _open_pair(pair)
        base = f"{pair.scene_id}_{pair.level}"

        # aug0: original, unmodified
        _save_pair(a_img, b_img, pair.level, train_out / f"{base}_aug0", jpeg_quality)

        for k in range(1, n_aug + 1):
            params = _sample_aug(rng)
            _save_pair(
                _apply_aug(a_img.copy(), params),
                _apply_aug(b_img.copy(), params),
                pair.level,
                train_out / f"{base}_aug{k}",
                jpeg_quality,
            )

    # --- test: originals only, no augmentation ---
    for pair in dataset.val_pairs:
        a_img, b_img = _open_pair(pair)
        base = f"{pair.scene_id}_{pair.level}"
        _save_pair(a_img, b_img, pair.level, test_out / base, jpeg_quality)

    n_train_written = len(dataset.train_pairs) * (n_aug + 1)
    n_test_written  = len(dataset.val_pairs)

    print(f"Done.")
    print(f"  train/ : {n_train_written} pairs written to {train_out}")
    print(f"  test/  : {n_test_written}  pairs written to {test_out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Geometry-only augmentation for A/B illumination dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--raw",      required=True,        help="Raw dataset root (data/raw)")
    p.add_argument("--out",      required=True,        help="Output directory  (data/augmented)")
    p.add_argument("--n-aug",    type=int,   default=5, help="Augmented copies per train pair (excl. original)")
    p.add_argument("--val-split",type=float, default=0.2, help="Fraction of scenes held out for test")
    p.add_argument("--seed",     type=int,   default=42,  help="RNG seed for split and augmentation")
    p.add_argument("--quality",  type=int,   default=95,  help="JPEG output quality (1-95)")
    p.add_argument("--dry-run",  action="store_true",     help="Print stats without writing files")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    augment_dataset(
        raw=Path(args.raw),
        out=Path(args.out),
        n_aug=args.n_aug,
        val_split=args.val_split,
        seed=args.seed,
        jpeg_quality=args.quality,
        dry_run=args.dry_run,
    )
