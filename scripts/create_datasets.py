#!/usr/bin/env python3
"""Create separate datasets for training level_2 and level_3 filters.

Splits the augmented dataset by illumination level into two independent experiments:
  - level_2/: trains on scene_*_2_aug* pairs
  - level_3/: trains on scene_*_3_aug* pairs

Each dataset has train/ and test/ splits with the same train/val scene division.

Output layout:
    data/datasets/
      level_2/
        train/scene_001_2_aug0/, scene_001_2_aug1/, ... (all aug variants)
        test/scene_007_2/, scene_011_2/, ...
      level_3/
        train/scene_001_3_aug0/, scene_001_3_aug1/, ...
        test/scene_007_3/, scene_011_3/, ...

Usage:
    uv run python scripts/create_datasets.py --augmented data/augmented --out data/datasets
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List


def discover_pairs_by_level(
    augmented_root: Path,
    level: int,
) -> tuple[List[Path], List[Path]]:
    """Discover train and test pairs for a given level."""
    train_root = augmented_root / "train"
    test_root = augmented_root / "test"

    train_dirs = sorted([
        d for d in train_root.iterdir()
        if d.is_dir() and f"_{level}_" in d.name
    ])

    test_dirs = sorted([
        d for d in test_root.iterdir()
        if d.is_dir() and d.name.endswith(f"_{level}")
    ])

    return train_dirs, test_dirs


def create_dataset(
    level: int,
    train_dirs: List[Path],
    test_dirs: List[Path],
    out_root: Path,
    dry_run: bool = False,
) -> None:
    """Create train/test structure for a specific illumination level."""
    level_dir = out_root / f"level_{level}"
    train_out = level_dir / "train"
    test_out = level_dir / "test"

    print(f"\n=== LEVEL {level} ===")
    print(f"Train pairs: {len(train_dirs)}")
    print(f"Test pairs:  {len(test_dirs)}")

    if dry_run:
        print("[dry-run] No files written")
        return

    # Create directories
    train_out.mkdir(parents=True, exist_ok=True)
    test_out.mkdir(parents=True, exist_ok=True)

    # Create symlinks for train
    for src_dir in train_dirs:
        dest_path = train_out / src_dir.name
        if not dest_path.exists():
            dest_path.symlink_to(src_dir)
            print(f"  train: {src_dir.name}")

    # Create symlinks for test
    for src_dir in test_dirs:
        dest_path = test_out / src_dir.name
        if not dest_path.exists():
            dest_path.symlink_to(src_dir)
            print(f"  test:  {src_dir.name}")

    print(f"[OK] Level {level} dataset created at {level_dir}")


def main(
    augmented: Path,
    out: Path,
    dry_run: bool,
) -> None:
    """Main pipeline."""
    print(f"Creating separate level datasets")
    print(f"  Input:  {augmented}")
    print(f"  Output: {out}")

    for level in [2, 3]:
        train_dirs, test_dirs = discover_pairs_by_level(augmented, level)
        create_dataset(level, train_dirs, test_dirs, out, dry_run)

    print(f"\n[OK] Done.")
    print(f"\nDatasets created:")
    print(f"  {out}/level_2/")
    print(f"  {out}/level_3/")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create separate datasets for level_2 and level_3 filter training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--augmented", required=True, help="Augmented dataset root (data/augmented)")
    p.add_argument("--out",       required=True, help="Output directory (data/datasets)")
    p.add_argument("--dry-run",   action="store_true", help="Print stats without writing files")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        augmented=Path(args.augmented),
        out=Path(args.out),
        dry_run=args.dry_run,
    )
