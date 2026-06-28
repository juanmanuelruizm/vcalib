#!/usr/bin/env python3
"""Create paired-level datasets for filter training: (level_1, level_2/3) tuples.

Stage 3 of data pipeline. Builds datasets where each training example is a PAIR
of illumination levels: level_1 (reference) + level_2 (or level_3). The filter
trains to transform level_2/3 activations toward level_1 activations.

PIPELINE:
  data/augmented/ (from augment_dataset.py)
    → Extract (level_1_aug*, level_2_aug*) pairs for experiment 1
    → Extract (level_1_aug*, level_3_aug*) pairs for experiment 2
    → Create symlinks to avoid duplication
    → data/datasets/level_1_vs_level_2/ and level_1_vs_level_3/

TRAINING FLOW (per pair):
  1. Pass level_1 → get reference activations A1
  2. Pass level_2/3 → get activations B2
  3. Apply filter to level_2/3 → get filtered_B2
  4. Pass filtered_B2 → get B2_filtered activations
  5. Loss = distance(B2_filtered_activations, A1_activations)
  6. Train filter to minimize loss

EXPERIMENTS:
  Experiment 1 (level_1 vs level_2):
    - Train: {scene_001_1_aug0, scene_001_2_aug0}, {scene_001_1_aug1, scene_001_2_aug1}, ...
    - Test:  {scene_007_1, scene_007_2}, {scene_011_1, scene_011_2}, ...

  Experiment 2 (level_1 vs level_3):
    - Train: {scene_001_1_aug0, scene_001_3_aug0}, {scene_001_1_aug1, scene_001_3_aug1}, ...
    - Test:  {scene_007_1, scene_007_3}, {scene_011_1, scene_011_3}, ...

KEY:
  - Each pair has identical augment indices (both level_1_aug0 and level_2_aug0)
  - Same scene split for both experiments → direct comparison
  - level_1 is the reference target; level_2/3 are the "bad" inputs to correct

OUTPUT (uses symlinks):
  data/datasets/
    level_1_vs_level_2/
      train/scene_001_aug0/
        level_1.jpg → ../../augmented/train/scene_001_1_aug0/level_1.jpg
        level_2.jpg → ../../augmented/train/scene_001_2_aug0/level_2.jpg
      test/scene_007/
        level_1.jpg
        level_2.jpg
    level_1_vs_level_3/
      train/scene_001_aug0/
        level_1.jpg
        level_3.jpg
      test/scene_007/
        level_1.jpg
        level_3.jpg

Usage:
    uv run python scripts/create_datasets.py --augmented data/augmented --out data/datasets
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple


def discover_pair_dirs(
    augmented_root: Path,
    level_b: int,
) -> Tuple[List[Path], List[Path]]:
    """Discover pair directories scene_XXX_{level_b}_augK that contain both level_1 and level_B.

    Each augmented pair already contains level_1.jpg (A) and level_{level_b}.jpg (B).

    Returns: (train_dirs, test_dirs)
    """
    train_root = augmented_root / "train"
    test_root = augmented_root / "test"

    # Train: find scene_XXX_{level_b}_augK
    train_dirs = sorted([
        d for d in train_root.iterdir()
        if d.is_dir() and f"_{level_b}_aug" in d.name and
        (d / "level_1.jpg").exists() and (d / f"level_{level_b}.jpg").exists()
    ])

    # Test: find scene_XXX_{level_b}
    test_dirs = sorted([
        d for d in test_root.iterdir()
        if d.is_dir() and d.name.endswith(f"_{level_b}") and
        (d / "level_1.jpg").exists() and (d / f"level_{level_b}.jpg").exists()
    ])

    return train_dirs, test_dirs


def create_dataset(
    level_b: int,
    train_dirs: List[Path],
    test_dirs: List[Path],
    out_root: Path,
    dry_run: bool = False,
) -> None:
    """Create train/test structure for (level_1 vs level_B) experiment.

    Each output symlink points to an augmented pair directory that contains
    both level_1.jpg and level_{level_b}.jpg.
    """
    exp_name = f"level_1_vs_level_{level_b}"
    exp_dir = out_root / exp_name
    train_out = exp_dir / "train"
    test_out = exp_dir / "test"

    print(f"\n=== EXPERIMENT: {exp_name} ===")
    print(f"Train pairs: {len(train_dirs)}")
    print(f"Test pairs:  {len(test_dirs)}")

    if dry_run:
        print("[dry-run] No files written")
        return

    # Create directories
    train_out.mkdir(parents=True, exist_ok=True)
    test_out.mkdir(parents=True, exist_ok=True)

    # Train: symlink to augmented pair directories
    for pair_dir in train_dirs:
        # scene_001_2_aug0 -> scene_001_aug0 (remove level suffix)
        new_name = pair_dir.name.replace(f"_{level_b}_", "_")  # scene_001_aug0

        link_path = train_out / new_name
        if not link_path.exists():
            link_path.symlink_to(pair_dir.resolve())  # Use absolute path

    # Test: symlink to test pair directories
    for pair_dir in test_dirs:
        # scene_007_2 -> scene_007 (remove level suffix)
        new_name = pair_dir.name.replace(f"_{level_b}", "")  # scene_007

        link_path = test_out / new_name
        if not link_path.exists():
            link_path.symlink_to(pair_dir.resolve())  # Use absolute path

    print(f"[OK] {exp_name} dataset created at {exp_dir}")


def main(
    augmented: Path,
    out: Path,
    dry_run: bool,
) -> None:
    """Main pipeline."""
    print(f"Creating paired-level datasets (level_1 vs level_2/3)")
    print(f"  Input:  {augmented}")
    print(f"  Output: {out}")

    for level_b in [2, 3]:
        train_dirs, test_dirs = discover_pair_dirs(augmented, level_b)
        create_dataset(level_b, train_dirs, test_dirs, out, dry_run)

    print(f"\n[OK] Done.")
    print(f"\nDatasets created:")
    print(f"  {out}/level_1_vs_level_2/")
    print(f"  {out}/level_1_vs_level_3/")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create paired-level datasets: (level_1, level_2) and (level_1, level_3)",
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
