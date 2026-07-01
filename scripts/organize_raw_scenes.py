#!/usr/bin/env python3
"""Stage 1 of the data pipeline: organize flat raw captures into scene folders.

The raw captures live flat as ``data/raw/IMG_<n>_I<level>.jpg`` (level 1 = A
reference, 2/3 = shifted B). The rest of the pipeline (``augment_dataset.py`` ->
``create_datasets.py``, via ``src.utils.data_pairs.discover_pairs``) expects a
per-scene layout ``<root>/scene_<NNN>/level_<level>.jpg``. This script bridges
the two with **relative** symlinks (portable, no copies):

  data/scenes/scene_007/level_1.jpg -> ../../raw/IMG_7_I1.jpg

Scene folders are zero-padded (``scene_%03d``) so the seed=42 scene split in
discover_pairs reproduces the original held-out test set (007,011,015,020,023,027).

Usage:
    uv run python scripts/organize_raw_scenes.py --raw data/raw --out data/scenes
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

RAW_RE = re.compile(r"IMG_(\d+)_I(\d+)\.jpg$", re.IGNORECASE)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="data/raw", help="flat raw capture dir")
    ap.add_argument("--out", default="data/scenes", help="scene-structured output dir")
    args = ap.parse_args()

    raw = Path(args.raw)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    made = 0
    scenes: set[str] = set()
    for p in sorted(raw.glob("*.jpg")):
        m = RAW_RE.search(p.name)
        if not m:
            continue
        n, lvl = int(m.group(1)), int(m.group(2))
        scene_dir = out / f"scene_{n:03d}"
        scene_dir.mkdir(exist_ok=True)
        scenes.add(scene_dir.name)
        link = scene_dir / f"level_{lvl}.jpg"
        if link.is_symlink() or link.exists():
            continue
        rel = os.path.relpath(p.resolve(), start=scene_dir.resolve())
        os.symlink(rel, link)
        made += 1

    print(f"organized {len(scenes)} scenes, {made} new symlinks under {out}")


if __name__ == "__main__":
    main()
