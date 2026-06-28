"""Discover and load real A/B image pairs from the captured dataset.

Expects the directory layout defined in ``tasks/todo.md`` (A7):

    data/raw/
      scenes_YYYYMMDD/
        scene_001/
          level_1.jpg    ← condition A (reference)
          level_2.jpg    ← condition B (shift level 2)
          level_3.jpg    ← condition B (shift level 3)
          ...
        scene_002/
          ...
      calibration_scene/
        reference.jpg    ← condition A (for the deploy tool)

The loader discovers this layout, groups images by scene, designates ``level_1`` as
the reference condition A, and yields ``(A_path, B_path, level, scene_id)`` pairs for
each (scene, level≥2) combination. The held-out validation split is **by scene** (entire
scenes held out, not individual levels) so val pairs test generalization to unseen scenes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
_LEVEL_RE = re.compile(r"level_(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class ImagePair:
    """A single A/B pair from the same scene at a given illumination level."""

    a_path: Path
    b_path: Path
    level: int
    scene_id: str

    def __repr__(self) -> str:
        return f"ImagePair(scene={self.scene_id}, level={self.level}, A={self.a_path.name}, B={self.b_path.name})"


@dataclass
class Dataset:
    """A discovered dataset of A/B pairs with a train/val scene split."""

    pairs: List[ImagePair]
    scene_ids: List[str]
    train_scenes: List[str]
    val_scenes: List[str]
    levels: List[int]
    root: Path

    @property
    def train_pairs(self) -> List[ImagePair]:
        return [p for p in self.pairs if p.scene_id in self.train_scenes]

    @property
    def val_pairs(self) -> List[ImagePair]:
        return [p for p in self.pairs if p.scene_id in self.val_scenes]

    def __len__(self) -> int:
        return len(self.pairs)

    def __repr__(self) -> str:
        return (
            f"Dataset({len(self.pairs)} pairs, {len(self.scene_ids)} scenes, "
            f"{len(self.train_scenes)} train / {len(self.val_scenes)} val, levels={self.levels})"
        )


def _find_scene_dirs(root: Path) -> List[Path]:
    """Find scene_* directories under scenes_YYYYMMDD/ (or directly under root)."""
    scene_dirs: List[Path] = []
    # Look for scenes_*/scene_* or scene_* directly
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("scene_"):
            scene_dirs.append(child)
        elif child.name.startswith("scenes_"):
            scene_dirs.extend(
                sorted(d for d in child.iterdir() if d.is_dir() and d.name.startswith("scene_"))
            )
    return scene_dirs


def _parse_level(path: Path) -> Optional[int]:
    """Extract the level integer from a filename like 'level_1.jpg' or 'level_03.png'."""
    m = _LEVEL_RE.search(path.stem)
    return int(m.group(1)) if m else None


def _images_in(scene_dir: Path) -> Dict[int, Path]:
    """Return {level: path} for all level_*.jpg images in a scene directory."""
    out: Dict[int, Path] = {}
    for f in sorted(scene_dir.iterdir()):
        if f.suffix.lower() in VALID_EXTS:
            lvl = _parse_level(f)
            if lvl is not None:
                out[lvl] = f
    return out


def discover_pairs(
    root: str | Path,
    val_split: float = 0.2,
    seed: int = 42,
) -> Dataset:
    """Discover A/B pairs from the dataset directory.

    Args:
        root: Path to ``data/raw/`` or ``data/raw/scenes_YYYYMMDD/``.
        val_split: Fraction of scenes to hold out for validation (by scene, not by level).
        seed: Reproducible scene split.

    Returns:
        :class:`Dataset` with train/val pairs split by scene.
    """
    import random

    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    scene_dirs = _find_scene_dirs(root)
    if not scene_dirs:
        raise FileNotFoundError(
            f"No scene_* directories found under {root}. "
            f"Expected layout: {root}/scenes_YYYYMMDD/scene_001/level_1.jpg, ..."
        )

    pairs: List[ImagePair] = []
    scene_ids: List[str] = []
    all_levels: set[int] = set()

    for sd in scene_dirs:
        levels = _images_in(sd)
        if 1 not in levels:
            continue  # need level_1 as reference A
        a_path = levels[1]
        scene_id = sd.name
        scene_ids.append(scene_id)
        for lvl, b_path in sorted(levels.items()):
            if lvl <= 1:
                continue
            pairs.append(ImagePair(a_path=a_path, b_path=b_path, level=lvl, scene_id=scene_id))
            all_levels.add(lvl)

    if not pairs:
        raise FileNotFoundError(
            f"No A/B pairs found (need level_1.jpg as A and level_2+.jpg as B in each scene). "
            f"Checked {len(scene_dirs)} scene dirs under {root}."
        )

    # Split by scene (reproducible)
    rng = random.Random(seed)
    scene_ids_sorted = sorted(scene_ids)
    rng.shuffle(scene_ids_sorted)
    n_val = max(1, int(len(scene_ids_sorted) * val_split))
    val_scenes = scene_ids_sorted[:n_val]
    train_scenes = scene_ids_sorted[n_val:]

    return Dataset(
        pairs=pairs,
        scene_ids=scene_ids,
        train_scenes=train_scenes,
        val_scenes=val_scenes,
        levels=sorted(all_levels),
        root=root,
    )


def load_pair_tensors(
    pair: ImagePair,
    input_size: int = 384,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load an ImagePair as (A, B) [0,1] (3, H, W) tensors (the filter insertion point)."""
    from src.utils.activations import to_unit_rgb

    a = to_unit_rgb(pair.a_path, input_size)
    b = to_unit_rgb(pair.b_path, input_size)
    return a, b


import torch  # noqa: E402
