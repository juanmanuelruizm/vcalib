"""Tests for the real A/B data-pair loader (no model needed)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.data_pairs import ImagePair, discover_pairs, _parse_level


def _make_fake_dataset(tmp_path: Path, n_scenes: int = 5, levels: list[int] = [1, 2, 3]) -> Path:
    """Create a fake dataset directory structure under tmp_path/data/raw/."""
    root = tmp_path / "data" / "raw" / "scenes_20260627"
    for s in range(1, n_scenes + 1):
        scene_dir = root / f"scene_{s:03d}"
        scene_dir.mkdir(parents=True)
        for lvl in levels:
            (scene_dir / f"level_{lvl}.jpg").write_bytes(b"fake")
    return root


def test_parse_level():
    assert _parse_level(Path("level_1.jpg")) == 1
    assert _parse_level(Path("level_03.png")) == 3
    assert _parse_level(Path("IMG_001.jpg")) is None


def test_discover_pairs_basic(tmp_path):
    root = _make_fake_dataset(tmp_path, n_scenes=5, levels=[1, 2, 3])
    ds = discover_pairs(root, val_split=0.2, seed=42)
    assert len(ds.pairs) == 5 * 2  # 5 scenes × 2 B levels
    assert len(ds.scene_ids) == 5
    assert ds.levels == [2, 3]
    assert len(ds.train_scenes) + len(ds.val_scenes) == 5
    assert len(ds.val_scenes) == 1  # 20% of 5 = 1


def test_discover_pairs_split_by_scene(tmp_path):
    root = _make_fake_dataset(tmp_path, n_scenes=10, levels=[1, 2])
    ds = discover_pairs(root, val_split=0.2, seed=42)
    # All pairs from val scenes should be in val_pairs
    val_scene_set = set(ds.val_scenes)
    train_scene_set = set(ds.train_scenes)
    assert val_scene_set.isdisjoint(train_scene_set)
    assert all(p.scene_id in train_scene_set for p in ds.train_pairs)
    assert all(p.scene_id in val_scene_set for p in ds.val_pairs)


def test_discover_pairs_reproducible_split(tmp_path):
    root = _make_fake_dataset(tmp_path, n_scenes=10, levels=[1, 2])
    ds1 = discover_pairs(root, val_split=0.2, seed=42)
    ds2 = discover_pairs(root, val_split=0.2, seed=42)
    assert ds1.val_scenes == ds2.val_scenes
    assert ds1.train_scenes == ds2.train_scenes


def test_discover_pairs_different_seed_different_split(tmp_path):
    root = _make_fake_dataset(tmp_path, n_scenes=10, levels=[1, 2])
    ds1 = discover_pairs(root, val_split=0.2, seed=42)
    ds2 = discover_pairs(root, val_split=0.2, seed=99)
    assert ds1.val_scenes != ds2.val_scenes


def test_discover_pairs_missing_level_1_skipped(tmp_path):
    root = tmp_path / "data" / "raw" / "scenes_test"
    s1 = root / "scene_001"
    s1.mkdir(parents=True)
    (s1 / "level_2.jpg").write_bytes(b"fake")  # no level_1 → skipped
    s2 = root / "scene_002"
    s2.mkdir(parents=True)
    (s2 / "level_1.jpg").write_bytes(b"fake")
    (s2 / "level_2.jpg").write_bytes(b"fake")
    ds = discover_pairs(root, val_split=0.5, seed=42)
    assert len(ds.pairs) == 1
    assert ds.pairs[0].scene_id == "scene_002"


def test_discover_pairs_no_data_raises(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    with pytest.raises(FileNotFoundError, match="No scene_"):
        discover_pairs(root)


def test_discover_pairs_no_pairs_raises(tmp_path):
    root = tmp_path / "data" / "raw" / "scenes_test"
    s1 = root / "scene_001"
    s1.mkdir(parents=True)
    (s1 / "level_1.jpg").write_bytes(b"fake")  # only A, no B
    with pytest.raises(FileNotFoundError, match="No A/B pairs"):
        discover_pairs(root)


def test_image_pair_repr():
    p = ImagePair(a_path=Path("a.jpg"), b_path=Path("b.jpg"), level=2, scene_id="scene_001")
    assert "scene_001" in repr(p) and "level=2" in repr(p)


def test_dataset_repr_and_props(tmp_path):
    root = _make_fake_dataset(tmp_path, n_scenes=4, levels=[1, 2, 3])
    ds = discover_pairs(root, val_split=0.25, seed=42)
    r = repr(ds)
    assert "pairs" in r and "scenes" in r
    assert len(ds.train_pairs) + len(ds.val_pairs) == len(ds.pairs)
