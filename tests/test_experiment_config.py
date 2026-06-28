"""Tests for the experiment config schema + loader (no model needed)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from src.experiment_config import (
    TrainingConfig,
    load_config,
    load_configs,
    summarize,
)


def _write_cfg(path: Path, data: Dict[str, Any]) -> Path:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)
    return path


def _minimal_cfg(tmp_path: Path, name: str = "test_run") -> Dict[str, Any]:
    dataset = tmp_path / "data" / "level_1_vs_level_2"
    dataset.mkdir(parents=True)
    (dataset / "train" / "scene_001").mkdir(parents=True)
    (dataset / "train" / "scene_001" / "level_1.jpg").write_bytes(b"x")
    (dataset / "train" / "scene_001" / "level_2.jpg").write_bytes(b"x")
    (dataset / "test" / "scene_002").mkdir(parents=True)
    (dataset / "test" / "scene_002" / "level_1.jpg").write_bytes(b"x")
    (dataset / "test" / "scene_002" / "level_2.jpg").write_bytes(b"x")
    return {
        "name": name,
        "dataset": str(dataset),
        "input_size": 384,
        "filter": {"type": "brightness_2param"},
        "layer_group": {"name": "projector", "layers": ["backbone.projector"]},
        "training": {
            "max_epochs": 20,
            "learning_rate": 0.005,
            "reg_weight": 0.01,
            "early_stopping_patience": 5,
            "seed": 42,
            "checkpoint_every": 5,
        },
        "output": {"results_dir": "results/experiments"},
    }


def test_load_minimal_valid(tmp_path: Path) -> None:
    cfg_data = _minimal_cfg(tmp_path)
    path = _write_cfg(tmp_path / "cfg.yaml", cfg_data)
    cfg = load_config(path)
    assert cfg.name == "test_run"
    assert cfg.dataset.is_dir()
    assert cfg.input_size == 384
    assert cfg.filter_spec == {"type": "brightness_2param"}
    assert cfg.group_name == "projector"
    assert cfg.group_layers == ["backbone.projector"]
    assert cfg.training.max_epochs == 20
    assert cfg.training.learning_rate == pytest.approx(0.005)
    assert cfg.training.reg_weight == pytest.approx(0.01)
    assert cfg.training.early_stopping_patience == 5
    assert cfg.training.seed == 42
    assert cfg.training.checkpoint_every == 5
    assert cfg.output_results_dir == Path("results/experiments")


def test_filter_display(tmp_path: Path) -> None:
    cfg_data = _minimal_cfg(tmp_path)
    cfg_data["filter"] = {"type": "lut_3d", "size": 9}
    path = _write_cfg(tmp_path / "cfg.yaml", cfg_data)
    cfg = load_config(path)
    assert cfg.filter_display == "lut_3d(size=9)"


def test_layer_range_expansion(tmp_path: Path) -> None:
    cfg_data = _minimal_cfg(tmp_path)
    cfg_data["layer_group"] = {
        "name": "backbone.early",
        "layers": ["backbone.layer.0..3"],
    }
    path = _write_cfg(tmp_path / "cfg.yaml", cfg_data)
    cfg = load_config(path)
    assert cfg.group_layers == [
        "backbone.layer.0",
        "backbone.layer.1",
        "backbone.layer.2",
        "backbone.layer.3",
    ]


def test_missing_name(tmp_path: Path) -> None:
    cfg_data = _minimal_cfg(tmp_path)
    del cfg_data["name"]
    path = _write_cfg(tmp_path / "cfg.yaml", cfg_data)
    with pytest.raises(ValueError, match="name"):
        load_config(path)


def test_missing_dataset(tmp_path: Path) -> None:
    cfg_data = _minimal_cfg(tmp_path)
    cfg_data["dataset"] = "nonexistent/path/xyz"
    path = _write_cfg(tmp_path / "cfg.yaml", cfg_data)
    with pytest.raises(ValueError, match="dataset path"):
        load_config(path)


def test_missing_dataset_skipped_in_dry_run(tmp_path: Path) -> None:
    cfg_data = _minimal_cfg(tmp_path)
    cfg_data["dataset"] = "nonexistent/path/xyz"
    path = _write_cfg(tmp_path / "cfg.yaml", cfg_data)
    cfg = load_config(path, check_dataset=False)
    assert cfg.name == "test_run"


def test_unknown_filter(tmp_path: Path) -> None:
    cfg_data = _minimal_cfg(tmp_path)
    cfg_data["filter"] = {"type": "no_such_filter"}
    path = _write_cfg(tmp_path / "cfg.yaml", cfg_data)
    with pytest.raises(ValueError, match="unknown filter"):
        load_config(path)


def test_unknown_layer(tmp_path: Path) -> None:
    cfg_data = _minimal_cfg(tmp_path)
    cfg_data["layer_group"] = {"name": "bad", "layers": ["no.such.layer"]}
    path = _write_cfg(tmp_path / "cfg.yaml", cfg_data)
    with pytest.raises(ValueError, match="Unknown layer"):
        load_config(path)


def test_negative_max_epochs(tmp_path: Path) -> None:
    cfg_data = _minimal_cfg(tmp_path)
    cfg_data["training"]["max_epochs"] = 0
    path = _write_cfg(tmp_path / "cfg.yaml", cfg_data)
    with pytest.raises(ValueError, match="max_epochs"):
        load_config(path)


def test_negative_learning_rate(tmp_path: Path) -> None:
    cfg_data = _minimal_cfg(tmp_path)
    cfg_data["training"]["learning_rate"] = -1.0
    path = _write_cfg(tmp_path / "cfg.yaml", cfg_data)
    with pytest.raises(ValueError, match="learning_rate"):
        load_config(path)


def test_load_configs_directory(tmp_path: Path) -> None:
    d = tmp_path / "exps"
    d.mkdir()
    cfg_data = _minimal_cfg(tmp_path, name="run_a")
    _write_cfg(d / "a.yaml", cfg_data)
    cfg_data["name"] = "run_b"
    _write_cfg(d / "b.yaml", cfg_data)
    configs = load_configs(d)
    assert len(configs) == 2
    assert {c.name for c in configs} == {"run_a", "run_b"}


def test_load_configs_single_file(tmp_path: Path) -> None:
    cfg_data = _minimal_cfg(tmp_path, name="solo")
    path = _write_cfg(tmp_path / "cfg.yaml", cfg_data)
    configs = load_configs(path)
    assert len(configs) == 1
    assert configs[0].name == "solo"


def test_duplicate_name_raises(tmp_path: Path) -> None:
    d = tmp_path / "exps"
    d.mkdir()
    cfg_data = _minimal_cfg(tmp_path, name="dup")
    _write_cfg(d / "a.yaml", cfg_data)
    _write_cfg(d / "b.yaml", cfg_data)
    with pytest.raises(ValueError, match="Duplicate experiment name"):
        load_configs(d)


def test_summarize(tmp_path: Path) -> None:
    cfg_data = _minimal_cfg(tmp_path, name="run_x")
    path = _write_cfg(tmp_path / "cfg.yaml", cfg_data)
    cfg = load_config(path)
    s = summarize(cfg)
    assert "run_x" in s
    assert "brightness_2param" in s
    assert "projector" in s
    assert "epochs=20" in s


def test_training_defaults(tmp_path: Path) -> None:
    cfg_data = _minimal_cfg(tmp_path)
    del cfg_data["training"]
    path = _write_cfg(tmp_path / "cfg.yaml", cfg_data)
    cfg = load_config(path)
    assert cfg.training == TrainingConfig()


def test_composite_filter_rejected(tmp_path: Path) -> None:
    cfg_data = _minimal_cfg(tmp_path)
    cfg_data["filter"] = {"composite": ["lut_3d", "tone_curve"]}
    path = _write_cfg(tmp_path / "cfg.yaml", cfg_data)
    with pytest.raises(ValueError, match="single filter"):
        load_config(path)


@pytest.mark.slow
def test_dry_run_on_starter_configs() -> None:
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "run_configs.py", "configs/experiments/", "--dry-run"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "Total runs:" in result.stdout
    assert "Dry run" in result.stdout
