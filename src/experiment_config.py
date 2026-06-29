"""Experiment config schema + loader for config-driven experiment runs.

Each YAML file in ``configs/experiments/`` describes **one atomic training run**:
one filter, one layer group, one dataset, plus training hyperparameters. The
runner (:file:`run_configs.py`) globs one or more YAML files and executes every
config, aggregating results into a single CSV.

Schema (see ``docs/specs/2026-06-28-config-driven-experiments.md``)::

    name: level2_lut3d_projector
    dataset: data/datasets/level_1_vs_level_2
    input_size: 384
    filter:
      type: lut_3d
      size: 9
    layer_group:
      name: projector
      layers: [backbone.projector]
    training:
      max_epochs: 20
      learning_rate: 0.005
      reg_weight: 0.01
      early_stopping_patience: 5
      seed: 42
      checkpoint_every: 5
    output:
      results_dir: results/experiments

Layer range syntax (``backbone.layer.0..3`` -> four names) is supported via
:func:`src.utils.layer_groups.expand_layer_spec`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from src.filters import FILTER_REGISTRY
from src.utils.layer_groups import expand_layer_spec


@dataclass
class TrainingConfig:
    """Optimizer + loop hyperparameters for one run."""

    max_epochs: int = 20
    learning_rate: float = 5e-3
    reg_weight: float = 0.01
    early_stopping_patience: int = 5
    seed: int = 42
    checkpoint_every: int = 5
    loss_mode: str = "activation"  # "activation" | "combined" | "detection"
    detection_weight: float = 1.0  # β in combined mode
    optimizer: str = "adam"                     # "adam" | "adamw"
    weight_decay: float = 0.0                   # used only with "adamw"
    scheduler: Optional[str] = None             # "cosine" | "step" | None
    scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)
    layer_weights: Optional[Dict[str, float]] = None


@dataclass(frozen=True)
class ExperimentConfig:
    """One atomic experiment run, parsed from a YAML file."""

    name: str
    dataset: Path
    input_size: int
    filter_spec: Dict[str, Any]
    group_name: str
    group_layers: List[str]
    training: TrainingConfig
    output_results_dir: Path
    config_file: Optional[Path] = None

    @property
    def filter_display(self) -> str:
        """Human-readable filter label (e.g. ``lut_3d(size=9)`` or ``affine_6param+gamma_3param``)."""
        if "composite" in self.filter_spec:
            return "+".join(str(s) for s in self.filter_spec["composite"])
        name = str(self.filter_spec["type"])
        extras = {k: v for k, v in self.filter_spec.items() if k != "type"}
        if extras:
            return f"{name}({','.join(f'{k}={v}' for k, v in extras.items())})"
        return name


def _require_keys(raw: Mapping[str, Any], required: Sequence[str], config_file: Path) -> None:
    for key in required:
        if key not in raw:
            raise ValueError(f"Config {config_file!s}: missing required field {key!r}")


def _validate_filter(raw_filter: Any, config_file: Path) -> Dict[str, Any]:
    if not isinstance(raw_filter, Mapping):
        raise ValueError(f"Config {config_file!s}: 'filter' must be a mapping")
    if "composite" in raw_filter:
        stages = raw_filter["composite"]
        if not isinstance(stages, list) or not stages:
            raise ValueError(
                f"Config {config_file!s}: 'filter.composite' must be a non-empty list of filter names"
            )
        for s in stages:
            if not isinstance(s, str) or s not in FILTER_REGISTRY:
                raise ValueError(
                    f"Config {config_file!s}: unknown composite stage {s!r}. "
                    f"Valid: {sorted(FILTER_REGISTRY)}"
                )
        return dict(raw_filter)
    if "type" not in raw_filter or not isinstance(raw_filter["type"], str):
        raise ValueError(f"Config {config_file!s}: 'filter.type' is required and must be a string")
    ftype = raw_filter["type"]
    if ftype not in FILTER_REGISTRY:
        raise ValueError(
            f"Config {config_file!s}: unknown filter type {ftype!r}. "
            f"Valid: {sorted(FILTER_REGISTRY)}"
        )
    return dict(raw_filter)


def _validate_layer_group(raw_group: Any, config_file: Path) -> tuple[str, List[str]]:
    if not isinstance(raw_group, Mapping):
        raise ValueError(f"Config {config_file!s}: 'layer_group' must be a mapping")
    if "name" not in raw_group or not isinstance(raw_group["name"], str) or not raw_group["name"]:
        raise ValueError(f"Config {config_file!s}: 'layer_group.name' is required and non-empty")
    if (
        "layers" not in raw_group
        or not isinstance(raw_group["layers"], list)
        or not raw_group["layers"]
    ):
        raise ValueError(
            f"Config {config_file!s}: 'layer_group.layers' is required and must be a non-empty list"
        )
    layers: List[str] = []
    for spec in raw_group["layers"]:
        if not isinstance(spec, str):
            raise ValueError(f"Config {config_file!s}: layer spec {spec!r} must be a string")
        layers.extend(expand_layer_spec(spec))
    return str(raw_group["name"]), layers


def _validate_training(raw_training: Any, config_file: Path) -> TrainingConfig:
    if raw_training is None:
        return TrainingConfig()
    if not isinstance(raw_training, Mapping):
        raise ValueError(f"Config {config_file!s}: 'training' must be a mapping")
    _VALID_LOSS_MODES = {"activation", "combined", "detection"}
    defaults = TrainingConfig()
    try:
        max_epochs = int(raw_training.get("max_epochs", defaults.max_epochs))
        learning_rate = float(raw_training.get("learning_rate", defaults.learning_rate))
        reg_weight = float(raw_training.get("reg_weight", defaults.reg_weight))
        early_stopping_patience = int(
            raw_training.get("early_stopping_patience", defaults.early_stopping_patience)
        )
        seed = int(raw_training.get("seed", defaults.seed))
        checkpoint_every = int(raw_training.get("checkpoint_every", defaults.checkpoint_every))
        optimizer = str(raw_training.get("optimizer", defaults.optimizer))
        weight_decay = float(raw_training.get("weight_decay", defaults.weight_decay))
        scheduler_raw = raw_training.get("scheduler", defaults.scheduler)
        scheduler: Optional[str] = str(scheduler_raw) if scheduler_raw is not None else None
        scheduler_kwargs_raw = raw_training.get("scheduler_kwargs", {})
        scheduler_kwargs: Dict[str, Any] = dict(scheduler_kwargs_raw) if scheduler_kwargs_raw else {}
        layer_weights_raw = raw_training.get("layer_weights", None)
        layer_weights: Optional[Dict[str, float]] = (
            {str(k): float(v) for k, v in layer_weights_raw.items()}
            if layer_weights_raw is not None
            else None
        )
        loss_mode = str(raw_training.get("loss_mode", TrainingConfig().loss_mode))
        detection_weight = float(
            raw_training.get("detection_weight", TrainingConfig().detection_weight)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Config {config_file!s}: invalid 'training' value: {exc}") from exc
    if max_epochs < 1:
        raise ValueError(f"Config {config_file!s}: training.max_epochs must be >= 1")
    if learning_rate <= 0:
        raise ValueError(f"Config {config_file!s}: training.learning_rate must be > 0")
    if reg_weight < 0:
        raise ValueError(f"Config {config_file!s}: training.reg_weight must be >= 0")
    if early_stopping_patience < 1:
        raise ValueError(f"Config {config_file!s}: training.early_stopping_patience must be >= 1")
    if checkpoint_every < 0:
        raise ValueError(f"Config {config_file!s}: training.checkpoint_every must be >= 0")
    if loss_mode not in _VALID_LOSS_MODES:
        raise ValueError(
            f"Config {config_file!s}: training.loss_mode must be one of {sorted(_VALID_LOSS_MODES)}"
        )
    if detection_weight <= 0:
        raise ValueError(f"Config {config_file!s}: training.detection_weight must be > 0")
    if optimizer not in ("adam", "adamw"):
        raise ValueError(f"Config {config_file!s}: training.optimizer must be 'adam' or 'adamw'")
    if weight_decay < 0:
        raise ValueError(f"Config {config_file!s}: training.weight_decay must be >= 0")
    if scheduler not in (None, "cosine", "step"):
        raise ValueError(
            f"Config {config_file!s}: training.scheduler must be 'cosine', 'step', or null"
        )
    return TrainingConfig(
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        reg_weight=reg_weight,
        early_stopping_patience=early_stopping_patience,
        seed=seed,
        checkpoint_every=checkpoint_every,
        loss_mode=loss_mode,
        detection_weight=detection_weight,
        optimizer=optimizer,
        weight_decay=weight_decay,
        scheduler=scheduler,
        scheduler_kwargs=scheduler_kwargs,
        layer_weights=layer_weights,
    )


def load_config(path: Union[str, Path], check_dataset: bool = True) -> ExperimentConfig:
    """Read + validate a single experiment YAML. Raises ``ValueError`` on any problem.

    ``check_dataset=False`` skips the on-disk dataset existence check (useful for
    ``--dry-run`` when the dataset is not materialized locally, e.g. on Windows
    checkouts of case-collision-renamed trees).
    """
    import yaml

    config_file = Path(path)
    if not config_file.is_file():
        raise ValueError(f"Config file not found: {config_file!s}")

    with open(config_file, "r", encoding="utf-8") as f:
        raw: Any = yaml.safe_load(f)
    if not isinstance(raw, Mapping):
        raise ValueError(f"Config {config_file!s}: top-level must be a mapping")

    _require_keys(raw, ["name", "dataset", "filter", "layer_group"], config_file)

    name = raw["name"]
    if not isinstance(name, str) or not name:
        raise ValueError(f"Config {config_file!s}: 'name' must be a non-empty string")

    dataset = Path(raw["dataset"])
    if check_dataset and not dataset.is_dir():
        raise ValueError(
            f"Config {config_file!s}: dataset path {dataset!s} does not exist or is not a directory"
        )

    input_size = int(raw.get("input_size", 384))
    if input_size <= 0:
        raise ValueError(f"Config {config_file!s}: input_size must be > 0")

    filter_spec = _validate_filter(raw["filter"], config_file)
    group_name, group_layers = _validate_layer_group(raw["layer_group"], config_file)
    training = _validate_training(raw.get("training"), config_file)

    out_dir = Path(raw.get("output", {}).get("results_dir", "results/experiments"))

    return ExperimentConfig(
        name=name,
        dataset=dataset,
        input_size=input_size,
        filter_spec=filter_spec,
        group_name=group_name,
        group_layers=group_layers,
        training=training,
        output_results_dir=out_dir,
        config_file=config_file,
    )


def load_configs(
    paths: Union[str, Path, Iterable[Union[str, Path]]], check_dataset: bool = True
) -> List[ExperimentConfig]:
    """Load one or more experiment configs.

    Accepts:
    - a single file path (returns one-element list)
    - a directory (globs ``*.yaml`` non-recursively, sorted by stem)
    - an iterable of file/directory paths

    Deduplicates on ``name`` across the loaded set (raises on collision).
    ``check_dataset=False`` skips the on-disk dataset existence check (for ``--dry-run``).
    """
    if isinstance(paths, (str, Path)):
        items: List[Union[str, Path]] = [paths]
    else:
        items = list(paths)
    if not items:
        return []

    files: List[Path] = []
    for p in items:
        path = Path(p)
        if path.is_dir():
            files.extend(sorted(path.glob("*.yaml")))
        elif path.is_file():
            files.append(path)
        else:
            raise ValueError(f"Path not found: {path!s}")

    configs = [load_config(f, check_dataset=check_dataset) for f in files]
    seen: Dict[str, Path] = {}
    for c in configs:
        if c.name in seen:
            raise ValueError(
                f"Duplicate experiment name {c.name!r} in {c.config_file!s} "
                f"(already defined in {seen[c.name]!s})"
            )
        seen[c.name] = c.config_file or Path("?")
    return configs


def summarize(config: ExperimentConfig) -> str:
    """One-line dry-run summary."""
    return (
        f"{config.name:<40s} dataset={config.dataset!s:<42s} "
        f"filter={config.filter_display:<28s} group={config.group_name:<16s} "
        f"epochs={config.training.max_epochs}"
    )
