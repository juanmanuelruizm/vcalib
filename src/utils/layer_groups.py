"""Layer-group encoding for the Phase 2 grid sweep.

The loss is computed over **groups of layers**, not a single layer. A group is an
ordered set of canonical layer names (see ``src/utils/activations.py::LAYER_PATHS``);
the calibration loss aggregates a per-layer distance across all layers in the group
(see ``configs/grid.yaml::loss``). This module turns the declarative config into a
concrete, deduplicated list of :class:`LayerGroup` objects that the grid executor and
an automatic layer-search loop can iterate over.

Two sources of groups, unioned (explicit baselines win on collisions):

1. **Explicit** (``layer_groups`` in the YAML): hand-curated named groups, e.g.
   ``backbone.early`` = blocks 0..3. Supports range syntax in the layer list:
   ``"backbone.layer.0..3"`` expands to ``backbone.layer.0,1,2,3`` (inclusive both ends).

2. **Auto-generated** (``layer_group_search`` in the YAML): programmatic candidates
   for the automatic search loop. Contiguous windows of the DINOv2 backbone blocks of
   configurable sizes/stride, optionally with the projector and/or decoder layers
   appended. Names are deterministic (``auto.b<start>_<end>``, ``...+proj``, ``...+dec``)
   so run results stay referenceable.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.utils.activations import LAYER_PATHS

_RANGE_RE = re.compile(r"^(?P<prefix>.+\.)(?P<start>\d+)\.\.(?P<end>\d+)$")


def expand_layer_spec(spec: str) -> List[str]:
    """Expand a layer spec to canonical layer names.

    Supports inclusive range syntax ``"prefix.start..end"`` (e.g.
    ``"backbone.layer.0..3"`` -> ``backbone.layer.0..3`` four names) and plain
    canonical names (e.g. ``"backbone.projector"``). Validates every result against
    :data:`src.utils.activations.LAYER_PATHS`.
    """
    m = _RANGE_RE.match(spec)
    if m:
        prefix = m.group("prefix")
        start, end = int(m.group("start")), int(m.group("end"))
        if end < start:
            raise ValueError(f"Range {spec!r}: end < start")
        names = [f"{prefix}{i}" for i in range(start, end + 1)]
    else:
        names = [spec]
    unknown = [n for n in names if n not in LAYER_PATHS]
    if unknown:
        raise ValueError(
            f"Unknown layer(s) {unknown!r}. Valid canonical names: {sorted(LAYER_PATHS)}"
        )
    return names


@dataclass(frozen=True)
class LayerGroup:
    """A named group of canonical layer names; the loss aggregates over its layers."""

    name: str
    layers: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("LayerGroup name must be non-empty")
        if not self.layers:
            raise ValueError(f"LayerGroup {self.name!r} has no layers")
        unknown = [layer for layer in self.layers if layer not in LAYER_PATHS]
        if unknown:
            raise ValueError(f"LayerGroup {self.name!r}: unknown layers {unknown!r}")

    @property
    def layer_set(self) -> frozenset:
        return frozenset(self.layers)


def load_explicit_groups(raw: Optional[Sequence[Mapping[str, Any]]]) -> List[LayerGroup]:
    """Load hand-curated groups from the ``layer_groups`` config block."""
    out: List[LayerGroup] = []
    if not raw:
        return out
    for entry in raw:
        name = entry.get("name")
        if not isinstance(name, str):
            raise ValueError(f"layer_groups entry missing 'name': {entry!r}")
        spec_list = entry.get("layers")
        if not isinstance(spec_list, list) or not spec_list:
            raise ValueError(f"layer_groups entry {name!r} missing non-empty 'layers'")
        layers: List[str] = []
        for spec in spec_list:
            layers.extend(expand_layer_spec(str(spec)))
        out.append(LayerGroup(name=name, layers=tuple(layers)))
    return out


def _decoder_layers() -> List[str]:
    return ["decoder.layer.0", "decoder.layer.1"]


def generate_search_groups(cfg: Optional[Mapping[str, Any]]) -> List[LayerGroup]:
    """Generate candidate groups from the ``layer_group_search`` config block.

    Config keys (all optional except ``backbone`` when ``enabled``):
      enabled (bool, default False)
      backbone (str)        -- range spec of the ordered DINOv2 blocks to window over
      window_sizes (list)   -- contiguous window lengths to try (e.g. [2, 4, 6])
      stride (int)          -- step between window starts (default 1)
      attach_projector (bool) -- also emit "<window>+proj" groups
      attach_decoder (bool)   -- also emit "<window>+dec" groups
      singletons (bool)       -- also emit each block alone as "auto.b<i>"
    """
    if not cfg or not cfg.get("enabled", False):
        return []
    backbone_spec = cfg.get("backbone")
    if not isinstance(backbone_spec, str):
        raise ValueError("layer_group_search.backbone is required when enabled")
    blocks = expand_layer_spec(backbone_spec)
    window_sizes = cfg.get("window_sizes", [2, 4])
    stride = int(cfg.get("stride", 1))
    if stride < 1:
        raise ValueError("layer_group_search.stride must be >= 1")
    attach_proj = bool(cfg.get("attach_projector", False))
    attach_dec = bool(cfg.get("attach_decoder", False))
    singletons = bool(cfg.get("singletons", False))

    out: List[LayerGroup] = []
    n = len(blocks)
    for w in sorted(set(int(x) for x in window_sizes)):
        if w < 1 or w > n:
            continue
        for s in range(0, n - w + 1, stride):
            e = s + w - 1
            window = tuple(blocks[s : s + w])
            out.append(LayerGroup(name=f"auto.b{s}_{e}", layers=window))
            if attach_proj:
                out.append(
                    LayerGroup(
                        name=f"auto.b{s}_{e}+proj",
                        layers=window + ("backbone.projector",),
                    )
                )
            if attach_dec:
                out.append(
                    LayerGroup(
                        name=f"auto.b{s}_{e}+dec",
                        layers=window + tuple(_decoder_layers()),
                    )
                )
    if singletons:
        for i, b in enumerate(blocks):
            out.append(LayerGroup(name=f"auto.b{i}", layers=(b,)))
    return out


def resolve_grid_groups(config: Mapping[str, Any]) -> List[LayerGroup]:
    """Union of explicit + auto-generated groups, deduped by layer-set (explicit wins)."""
    explicit = load_explicit_groups(config.get("layer_groups"))
    generated = generate_search_groups(config.get("layer_group_search"))
    seen_sets: set = set()
    seen_names: set = set()
    out: List[LayerGroup] = []
    for g in explicit + generated:
        if g.layer_set in seen_sets or g.name in seen_names:
            continue
        seen_sets.add(g.layer_set)
        seen_names.add(g.name)
        out.append(g)
    return out


def load_grid_config(path: str | Path) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return dict(yaml.safe_load(f))


def group_summary(groups: Sequence[LayerGroup]) -> List[Dict[str, Any]]:
    """Serializable summary (name + layer list) for logging / CSV."""
    return [{"name": g.name, "layers": list(g.layers), "n": len(g.layers)} for g in groups]


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand layer-group config into concrete groups.")
    parser.add_argument("--config", type=str, default="configs/grid.yaml")
    parser.add_argument("--dump", action="store_true", help="print resolved groups")
    args = parser.parse_args()
    cfg = load_grid_config(args.config)
    groups = resolve_grid_groups(cfg)
    print(f"Resolved {len(groups)} layer group(s):")
    for g in groups:
        print(f"  {g.name:28s} n={len(g.layers):2d}  {list(g.layers)}")


if __name__ == "__main__":
    main()
