"""Model-free unit tests for layer-group encoding and search generation."""

import pytest

from src.utils.layer_groups import (
    LayerGroup,
    expand_layer_spec,
    generate_search_groups,
    group_summary,
    load_explicit_groups,
    resolve_grid_groups,
)


def test_expand_range_inclusive():
    assert expand_layer_spec("backbone.layer.0..3") == [
        "backbone.layer.0",
        "backbone.layer.1",
        "backbone.layer.2",
        "backbone.layer.3",
    ]
    assert expand_layer_spec("decoder.layer.0..1") == ["decoder.layer.0", "decoder.layer.1"]


def test_expand_single_name_passthrough():
    assert expand_layer_spec("backbone.projector") == ["backbone.projector"]


def test_expand_rejects_unknown_layer():
    with pytest.raises(ValueError, match="Unknown layer"):
        expand_layer_spec("backbone.layer.99..99")
    with pytest.raises(ValueError, match="Unknown layer"):
        expand_layer_spec("not.a.layer")


def test_expand_rejects_inverted_range():
    with pytest.raises(ValueError, match="end < start"):
        expand_layer_spec("backbone.layer.3..0")


def test_layer_group_validates_layers():
    g = LayerGroup(name="g", layers=("backbone.layer.0", "backbone.projector"))
    assert g.layer_set == frozenset({"backbone.layer.0", "backbone.projector"})
    with pytest.raises(ValueError, match="unknown layers"):
        LayerGroup(name="g", layers=("backbone.layer.99",))
    with pytest.raises(ValueError, match="no layers"):
        LayerGroup(name="g", layers=())


def test_load_explicit_groups_with_ranges():
    raw = [{"name": "backbone.early", "layers": ["backbone.layer.0..1"]}]
    groups = load_explicit_groups(raw)
    assert len(groups) == 1
    assert groups[0].name == "backbone.early"
    assert groups[0].layers == ("backbone.layer.0", "backbone.layer.1")


def test_load_explicit_groups_mixed_specs():
    raw = [{"name": "mix", "layers": ["backbone.layer.0..2", "backbone.projector"]}]
    groups = load_explicit_groups(raw)
    assert groups[0].layers == (
        "backbone.layer.0",
        "backbone.layer.1",
        "backbone.layer.2",
        "backbone.projector",
    )


def test_load_explicit_groups_empty_returns_empty():
    assert load_explicit_groups(None) == []
    assert load_explicit_groups([]) == []


def test_load_explicit_groups_rejects_missing_name():
    with pytest.raises(ValueError, match="missing 'name'"):
        load_explicit_groups([{"layers": ["backbone.projector"]}])

    with pytest.raises(ValueError, match="missing non-empty 'layers'"):
        load_explicit_groups([{"name": "g", "layers": []}])


def test_generate_search_disabled_returns_empty():
    assert generate_search_groups(None) == []
    assert generate_search_groups({"enabled": False}) == []


def test_generate_search_windows():
    cfg = {
        "enabled": True,
        "backbone": "backbone.layer.0..5",  # 6 blocks
        "window_sizes": [2],
        "stride": 1,
        "attach_projector": False,
        "attach_decoder": False,
        "singletons": False,
    }
    groups = generate_search_groups(cfg)
    names = [g.name for g in groups]
    assert names == ["auto.b0_1", "auto.b1_2", "auto.b2_3", "auto.b3_4", "auto.b4_5"]
    assert groups[0].layers == ("backbone.layer.0", "backbone.layer.1")


def test_generate_search_stride_and_proj_decoder():
    cfg = {
        "enabled": True,
        "backbone": "backbone.layer.0..5",
        "window_sizes": [2],
        "stride": 2,
        "attach_projector": True,
        "attach_decoder": True,
        "singletons": False,
    }
    groups = generate_search_groups(cfg)
    names = [g.name for g in groups]
    # stride 2 over 6 blocks with size 2 -> starts 0,2,4
    assert names == [
        "auto.b0_1",
        "auto.b0_1+proj",
        "auto.b0_1+dec",
        "auto.b2_3",
        "auto.b2_3+proj",
        "auto.b2_3+dec",
        "auto.b4_5",
        "auto.b4_5+proj",
        "auto.b4_5+dec",
    ]
    proj = next(g for g in groups if g.name == "auto.b0_1+proj")
    assert proj.layers == ("backbone.layer.0", "backbone.layer.1", "backbone.projector")
    dec = next(g for g in groups if g.name == "auto.b0_1+dec")
    assert dec.layers == (
        "backbone.layer.0",
        "backbone.layer.1",
        "decoder.layer.0",
        "decoder.layer.1",
    )


def test_generate_search_singletons():
    cfg = {
        "enabled": True,
        "backbone": "backbone.layer.0..2",
        "window_sizes": [],  # no windows
        "singletons": True,
    }
    groups = generate_search_groups(cfg)
    assert [g.name for g in groups] == ["auto.b0", "auto.b1", "auto.b2"]
    assert groups[1].layers == ("backbone.layer.1",)


def test_generate_search_dedups_window_sizes():
    cfg = {
        "enabled": True,
        "backbone": "backbone.layer.0..3",
        "window_sizes": [2, 2, 2],
        "stride": 1,
    }
    groups = generate_search_groups(cfg)
    assert [g.name for g in groups] == ["auto.b0_1", "auto.b1_2", "auto.b2_3"]


def test_resolve_grid_groups_dedups_by_layer_set_explicit_wins():
    config = {
        "layer_groups": [
            {"name": "projector", "layers": ["backbone.projector"]},
        ],
        "layer_group_search": {
            "enabled": True,
            "backbone": "backbone.layer.0..1",
            "window_sizes": [1],
            "stride": 1,
            "singletons": True,
        },
    }
    groups = resolve_grid_groups(config)
    names = [g.name for g in groups]
    # explicit baseline kept first
    assert names[0] == "projector"
    # size-1 windows are emitted before singletons, so they win the layer-set dedup;
    # the singleton names auto.b0 / auto.b1 are dropped (same layer-set as the windows)
    assert "auto.b0_0" in names and "auto.b1_1" in names
    assert "auto.b0" not in names and "auto.b1" not in names
    assert len(names) == len(set(names))


def test_resolve_grid_groups_empty_config():
    assert resolve_grid_groups({}) == []


def test_group_summary_serializable():
    groups = [LayerGroup(name="g", layers=("backbone.layer.0", "backbone.layer.1"))]
    s = group_summary(groups)
    assert s == [{"name": "g", "layers": ["backbone.layer.0", "backbone.layer.1"], "n": 2}]
