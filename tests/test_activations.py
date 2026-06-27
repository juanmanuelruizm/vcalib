"""Model-free unit tests for the activation utilities.

These tests exercise the pure helpers (preprocessing, output reduction, layer-path
resolution, synthetic image) without loading the 366 MB RF-DETR checkpoint, so they
run fast in the default suite. The full forward-pass pipeline is verified manually
via ``uv run python src/utils/activations.py --test``.
"""

from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from src.utils.activations import (
    IMAGENET_MEAN,
    LAYER_PATHS,
    _to_tensor,
    load_image,
    normalize,
    preprocess,
    resolve_layer_path,
    to_unit_rgb,
)
from src.utils.activations import _synthetic_image


def test_layer_paths_cover_backbone_projector_and_decoder():
    backbone = [k for k in LAYER_PATHS if k.startswith("backbone.layer.")]
    assert len(backbone) == 12
    assert "backbone.projector" in LAYER_PATHS
    decoder = [k for k in LAYER_PATHS if k.startswith("decoder.layer.")]
    assert len(decoder) == 2
    assert all(
        LAYER_PATHS[k].startswith("backbone.0.encoder.encoder.encoder.layer.") for k in backbone
    )
    assert LAYER_PATHS["backbone.projector"] == "backbone.0.projector"
    assert LAYER_PATHS["decoder.layer.0"] == "transformer.decoder.layers.0"


def test_resolve_layer_path_canonical_and_full_passthrough():
    assert resolve_layer_path("backbone.layer.0") == "backbone.0.encoder.encoder.encoder.layer.0"
    assert resolve_layer_path("decoder.layer.1") == "transformer.decoder.layers.1"
    assert resolve_layer_path("backbone.0.projector") == "backbone.0.projector"


def test_to_tensor_with_tensor_returns_it():
    t = torch.zeros(2, 3)
    assert _to_tensor(t) is t


def test_to_tensor_with_tuple_returns_first_tensor():
    t = torch.ones(4, 4)
    assert torch.equal(_to_tensor((t, "meta", None)), t)


def test_to_tensor_with_last_hidden_state():
    t = torch.ones(5, 5)
    obj = SimpleNamespace(last_hidden_state=t, hidden_states=None)
    assert torch.equal(_to_tensor(obj), t)


def test_to_tensor_with_hidden_states_list():
    t = torch.ones(7, 7)
    obj = SimpleNamespace(last_hidden_state=None, hidden_states=[torch.zeros(1, 1), t])
    assert torch.equal(_to_tensor(obj), t)


def test_to_tensor_with_no_tensor_returns_none():
    assert _to_tensor(("only", "strings")) is None
    assert _to_tensor(None) is None


def test_to_unit_rgb_shape_and_range():
    img = (np.random.default_rng(0).random((40, 60, 3)) * 255).astype(np.uint8)
    out = to_unit_rgb(img, input_size=32)
    assert out.shape == (3, 32, 32)
    assert out.dtype == torch.float32
    assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0


def test_normalize_applies_imagenet_stats():
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    unit = mean.expand(3, 4, 4).contiguous()
    out = normalize(unit)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_preprocess_shape_and_channels():
    img = (np.random.default_rng(1).random((20, 20, 3)) * 255).astype(np.uint8)
    out = preprocess(img, size="n")
    assert out.shape == (1, 3, 384, 384)
    assert out.dtype == torch.float32


def test_synthetic_image_shape_and_dtype():
    img = _synthetic_image(seed=42)
    assert img.shape == (384, 384, 3)
    assert img.dtype == np.uint8


def test_load_image_from_ndarray_and_tensor():
    arr = (np.random.default_rng(2).random((8, 8, 3)) * 255).astype(np.uint8)
    pil = load_image(arr)
    assert isinstance(pil, Image.Image)
    assert pil.size == (8, 8)

    t = torch.rand(3, 8, 8)
    pil_t = load_image(t)
    assert isinstance(pil_t, Image.Image)
    assert pil_t.size == (8, 8)
