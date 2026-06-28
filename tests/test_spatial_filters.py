"""Tests for the spatially-varying (control-grid) filters.

Covers: identity init = no-op, output range, shape preserved, param counts at K=2/K=3,
spatial non-uniformity (different zones get different corrections), registry/factory
with grid_size, composite with spatial, and a smoke pass on a real photo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.filters import (
    FILTER_REGISTRY,
    build_filter,
    get_filter,
    make_composite,
)
from src.filters.spatial import (
    SpatialAffine,
    SpatialBrightness,
    SpatialGamma,
    SpatialWhiteBalance,
)

SPATIAL = [SpatialBrightness, SpatialWhiteBalance, SpatialAffine, SpatialGamma]
PARAM_COUNTS_K2 = {
    SpatialBrightness: 2 * 4,  # n_field=2, K^2=4
    SpatialWhiteBalance: 3 * 4,
    SpatialAffine: 6 * 4,
    SpatialGamma: 3 * 4,
}


def _rand_image(seed: int = 0, H: int = 32, W: int = 32) -> torch.Tensor:
    return torch.rand(1, 3, H, W, generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("cls", SPATIAL, ids=lambda c: c.__name__)
def test_identity_init_is_noop(cls):
    f = cls(grid_size=2, init_identity=True)
    f.eval()
    x = _rand_image(1)
    out = f(x)
    assert torch.allclose(out, x, atol=1e-5), f"{cls.__name__} not identity at init"


@pytest.mark.parametrize("cls", SPATIAL, ids=lambda c: c.__name__)
def test_output_in_unit_range(cls):
    f = cls(grid_size=3, init_identity=False)
    x = _rand_image(2)
    out = f(x)
    assert out.shape == x.shape
    d = out.detach()
    assert float(d.min()) >= 0.0 and float(d.max()) <= 1.0


@pytest.mark.parametrize("cls", SPATIAL, ids=lambda c: c.__name__)
def test_shape_preserved_3d_and_4d(cls):
    f = cls(grid_size=2)
    x4 = _rand_image(3, 24, 24)
    assert f(x4).shape == x4.shape
    x3 = x4[0]
    assert f(x3).shape == x3.shape


@pytest.mark.parametrize("cls", SPATIAL, ids=lambda c: c.__name__)
def test_param_counts_k2(cls):
    assert cls(grid_size=2).num_params == PARAM_COUNTS_K2[cls]


def test_param_counts_scale_with_k():
    assert SpatialAffine(grid_size=2).num_params == 6 * 4
    assert SpatialAffine(grid_size=3).num_params == 6 * 9
    assert SpatialAffine(grid_size=4).num_params == 6 * 16
    assert SpatialGamma(grid_size=3).num_params == 3 * 9


def test_grid_size_minimum():
    with pytest.raises(ValueError, match="grid_size"):
        SpatialAffine(grid_size=1)


def test_spatial_non_uniformity_affine():
    """A gain field that is 1 at one corner and 2 at the opposite must differ by zone."""
    f = SpatialAffine(grid_size=2, init_identity=False)
    with torch.no_grad():
        f.control_grid.zero_()
        # gains: top-left corner R gain = 1, bottom-right R gain = 2
        f.control_grid[0, 0, 0] = 1.0  # R gain @ top-left
        f.control_grid[0, 1, 1] = 2.0  # R gain @ bottom-right
        # offsets stay 0
    x = torch.full((1, 3, 16, 16), 0.5)
    out = f(x).detach()
    tl = float(out[0, 0, 0, 0])  # top-left R
    br = float(out[0, 0, 15, 15])  # bottom-right R
    # top-left ~ 1*0.5 = 0.5; bottom-right ~ 2*0.5 = 1.0 (clamped)
    assert abs(tl - 0.5) < 0.05
    assert br > tl + 0.3, "spatial field did not vary across zones"


def test_spatial_non_uniformity_gamma():
    f = SpatialGamma(grid_size=2, init_identity=False)
    with torch.no_grad():
        f.control_grid.zero_()
        f.control_grid[0, 0, 0] = 1.0  # gamma=1 -> identity
        f.control_grid[0, 1, 1] = 2.5  # gamma=2.5 -> darkens
    x = torch.full((1, 3, 16, 16), 0.5)
    out = f(x).detach()
    tl = float(out[0, 0, 0, 0])
    br = float(out[0, 0, 15, 15])
    assert abs(tl - 0.5) < 0.05
    assert br < tl - 0.2, "spatial gamma did not vary across zones"


def test_spatial_field_smooth_no_seams():
    """Bilinear interpolation: adjacent pixels differ by less than the control-point gap."""
    f = SpatialWhiteBalance(grid_size=2, init_identity=False)
    with torch.no_grad():
        f.control_grid.zero_()
        f.control_grid[0, 0, 0] = 1.0
        f.control_grid[0, 1, 1] = 2.0
    x = torch.full((1, 3, 32, 32), 0.5)
    out = f(x).detach()
    diff = (out[0, 0, :, 1:] - out[0, 0, :, :-1]).abs().max()
    assert float(diff) < 0.1, "field not smooth (seam detected)"


def test_registry_covers_spatial():
    expected = {"spatial_brightness", "spatial_whitebalance", "spatial_affine", "spatial_gamma"}
    assert expected.issubset(set(FILTER_REGISTRY))


def test_get_filter_default_grid_size_2():
    f = get_filter("spatial_affine")
    assert isinstance(f, SpatialAffine)
    assert f.grid_size == 2
    assert f.num_params == 6 * 4


def test_build_filter_with_grid_size():
    f = build_filter({"type": "spatial_affine", "grid_size": 3})
    assert isinstance(f, SpatialAffine)
    assert f.grid_size == 3
    assert f.num_params == 6 * 9


def test_build_filter_spatial_gamma_k4():
    f = build_filter({"type": "spatial_gamma", "grid_size": 4})
    assert isinstance(f, SpatialGamma)
    assert f.num_params == 3 * 16


def test_composite_with_spatial():
    comp = make_composite(["spatial_whitebalance", "gamma_3param"])
    assert comp.num_params == (3 * 4) + 3
    x = _rand_image(7)
    out = comp(x)
    assert out.shape == x.shape
    assert torch.allclose(out, x, atol=1e-5)  # both identity


def test_get_params_keys():
    f = SpatialAffine(grid_size=2)
    p = f.get_params()
    assert set(p.keys()) == {"gains", "offsets"}
    assert p["gains"].shape == (3, 2, 2)
    assert p["offsets"].shape == (3, 2, 2)
    # identity: gains ~1, offsets ~0
    assert torch.allclose(p["gains"], torch.ones(3, 2, 2), atol=1e-6)
    assert torch.allclose(p["offsets"], torch.zeros(3, 2, 2), atol=1e-6)


_REAL_PHOTOS = sorted(
    set(list(Path("data/raw").glob("*.jpg")) + list(Path("data/raw").glob("*.JPG")))
)


@pytest.mark.skipif(not _REAL_PHOTOS, reason="no photos in data/raw")
@pytest.mark.parametrize("cls", SPATIAL, ids=lambda c: c.__name__)
def test_smoke_on_real_photo(cls):
    from PIL import Image

    img = Image.open(_REAL_PHOTOS[0]).convert("RGB").resize((96, 96))
    arr = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

    f = cls(grid_size=3)
    out = f(arr)
    assert out.shape == arr.shape
    d = out.detach()
    assert float(d.min()) >= 0.0 and float(d.max()) <= 1.0
    assert torch.allclose(out, arr, atol=1e-4), f"{cls.__name__} not identity on real photo"
