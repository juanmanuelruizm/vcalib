"""Tests for the parametric filter library.

Every filter must satisfy: identity init = no-op, output stays in [0, 1], shape
preserved, params reported in range. Also covers the registry factory, composite
chaining, and a smoke pass on a real photo from data/raw.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.filters import (
    Affine6Param,
    Brightness,
    CompositeFilter,
    Contrast,
    FILTER_REGISTRY,
    Filter,
    Gamma,
    Matrix12Param,
    Saturation,
    WhiteBalance,
    build_filter,
    get_filter,
    make_composite,
)

ALL_FILTERS = [
    Brightness,
    WhiteBalance,
    Affine6Param,
    Saturation,
    Contrast,
    Gamma,
    Matrix12Param,
]

SAMPLE_IMAGE = torch.rand(1, 3, 16, 16)


def _rand_image(seed: int = 0) -> torch.Tensor:
    return torch.rand(1, 3, 16, 16, generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("cls", ALL_FILTERS, ids=lambda c: c.__name__)
def test_identity_init_is_noop(cls):
    f = cls(init_identity=True)
    f.eval()
    x = _rand_image(1)
    out = f(x)
    assert torch.allclose(out, x, atol=1e-6), f"{cls.__name__} not identity at init"


@pytest.mark.parametrize("cls", ALL_FILTERS, ids=lambda c: c.__name__)
def test_output_in_unit_range(cls):
    f = cls(init_identity=False)
    x = _rand_image(2)
    out = f(x)
    assert out.shape == x.shape
    out_d = out.detach()
    assert float(out_d.min()) >= 0.0
    assert float(out_d.max()) <= 1.0


@pytest.mark.parametrize("cls", ALL_FILTERS, ids=lambda c: c.__name__)
def test_shape_preserved_3d_and_4d(cls):
    f = cls()
    x4 = _rand_image(3)
    assert f(x4).shape == x4.shape
    x3 = x4[0]
    assert f(x3).shape == x3.shape


@pytest.mark.parametrize("cls", ALL_FILTERS, ids=lambda c: c.__name__)
def test_get_params_in_range(cls):
    f = cls(init_identity=False)
    # push params toward extremes by setting raw values far out
    for p in f.parameters():
        with torch.no_grad():
            p.fill_(10.0)
    for k, v in f.get_params().items():
        assert torch.isfinite(v).all(), f"{cls.__name__}.{k} not finite"
        assert float(v.min()) >= -2.5 and float(v.max()) <= 2.5, f"{cls.__name__}.{k} out of range"


@pytest.mark.parametrize("cls", ALL_FILTERS, ids=lambda c: c.__name__)
def test_num_params_matches(cls):
    expected = {
        Brightness: 2,
        WhiteBalance: 3,
        Affine6Param: 6,
        Saturation: 1,
        Contrast: 1,
        Gamma: 3,
        Matrix12Param: 12,
    }
    assert cls().num_params == expected[cls]


def test_filters_are_nn_modules_and_trainable():
    f = Affine6Param()
    assert isinstance(f, torch.nn.Module)
    assert isinstance(f, Filter)
    assert all(p.requires_grad for p in f.parameters())


def test_brightness_values():
    f = Brightness(init_identity=False)
    with torch.no_grad():
        f.gain.fill_(2.0)
        f.offset.fill_(0.1)
    out = f(SAMPLE_IMAGE)
    expected = torch.clamp(2.0 * SAMPLE_IMAGE + 0.1, 0, 1)
    assert torch.allclose(out, expected, atol=1e-6)


def test_white_balance_per_channel():
    f = WhiteBalance(init_identity=False)
    with torch.no_grad():
        f.gains.copy_(torch.tensor([0.5, 1.0, 1.5]))
    out = f(SAMPLE_IMAGE)
    expected = torch.clamp(SAMPLE_IMAGE * torch.tensor([0.5, 1.0, 1.5]).view(1, 3, 1, 1), 0, 1)
    assert torch.allclose(out, expected, atol=1e-6)


def test_saturation_grayscale_at_zero():
    f = Saturation(init_identity=False)
    with torch.no_grad():
        f.sat.fill_(0.0)
    out = f(SAMPLE_IMAGE)
    luma = SAMPLE_IMAGE[:, 0] * 0.299 + SAMPLE_IMAGE[:, 1] * 0.587 + SAMPLE_IMAGE[:, 2] * 0.114
    for c in range(3):
        assert torch.allclose(out[:, c], luma, atol=1e-6)


def test_contrast_identity_and_boost():
    x = _rand_image(7)
    f = Contrast()
    assert torch.allclose(f(x), x, atol=1e-5)
    f2 = Contrast(init_identity=False)
    with torch.no_grad():
        f2.contrast.fill_(1.5)
    out = f2(x)
    assert out.shape == x.shape
    assert float(out.detach().max()) <= 1.0 and float(out.detach().min()) >= 0.0


def test_gamma_identity_and_curve():
    x = _rand_image(8)
    f = Gamma()
    assert torch.allclose(f(x), x, atol=1e-5)
    f2 = Gamma(init_identity=False)
    with torch.no_grad():
        f2.gamma.fill_(2.0)
    out = f2(x)
    # gamma=2 darkens midtones; output <= input almost everywhere for x in (0,1)
    assert (out <= x + 1e-6).float().mean() > 0.9
    assert float(out.detach().min()) >= 0.0 and float(out.detach().max()) <= 1.0


def test_matrix_identity_and_rotation():
    f = Matrix12Param()
    assert torch.allclose(f(SAMPLE_IMAGE), SAMPLE_IMAGE, atol=1e-6)
    f2 = Matrix12Param(init_identity=False)
    with torch.no_grad():
        f2.matrix.copy_(torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
    out = f2(SAMPLE_IMAGE)
    # swap R and G channels
    swapped = torch.stack([SAMPLE_IMAGE[:, 1], SAMPLE_IMAGE[:, 0], SAMPLE_IMAGE[:, 2]], dim=1)
    assert torch.allclose(out, torch.clamp(swapped, 0, 1), atol=1e-6)


def test_composite_identity_chain():
    comp = CompositeFilter([Brightness(), Gamma(), Saturation()])
    x = _rand_image(9)
    assert torch.allclose(comp(x), x, atol=1e-5)
    assert comp.num_params == 2 + 3 + 1


def test_composite_chains_transforms():
    comp = CompositeFilter([WhiteBalance(), Affine6Param()])
    assert comp.num_params == 3 + 6
    x = _rand_image(10)
    out = comp(x)
    assert out.shape == x.shape
    assert float(out.detach().min()) >= 0.0 and float(out.detach().max()) <= 1.0


def test_composite_empty_raises():
    with pytest.raises(ValueError, match="at least one"):
        CompositeFilter([])


def test_composite_get_params_namespaced():
    comp = CompositeFilter([Brightness(), Gamma()])
    params = comp.get_params()
    assert "f0_Brightness_gain" in params
    assert "f1_Gamma_gamma" in params


def test_registry_covers_all_filters():
    expected = {
        "brightness_2param",
        "white_balance_3param",
        "affine_6param",
        "saturation_1param",
        "contrast_1param",
        "gamma_3param",
        "matrix_12param",
    }
    assert expected.issubset(set(FILTER_REGISTRY))


def test_get_filter_returns_fresh_identity_instances():
    a = get_filter("affine_6param")
    b = get_filter("affine_6param")
    assert a is not b
    x = _rand_image(11)
    assert torch.allclose(a(x), x, atol=1e-6)


def test_get_filter_rejects_unknown():
    with pytest.raises(KeyError, match="Unknown filter"):
        get_filter("does_not_exist")


def test_build_filter_variants():
    assert isinstance(build_filter("affine_6param"), Affine6Param)
    comp = build_filter(["affine_6param", "gamma_3param"])
    assert isinstance(comp, CompositeFilter) and comp.num_params == 9
    comp2 = build_filter({"composite": ["brightness_2param", "saturation_1param"]})
    assert isinstance(comp2, CompositeFilter) and comp2.num_params == 3
    assert isinstance(build_filter({"type": "matrix_12param"}), Matrix12Param)


def test_make_composite_order():
    comp = make_composite(["brightness_2param", "gamma_3param"])
    assert [type(f).__name__ for f in comp.filters] == ["Brightness", "Gamma"]


def test_rejects_non_nchw():
    f = Affine6Param()
    with pytest.raises(ValueError, match="NCHW"):
        f(torch.rand(3, 16, 16, 16))


_REAL_PHOTOS = sorted(Path("data/raw").glob("*.jpg")) + sorted(Path("data/raw").glob("*.JPG"))


@pytest.mark.skipif(not _REAL_PHOTOS, reason="no photos in data/raw")
@pytest.mark.parametrize("cls", ALL_FILTERS, ids=lambda c: c.__name__)
def test_smoke_on_real_photo(cls):
    """Run each filter on a real photo: shape/range preserved, identity = no-op."""
    from PIL import Image

    img = Image.open(_REAL_PHOTOS[0]).convert("RGB").resize((64, 64))
    arr = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

    f = cls()
    out = f(arr)
    assert out.shape == arr.shape
    assert float(out.detach().min()) >= 0.0 and float(out.detach().max()) <= 1.0
    assert torch.allclose(out, arr, atol=1e-5), f"{cls.__name__} not identity on real photo"
