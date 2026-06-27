"""Tests for the monotone tone curve filter (F2)."""

from __future__ import annotations

import pytest
import torch

from src.filters import build_filter, get_filter
from src.filters.tone_curve import ToneCurve
from tests._filter_helpers import (
    check_composite,
    check_differentiability,
    check_identity_on_real_photo,
    check_output_in_range,
    rand_image,
)


def test_identity_noop_synthetic():
    f = ToneCurve(P=16)
    x = rand_image(1)
    assert torch.allclose(f(x), x, atol=1e-5)


def test_identity_noop_real_photo():
    check_identity_on_real_photo(ToneCurve(P=16))


def test_output_in_range():
    f = ToneCurve(P=8, init_identity=False)
    check_output_in_range(f, rand_image(2))


def test_param_count():
    assert ToneCurve(P=16).num_params == 3 * 15  # 3*(P-1) deltas
    assert ToneCurve(P=32).num_params == 3 * 31


def test_monotone_increasing():
    f = ToneCurve(P=16, init_identity=False)
    curves = f._curves()  # (3, P)
    diffs = curves[:, 1:] - curves[:, :-1]
    assert float(diffs.min()) >= 0.0  # monotone


def test_curves_normalized_01():
    f = ToneCurve(P=16, init_identity=False)
    curves = f._curves()
    assert float(curves[:, 0].max()) < 1e-6  # start at 0
    assert float((curves[:, -1] - 1.0).abs().max()) < 1e-5  # end at 1


def test_differentiability():
    f = ToneCurve(P=8)
    frac = check_differentiability(f, rand_image(3), min_nonzero=0.90)
    assert frac > 0.90


def test_registry_and_build():
    assert isinstance(get_filter("tone_curve"), ToneCurve)
    f = build_filter({"type": "tone_curve", "P": 32})
    assert isinstance(f, ToneCurve) and f.P == 32


def test_composite():
    check_composite("tone_curve", "brightness_2param", 3 * 15 + 2)


def test_shape_3d_and_4d():
    f = ToneCurve(P=8)
    x4 = rand_image(6, 12, 12)
    assert f(x4).shape == x4.shape
    assert f(x4[0]).shape == x4[0].shape


@pytest.mark.slow
def test_smoke_calibration_reduces_distance():
    from tests._filter_helpers import run_smoke_calibration

    f = ToneCurve(P=16)
    res = run_smoke_calibration(f, "tone_curve")
    assert res["train_reduction"] is not None and res["train_reduction"] > 0.0
    print(f"[tone_curve smoke] {res}")
