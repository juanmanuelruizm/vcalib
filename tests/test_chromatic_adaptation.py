"""Tests for the chromatic adaptation (Bradford LMS) filter (F4)."""

from __future__ import annotations

import pytest
import torch

from src.filters import build_filter, get_filter
from src.filters.chromatic_adaptation import ChromaticAdaptation
from tests._filter_helpers import (
    check_composite,
    check_differentiability,
    check_identity_on_real_photo,
    check_output_in_range,
    rand_image,
)


def test_identity_noop_synthetic():
    f = ChromaticAdaptation(mode="diagonal")
    x = rand_image(1)
    assert torch.allclose(f(x), x, atol=1e-4)


def test_identity_noop_real_photo():
    check_identity_on_real_photo(ChromaticAdaptation(mode="diagonal"))


def test_identity_full_matrix():
    f = ChromaticAdaptation(mode="full")
    x = rand_image(7)
    assert torch.allclose(f(x), x, atol=1e-4)


def test_output_in_range():
    f = ChromaticAdaptation(mode="diagonal", init_identity=False)
    check_output_in_range(f, rand_image(2))


def test_param_count():
    assert ChromaticAdaptation(mode="diagonal").num_params == 3
    assert ChromaticAdaptation(mode="full").num_params == 9


def test_mode_validation():
    with pytest.raises(ValueError, match="mode"):
        ChromaticAdaptation(mode="invalid")


def test_differentiability():
    f = ChromaticAdaptation(mode="diagonal")
    check_differentiability(f, rand_image(3), min_nonzero=0.9)


def test_differentiability_full():
    f = ChromaticAdaptation(mode="full")
    check_differentiability(f, rand_image(4), min_nonzero=0.9)


def test_registry_and_build():
    assert isinstance(get_filter("chromatic_adaptation"), ChromaticAdaptation)
    f = build_filter({"type": "chromatic_adaptation", "mode": "full"})
    assert isinstance(f, ChromaticAdaptation) and f.mode == "full"


def test_composite():
    check_composite("chromatic_adaptation", "brightness_2param", 3 + 2)


@pytest.mark.slow
def test_smoke_calibration_reduces_distance():
    from tests._filter_helpers import run_smoke_calibration

    f = ChromaticAdaptation(mode="diagonal")
    res = run_smoke_calibration(f, "chromatic_adaptation")
    assert res["train_reduction"] is not None and res["train_reduction"] > 0.0
    print(f"[chromatic_adaptation smoke] {res}")
