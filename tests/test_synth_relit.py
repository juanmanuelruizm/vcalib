"""Tests for the stand-in re-lit data (A8). Pure tensor ops, no model needed."""

from pathlib import Path

import pytest
import torch

from src.utils.synth_relit import (
    make_relit_pair,
    default_photo_path,
)

_RAW = sorted(set(list(Path("data/raw").glob("*.jpg")) + list(Path("data/raw").glob("*.JPG"))))


def test_make_relit_pair_shapes_and_range():
    a, b_train, b_val = make_relit_pair(torch.rand(3, 64, 64))
    assert a.shape == (3, 64, 64)
    assert b_train.shape == a.shape and b_val.shape == a.shape
    for t in (a, b_train, b_val):
        assert float(t.min()) >= 0.0 and float(t.max()) <= 1.0


def test_a_is_unchanged_original():
    x = torch.rand(3, 32, 32)
    a, _, _ = make_relit_pair(x)
    assert torch.allclose(a, x, atol=1e-6)


def test_b_train_differs_from_a():
    a, b_train, _ = make_relit_pair(torch.rand(3, 32, 32))
    assert not torch.allclose(b_train, a, atol=1e-4)
    assert float((b_train - a).abs().mean()) > 0.01


def test_b_val_differs_and_milder_than_b_train():
    a, b_train, b_val = make_relit_pair(torch.rand(3, 32, 32))
    assert not torch.allclose(b_val, a, atol=1e-4)
    # B_val shift magnitude (gains closer to 1, gamma closer to 1) → smaller delta than B_train
    d_train = float((b_train - a).abs().mean())
    d_val = float((b_val - a).abs().mean())
    assert d_val < d_train


def test_deterministic_same_input_same_output():
    x = torch.rand(3, 32, 32)
    a1, bt1, bv1 = make_relit_pair(x)
    a2, bt2, bv2 = make_relit_pair(x)
    assert torch.allclose(a1, a2) and torch.allclose(bt1, bt2) and torch.allclose(bv1, bv2)


def test_default_photo_path_raises_when_empty(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "raw").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="No photos"):
        default_photo_path()


@pytest.mark.skipif(not _RAW, reason="no photos in data/raw")
def test_make_relit_pair_from_real_photo():
    a, b_train, b_val = make_relit_pair(_RAW[0], input_size=128)
    assert a.shape == (3, 128, 128)
    assert float(a.min()) >= 0.0 and float(a.max()) <= 1.0
    assert not torch.allclose(b_train, a, atol=1e-3)
