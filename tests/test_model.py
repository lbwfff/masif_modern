"""Tests for the PyTorch MaSIF-site model."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import torch as t
from masif.models.masif_site import MaSIFSite


def _rand_batch(B=4, V=8, F=5):
    def r(*shape):
        return t.randn(*shape)

    return dict(
        input_feat=r(B, V, F),
        rho=t.rand(B, V),
        theta=t.rand(B, V) * 2 * np.pi,
        mask=t.ones(B, V, 1),
        # neighbour row indices within [0, B): default = self
        indices=t.zeros(B, V, dtype=t.long),
    )


def test_forward_shapes_single_layer():
    model = MaSIFSite(max_rho=3.0, n_thetas=4, n_rhos=3, n_rotations=4, n_conv_layers=1)
    b = _rand_batch()
    logits = model(**b)
    assert logits.shape == (4, 2)
    assert model.score(b["input_feat"], b["rho"], b["theta"], b["mask"], b["indices"]).shape == (4,)


def test_forward_stacked_layers():
    model = MaSIFSite(max_rho=3.0, n_thetas=4, n_rhos=3, n_rotations=4, n_conv_layers=3)
    b = _rand_batch()
    logits = model(**b)
    assert logits.shape == (4, 2)


def test_backprop_runs():
    model = MaSIFSite(max_rho=3.0, n_thetas=4, n_rhos=3, n_rotations=2, n_conv_layers=2)
    b = _rand_batch()
    labels = t.randint(0, 2, (4,)).float()
    loss = t.nn.functional.binary_cross_entropy_with_logits(model(**b)[:, 0], labels)
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_feat_mask_reduces_channels():
    model = MaSIFSite(max_rho=3.0, n_thetas=4, n_rhos=3, feat_mask=[1, 1, 0, 0, 1])
    b = _rand_batch(F=3)  # only 3 channels used
    out = model(**b)
    assert out.shape == (4, 2)