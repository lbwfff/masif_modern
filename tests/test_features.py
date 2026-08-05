"""Tests for surface feature computation."""

import numpy as np
import pytest

from masif.surface.features import (
    shape_index,
    normalize_electrostatics,
    vertex_curvature,
    _angle_penalty,
)
from masif.patches import compute_ddc


def test_normalize_electrostatics_clamps_and_rescales():
    x = np.array([5.0, -5.0, 0.0, 3.0, -3.0])
    out = normalize_electrostatics(x)
    np.testing.assert_allclose(out, [1.0, -1.0, 0.0, 1.0, -1.0], atol=1e-6)


def test_angle_penalty():
    assert _angle_penalty(np.array([0.0])) == pytest.approx(1.0)
    assert _angle_penalty(np.array([np.pi])) == pytest.approx(0.0)


def test_shape_index_saddle_is_zero():
    # z = x^2 - y^2 has principal curvatures +2 and -2 -> shape index 0.
    xs = np.linspace(-1, 1, 9)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    verts = np.stack([gx.ravel(), gy.ravel(), (gx**2 - gy**2).ravel()], axis=1)
    # crude normal estimate: normal to the surface points mostly +z
    normals = np.zeros_like(verts)
    normals[:, 2] = 1.0
    H, K = vertex_curvature(verts, _faces_for_grid(9), normals)
    si = shape_index(H, K)
    # center vertex (grid centre) should be near the saddle value 0
    c = 9 * 4 + 4
    assert abs(si[c]) < 0.1


def _faces_for_grid(n):
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            b = a + 1
            c = a + n
            d = c + 1
            faces.append([a, b, c])
            faces.append([b, d, c])
    return np.asarray(faces, dtype=np.int64)


def test_ddc_zero_on_flat_patch():
    patch_v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
    patch_n = np.tile(np.array([0.0, 0.0, 1.0]), (4, 1))
    patch_rho = np.array([0.0, 1.0, 1.0, np.sqrt(2.0)])
    ddc = compute_ddc(patch_v, patch_n, 0, patch_rho)
    np.testing.assert_allclose(ddc, 0.0, atol=1e-6)
