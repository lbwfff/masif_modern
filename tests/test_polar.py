"""Tests for patch geometry (polar coordinates)."""

import numpy as np
import pytest

from masif.geometry.polar import compute_polar_coordinates, geodesic_distances


def _flat_grid_mesh(n=6, spacing=1.0):
    """A regular triangulated grid in the z=0 plane."""
    xs = np.arange(n) * spacing
    ys = np.arange(n) * spacing
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    verts = np.stack([gx.ravel(), gy.ravel(), np.zeros(n * n)], axis=1).astype(np.float32)
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            b = a + 1
            c = a + n
            d = c + 1
            faces.append([a, b, c])
            faces.append([b, d, c])
    return verts.astype(np.float32), np.asarray(faces, dtype=np.int64)


def test_geodesic_distances_symmetric_positive():
    verts, faces = _flat_grid_mesh()
    D = geodesic_distances(verts, faces, radius=3.0)
    Dd = D.toarray()
    assert Dd.shape == (verts.shape[0], verts.shape[0])
    # symmetry
    np.testing.assert_allclose(Dd, Dd.T, atol=1e-5)
    # center-to-center small positive values on the diagonal
    assert np.all(np.diag(Dd) == 1e-8)


def test_polar_coordinates_shapes_and_sorting():
    verts, faces = _flat_grid_mesh(n=6)
    normals = np.zeros_like(verts)
    normals[:, 2] = 1.0
    rho, theta, neigh, mask = compute_polar_coordinates(
        verts, faces, normals, radius=3.0, max_vertices=10
    )
    n = verts.shape[0]
    assert rho.shape == (n, 10)
    assert theta.shape == (n, 10)
    assert neigh.shape == (n, 10)
    assert mask.shape == (n, 10)

    # binary mask, center first, rho ascending per row
    assert set(np.unique(mask)) <= {0.0, 1.0}
    for i in range(n):
        k = int(mask[i].sum())
        assert k >= 1
        assert neigh[i, 0] == i
        assert rho[i, 0] == 0.0
        assert np.all(np.diff(rho[i, :k]) >= -1e-6)
        assert np.all(theta[i, :k] >= 0.0) and np.all(theta[i, :k] <= 2 * np.pi + 1e-6)


def test_polar_patch_contains_geodesic_neighbors():
    verts, faces = _flat_grid_mesh(n=6)
    normals = np.zeros_like(verts)
    normals[:, 2] = 1.0
    rho, _, neigh, mask = compute_polar_coordinates(
        verts, faces, normals, radius=2.0, max_vertices=50
    )
    # center (0,0) at index 0 -> within radius 2 its patch includes (0,1) and (1,0)
    k = int(mask[0].sum())
    members = set(neigh[0, :k].tolist())
    assert 1 in members  # (0,1)
    assert 6 in members  # (1,0)
