"""Geodesic distances and angular coordinates over a surface mesh.

This module replaces the original ``compute_polar_coordinates.py`` (networkx
all-pairs Dijkstra + per-patch MDS embedding), which was the main preprocessing
bottleneck:

* Geodesic distances use ``scipy.sparse.csgraph.dijkstra`` (compiled, chunked) instead
  of ``networkx``.
* Angular coordinates use a lightweight discrete-exponential-map-style propagation
  instead of a per-patch MDS solve: the 1-ring is embedded on the center tangent plane
  and outer vertices inherit a direction that is a distance-weighted blend of their
  already-embedded in-patch neighbours.

The angle is only required to be locally consistent: the network applies
``n_rotations`` rotations of ``theta`` and max-pools over them, which makes the final
descriptor invariant to the arbitrary global reference of ``theta``.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

logger = logging.getLogger(__name__)


def build_adjacency(faces: np.ndarray, n_vertices: int) -> list[np.ndarray]:
    """Undirected 1-ring adjacency list built from the face index array."""
    n = len(faces)
    row = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2], faces[:, 1], faces[:, 2], faces[:, 0]])
    col = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0], faces[:, 0], faces[:, 1], faces[:, 2]])
    mask = row != col
    row, col = row[mask], col[mask]
    adj = [[] for _ in range(n_vertices)]
    for r, c in zip(row.tolist(), col.tolist()):
        adj[r].append(c)
    return [np.asarray(a, dtype=np.int64) for a in adj]


def geodesic_distances(
    vertices: np.ndarray,
    faces: np.ndarray,
    radius: float,
    chunk_size: int = 512,
) -> sp.csr_matrix:
    """Pairwise geodesic distances truncated at ``radius``.

    Returns a sparse CSR matrix ``D`` where ``D[i, j]`` is the geodesic distance from
    ``i`` to ``j`` (0 when ``j`` is not within ``radius`` of ``i``). ``D[i, i]`` is set
    to a small positive value to mirror the original implementation.
    """
    n = vertices.shape[0]
    row = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2], faces[:, 1], faces[:, 2], faces[:, 0]])
    col = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0], faces[:, 0], faces[:, 1], faces[:, 2]])
    mask = row != col
    row, col = row[mask], col[mask]
    edge_w = np.linalg.norm(vertices[row] - vertices[col], axis=1)
    graph = sp.coo_matrix((edge_w, (row, col)), shape=(n, n)).tocsr()

    data, rows, cols = [], [], []
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        idx = np.arange(start, stop)
        dist = dijkstra(graph, directed=False, indices=idx, limit=radius)
        dist = np.asarray(dist)
        for k, src in enumerate(idx):
            d = dist[k]
            nz = np.where((d > 0) & (d <= radius))[0]
            rows.extend([src] * len(nz))
            cols.extend(nz.tolist())
            data.extend(d[nz].tolist())
        # keep the center itself at a tiny distance (as the original D matrix did)
        rows.extend(idx.tolist())
        cols.extend(idx.tolist())
        data.extend([1e-8] * len(idx))

    D = sp.coo_matrix((np.asarray(data), (np.asarray(rows), np.asarray(cols))), shape=(n, n))
    D.sum_duplicates()
    return D.tocsr()


def compute_polar_coordinates(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    radius: float,
    max_vertices: int = 100,
    max_distance: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute polar coordinates for every patch.

    Returns ``(rho, theta, neigh_indices, mask)``:
    * ``rho``        ``(n, max_vertices)`` geodesic distance to the center (ascending)
    * ``theta``      ``(n, max_vertices)`` angle in ``[0, 2*pi)``
    * ``neigh_indices`` ``(n, max_vertices)`` int, padded vertex indices
    * ``mask``       ``(n, max_vertices)`` 1.0 where the patch entry is valid

    ``max_distance`` mirrors the original ``radius`` for the neural network patch;
    if not given, ``radius`` is used.
    """
    if max_distance is None:
        max_distance = radius
    n = vertices.shape[0]
    D = geodesic_distances(vertices, faces, radius=radius)
    adj = build_adjacency(faces, n)

    rho = np.zeros((n, max_vertices))
    theta = np.zeros((n, max_vertices))
    mask = np.zeros((n, max_vertices))
    neigh = np.zeros((n, max_vertices), dtype=np.int64)

    # Pre-allocate temporary arrays for per-center propagation.
    import time

    t0 = time.time()
    for i in range(n):
        if i % 500 == 0 and i > 0:
            logger.info(
                "  polar coords: %d/%d vertices (%.1fs)", i, n, time.time() - t0
            )
        d_row = D.getrow(i).toarray().ravel()
        nz = np.where((d_row > 0) & (d_row <= max_distance))[0]
        # sort by distance ascending
        order = np.argsort(d_row[nz], kind="stable")
        nz = nz[order]
        # include center as member 0
        members = np.concatenate([[i], nz])
        # cap the patch size
        if len(members) > max_vertices:
            members = members[:max_vertices]
        rho_vals = np.concatenate([[0.0], d_row[nz][: max(0, len(members) - 1)]])

        th = _propagate_angles(i, members, rho_vals, vertices, normals, adj)

        k = len(members)
        neigh[i, :k] = members
        rho[i, :k] = rho_vals
        theta[i, :k] = th
        mask[i, :k] = 1.0

    theta[theta < 0] += 2 * np.pi
    theta[theta > 2 * np.pi] -= 2 * np.pi
    return rho, theta, neigh, mask


def _propagate_angles(
    center: int,
    members: np.ndarray,
    rho_vals: np.ndarray,
    vertices: np.ndarray,
    normals: np.ndarray,
    adj: list[np.ndarray],
) -> np.ndarray:
    """Angles of all patch members around ``center`` via direction propagation.

    Members are processed in ascending geodesic distance. The 1-ring (direct mesh
    neighbours of the center) is embedded by projecting onto the center's tangent
    plane; deeper vertices inherit a direction that is the weighted blend of the
    already-embedded neighbours inside the patch.
    """
    n = len(members)
    out = np.zeros(n)
    if n <= 1:
        return out

    member_set = set(members.tolist())
    tangent_u = _orthonormal(normals[center])
    tangent_v = np.cross(normals[center], tangent_u)
    center_pos = vertices[center]

    # dir[c] = (dx, dy) unit direction of member c in the tangent plane (or None)
    dirs = {center: (1.0, 0.0)}  # center acts as anchor for 1-ring embedding
    for idx in range(1, n):
        v = int(members[idx])
        rv = rho_vals[idx]
        nbrs = [p for p in adj[v] if p in member_set]
        embedded = [p for p in nbrs if p in dirs and p != center]
        # 1-ring of center: project onto tangent plane
        if center in nbrs and len(embedded) == 0:
            d = vertices[v] - center_pos
            out[idx] = np.arctan2(np.dot(d, tangent_v), np.dot(d, tangent_u))
            dirs[v] = (np.cos(out[idx]), np.sin(out[idx]))
            continue
        if not embedded:
            # disconnected member (should be rare) -> keep previous angle
            if idx > 1:
                out[idx] = out[idx - 1]
            continue
        # weighted blend of parent directions
        sx = sy = 0.0
        for p in embedded:
            w = 1.0 / (rv - _rho_of(p, members, rho_vals) + 1e-3)
            dx, dy = dirs[p]
            sx += w * dx
            sy += w * dy
        norm = np.hypot(sx, sy)
        if norm < 1e-8:
            out[idx] = out[idx - 1] if idx > 1 else 0.0
        else:
            out[idx] = np.arctan2(sy / norm, sx / norm)
        dirs[v] = (sx / norm, sy / norm)
    return out


def _rho_of(v: int, members: np.ndarray, rho_vals: np.ndarray) -> float:
    pos = int(np.where(members == v)[0][0])
    return float(rho_vals[pos])


def _orthonormal(n: np.ndarray) -> np.ndarray:
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(n, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, ref)
    u /= np.linalg.norm(u)
    return u
