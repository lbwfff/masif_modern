"""Per-vertex chemical / geometric features.

Ported from the original ``computeCharges.py``, ``computeHydrophobicity.py`` and the
curvature section of ``read_data_from_surface.py``. APBS electrostatics is replaced by
a fast Coulombic proxy (see :func:`coulombic_potential`).

The five channels mirror the original MaSIF feature vector:

0. shape index          (from principal curvatures)
1. distance-dependent curvature  (computed per patch, in :mod:`masif.patches`)
2. hydrogen-bond potential       (donors/acceptors)
3. electrostatics charge         (Coulombic proxy)
4. hydrophobicity                (Kyte-Doolittle / 4.5)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from Bio.PDB.Residue import Residue

from masif.chemistry import (
    KD_SCALE,
    PARTIAL_CHARGES,
    HBOND_STD_DEV,
    ACCEPTOR_ANGLE_ATOM,
    ACCEPTOR_PLANE_ATOM,
    DONOR_ATOM,
    POLAR_HYDROGENS,
)
from masif.config import SurfaceConfig
from masif.surface.generate import MolecularSurface


# --------------------------------------------------------------------------- #
# Curvature / shape index
# --------------------------------------------------------------------------- #
def vertex_curvature(
    vertices: np.ndarray, faces: np.ndarray, normals: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the mean (H) and Gaussian (K) curvature at every vertex.

    Uses the classical local quadric fit in a tangent frame over the 1-ring,
    vectorised with a padded fixed-size neighbourhood and batched least squares.
    """
    n = vertices.shape[0]
    # 1-ring neighbourhood (padded, up to MAX_RING neighbours)
    ring = [set() for _ in range(n)]
    for f in faces:
        for i in range(3):
            ring[f[i]].add(int(f[(i + 1) % 3]))
            ring[f[i]].add(int(f[(i + 2) % 3]))
    MAX_RING = 12
    nb = np.full((n, MAX_RING), -1, dtype=np.int64)
    for i, r in enumerate(ring):
        r = sorted(r)
        k = min(len(r), MAX_RING)
        nb[i, :k] = r[:k]

    valid = nb >= 0
    nb_safe = np.where(valid, nb, 0)
    du = vertices[nb_safe] - vertices[:, None, :]  # (n, K, 3)
    nrm = normals
    # tangent frame (u, v) with w = normal
    u = np.array([_orthonormal(nrm[i]) for i in range(n)])  # (n, 3)
    v = np.cross(nrm, u)
    uu = np.einsum("nki,ni->nk", du, u)
    vv = np.einsum("nki,ni->nk", du, v)
    ww = np.einsum("nki,ni->nk", du, nrm)

    # design matrix A: w = a*u^2 + b*v^2 + c*u*v
    A = np.stack([uu * uu, vv * vv, uu * vv], axis=-1)  # (n, K, 3)
    A = np.where(valid[..., None], A, 0.0)
    ww = np.where(valid, ww, 0.0)

    AtA = np.einsum("nki,nkj->nij", A, A)  # (n, 3, 3)
    Atw = np.einsum("nki,nk->ni", A, ww)  # (n, 3)
    # batched solve; fall back to 0 where singular
    eye = np.eye(3)[None, :, :] * 1e-8
    try:
        coeff = np.linalg.solve(AtA + eye, Atw[..., None])[..., 0]  # (n, 3)
    except np.linalg.LinAlgError:
        coeff = np.linalg.pinv(AtA + eye) @ Atw[..., None]
        coeff = coeff[..., 0]
    singular = np.linalg.det(AtA) < 1e-12
    coeff = np.where(singular[:, None], 0.0, coeff)

    a, b, c = coeff[:, 0], coeff[:, 1], coeff[:, 2]
    # principal curvatures = eigenvalues of [[a, c/2], [c/2, b]]
    M = np.zeros((n, 2, 2))
    M[:, 0, 0] = a
    M[:, 0, 1] = c / 2.0
    M[:, 1, 0] = c / 2.0
    M[:, 1, 1] = b
    eig = np.linalg.eigvalsh(M)  # ascending
    k1, k2 = eig[:, 1], eig[:, 0]
    H = (k1 + k2) / 2.0
    K = k1 * k2
    return H, K


def _orthonormal(n: np.ndarray) -> np.ndarray:
    """A unit vector perpendicular to ``n``."""
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(n, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, ref)
    u /= np.linalg.norm(u)
    return u


def shape_index(H: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Shape index from principal curvatures (original formula)."""
    elem = np.square(H) - K
    elem[elem < 0] = 1e-8
    k1 = H + np.sqrt(elem)
    k2 = H - np.sqrt(elem)
    denom = k1 - k2
    si = np.zeros_like(k1)
    nz = np.abs(denom) > 1e-12  # flat regions (k1 == k2) -> shape index 0
    si[nz] = (k1[nz] + k2[nz]) / denom[nz]
    return np.arctan(si) * (2.0 / np.pi)


# --------------------------------------------------------------------------- #
# Hydrogen-bond potential
# --------------------------------------------------------------------------- #
def _angle_penalty(deviation: np.ndarray) -> np.ndarray:
    """Quadratic penalty for a hydrogen-bond angle deviation (original formula)."""
    return np.maximum(0.0, 1.0 - (deviation / HBOND_STD_DEV) ** 2)


def _atom_hbond_charge(atom_name: str, res: Residue, v: np.ndarray) -> float:
    """Hydrogen-bond donor/acceptor potential of a single surface point ``v``."""
    res_type = res.get_resname()
    # polar hydrogen -> donor
    if res_type in POLAR_HYDROGENS and atom_name in POLAR_HYDROGENS[res_type]:
        donor_name = DONOR_ATOM[atom_name]
        if donor_name not in res:
            return 0.0
        a = res[donor_name].get_coord()
        b = res[atom_name].get_coord()
        dev = _angle_deviation(a, b, v, np.pi)
        return 1.0 * _angle_penalty(dev)
    # acceptor (backbone O / side chain O / His N)
    if _is_acceptor(atom_name, res):
        if atom_name not in ACCEPTOR_ANGLE_ATOM:
            return 0.0
        b = res[atom_name].get_coord()
        a_name = ACCEPTOR_ANGLE_ATOM[atom_name]
        if a_name not in res:
            return 0.0
        a = res[a_name].get_coord()
        dev = _angle_deviation(a, b, v, 2 * np.pi / 3)
        angle_penalty = _angle_penalty(dev)
        plane_penalty = 1.0
        if atom_name in ACCEPTOR_PLANE_ATOM:
            p_name = ACCEPTOR_PLANE_ATOM[atom_name]
            if p_name in res:
                d = res[p_name].get_coord()
                plane_penalty = _angle_penalty(_plane_deviation(d, a, b, v))
        return -1.0 * angle_penalty * plane_penalty
    return 0.0


def _is_acceptor(atom_name: str, res: Residue) -> bool:
    if atom_name.startswith("O"):
        return True
    if res.get_resname() == "HIS":
        if atom_name == "ND1" and "HD1" not in res:
            return True
        if atom_name == "NE2" and "HE2" not in res:
            return True
    return False


def _angle_deviation(a, b, c, theta: float) -> float:
    import math

    va = a - b
    vc = c - b
    na = np.linalg.norm(va)
    nc = np.linalg.norm(vc)
    if na == 0 or nc == 0:
        return 0.0
    cosang = np.clip(np.dot(va, vc) / (na * nc), -1.0, 1.0)
    return abs(math.acos(cosang) - theta)


def _dihedral(a, b, c, d) -> float:
    """Signed dihedral angle of four points (same convention as Bio.PDB.calc_dihedral)."""
    v1 = b - a
    v2 = c - b
    v3 = d - c
    n1 = np.cross(v1, v2)
    n2 = np.cross(v2, v3)
    n1 = n1 / (np.linalg.norm(n1) + 1e-12)
    n2 = n2 / (np.linalg.norm(n2) + 1e-12)
    v2n = v2 / (np.linalg.norm(v2) + 1e-12)
    x = float(np.dot(n1, n2))
    y = float(np.dot(np.cross(n1, n2), v2n))
    return float(np.arctan2(y, x))


def _plane_deviation(a, b, c, d) -> float:
    """Deviation of point ``d`` from the plane defined by ``a,b,c`` (original formula)."""
    dih = _dihedral(a, b, c, d)
    dev1 = abs(dih)
    dev2 = np.pi - abs(dih)
    return min(dev1, dev2)


def compute_hbond_potential(surface: MolecularSurface) -> np.ndarray:
    """Hydrogen-bond potential for every vertex (from its nearest atom's residue)."""
    assert surface.atoms is not None
    atoms = surface.atoms
    idx = surface.atom_idx[:, 0]
    out = np.zeros(surface.n_vertices)
    for i in range(surface.n_vertices):
        ai = idx[i]
        out[i] = _atom_hbond_charge(atoms.atom_names[ai], atoms.residues[ai], surface.vertices[i])
    return out


# --------------------------------------------------------------------------- #
# Electrostatics (Coulombic proxy replacing APBS)
# --------------------------------------------------------------------------- #
def coulombic_potential(
    surface: MolecularSurface, cutoff: float = 15.0, dielectric: float = 4.0
) -> np.ndarray:
    """Electrostatic potential at each surface vertex.

    Approximates the Poisson-Boltzmann electrostatics used by the paper with a simple
    screened Coulombic sum over the protein's partial charges:

        V(v) = sum_i  q_i / (dielectric * (|r_i - v| + 0.5))

    This is a coarse but free-of-external-tools proxy; swap in a real PB solver if
    higher fidelity is required.
    """
    assert surface.atoms is not None
    atoms = surface.atoms
    q = np.zeros(atoms.coords.shape[0])
    for i, name in enumerate(atoms.atom_names):
        q[i] = PARTIAL_CHARGES.get(name, 0.0)

    # sparse pairwise distances (atom rows, vertex cols) within `cutoff`, vectorised
    from scipy.spatial import cKDTree

    atom_tree = cKDTree(atoms.coords)
    vert_tree = cKDTree(surface.vertices)
    sm = atom_tree.sparse_distance_matrix(vert_tree, cutoff).tocoo()  # (atom, vertex, dist)
    rows, cols = sm.row, sm.col
    d = sm.data
    weights = q[rows] / (dielectric * (d + 0.5))
    out = np.bincount(cols, weights=weights, minlength=surface.n_vertices)
    return out


def normalize_electrostatics(in_elec: np.ndarray) -> np.ndarray:
    """Clamp to [-3, 3] and rescale to [-1, 1] (original ``normalize_electrostatics``)."""
    elec = np.copy(in_elec)
    upper, lower = 3.0, -3.0
    elec[elec > upper] = upper
    elec[elec < lower] = lower
    elec = (elec - lower) / (upper - lower)
    return 2.0 * elec - 1.0


# --------------------------------------------------------------------------- #
# Hydrophobicity
# --------------------------------------------------------------------------- #
def compute_hydrophobicity(surface: MolecularSurface) -> np.ndarray:
    """Kyte-Doolittle hydropathy of the nearest atom's residue, / 4.5."""
    assert surface.atoms is not None
    atoms = surface.atoms
    idx = surface.atom_idx[:, 0]
    resnames = atoms.resnames[idx]
    out = np.zeros(surface.n_vertices)
    for i, rn in enumerate(resnames):
        out[i] = KD_SCALE.get(rn, 0.0)
    return out / 4.5


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
@dataclass
class VertexFeatures:
    """The five per-vertex feature channels plus interface labels."""

    shape_index: np.ndarray
    ddc: np.ndarray  # computed later per patch; zero-fill here
    hbond: np.ndarray
    charge: np.ndarray
    hphob: np.ndarray
    iface: np.ndarray

    def stack(self) -> np.ndarray:
        return np.stack([self.shape_index, self.ddc, self.hbond, self.charge, self.hphob], axis=1)


def compute_vertex_features(
    surface: MolecularSurface, config: Optional[SurfaceConfig] = None
) -> VertexFeatures:
    """Compute all five per-vertex features for a generated surface."""
    config = config or SurfaceConfig()
    H, K = vertex_curvature(surface.vertices, surface.faces, surface.normals)
    si = shape_index(H, K)
    hbond = compute_hbond_potential(surface)
    charge = normalize_electrostatics(coulombic_potential(surface))
    hphob = compute_hydrophobicity(surface)
    return VertexFeatures(
        shape_index=si,
        ddc=np.zeros(surface.n_vertices),
        hbond=hbond,
        charge=charge,
        hphob=hphob,
        iface=np.zeros(surface.n_vertices),
    )
