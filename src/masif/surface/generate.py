"""Molecular surface generation without external binaries.

Replaces the original ``MSMS + PyMesh`` stack with a pure-Python pipeline:

1. Voxelise the fused vdW spheres of the protein atoms into an occupancy grid.
2. Compute the exact Euclidean distance transform of the complement (``scipy`` or the
   optional ``edt`` package) to obtain a distance-from-surface field.
3. Extract the solvent-accessible surface (isosurface at vdW + probe) with Marching
   Cubes (``scikit-image``).
4. Associate each surface vertex with its nearest atom(s) via a KD-tree so chemical
   features can be interpolated.

This approximates the MSMS solvent-excluded surface for feature computation purposes;
it is far easier to install and parallelise than the original dependency chain.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from Bio.PDB import Structure
from scipy.ndimage import distance_transform_edt

from masif.chemistry import ELEMENT_RADIUS
from masif.config import SurfaceConfig

try:
    import edt  # type: ignore
    _HAS_EDT = True
except Exception:  # pragma: no cover
    _HAS_EDT = False

logger = logging.getLogger(__name__)


@dataclass
class AtomTable:
    """Flat, numpy-friendly table of the protein atoms used for feature mapping."""

    coords: np.ndarray  # (M, 3)
    atom_names: np.ndarray  # (M,) str
    resnames: np.ndarray  # (M,) str
    elements: np.ndarray  # (M,) str
    chain_ids: np.ndarray  # (M,) str
    residues: list  # (M,) BioPython Residue objects, aligned with ``coords``


def atom_table(structure: Structure, include_hetatm: bool = False) -> AtomTable:
    """Build an :class:`AtomTable` from a BioPython structure.

    Only heavy atoms whose element has a vdW radius are retained; this matches the
    original ``xyzrn`` behaviour of ignoring water/HETATM for the surface.
    """
    coords, names, resnames, elements, chains, residues = [], [], [], [], [], []
    for atom in structure.get_atoms():
        if (not include_hetatm) and atom.get_parent().id[0] != " ":
            continue
        element = atom.element
        if element not in ELEMENT_RADIUS:
            continue
        coords.append(atom.get_coord())
        names.append(atom.get_name())
        resnames.append(atom.get_parent().get_resname())
        elements.append(element)
        chains.append(atom.get_parent().get_parent().get_id())
        residues.append(atom.get_parent())
    return AtomTable(
        coords=np.asarray(coords, dtype=np.float32),
        atom_names=np.asarray(names, dtype=object),
        resnames=np.asarray(resnames, dtype=object),
        elements=np.asarray(elements, dtype=object),
        chain_ids=np.asarray(chains, dtype=object),
        residues=residues,
    )


@dataclass
class MolecularSurface:
    """Triangular molecular surface with per-vertex atom association."""

    vertices: np.ndarray  # (N, 3)
    faces: np.ndarray  # (M, 3) int
    normals: np.ndarray  # (N, 3), unit, outward
    atom_idx: np.ndarray  # (N, k) int -> row index into an AtomTable
    atom_weights: np.ndarray  # (N, k) float, inverse-distance interpolation weights
    atoms: Optional[AtomTable] = None  # the AtomTable the indices refer to

    @property
    def n_vertices(self) -> int:
        return self.vertices.shape[0]


def _mark_occupied(atom_coords: np.ndarray, elements: np.ndarray, grid: dict) -> np.ndarray:
    """Return a boolean occupancy grid (True = inside any vdW sphere)."""
    shape = grid["shape"]
    occupied = np.zeros(shape, dtype=bool)
    origin = grid["origin"]
    spacing = grid["spacing"]
    for i in range(atom_coords.shape[0]):
        c = atom_coords[i]
        r = ELEMENT_RADIUS[elements[i]]
        # voxel index range covering the atom sphere
        imin = np.floor((c - r - origin) / spacing).astype(int)
        imax = np.ceil((c + r - origin) / spacing).astype(int) + 1
        imin = np.clip(imin, 0, None)
        imax = np.clip(imax, None, np.array(shape))
        if np.any(imax <= imin):
            continue
        ix = np.arange(imin[0], imax[0])
        iy = np.arange(imin[1], imax[1])
        iz = np.arange(imin[2], imax[2])
        nx, ny, nz = len(ix), len(iy), len(iz)
        if nx * ny * nz == 0:
            continue
        gx, gy, gz = np.meshgrid(ix, iy, iz, indexing="ij")
        coords = np.stack(
            [origin[0] + gx * spacing, origin[1] + gy * spacing, origin[2] + gz * spacing],
            axis=-1,
        ).reshape(-1, 3)
        inside = np.sum((coords - c) ** 2, axis=1) <= r * r
        filled = inside.reshape(nx, ny, nz)
        occupied[ix[:, None, None], iy[None, :, None], iz[None, None, :]] |= filled
    return occupied


def _distance_field(occupied: np.ndarray, config: SurfaceConfig) -> np.ndarray:
    """Distance of each outside voxel to the nearest occupied (atom) voxel."""
    complement = ~occupied
    if config.prefer_edt and _HAS_EDT:
        return edt.edt(complement.astype(np.int8))
    return distance_transform_edt(complement)


def _fix_outward_normals(vertices: np.ndarray, normals: np.ndarray) -> np.ndarray:
    centroid = vertices.mean(axis=0)
    inward = np.einsum("ij,ij->i", vertices - centroid, normals) < 0
    normals = normals.copy()
    normals[inward] = -normals[inward]
    return normals


def _associate_atoms(
    vertices: np.ndarray, atoms: AtomTable, k: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """Find the k nearest atoms to each vertex and inverse-distance weights."""
    from sklearn.neighbors import KDTree

    kdt = KDTree(atoms.coords)
    dists, idx = kdt.query(vertices.astype(np.float64), k=k)
    dists = np.square(dists)  # square distances, as in assignChargesToNewMesh
    weights = np.zeros_like(dists)
    for i in range(vertices.shape[0]):
        d = dists[i]
        if d[0] == 0.0:
            weights[i, 0] = 1.0
            continue
        inv = 1.0 / d
        weights[i] = inv / inv.sum()
    return idx, weights


def generate_surface(
    structure: Structure,
    config: Optional[SurfaceConfig] = None,
    include_hetatm: bool = False,
) -> MolecularSurface:
    """Generate the solvent-accessible surface of a protein structure.

    Returns a :class:`MolecularSurface` with vertices, faces, outward normals and
    per-vertex nearest-atom interpolation weights.
    """
    config = config or SurfaceConfig()
    atoms = atom_table(structure, include_hetatm=include_hetatm)
    coords = atoms.coords
    if coords.shape[0] == 0:
        raise ValueError("no usable protein atoms found in structure")

    spacing = config.voxel_size
    probe = config.probe_radius
    pad = config.padding
    max_radius = max(ELEMENT_RADIUS.values())

    lo = coords.min(axis=0) - (max_radius + probe + pad)
    hi = coords.max(axis=0) + (max_radius + probe + pad)
    shape = np.ceil((hi - lo) / spacing).astype(int) + 1
    grid = {"shape": shape, "origin": lo, "spacing": spacing}

    occupied = _mark_occupied(coords, atoms.elements, grid)
    dist = _distance_field(occupied, config)

    from skimage import measure

    # Signal-to-noise: remove small grid-cell noise before the isosurface.
    verts, faces, normals_np, _ = measure.marching_cubes(
        dist, level=float(probe), spacing=(spacing, spacing, spacing)
    )
    verts = verts + lo  # back to real coordinates
    normals = _fix_outward_normals(verts, normals_np)

    idx, weights = _associate_atoms(verts, atoms, k=4)
    return MolecularSurface(
        vertices=verts.astype(np.float32),
        faces=faces.astype(np.int64),
        normals=normals.astype(np.float32),
        atom_idx=idx,
        atom_weights=weights,
        atoms=atoms,
    )