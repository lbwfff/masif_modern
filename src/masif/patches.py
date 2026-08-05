"""Per-protein patch assembly and PyTorch dataset.

Ports ``read_data_from_surface.py`` (patch building, DDC computation, interface
labels) and adds a PyTorch ``Dataset``/collate layer that feeds the model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from masif.geometry.polar import compute_polar_coordinates
from masif.surface.features import VertexFeatures, compute_vertex_features
from masif.surface.generate import MolecularSurface

logger = logging.getLogger(__name__)

N_FEATURES = 5
_CHANNEL_NAMES = ["shape_index", "ddc", "hbond", "charge", "hphob"]


# --------------------------------------------------------------------------- #
# Distance dependent curvature (ported from read_data_from_surface.py)
# --------------------------------------------------------------------------- #
def mean_normal_center_patch(D: np.ndarray, n: np.ndarray, r: float) -> np.ndarray:
    """Mean normal of vertices within ``r`` of the patch center."""
    c_normal = [n[i] for i in range(len(D)) if D[i] <= r]
    if len(c_normal) == 0:
        return n[0] / np.linalg.norm(n[0])
    mean_normal = np.mean(c_normal, axis=0)
    norm = np.linalg.norm(mean_normal)
    return mean_normal / norm if norm > 0 else n[0]


def compute_ddc(
    patch_v: np.ndarray, patch_n: np.ndarray, patch_cp: int, patch_rho: np.ndarray
) -> np.ndarray:
    """Distance-dependent curvature (Yin et al. PNAS 2009), vectorised.

    ``patch_v``: patch vertex coordinates; ``patch_n``: their normals;
    ``patch_cp``: index of the central point; ``patch_rho``: geodesic distances to the
    center. Returns one DDC value per patch member.
    """
    n = patch_n
    r = patch_v
    i = patch_cp
    ni = mean_normal_center_patch(patch_rho, n, 2.5)
    dij = np.linalg.norm(r - r[i], axis=1)
    sf = np.linalg.norm(r + n - (ni + r[i]), axis=1) - dij
    sf[sf > 0] = 1
    sf[sf < 0] = -1
    dij[dij == 0] = 1e-8
    kij = sf * np.linalg.norm(n - ni, axis=1) / dij
    kij[(kij > 0.7) | (kij < -0.7)] = 0
    return kij


# --------------------------------------------------------------------------- #
# Interface labels
# --------------------------------------------------------------------------- #
def compute_interface_labels(
    surface: MolecularSurface, complex_vertices: np.ndarray, cutoff_sq: float = 2.0
) -> np.ndarray:
    """Interface ground truth.

    A vertex of the chain surface is on the interface if it is far from the surface of
    the full complex (buried by the partner), i.e. if its distance to the nearest
    complex-surface vertex squared exceeds ``cutoff_sq`` (mirrors the original code).
    """
    from scipy.spatial import cKDTree

    kdt = cKDTree(complex_vertices)
    d, _ = kdt.query(surface.vertices)
    d = np.square(d)
    return (d >= cutoff_sq).astype(np.float32)


# --------------------------------------------------------------------------- #
# Patch assembly
# --------------------------------------------------------------------------- #
@dataclass
class ProteinPatches:
    """Precomputed per-vertex patches for one protein."""

    pdb_id: str
    chain_id: str
    input_feat: np.ndarray  # (N, maxv, 5)
    rho: np.ndarray  # (N, maxv)
    theta: np.ndarray  # (N, maxv)
    mask: np.ndarray  # (N, maxv)
    indices: np.ndarray  # (N, maxv) int
    labels: np.ndarray  # (N,) float, interface ground truth
    vertices: np.ndarray  # (N, 3)

    @property
    def n_vertices(self) -> int:
        return self.labels.shape[0]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            input_feat=self.input_feat,
            rho=self.rho,
            theta=self.theta,
            mask=self.mask,
            indices=self.indices,
            labels=self.labels,
            vertices=self.vertices,
            pdb_id=self.pdb_id,
            chain_id=self.chain_id,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ProteinPatches":
        z = np.load(path, allow_pickle=True)
        return cls(
            pdb_id=str(z["pdb_id"]),
            chain_id=str(z["chain_id"]),
            input_feat=z["input_feat"],
            rho=z["rho"],
            theta=z["theta"],
            mask=z["mask"],
            indices=z["indices"],
            labels=z["labels"].astype(np.float32),
            vertices=z["vertices"],
        )


def assemble_patches(
    surface: MolecularSurface,
    features: VertexFeatures,
    pdb_id: str,
    chain_id: str,
    radius: float,
    max_vertices: int,
    labels: Optional[np.ndarray] = None,
) -> ProteinPatches:
    """Decompose a surface into patches with the 5-channel feature tensor.

    This is the modern replacement for ``read_data_from_surface``: polar coordinates
    come from :func:`masif.geometry.polar.compute_polar_coordinates` and the DDC
    channel is filled here from patch geometry.
    """
    rho, theta, neigh, mask = compute_polar_coordinates(
        surface.vertices, surface.faces, surface.normals, radius=radius, max_vertices=max_vertices
    )
    n = surface.n_vertices
    input_feat = np.zeros((n, max_vertices, N_FEATURES), dtype=np.float32)

    per_vertex = np.stack(
        [features.shape_index, features.ddc, features.hbond, features.charge, features.hphob], axis=1
    )  # (N, 5)

    # channel 1 (DDC) depends on the patch; compute it per center.
    for vix in range(n):
        members = neigh[vix]
        k = int(mask[vix].sum())
        if k == 0:
            continue
        patch_v = surface.vertices[members[:k]]
        patch_n = surface.normals[members[:k]]
        patch_rho = rho[vix][:k]
        ddc = compute_ddc(patch_v, patch_n, 0, patch_rho)
        input_feat[vix, :k, 1] = ddc
        for c in (0, 2, 3, 4):
            input_feat[vix, :k, c] = per_vertex[members[:k], c]

    if labels is None:
        labels = np.zeros(n, dtype=np.float32)
    return ProteinPatches(
        pdb_id=pdb_id,
        chain_id=chain_id,
        input_feat=input_feat,
        rho=rho,
        theta=theta,
        mask=mask,
        indices=neigh,
        labels=labels.astype(np.float32),
        vertices=surface.vertices.copy(),
    )


# --------------------------------------------------------------------------- #
# PyTorch dataset
# --------------------------------------------------------------------------- #
def remap_indices(indices: np.ndarray, subset: np.ndarray) -> np.ndarray:
    """Remap global vertex indices to row positions within ``subset``.

    Neighbours not present in ``subset`` are mapped to 0 (they are excluded by the
    mask at model time).
    """
    n_total = int(indices.max()) + 1
    pos = np.full(n_total, -1, dtype=np.int64)
    pos[subset] = np.arange(len(subset))
    out = pos[indices]
    out[out < 0] = 0
    return out


class SiteDataset:
    """Map-style dataset over a directory of precomputed ``ProteinPatches`` npz files."""

    def __init__(self, precomputation_dir: str | Path, sample_proteins=None):
        precomputation_dir = Path(precomputation_dir)
        self.files = sorted(precomputation_dir.glob("*.npz"))
        if sample_proteins is not None:
            ids = set(sample_proteins)
            self.files = [f for f in self.files if f.stem in ids]
        logger.info("SiteDataset: %d proteins under %s", len(self.files), precomputation_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> ProteinPatches:
        return ProteinPatches.load(self.files[idx])


def build_batch(
    patches: ProteinPatches,
    max_batch: int,
    seed: int = 0,
) -> dict:
    """Build a training batch from one protein by sampling balanced pos/neg vertices.

    Returns tensors with ``B = min(2 * n_pos, 2 * n_neg, max_batch)`` centers:
    ``input_feat (B, V, 5)``, ``rho (B, V)``, ``theta (B, V)``, ``mask (B, V)``,
    ``indices (B, V)`` (remapped to batch rows) and ``labels (B,)``.
    """
    import torch

    rng = np.random.default_rng(seed)
    labels = patches.labels
    pos = np.where(labels == 1.0)[0]
    neg = np.where(labels == 0.0)[0]
    rng.shuffle(neg)
    rng.shuffle(pos)
    n = min(len(pos), len(neg), max_batch // 2)
    if n == 0:
        # degenerate protein: fall back to whatever is available
        n = min(len(labels), max_batch)
        subset = rng.choice(len(labels), size=n, replace=False)
    else:
        subset = np.concatenate([pos[:n], neg[:n]])

    feats = patches.input_feat[subset]
    rho = patches.rho[subset]
    theta = patches.theta[subset]
    mask = patches.mask[subset]
    indices = remap_indices(patches.indices, subset)
    labels_sub = labels[subset]

    B, V = feats.shape[0], feats.shape[1]
    return {
        "input_feat": torch.from_numpy(feats).float(),
        "rho": torch.from_numpy(rho).float(),
        "theta": torch.from_numpy(theta).float(),
        "mask": torch.from_numpy(mask).float().unsqueeze(-1),
        "indices": torch.from_numpy(indices).long(),
        "labels": torch.from_numpy(labels_sub).float(),
        "B": B,
        "V": V,
    }
