"""Typed configuration for MaSIF, replacing the original mutable ``masif_opts`` dict.

Every option is a dataclass field with a documented default, so configuration is
self-describing, validated, and can be overridden per-application without mutating
a shared global dict (the original ``default_config.masif_opts``).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class SurfaceConfig:
    """Parameters controlling molecular surface generation."""

    #: Voxel size (Angstrom) of the occupancy grid used by the surface generator.
    #: 1.0 roughly matches the original mesh resolution (~1 vertex per A^2);
    #: smaller values (0.5) yield much finer meshes and are slow.
    voxel_size: float = 1.0
    #: Probe radius (Angstrom); the isosurface is drawn at vdW + probe (solvent accessible surface).
    probe_radius: float = 1.4
    #: Pad the bounding box by this margin (Angstrom) around the atoms.
    padding: float = 4.0
    #: If the ``edt`` package is importable, use it (faster) instead of scipy.ndimage.
    prefer_edt: bool = True
    #: Only keep atoms within this radius of the surface when assigning per-vertex features.
    feature_radius: float = 4.0
    #: Interpolate features from k nearest atoms (True) or take the single nearest atom.
    feature_interpolation: bool = True


@dataclass
class PatchConfig:
    """Patch decomposition parameters (radius/angular coords)."""

    #: Geodesic radius of each patch (Angstrom).
    radius: float = 9.0
    #: Maximum number of vertices retained in a patch.
    max_vertices: int = 100
    #: Number of grid rings for rho in the geodesic convolution.
    n_rhos: int = 5
    #: Number of angular sectors for theta in the geodesic convolution.
    n_thetas: int = 16
    #: Rotations of the theta coordinate used for the rotation-invariant pooling.
    n_rotations: int = 16
    #: Fallback geodesic cutoff used when building the mesh graph (>= radius).
    graph_cutoff: float = 0.0  # 0 -> radius


@dataclass
class ModelConfig:
    """Neural network hyper-parameters (MaSIF-site)."""

    #: Number of stacked geodesic-convolution layers.
    n_conv_layers: int = 3
    #: Binary mask over the 5 input features (shape_index, ddc, hbond, charge, hphob).
    feat_mask: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0)
    #: Hidden size used by the refinement MLP.
    hidden_size: int = 64
    learning_rate: float = 1e-3
    dropout: float = 0.0
    weight_decay: float = 0.0


@dataclass
class TrainConfig:
    """Training loop parameters."""

    batch_size: int = 100
    epochs: int = 1
    seed: int = 0
    num_workers: int = 0
    log_every: int = 10
    #: Fraction of precomputed proteins held out as validation.
    val_fraction: float = 0.1
    #: Skip proteins whose interface covers more than this fraction of the surface.
    max_pos_frac: float = 0.75
    #: Skip proteins with fewer than this many positive interface vertices.
    min_pos_vertices: int = 30
    #: Skip proteins with more than this many surface vertices.
    max_vertices: int = 8000


@dataclass
class Paths:
    """Directory layout for a given application (e.g. ``masif_site``)."""

    root: Path
    #: Where raw downloaded PDB files live.
    raw_pdb_dir: str = "data_preparation/00-raw_pdbs"
    #: Where chain-extracted / protonated PDB files live.
    pdb_chain_dir: str = "data_preparation/01-benchmark_pdbs"
    #: Where generated surface PLY files live.
    ply_chain_dir: str = "data_preparation/01-benchmark_surfaces"
    #: Where the per-protein patch tensors (npz) live.
    precomputation_dir: str = "data_preparation/04-precomputation"
    #: Where model checkpoints and logs are written.
    model_dir: str = "nn_models/site"
    #: Where per-protein prediction npy files are written.
    out_pred_dir: str = "output/pred_data"
    #: Where colored surface PLY files are written.
    out_surf_dir: str = "output/pred_surfaces"
    #: Training protein list (one ``PDBID_CHAIN`` per line).
    training_list: str = "lists/training.txt"
    #: Testing protein list.
    testing_list: str = "lists/testing.txt"

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser().resolve()


@dataclass
class MaSIFConfig:
    """Top-level configuration bundle."""

    surface: SurfaceConfig = field(default_factory=SurfaceConfig)
    patch: PatchConfig = field(default_factory=PatchConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    paths: Paths = field(default_factory=lambda: Paths(Path(".")))

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "MaSIFConfig":
        out = cls()
        for section in ("surface", "patch", "model", "train", "paths"):
            if section in d:
                setattr(
                    out,
                    section,
                    dataclasses.replace(getattr(out, section), **d[section]),
                )
        return out
