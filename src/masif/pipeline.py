"""End-to-end data preparation pipeline for one protein (chain).

Replaces the original sequence of ``00-pdb_download``, ``01-pdb_extract_and_triangulate``
and ``04-masif_precompute`` with a single function that produces:

* the chain PDB (protonated + chain-extracted),
* a surface PLY (for visualization),
* the precomputed patch ``.npz`` used by the dataset.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import trimesh

from masif.config import MaSIFConfig
from masif.patches import ProteinPatches, assemble_patches, compute_interface_labels
from masif.surface.features import compute_vertex_features
from masif.surface.generate import generate_surface
from masif.surface.pdb import extract_chains, load_pdb, protonate, write_pdb

logger = logging.getLogger(__name__)


def _ply_name(cfg: MaSIFConfig, pdb_id: str, chain_id: str) -> Path:
    return Path(cfg.paths.root) / cfg.paths.ply_chain_dir / f"{pdb_id}_{chain_id}.ply"


def prepare_protein(
    pdb_id: str,
    chain_ids: str,
    cfg: MaSIFConfig,
    raw_pdb: str | Path,
    out_prefix: str,
    partner_chain_ids: str = "",
    save_surface: bool = True,
    complex_surface=None,
    structure=None,
) -> tuple:
    """Prepare one protein chain end-to-end.

    Args:
        pdb_id: PDB identifier used for file names.
        chain_ids: chain(s) whose surface/patch is computed (e.g. ``"A"``).
        cfg: configuration bundle.
        raw_pdb: path to the raw (unprotonated) PDB.
        out_prefix: prefix for the chain pdb / patch files.
        partner_chain_ids: chains treated as the interaction partner (used only to
            compute interface labels). Empty => no ground truth.
        complex_surface: precomputed surface of the full complex (``chain_ids +
            partner_chain_ids``). If ``None`` and a partner is given, it is generated
            internally. Reusing a shared complex surface avoids regenerating it for the
            partner chain.
        structure: pre-loaded, protonated BioPython structure. If ``None``, ``raw_pdb``
            is loaded and protonated internally. Passing a shared structure avoids
            re-reading the PDB for the partner chain.

    Returns:
        ``(patches, complex_surface)`` where ``complex_surface`` is the complex surface
        used for interface labels (``None`` if no partner was given).
    """
    root = cfg.paths.root
    if structure is None:
        structure = load_pdb(raw_pdb)
        protonate(structure)
    struct = structure
    chain_struct = extract_chains(struct, chain_ids)
    logger.info("[%s] surface generation ...", out_prefix)
    # 1. chain PDB (protonated, chain-extracted)
    pdb_out = Path(root) / cfg.paths.pdb_chain_dir / f"{out_prefix}.pdb"
    write_pdb(chain_struct, pdb_out)

    # 2. surface of the chain
    surface = generate_surface(chain_struct, cfg.surface)
    logger.info("[%s] surface done: %d vertices", out_prefix, surface.n_vertices)
    if save_surface:
        _save_ply(surface, _ply_name(cfg, pdb_id, chain_ids))

    # 3. per-vertex features
    logger.info("[%s] computing vertex features ...", out_prefix)
    features = compute_vertex_features(surface, cfg.surface)

    # 4. interface ground truth from the full complex surface
    labels = None
    if partner_chain_ids:
        logger.info("[%s] computing interface labels ...", out_prefix)
        if complex_surface is None:
            complex_struct = extract_chains(struct, chain_ids + partner_chain_ids)
            complex_surface = generate_surface(complex_struct, cfg.surface)
        labels = compute_interface_labels(surface, complex_surface.vertices)
        features.iface = labels

    # 5. patch decomposition
    logger.info("[%s] assembling patches (polar coords) ...", out_prefix)
    patches = assemble_patches(
        surface,
        features,
        pdb_id=pdb_id,
        chain_id=chain_ids,
        radius=cfg.patch.radius,
        max_vertices=cfg.patch.max_vertices,
        labels=labels,
    )

    patch_out = Path(root) / cfg.paths.precomputation_dir / f"{out_prefix}.npz"
    patches.save(patch_out)
    logger.info("prepared %s -> %d vertices", out_prefix, patches.n_vertices)
    return patches, complex_surface


def _save_ply(surface, filename: str | Path) -> None:
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.Trimesh(
        vertices=surface.vertices,
        faces=surface.faces,
        vertex_normals=surface.normals,
        process=False,
    )
    mesh.export(filename)
