"""``masif`` command line interface.

Subcommands:

* ``prepare``  PDB -> surface -> features -> patches (.npz)
* ``train``    train MaSIF-site on precomputed patches
* ``predict``  score every vertex of precomputed proteins
* ``color``    write a coloured PLY from predictions

Configuration can be given as a JSON file (``--config``) or individual flags; missing
fields fall back to :class:`masif.config.MaSIFConfig` defaults.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from masif.config import MaSIFConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("masif")


def _config_from_args(args) -> MaSIFConfig:
    cfg = MaSIFConfig()
    if args.config:
        cfg = MaSIFConfig.from_dict(json.loads(Path(args.config).read_text()))
    if getattr(args, "root", None):
        cfg.paths.root = args.root
    return cfg


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=str, default=".", help="application data root")
    parser.add_argument("--config", type=str, default=None, help="JSON config file")


def cmd_prepare(args) -> None:
    from masif.pipeline import prepare_protein
    from masif.surface.pdb import load_pdb, protonate

    cfg = _config_from_args(args)
    for name in args.proteins:
        fields = name.split("_")
        pdb_id, chains = fields[0], fields[1]
        partner = fields[2] if len(fields) > 2 else ""
        raw = Path(args.raw_pdb_dir) / f"{pdb_id}.pdb"
        if not raw.exists() and (cfg.paths.root / "raw_pdbs" / f"{pdb_id}.pdb").exists():
            raw = cfg.paths.root / "raw_pdbs" / f"{pdb_id}.pdb"
        if not raw.exists():
            logger.error("raw PDB not found: %s", raw)
            continue

        # Load and protonate once, then reuse for both chains.
        struct = load_pdb(raw)
        protonate(struct)

        # Generate the full-complex surface once (if there is a partner).
        complex_surface = None
        if partner:
            from masif.surface.generate import generate_surface
            from masif.surface.pdb import extract_chains

            complex_struct = extract_chains(struct, chains + partner)
            complex_surface = generate_surface(complex_struct, cfg.surface)

        # Process the query chain (e.g. AB).
        prepare_protein(
            pdb_id=pdb_id,
            chain_ids=chains,
            cfg=cfg,
            raw_pdb=raw,
            out_prefix=f"{pdb_id}_{chains}",
            partner_chain_ids=partner,
            complex_surface=complex_surface,
            structure=struct,
        )

        # Process the partner chain (e.g. DE) as its own sample, reusing the same
        # complex surface so its interface labels are computed against the query.
        if partner:
            prepare_protein(
                pdb_id=pdb_id,
                chain_ids=partner,
                cfg=cfg,
                raw_pdb=raw,
                out_prefix=f"{pdb_id}_{partner}",
                partner_chain_ids=chains,
                complex_surface=complex_surface,
                structure=struct,
            )


def cmd_train(args) -> None:
    import torch

    from masif.models.masif_site import MaSIFSite
    from masif.trainer import train

    cfg = _config_from_args(args)
    precomp = Path(cfg.paths.root) / cfg.paths.precomputation_dir
    files = sorted(precomp.glob("*.npz"))
    if not files:
        raise SystemExit(f"no precomputed patches under {precomp}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MaSIFSite(
        max_rho=cfg.patch.radius,
        n_thetas=cfg.patch.n_thetas,
        n_rhos=cfg.patch.n_rhos,
        n_rotations=cfg.patch.n_rotations,
        n_conv_layers=cfg.model.n_conv_layers,
        feat_mask=list(cfg.model.feat_mask),
    )
    out_dir = Path(cfg.paths.root) / cfg.paths.model_dir
    train(model, files, cfg.model, cfg.train, device, out_dir)


def cmd_predict(args) -> None:
    import torch

    from masif.models.masif_site import MaSIFSite
    from masif.patches import ProteinPatches
    from masif.trainer import _eval_full

    cfg = _config_from_args(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MaSIFSite(
        max_rho=cfg.patch.radius,
        n_thetas=cfg.patch.n_thetas,
        n_rhos=cfg.patch.n_rhos,
        n_rotations=cfg.patch.n_rotations,
        n_conv_layers=cfg.model.n_conv_layers,
        feat_mask=list(cfg.model.feat_mask),
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)

    precomp = Path(cfg.paths.root) / cfg.paths.precomputation_dir
    out = Path(cfg.paths.root) / cfg.paths.out_pred_dir
    out.mkdir(parents=True, exist_ok=True)
    for f in sorted(precomp.glob("*.npz")):
        patches = ProteinPatches.load(f)
        scores = _eval_full(model, patches, device, cfg.model.feat_mask)
        np.save(out / f"pred_{f.stem}.npy", scores)
        auc = _roc_auc(patches.labels, scores)
        logger.info("%s AUC=%.4f", f.stem, auc)


def _roc_auc(labels, scores):
    from sklearn.metrics import roc_auc_score

    if len(np.unique(labels)) < 2:
        return float("nan")
    return roc_auc_score(labels, scores)


def cmd_color(args) -> None:
    import trimesh

    cfg = _config_from_args(args)
    surf_dir = Path(cfg.paths.root) / cfg.paths.ply_chain_dir
    pred_dir = Path(cfg.paths.root) / cfg.paths.out_pred_dir
    out_dir = Path(cfg.paths.root) / cfg.paths.out_surf_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for pred in sorted(pred_dir.glob("pred_*.npy")):
        stem = pred.stem[len("pred_") :]
        surf = surf_dir / f"{stem}.ply"
        if not surf.exists():
            continue
        scores = np.load(pred)
        mesh = trimesh.load(surf, process=False)
        # map scores to a blue-white-red colormap
        s = np.clip((scores - scores.min()) / max(scores.ptp(), 1e-6), 0, 1)
        colors = np.zeros((len(s), 4))
        colors[:, 2] = 1.0 - s
        colors[:, 0] = s
        colors[:, 3] = 1.0
        mesh.visual.vertex_colors = (colors * 255).astype(np.uint8)
        mesh.export(out_dir / f"{stem}_pred.ply")
        logger.info("colored %s", stem)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="masif", description="Modernized MaSIF CLI")
    sub = p.add_subparsers(dest="command", required=True)

    pp = sub.add_parser("prepare", help="PDB -> patches (.npz)")
    pp.add_argument("proteins", nargs="+", help="PDBID_CHAIN[+PARTNER], e.g. 1AKJ_AB_DE")
    pp.add_argument("--raw-pdb-dir", default="raw_pdbs", help="directory with raw PDBs")
    _add_common(pp)
    pp.set_defaults(func=cmd_prepare)

    pt = sub.add_parser("train", help="train MaSIF-site")
    _add_common(pt)
    pt.set_defaults(func=cmd_train)

    pe = sub.add_parser("predict", help="score precomputed proteins")
    pe.add_argument("--checkpoint", default="model.pt", help="path to model.pt")
    _add_common(pe)
    pe.set_defaults(func=cmd_predict)

    pc = sub.add_parser("color", help="colored PLY from predictions")
    _add_common(pc)
    pc.set_defaults(func=cmd_color)
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
