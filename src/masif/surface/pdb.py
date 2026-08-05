"""PDB parsing utilities built on BioPython.

Replaces the original ``input_output/extractPDB.py`` and the ``reduce`` binary with a
self-contained implementation. Polar hydrogens are added deterministically (see
:func:`protonate`) so the hydrogen-bond feature can be computed without installing
Reduce.
"""

from __future__ import annotations

import numpy as np
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.Structure import Structure
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue
from pathlib import Path

from masif.chemistry import POLAR_HYDROGENS, DONOR_ATOM

BOND_LEN = 1.01  # typical X-H bond length (Angstrom)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else np.zeros_like(v)


def _donor_h_position(res: Residue, hydrogen_name: str) -> np.ndarray:
    """Deterministic polar-hydrogen position for a given donor hydrogen name.

    Places the hydrogen one bond-length away from its donor atom, pointing away from
    the donor's average bonded-atom direction (a simple, robust model that gives the
    geometry the hbond angular penalties need).
    """
    donor_name = DONOR_ATOM[hydrogen_name]
    donor = res[donor_name]
    d_coord = donor.get_coord()
    bonded = [
        a.get_coord() for a in res.get_atoms() if a is not donor and a.get_name() != hydrogen_name
    ]
    if len(bonded) == 0:
        direction = np.array([0.0, 0.0, 1.0])
    else:
        direction = _unit(d_coord - np.mean(np.stack(bonded), axis=0))
    return d_coord + BOND_LEN * direction


def protonate(structure: Structure) -> None:
    """Add missing polar hydrogens to ``structure`` in place.

    Only polar donors (from :data:`masif.chemistry.POLAR_HYDROGENS`) that are missing
    their hydrogen are added. This removes the dependency on the external ``reduce``
    binary while producing geometrically reasonable hydrogen positions.
    """
    for res in structure.get_residues():
        resname = res.get_resname()
        if resname not in POLAR_HYDROGENS:
            continue
        for h_name in POLAR_HYDROGENS[resname]:
            if h_name in res:
                continue
            donor_name = DONOR_ATOM[h_name]
            if donor_name not in res:
                continue
            pos = _donor_h_position(res, h_name)
            atom = Atom(h_name, pos, 0.0, 1.0, " ", h_name, 1, element="H")
            res.add(atom)


def extract_chains(structure: Structure, chain_ids: str) -> Structure:
    """Return a new structure containing only the requested chains."""
    from Bio.PDB.Structure import Structure as S
    from Bio.PDB.Model import Model
    from Bio.PDB.Chain import Chain

    new_struct = S(structure.id)
    for model in structure:
        new_model = Model(model.id)
        for chain in model:
            if chain.get_id() in chain_ids:
                new_chain = Chain(chain.get_id())
                for res in chain:
                    new_chain.add(res)
                new_model.add(new_chain)
        new_struct.add(new_model)
    return new_struct


def write_pdb(structure: Structure, filename: str | Path) -> None:
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(filename))


def load_pdb(filename: str | Path) -> Structure:
    return PDBParser(QUIET=True).get_structure(Path(filename).stem, str(filename))


def save_chains(pdb_file: str | Path, out_file: str | Path, chain_ids: str) -> None:
    """Extract ``chain_ids`` from ``pdb_file``, protonate, and save to ``out_file``."""
    struct = load_pdb(pdb_file)
    protonate(struct)
    out = extract_chains(struct, chain_ids)
    write_pdb(out, out_file)
