"""Chemical parameters for MaSIF.

Ported from ``source/default_config/chemistry.py`` (Pablo Gainza, LPDI EPFL) and
``source/triangulation/computeHydrophobicity.py``. Only data tables are kept here;
the computation that used these tables lives in :mod:`masif.surface.features`.
"""

from __future__ import annotations

import numpy as np

# van der Waals radii (Angstrom) per atom element.
RADII: dict[str, str] = {
    "N": "1.540000",
    "O": "1.400000",
    "C": "1.740000",
    "H": "1.200000",
    "S": "1.800000",
    "P": "1.800000",
    "Z": "1.39",
    "X": "0.770000",  # radius of CB/CA in the disembodied case.
}


#: Polar hydrogen atom names per residue (names used by the program Reduce).
POLAR_HYDROGENS: dict[str, list[str]] = {
    "ALA": ["H"],
    "GLY": ["H"],
    "SER": ["H", "HG"],
    "THR": ["H", "HG1"],
    "LEU": ["H"],
    "ILE": ["H"],
    "VAL": ["H"],
    "ASN": ["H", "HD21", "HD22"],
    "GLN": ["H", "HE21", "HE22"],
    "ARG": ["H", "HH11", "HH12", "HH21", "HH22", "HE"],
    "HIS": ["H", "HD1", "HE2"],
    "TRP": ["H", "HE1"],
    "PHE": ["H"],
    "TYR": ["H", "HH"],
    "GLU": ["H"],
    "ASP": ["H"],
    "LYS": ["H", "HZ1", "HZ2", "HZ3"],
    "PRO": [],
    "CYS": ["H"],
    "MET": ["H"],
}

HBOND_STD_DEV = np.pi / 3

#: Maps an acceptor atom to the atom directly bonded to it on which the ideal
#: ~120 deg acceptor angle is measured.
ACCEPTOR_ANGLE_ATOM: dict[str, str] = {
    "O": "C",
    "O1": "C",
    "O2": "C",
    "OXT": "C",
    "OT1": "C",
    "OT2": "C",
    "OD1": "CG",
    "OD2": "CG",
    "OE1": "CD",
    "OE2": "CD",
    "ND1": "CE1",
    "NE2": "CE1",
    "OH": "CZ",
    "OG": "CB",
    "OG1": "CB",
}

#: Maps an acceptor atom to a third atom defining the acceptor plane.
ACCEPTOR_PLANE_ATOM: dict[str, str] = {
    "O": "CA",
    "OD1": "CB",
    "OD2": "CB",
    "OE1": "CG",
    "OE2": "CG",
    "ND1": "NE2",
    "NE2": "ND1",
    "OH": "CE1",
}

#: Maps a hydrogen atom to its donor atom.
DONOR_ATOM: dict[str, str] = {
    "H": "N",
    "HH11": "NH1",
    "HH12": "NH1",
    "HH21": "NH2",
    "HH22": "NH2",
    "HE": "NE",
    "HD21": "ND2",
    "HD22": "ND2",
    "HE21": "NE2",
    "HE22": "NE2",
    "HD1": "ND1",
    "HE2": "NE2",
    "HE1": "NE1",
    "HZ1": "NZ",
    "HZ2": "NZ",
    "HZ3": "NZ",
    "HH": "OH",
    "HG": "OG",
    "HG1": "OG1",
}

#: Kyte-Doolittle hydropathy scale (indexed by residue name).
KD_SCALE: dict[str, float] = {
    "ILE": 4.5,
    "VAL": 4.2,
    "LEU": 3.8,
    "PHE": 2.8,
    "CYS": 2.5,
    "MET": 1.9,
    "ALA": 1.8,
    "GLY": -0.4,
    "THR": -0.7,
    "SER": -0.8,
    "TRP": -0.9,
    "TYR": -1.3,
    "PRO": -1.6,
    "HIS": -3.2,
    "GLU": -3.5,
    "GLN": -3.5,
    "ASP": -3.5,
    "ASN": -3.5,
    "LYS": -3.9,
    "ARG": -4.5,
}

#: Mean elemental vdW radius per element, used for grid-based surface generation.
ELEMENT_RADIUS: dict[str, float] = {"H": 1.2, "C": 1.7, "N": 1.55, "O": 1.52, "S": 1.8, "P": 1.8}

#: Lightweight partial charges (electrons) per atom name, used by the Coulombic
#: electrostatics proxy that replaces the APBS binary. This is a coarse PARSE-like
#: scheme sufficient for a surface *feature*, not a replacement for PB solvers.
PARTIAL_CHARGES: dict[str, float] = {
    # backbone
    "N": -0.30,
    "H": +0.30,
    "CA": +0.10,
    "C": +0.55,
    "O": -0.55,
    # arg
    "CB": +0.00,
    "CG": +0.00,
    "CD": +0.00,
    "NE": -0.30,
    "CZ": +0.40,
    "NH1": -0.55,
    "NH2": -0.55,
    "HH11": +0.30,
    "HH12": +0.30,
    "HH21": +0.30,
    "HH22": +0.30,
    "HE": +0.30,
    # asp / glu
    "OD1": -0.55,
    "OD2": -0.55,
    "OE1": -0.55,
    "OE2": -0.55,
    "CG": -0.10,
    "CD": -0.10,
    # lys
    "NZ": +0.50,
    "HZ1": +0.30,
    "HZ2": +0.30,
    "HZ3": +0.30,
    # his
    "ND1": -0.30,
    "CE1": +0.20,
    "NE2": -0.30,
    "CD2": +0.10,
    # tyroser
    "OG": -0.55,
    "OG1": -0.55,
    "HG": +0.30,
    "HG1": +0.30,
    "OH": -0.55,
    "HH": +0.30,
    # asn / gln
    "OD1": -0.55,
    "OE1": -0.55,
    "ND2": -0.60,
    "NE2": -0.60,
    "HD21": +0.30,
    "HD22": +0.30,
    "HE21": +0.30,
    "HE22": +0.30,
}
