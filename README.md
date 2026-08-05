# masif (modernized)

A modern, dependency-light rewrite of **MaSIF** (Molecular Surface Interaction
Fingerprints, Gainza et al., *Nat. Methods* 2020) focused on the **MaSIF-site**
interface-prediction application. It keeps MaSIF's core idea — geometric deep learning
over overlapping patches of the protein molecular surface using a polar-coordinate
geodesic convolution — while replacing the original, hard-to-install machinery.

This was written *in place of* the original codebase at `../masif`; it does not require
compiling or invoking MSMS / PyMesh / APBS / Reduce.

## What changed vs. the original

| Original | Modernised |
| --- | --- |
| TensorFlow 1.x (`tf.contrib`, Sessions, placeholders) | PyTorch (`nn.Module`) |
| MSMS + PyMesh surface mesh | `edt`/`scipy` distance transform + Marching Cubes + `trimesh` |
| APBS Poisson–Boltzmann electrostatics | fast Coulombic proxy (`features.coulombic_potential`) |
| `reduce` binary for protonation | in-house deterministic polar-H placement (`surface.pdb`) |
| networkx all-pairs Dijkstra + per-patch MDS | `scipy.sparse.csgraph` geodesics + exponential-map style angles (`geometry.polar`) |
| global mutable `masif_opts` dict | typed dataclasses (`config.py`) |
| bash/slurm scripts per app | a single `masif` CLI |
| `.npy` files + TFRecords, off-disk duplication | per-protein `.npz` patches, on-the-fly dataset |

The network architecture is a faithful port of `MaSIF_site.py`: a per-channel Gaussian
grid convolution in the first layer, rotation-invariant max-pooling over `n_rotations`,
and stacked geodesic-convolution layers that rebuild each patch via its neighbour index
tensor.

## Install

```bash
cd masif_modern
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]      # add the optional `fast` extra to use the `edt` package
```

## Usage

Prepare patches for one protein (chain A of `1MBN`; `_DE` are the interaction partner
chains used to mark the interface):

```bash
masif prepare 1MBN_A                                  --root data/site --raw-pdb-dir raw_pdbs
masif prepare 1AKJ_AB_DE                              --root data/site --raw-pdb-dir raw_pdbs
```

Train a model on every `.npz` under the precomputation directory:

```bash
masif train --root data/site
```

Score every vertex of the precomputed proteins and write a report:

```bash
masif predict --root data/site --checkpoint data/site/nn_models/site/model.pt
```

Colour predicted surfaces to coloured PLY:

```bash
masif color --root data/site
```

Settings (grid resolution, probe radius, patch geometry, NN hyper-parameters) live in
`masif/config.py` and can be overridden via a JSON `--config`.

## Layout

```
src/masif/
  config.py              typed dataclasses (surface / patch / model / train / paths)
  chemistry.py           radii, polar H, donors/acceptors, hydropathy, partial charges
  surface/               PDB, surface generation, per-vertex features
    generate.py          EDT + Marching Cubes + trimesh surface
    features.py          shape index, hbond, electrostatics, hydrophobicity
    pdb.py               BioPython parsing + deterministic protonation
  geometry/polar.py      geodesic distances + angular coordinates (no MDS)
  patches.py             patch assembly, DDC, interface labels, torch dataset
  models/
    geodesic_conv.py     Gaussian-grid rotation-invariant geodesic convolution
    masif_site.py        the MaSIF-site network
  trainer.py             training / evaluation loop
  pipeline.py            PDB -> surface -> features -> patches (.npz)
  cli/main.py            `masif` command-line interface
tests/                   unit tests (polar coords, features, model)
```

## Tests

```bash
pytest
```

## Testing status

* `masif prepare` — **tested** on single-chain (1MBN_A) and two-chain complexes (1AKJ_AB_DE);
  produces valid `.ply` surfaces and `.npz` patches with correct tensor shapes, value
  ranges, and no NaN.
* `masif train` — **not yet tested end-to-end**. The training loop (`trainer.py`) is a
  faithful port but has not been run on real data yet. Known limitations to watch for:
  BCE loss and per-protein balanced sampling are implemented; the original ranking loss
  is not.
* `masif predict` — **not yet tested end-to-end**. The prediction path (`_eval_full`) now
  loops over Gaussian-rotation iterations to avoid materialising the full `(B, R, V, G)`
  tensor; this is memory-safe but slower than a fully vectorised version.
* `masif color` — **not yet tested**.

Before relying on any of the untested commands, spot-check a small dataset (2–3 complexes)
and inspect `history.json` / prediction AUCs.

## Scope & caveats

* **Surface** is a solvent-accessible surface (vdW + probe), an approximation of the
  original MSMS solvent-**excluded** surface. Reenter the `SES`/`SAS` choice in
  `SurfaceConfig` if strict comparability is needed.
* **Electrostatics** is a Coulombic proxy, not a PB solution; `APBS` gives more
  physical charges if you have it.
* **Angular coordinates** use an exponential-map-style propagation rather than MDS;
  only local consistency is required because the network is made rotation-invariant by
  multi-rotation max-pooling.
* **Training loss** uses standard balanced BCE (+ ROC-AUC reporting) in place of the
  original ranking-style `neg_scores - pos_scores` loss. Restore the ranking loss in
  `trainer.py` if you must match the paper exactly.

## License / credits

Apache-2.0. Original MaSIF by Pablo Gainza, Freyr Sverrisson et al., LPDI/EPFL.