# WaSP reference structure (`data/wasp_reference/`)

The starting structure for the workshop's explicit-solvent MD: a single PEG chain
from [Davel et al.](https://github.com/shirtsgroup/WaSP_simulations).

## Contents

| File | What |
|---|---|
| `peg_solv_36mer.pdb` | the MD starting structure — **42 444 atoms** (1 PEG + 14 061 water), re-residued so **openff-pablo** can read it: a fused start cap `MES` (CH3-O-CH2-CH2), 35 `OCC` (O-CH2-CH2) monomers, an end cap `MEE` (O-CH3), plus `HOH` water. Read by `scripts/run_md_explicit.py` and the notebook's Part 1.3 (`build_pablo_topology`). |
| `build_peg_solv_36mer.py` | regenerates `peg_solv_36mer.pdb` from the WaSP source (below). |

## Regenerating `peg_solv_36mer.pdb`

```bash
pixi run python data/wasp_reference/build_peg_solv_36mer.py
```

The script:
1. Downloads the original `production_topology.pdb` from the public WaSP repo
   via its raw URL (`raw.githubusercontent.com/shirtsgroup/WaSP_simulations/main/wasp_sims/peg_modified/Espaloma-AM1-BCC/conf1/production/production_topology.pdb`),
   to a transient `raw_peg.pdb`.
2. Reformats the file with MDAnalysis: reorders, re-residues, renames the 261 PEG atoms into the regular
   `MES`/`OCC`/`MEE` tiling that openff-pablo's residue definitions match in the notebook
   (so every linking residue carries both its `C2` + `O1` linking atoms).
   Water atoms are renamed to the canonical `O`/`H1`/`H2`; PEG `CONECT` records are written so
   pablo can establish the inter-residue links.
