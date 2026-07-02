# WaSP reference structure (`data/wasp_reference/`)

The starting structure for the workshop's explicit-solvent MD: a single PEG chain
from [Davel et al.](https://github.com/shirtsgroup/WaSP_simulations).

## Contents

| File | What |
|---|---|
| `peg_solv_36mer.pdb` | the MD starting structure — **42 444 atoms** (1 PEG + 14 061 water), re-residued so **openff-pablo** can read it: a start cap `CPL` (CH3, resid 1), 36 `OCC` (O-CH2-CH2) monomers (resids 2–37), an end cap `CPR` (O-CH3, resid 38), plus `HOH` water. |
| `build_peg_solv_36mer.py` | regenerates `peg_solv_36mer.pdb` from the WaSP source (below). |

## Regenerating `peg_solv_36mer.pdb`

```bash
pixi run python data/wasp_reference/build_peg_solv_36mer.py
```

The script:
1. Uses the local `raw_peg.pdb` if present; otherwise downloads the original
   `production_topology.pdb` from the public WaSP repo via its raw URL
   (`raw.githubusercontent.com/shirtsgroup/WaSP_simulations/main/wasp_sims/peg_modified/Espaloma-AM1-BCC/conf1/production/production_topology.pdb`)
   to `raw_peg.pdb`.
2. Reformats the file with MDAnalysis: reorders, re-residues, renames the 261 PEG atoms into the regular
   `CPL`/`OCC`/`CPR` tiling (resids numbered from 1) just to streamline the demo a bit;
   we want to demonstrate the Pablo pathway of parsing via atom + residue name,
   and we want to run analysis easily with MDAnalysis.