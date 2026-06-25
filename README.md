# PEG force-field fitting workshop

Fit an OpenFF SMIRNOFF force field to a polymer (poly(ethylene glycol), PEG) and
test it on physical observables. The single notebook
[`peg_workshop_combined.ipynb`](peg_workshop_combined.ipynb) walks through:

1. **Part 1 — OpenFF basics:** loading from various formats such as
   SMILES or PDB, assigning parameters, and outputting to simulation engines
   such as OpenMM/GROMACS. We also show an example of how to build a PEG-36 chain
   with SwiftPol; read a solvated structure from PDB with
   `openff-pablo`; run a short explicit-solvent MD; and do some quick analysis.
2. **Part 2 — Fitting:** refit the O–C–C–O `ProperTorsions.k` with
   `smee` + `descent` against DFT data (QCArchive torsiondrives).
   We also extend this optionally to show fits against
   Egret-1 (a MACE MLIP), and then a bespoke all-valence PRESTO fit.

DataFrames use **pandas**; helper functions live in
[`peg_workshop_utils.py`](peg_workshop_utils.py).

---

## Quickstart (pixi — recommended)

[pixi](https://pixi.sh) reads `pixi.toml` and builds an isolated environment under
`.pixi/` (run `pixi install` to solve it; no lockfile is committed).

```bash
# 1. Install pixi (https://pixi.sh/latest/#installation)
# if you haven't already
curl -fsSL https://pixi.sh/install.sh | bash        # macOS/Linux; or: brew install pixi

# 2. From the repo root, build the environment (installs everything in pixi.toml,
#    including openff-pablo + the other git packages)
pixi install

# 3. Launch the workshop
pixi run workshop          # == jupyter lab peg_workshop_combined.ipynb
```

---

## Alternative: conda / mamba

If you prefer conda/mamba, an equivalent spec is in
[`environment.yaml`](environment.yaml):

```bash
# Install miniforge (provides mamba): https://github.com/conda-forge/miniforge
# if you don't already have conda/mamba
mamba env create -f environment.yaml      # or: conda env create -f environment.yaml
mamba activate peg-workshop
jupyter lab peg_workshop_combined.ipynb
```

---

## GPU / cluster steps

The PRESTO fit and the production explicit-solvent MD need CUDA.
We cache some data run on a CPU cluster, not the workshop laptop — see
[`scripts/`](scripts/) (`train_presto.py` / `run_md_explicit.py` + their `.sbatch` wrappers).
Their outputs are vendored under `data/` and `artifacts/`, and the notebook loads
them if present (otherwise it gates those cells and explains how to produce them).

---

## Layout

| Path | What |
|------|------|
| `peg_workshop_combined.ipynb` | the workshop notebook (run from the repo root) |
| `peg_workshop_utils.py` | helper functions imported by the notebook |
| `data/` | torsiondrive datasets, the shared train/test `split.json`, WaSP reference PDB |
| `artifacts/` | fitted force fields (`peg_fitted_*`, `peg_presto.offxml`) + explicit-MD outputs (`md_explicit/`) |
| `scripts/` | CUDA data-generation + MD scripts (`train_presto.py`, `run_md_explicit.py`) + SLURM wrappers |
| `pixi.toml` / `pixi.lock` | the pixi environment spec |
| `environment.yaml` | conda/mamba equivalent |
