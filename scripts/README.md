# GPU scripts (`scripts/`)

Scripts for running steps that are much more efficient on CUDA than CPU.
We used these scripts to run the files in `artifacts/`.

The two CUDA-only steps the combined notebook *consumes* but cannot compute live
in a 90-min CPU workshop — the bespoke **PRESTO** fit and the explicit-solvent
**MD** — plus their SLURM wrappers. Run them on **a GPU machine**, then copy the
artifacts back into `artifacts/`.

| File | Runs | Where | Produces |
|------|------|-------|----------|
| `train_presto.py` | Fit 3 (PRESTO) | **GPU (CUDA only)** | `artifacts/peg_presto.offxml` |
| `run_md_explicit.py` | explicit-solvent MD | **GPU (CUDA)** | `artifacts/md_explicit/` (Rg + O–C–C–O) |
| `train_presto.sbatch [n_iter]` | `train_presto.py --device cuda` | 48 h GPU job | `artifacts/peg_presto.offxml` |
| `run_md_explicit.sbatch <offxml> <label> <ns>` | `run_md_explicit.py --device CUDA` | 24 h GPU job | `artifacts/md_explicit/explicit_<label>_*` (PEG-only) |

## Run directly

```bash
# PRESTO bespoke FF on the held-out molecules (Egret-1 reference)        [GPU]
pixi run python -u scripts/train_presto.py --device cuda

# Explicit-solvent MD; topology built with openff-pablo from the reformatted
# data/wasp_reference/peg_solv_36mer.pdb. dft/egret use snapshots of the notebook
# fits (artifacts/peg_md_*.offxml); presto uses peg_presto.offxml. One job per FF.
pixi run python -u scripts/run_md_explicit.py \
    --offxml artifacts/peg_md_dft.offxml --label dft --prod-ns 100 --device CUDA
```

## SLURM

Standard GPU `sbatch` wrappers (GPU partition, one GPU per task, pixi env,
`nvidia-smi` check). Submit from the repo root so `$SLURM_SUBMIT_DIR` points at
the checkout.

```bash
# from the repo root on the GPU machine:
sbatch scripts/train_presto.sbatch 2

# Explicit MD with the notebook-trained fits + PRESTO (baseline left as-is):
sbatch scripts/run_md_explicit.sbatch artifacts/peg_md_dft.offxml    dft     100
sbatch scripts/run_md_explicit.sbatch artifacts/peg_md_egret.offxml  egret   100
sbatch scripts/run_md_explicit.sbatch artifacts/peg_presto.offxml    presto  100

# ...or submit all three newly-trained FFs in one go:
for fl in "peg_md_dft dft" "peg_md_egret egret" "peg_presto presto"; do
  set -- $fl
  sbatch scripts/run_md_explicit.sbatch "artifacts/$1.offxml" "$2" 100
done
```

### Prerequisites

- **The torsiondrive data + split must be present** (`data/qca_peg_td/`,
  `data/mlip_peg_td/`, `data/split.json`)
- `train_presto.py` consumes `data/split.json`; `run_md_explicit.py` consumes the
  reformatted `data/wasp_reference/peg_solv_36mer.pdb` (read with openff-pablo) and
  a fitted offxml from `artifacts/`.
- **The dft/egret MD inputs are snapshots of the notebook fits.** The notebook
  overwrites `artifacts/peg_fitted_{dft,egret}_live.offxml` on every run, so before
  submitting MD, snapshot them to stable names:
  `cp artifacts/peg_fitted_{dft,egret}_live.offxml artifacts/peg_md_{dft,egret}.offxml`
  (re-snapshot whenever you re-run the fits). PRESTO uses the stable
  `artifacts/peg_presto.offxml` directly.
