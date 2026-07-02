#!/usr/bin/env bash
#SBATCH -J peg-md
#SBATCH -p gpu
#SBATCH -t 24:00:00                  # took me ~5 hours
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-3
#SBATCH --gpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --output slurm-%x.%A_%a.out
# #SBATCH --exclude=iscc006          # Iris-specific

# --- info ----------------------------------------------------------------------
# Explicit-solvent MD on GPU
#
# Array tasks:
#   0 = dft      artifacts/peg_fitted_dft_live.offxml
#   1 = egret    artifacts/peg_fitted_egret_live.offxml
#   2 = presto   artifacts/peg_presto.offxml
#   3 = baseline openff-2.3.0.offxml

# --- setup --------------------------------------------------------------------
set -eo pipefail

LABELS=(dft egret presto baseline)
OFFXMLS=(
    artifacts/peg_fitted_dft_live.offxml
    artifacts/peg_fitted_egret_live.offxml
    artifacts/peg_presto.offxml
    openff-2.3.0.offxml
)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
LABEL="${LABELS[$TASK_ID]:-}"
OFFXML="${OFFXMLS[$TASK_ID]:-}"

if [[ -z "$LABEL" || -z "$OFFXML" ]]; then
    echo "invalid SLURM_ARRAY_TASK_ID=${TASK_ID}; expected 0-3" >&2
    exit 2
fi

# --- environment --------------------------------------------------------------
source ~/.bashrc

# --- GPU sanity ---------------------------------------------------------------
nvidia-smi || { echo "no GPU visible -- check partition/--gpus-per-task"; exit 1; }

# --- run ----------------------------------------------------------------------
# Reads the WaSP conf1 PDB (vendored under data/wasp_reference/), parameterises
# PEG + TIP3P with the chosen FF, NPT-equilibrates then produces, writing
# trajectory + state CSV under artifacts/md_explicit/.
echo "array task ${TASK_ID}: ${LABEL} (${OFFXML})"
pixi run -e cuda python -u scripts/run_md_explicit.py \
    --offxml    "$OFFXML" \
    --pdb       "data/wasp_reference/peg_solv_36mer.pdb" \
    --label     "$LABEL" \
    --out-dir   "artifacts/cached_md" \
    --equil-ps  100 \
    --prod-ns   100 \
    --device    "CUDA"

echo "done: MD ${LABEL} (100 ns) -> artifacts/cached_md/explicit_${LABEL}_*"
