#!/usr/bin/env bash
# Explicit-solvent MD (density + Rg) on a GPU node. Based on
# /Volumes/lilywang/pydev/ffvq/slurm/train.sbatch (pixi env, GPU).
#
# Submit one job per force field. The dft/egret inputs are SNAPSHOTS of the
# notebook-trained fits (artifacts/peg_md_{dft,egret}.offxml); presto is the
# bespoke artifact. e.g.:
#   sbatch scripts/run_md_explicit.sbatch artifacts/peg_md_dft.offxml   dft      100
#   sbatch scripts/run_md_explicit.sbatch artifacts/peg_md_egret.offxml egret    100
#   sbatch scripts/run_md_explicit.sbatch artifacts/peg_presto.offxml   presto   100
#   sbatch scripts/run_md_explicit.sbatch openff-2.3.0.offxml           baseline 100
#
# args: $1 = offxml (default artifacts/peg_md_dft.offxml)
#       $2 = label  (default dft) -> artifacts/md_explicit/explicit_<label>_*
#       $3 = production ns (default 100)
#
# FIXME(cluster): lines marked FIXME are site-specific. See scripts/README.md.

#SBATCH -J peg-md
#SBATCH -p gpu                       # FIXME(cluster): GPU partition/queue ("gpu" on Iris)
#SBATCH -t 24:00:00                  # 100 ns of ~42k atoms is hours on a GPU; pad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1            # FIXME(cluster): Iris-specific GPU request
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --output slurm-%x.%A.out
# #SBATCH --exclude=iscc006          # FIXME(cluster): uncomment if needed (Iris-specific)
# #SBATCH --account=<your-account>   # FIXME(cluster): set if your site requires it

set -eo pipefail

OFFXML="${1:-artifacts/peg_md_dft.offxml}"
LABEL="${2:-dft}"
PROD_NS="${3:-100}"

# --- repo location ------------------------------------------------------------
REPO="${POLYMERS_REPO:-$SLURM_SUBMIT_DIR}"   # FIXME(cluster): your Iris checkout
cd "$REPO"

# --- environment --------------------------------------------------------------
source ~/.bashrc
# CUDA-enabled pixi env, default; override with POLYMERS_PIXI_ENV if needed
# (see scripts/README.md item 2-3 -- the cuda env's lock entries must be
# produced on Iris, not the Mac).
PIXI_ENV="${POLYMERS_PIXI_ENV:-cuda}"

# --- GPU sanity ---------------------------------------------------------------
nvidia-smi || { echo "no GPU visible -- check partition/--gpus-per-task"; exit 1; }

# --- run ----------------------------------------------------------------------
# Reads the WaSP conf1 PDB (vendored under data/wasp_reference/), parameterises
# PEG + TIP3P with the chosen FF, NPT-equilibrates then produces, writing
# trajectory + state CSV under artifacts/md_explicit/.
pixi run -e "$PIXI_ENV" python -u scripts/run_md_explicit.py \
    --offxml "$OFFXML" \
    --label "$LABEL" \
    --prod-ns "$PROD_NS" \
    --device CUDA

echo "done: MD ${LABEL} (${PROD_NS} ns) -> artifacts/md_explicit/explicit_${LABEL}_*"
