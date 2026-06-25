#!/usr/bin/env bash
# PRESTO bespoke-FF fit (Fit 3) on a GPU node. Based on
# /Volumes/lilywang/pydev/ffvq/slurm/train.sbatch (pixi env, GPU).
#
# Submit from the repo root:
#   sbatch scripts/train_presto.sbatch [n_iterations]
#
# arg: $1 = PRESTO refinement iterations (default 2 = PRESTO default).
#
# Runs the FULL production fit (PRESTO defaults: 10 conformers x 100 ps MLP-MD
# per molecule) on the 5 held-out molecules in data/split.json -> writes
# artifacts/peg_presto.offxml. This is the slow one; pad the wall-clock.
#
# FIXME(cluster): lines marked FIXME are site-specific. See scripts/README.md for
# the Iris setup (CUDA-enabled pixi env + the presto editable-path fix).

#SBATCH -J peg-presto
#SBATCH -p gpu                       # FIXME(cluster): GPU partition/queue ("gpu" on Iris)
#SBATCH -t 48:00:00                  # full PRESTO fit is many GPU-hours; pad generously
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1            # FIXME(cluster): Iris-specific GPU request
#SBATCH --cpus-per-task=8            # per-molecule Python loop + sampling are CPU-bound too
#SBATCH --mem=32gb
#SBATCH --output slurm-%x.%A.out
# #SBATCH --exclude=iscc006          # FIXME(cluster): uncomment if needed (Iris-specific)
# #SBATCH --account=<your-account>   # FIXME(cluster): set if your site requires it

set -eo pipefail

N_ITER="${1:-2}"

# --- repo location ------------------------------------------------------------
# FIXME(cluster): point REPO at your Iris checkout of 2026-polymers.
REPO="${POLYMERS_REPO:-$SLURM_SUBMIT_DIR}"
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
pixi run -e "$PIXI_ENV" python -u scripts/train_presto.py \
    --device cuda \
    --n-iterations "$N_ITER"

echo "done: PRESTO fit (${N_ITER} iterations) -> artifacts/peg_presto.offxml"
