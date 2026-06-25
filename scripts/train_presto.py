"""PRESTO bespoke force field for the 5 held-out PEG molecules.

This script runs a **proper, production-quality** PRESTO fit: it uses PRESTO's
default sampling settings (10 conformers x 100 ps MLP-MD per molecule, 2
refinement iterations).

REQUIRES CUDA. PRESTO drives Egret-1 sampling through OpenMM's PythonForce,
which needs a CUDA build (>= 12.9); it does not run on the CPU-only workshop
Mac (metadynamics crashes). To run:

    pixi run python -u scripts/train_presto.py --device cuda

Output: ``artifacts/peg_presto.offxml``.
"""

import argparse
import json
import pathlib

from openff.toolkit import Molecule

from presto.settings import (
    MLMDSamplingSettings,
    MLPSettings,
    MMMDMetadynamicsTorsionMinimisationSamplingSettings,
    MSMSettings,
    ParamSettings,
    WorkflowSettings,
)
from presto.workflow import get_bespoke_force_field

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SPLIT_PATH = DATA / "split.json"
OUT_DIR = ROOT / "artifacts"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--device", default="cuda", help="cuda (Iris) or cpu (unsupported)"
    )
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=2,
        help="PRESTO refinement iterations (default 2)",
    )
    args = parser.parse_args()

    if not SPLIT_PATH.exists():
        raise SystemExit(
            f"Missing {SPLIT_PATH}; run the notebook's section 2.1 download cell "
            "(DOWNLOAD_QCA=True) first, or sync data/split.json from the repo."
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    split = json.loads(SPLIT_PATH.read_text())
    # Canonical SMILES (no atom maps) of the held-out torsiondrive molecules.
    # `test_smiles` holds the mapped SMILES (new split schema; was `test_cmiles`).
    molecules = [
        Molecule.from_mapped_smiles(c, allow_undefined_stereo=True).to_smiles(
            mapped=False
        )
        for c in split["test_smiles"]
    ]
    print(f"Training PRESTO + Egret-1 on the {len(molecules)} held-out molecules:")
    for s in molecules:
        print(f"  {s}")

    # Egret-1 is the only override -- everything else uses PRESTO's production
    # defaults (10 conformers x 100 ps MLP-MD per molecule; see presto.settings).
    egret = MLPSettings(ml_potential="egret-1")
    settings = WorkflowSettings(
        param_settings=ParamSettings(
            molecule_input_type="smiles",
            molecules=molecules,
            msm_settings=MSMSettings(mlp_settings=egret),
        ),
        device_type=args.device,
        n_iterations=args.n_iterations,
        training_sampling_settings=MMMDMetadynamicsTorsionMinimisationSamplingSettings(
            mlp_settings=egret
        ),
        testing_sampling_settings=MLMDSamplingSettings(mlp_settings=egret),
    )

    out = OUT_DIR / "peg_presto.offxml"
    print(
        f"\nRunning a full PRESTO fit on {args.device} ({args.n_iterations} iterations, "
        f"default sampling -- expect many GPU-hours)..."
    )
    presto_ff = get_bespoke_force_field(settings)
    presto_ff.to_file(str(out))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
