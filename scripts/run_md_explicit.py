"""
Explicit-solvent PEG MD from the reformatted PEG PDB, with a fitted force field.

The topology is built with openff-pablo from the residue-tiled
``data/wasp_reference/peg_solv_36mer.pdb`` (data/wasp_reference/build_peg_solv_36mer.py), exactly
as in the notebook's Part 1.3 cell. Equilibration is 100 ps by default; production defaults to a quick 2 ns
(pass --prod-ns 100 for a full run).

To run::

    pixi run python -u scripts/run_md_explicit.py \
        --offxml artifacts/peg_md_dft.offxml --label dft --prod-ns 100 --device CUDA
"""

import argparse
import pathlib
import time

import MDAnalysis as mda
import openmm
from openff.pablo import ResidueDefinition, topology_from_pdb
from openff.pablo.residue import BondDefinition
from openff.toolkit import ForceField
from openmm import MonteCarloBarostat, app
from openmm import unit as ommunit

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Reformatted, residue-tiled PEG PDB (data/wasp_reference/build_peg_solv_36mer.py) that Pablo can
# read: a fused start residue MES = CH3-O-CH2-CH2, 35 OCC = O-CH2-CH2 monomers,
# and an end cap MEE = O-CH3.
WASP = ROOT / "data" / "wasp_reference"
DEFAULT_PDB = WASP / "peg_solv_36mer.pdb"


def build_pablo_topology(pdb_path):
    """Solvated PEG topology via openff-pablo (mirrors the notebook's Part 1.3 cell).

    Every residue carrying the C2->O1 ether linking_bond holds BOTH linking atoms
    as retained atoms (required by openff-pablo >= 0.2), so the terminal methyls
    are fused into the end residues. Water resolves from the CCD automatically.
    """
    ether = BondDefinition.with_defaults("C2", "O1", order=1)
    mes = ResidueDefinition.from_smiles(
        mapped_smiles="[C:1]([H:2])([H:3])([H:4])[O:5][C:6]([H:7])([H:8])[C:9]([H:10])([H:11])[H:12]",
        atom_names={1: "CM", 2: "HM1", 3: "HM2", 4: "HM3", 5: "O1", 6: "C1",
                    7: "H11", 8: "H12", 9: "C2", 10: "H21", 11: "H22", 12: "H23"},
        residue_name="MES", leaving_atoms=(12,), linking_bond=ether,
    )
    occ = ResidueDefinition.from_smiles(
        mapped_smiles="[O:1]([C:2]([H:4])([H:5])[C:3]([H:6])([H:7])[H:9])[H:8]",
        atom_names={1: "O1", 2: "C1", 3: "C2", 4: "H11", 5: "H12",
                    6: "H21", 7: "H22", 8: "HO1", 9: "H23"},
        residue_name="OCC", leaving_atoms=(8, 9), linking_bond=ether,
    )
    mee = ResidueDefinition.from_smiles(
        mapped_smiles="[O:1]([C:2]([H:3])([H:4])[H:5])[H:6]",
        atom_names={1: "O1", 2: "C2", 3: "H21", 4: "H22", 5: "H23", 6: "HO1"},
        residue_name="MEE", leaving_atoms=(6,), linking_bond=ether,
    )
    return topology_from_pdb(str(pdb_path), additional_definitions=[mes, occ, mee])


TEMPERATURE_K = 300.0
PRESSURE_ATM = 1.0
TIMESTEP_FS = 2.0
FRICTION_PER_PS = 1.0
BAROSTAT_FREQ = 25


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--offxml",
        default="openff-2.3.0.offxml",
        help="Fitted SMIRNOFF FF (or baseline name). Default Sage 2.3.0.",
    )
    parser.add_argument("--pdb", type=pathlib.Path, default=DEFAULT_PDB)
    parser.add_argument("--device", default="CUDA", help="CUDA (Iris) / CPU / OpenCL")
    parser.add_argument(
        "--equil-ps", type=float, default=100.0, help="NPT equilibration"
    )
    parser.add_argument("--prod-ns", type=float, default=2.0, help="NPT production")
    parser.add_argument("--report-ps", type=float, default=10.0)
    parser.add_argument("--label", default="dft")
    parser.add_argument(
        "--out-dir", type=pathlib.Path, default=ROOT / "artifacts" / "md_explicit"
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Force field : {args.offxml}")
    print(f"Start PDB   : {args.pdb}")
    print("Building topology with openff-pablo (fused-cap residue definitions)...")
    top = build_pablo_topology(args.pdb)
    print(f"  topology: {top.n_atoms} atoms, {top.n_molecules} molecules")

    # Parameterise: Sage 2.3.0 (+ fitted torsions) carries TIP3P water itself.
    ff = ForceField(args.offxml)
    print("Building Interchange / OpenMM system (PME, explicit solvent)...")
    interchange = ff.create_interchange(top)
    system = interchange.to_openmm_system()
    omm_top = interchange.to_openmm_topology()
    positions = interchange.positions.to_openmm()

    barostat = MonteCarloBarostat(
        PRESSURE_ATM * ommunit.atmosphere, TEMPERATURE_K * ommunit.kelvin, BAROSTAT_FREQ
    )
    system.addForce(barostat)
    integrator = openmm.LangevinMiddleIntegrator(
        TEMPERATURE_K * ommunit.kelvin,
        FRICTION_PER_PS / ommunit.picosecond,
        TIMESTEP_FS * ommunit.femtosecond,
    )
    platform = openmm.Platform.getPlatformByName(args.device)
    sim = app.Simulation(omm_top, system, integrator, platform)
    sim.context.setPositions(positions)

    print("Minimising...")
    sim.minimizeEnergy(maxIterations=1000)
    sim.context.setVelocitiesToTemperature(TEMPERATURE_K * ommunit.kelvin)

    report_steps = int(args.report_ps * 1000 / TIMESTEP_FS)
    n_equil = int(args.equil_ps * 1000 / TIMESTEP_FS)
    n_prod = int(args.prod_ns * 1e6 / TIMESTEP_FS)

    # Full system is simulated, then water is stripped afterwards -- so the
    # topology + trajectory are written to temporary files first.
    full_top = args.out_dir / f"explicit_{args.label}_full_top.pdb"
    with open(full_top, "w") as fh:
        app.PDBFile.writeFile(
            omm_top, sim.context.getState(getPositions=True).getPositions(), fh
        )

    state_csv = args.out_dir / f"explicit_{args.label}_state.csv"
    sim.reporters.append(
        app.StateDataReporter(
            str(state_csv),
            report_steps,
            step=True,
            time=True,
            temperature=True,
            speed=True,
        )
    )
    print(f"NPT equilibration: {args.equil_ps:.0f} ps ({n_equil:,} steps)...")
    t0 = time.time()
    sim.step(n_equil)

    full_traj = args.out_dir / f"explicit_{args.label}_full_traj.dcd"
    sim.reporters.append(app.DCDReporter(str(full_traj), report_steps))
    print(f"NPT production: {args.prod_ns:.2f} ns ({n_prod:,} steps)...")
    sim.step(n_prod)
    print(f"Done in {(time.time() - t0) / 60:.1f} min")

    # Strip water with MDAnalysis: write just the PEG polymer (the first 261
    # atoms) to a compact topology + trajectory -- all the notebook's Rg /
    # O-C-C-O analyses need, and ~99% smaller than the solvated system.
    print("Writing water-stripped PEG-only topology + trajectory (MDAnalysis)...")
    u = mda.Universe(str(full_top), str(full_traj))
    peg = u.select_atoms("resname MES OCC MEE")  # PEG = MES + 35 OCC + MEE
    assert peg.n_atoms == 261, f"expected 261 PEG atoms, got {peg.n_atoms}"
    peg_top = args.out_dir / f"explicit_{args.label}_top.pdb"
    peg_traj = args.out_dir / f"explicit_{args.label}_traj.dcd"
    peg.write(str(peg_top))
    with mda.Writer(str(peg_traj), peg.n_atoms) as writer:
        for _ in u.trajectory:
            writer.write(peg)
    n_frames = len(u.trajectory)
    full_top.unlink()
    full_traj.unlink()
    print(f"  -> {peg_top.name}, {peg_traj.name}  ({peg.n_atoms} atoms, {n_frames} frames)")


if __name__ == "__main__":
    main()
