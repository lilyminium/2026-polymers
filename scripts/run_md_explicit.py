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
from openff.pablo import STD_CCD_CACHE, ResidueDefinition, topology_from_pdb
from openff.pablo.residue import BondDefinition
from openff.toolkit import ForceField
from openmm import MonteCarloBarostat, app
from openmm import unit as ommunit

def build_pablo_topology(pdb_path):
    """Solvated PEG topology via openff-pablo (mirrors the notebook's Part 1.3 cell).

    The PEG is tiled CPL -> OCC x36 -> CPR, linked head-to-tail by the C2->O1
    ether bond (a residue's C2 bonds the NEXT residue's O1). Water resolves from
    the CCD automatically.
    """
    ether = BondDefinition.with_defaults("C2", "O1", order=1)
    cpl = ResidueDefinition.from_smiles(
        mapped_smiles="[C:1]([H:2])([H:3])([H:4])[H:5]",
        atom_names={1: "C2", 2: "H21", 3: "H22", 4: "H23", 5: "H24"},
        residue_name="CPL", leaving_atoms=(5,), linking_bond=ether,
    )
    occ = ResidueDefinition.from_smiles(
        mapped_smiles="[O:1]([C:2]([H:4])([H:5])[C:3]([H:6])([H:7])[H:9])[H:8]",
        atom_names={1: "O1", 2: "C1", 3: "C2", 4: "H11", 5: "H12",
                    6: "H21", 7: "H22", 8: "HO1", 9: "H23"},
        residue_name="OCC", leaving_atoms=(8, 9), linking_bond=ether,
    )
    cpr = ResidueDefinition.from_smiles(
        mapped_smiles="[O:1]([C:2]([H:3])([H:4])[H:5])[H:6]",
        atom_names={1: "O1", 2: "C2", 3: "H21", 4: "H22", 5: "H23", 6: "HO1"},
        residue_name="CPR", leaving_atoms=(6,), linking_bond=ether,
    )
    # Water resolves from the CCD; we pass only the three custom PEG residues.
    return topology_from_pdb(
        str(pdb_path), residue_library=STD_CCD_CACHE.with_([cpl, occ, cpr])
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--offxml",
        default="openff-2.3.0.offxml",
        help="Fitted SMIRNOFF FF (or baseline name). Default Sage 2.3.0.",
    )
    parser.add_argument(
        "--pdb",
        type=pathlib.Path,
        default=pathlib.Path("data/wasp_reference/peg_solv_36mer.pdb"),
        help="Starting solvated PEG PDB. Default: %(default)s.",
    )
    parser.add_argument("--device", default="CUDA", help="CUDA / CPU / OpenCL")
    parser.add_argument(
        "--equil-ps", type=float, default=100.0, help="NPT equilibration"
    )
    parser.add_argument("--prod-ns", type=float, default=2.0, help="NPT production")
    parser.add_argument("--report-ps", type=float, default=10.0)
    parser.add_argument(
        "--temperature-k",
        type=float,
        default=300.0,
        help="Temperature in K. Default: %(default)s.",
    )
    parser.add_argument(
        "--pressure-atm",
        type=float,
        default=1.0,
        help="Pressure in atm. Default: %(default)s.",
    )
    parser.add_argument(
        "--timestep-fs",
        type=float,
        default=2.0,
        help="Integrator timestep in fs. Default: %(default)s.",
    )
    parser.add_argument(
        "--friction-per-ps",
        type=float,
        default=1.0,
        help="Langevin friction in 1/ps. Default: %(default)s.",
    )
    parser.add_argument(
        "--barostat-freq",
        type=int,
        default=25,
        help="Monte Carlo barostat frequency in steps. Default: %(default)s.",
    )
    parser.add_argument("--label", default="dft")
    parser.add_argument(
        "--out-dir", type=pathlib.Path, default="artifacts/md_explicit"
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Force field : {args.offxml}")
    print(f"Start PDB   : {args.pdb}")
    print("Building topology with openff-pablo (CPL/OCC/CPR residue definitions)...")
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
        args.pressure_atm * ommunit.atmosphere,
        args.temperature_k * ommunit.kelvin,
        args.barostat_freq,
    )
    system.addForce(barostat)
    integrator = openmm.LangevinMiddleIntegrator(
        args.temperature_k * ommunit.kelvin,
        args.friction_per_ps / ommunit.picosecond,
        args.timestep_fs * ommunit.femtosecond,
    )
    platform = openmm.Platform.getPlatformByName(args.device)
    sim = app.Simulation(omm_top, system, integrator, platform)
    sim.context.setPositions(positions)

    print("Minimising...")
    sim.minimizeEnergy(maxIterations=1000)
    sim.context.setVelocitiesToTemperature(args.temperature_k * ommunit.kelvin)

    report_steps = int(args.report_ps * 1000 / args.timestep_fs)
    n_equil = int(args.equil_ps * 1000 / args.timestep_fs)
    n_prod = int(args.prod_ns * 1e6 / args.timestep_fs)

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
    peg_selection = "resname CPL OCC CPR"  # PEG = CPL + 36 OCC + CPR
    peg = u.select_atoms(peg_selection)
    expected_peg_atoms = 261
    if peg.n_atoms != expected_peg_atoms:
        raise ValueError(f"expected {expected_peg_atoms} PEG atoms, got {peg.n_atoms}")

    peg_top = args.out_dir / f"explicit_{args.label}_top.pdb"
    peg_traj = args.out_dir / f"explicit_{args.label}_traj.dcd"
    n_frames = len(u.trajectory)

    peg.write(str(peg_top))
    with mda.Writer(str(peg_traj), peg.n_atoms) as writer:
        for _ in u.trajectory:
            writer.write(peg)
    print(f"  -> {peg_top.name}, {peg_traj.name}  ({peg.n_atoms} atoms, {n_frames} frames)")


if __name__ == "__main__":
    main()
