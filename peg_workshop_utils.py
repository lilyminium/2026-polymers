"""Boilerplate helpers for the combined PEG workshop notebook.

The notebook keeps the *teaching* code (parameter-config, loss terms,
training loop, cross-validation interpretation) inline; everything in this
module is mechanical plumbing the participant doesn't need to read line by
line. Imports here mirror what the notebook already has at the top.

Participant-facing API:
    - build_tensor_ff, write_smirnoff
    - mm_energies_at_coords, evaluate_on_parquet, held_out_grid
    - plot_cross_benchmark, scan_panel
    - run_md
    - density_from_state_csv, rg_per_frame_mda
    - block_average_se, integrated_autocorr_time
    - draw_dihedral, draw_torsions_by_id, load_split

Sections:
    - Force-field round-tripping  (tensor params -> .offxml)
    - Tensor force-field construction + descent dataset building
    - MM single-point energy evaluation
    - GBSA implicit-solvent MD (conformational sampling)
    - Explicit-solvent reading, density + Rg, convergence diagnostics
    - 2D depictions of the rotated dihedral
"""

import io
import json
import pathlib
import time

import descent.targets.energy
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import openmm
import pandas as pd
import polars as pl
import smee.converters
import tqdm
from openff.interchange import Interchange
from openff.toolkit import ForceField, Molecule, Quantity
from openmm import GBSAOBCForce, LangevinMiddleIntegrator, app
from openmm import unit as ommunit
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import Draw

__all__ = [
    "block_average_se",
    "build_tensor_ff",
    "density_from_state_csv",
    "draw_dihedral",
    "draw_torsions_by_id",
    "evaluate_on_parquet",
    "held_out_grid",
    "integrated_autocorr_time",
    "load_split",
    "mm_energies_at_coords",
    "plot_cross_benchmark",
    "rg_per_frame_mda",
    "run_md",
    "scan_panel",
    "write_smirnoff",
]

# ============================================================
# Force-field round-tripping (tensor params -> .offxml)
# ============================================================

# Some handlers live entirely in code (NAGL charges, AM1BCC) and have no
# XML attribute to write back to. We skip them when round-tripping params
# from the tensor FF back into an .offxml.
_HANDLERS_WITHOUT_XML = {"NAGLChargesHandler", "ToolkitAM1BCCHandler"}


def _update_handler(handler, potential, config):
    """Write trained values from one TensorPotential back into its SMIRNOFF handler."""
    for key, values in zip(potential.parameter_keys, potential.parameters, strict=True):
        if key.associated_handler in _HANDLERS_WITHOUT_XML:
            continue
        param = handler[key.id]
        for name, unit_, value in zip(
            potential.parameter_cols, potential.parameter_units, values, strict=True
        ):
            if name not in config.cols:
                continue
            # Multi-term torsions index each (param_id, mult) pair separately;
            # the column name picks up a 1-based suffix (k1, k2, ...) when
            # written into the SMIRNOFF param.
            name = name if key.mult is None else f"{name}{key.mult + 1}"
            try:
                setattr(param, name, Quantity(value, unit_))
            except Exception:
                # Some attributes (e.g. fixed phase/periodicity) are
                # read-only on the SMIRNOFF side; silently skip them.
                pass


def write_smirnoff(initial_ff, optimized_tensor_ff, parameters_config):
    """Round-trip trained params back into a fresh .offxml on top of the baseline."""
    out = ForceField(initial_ff.to_string())  # clone the baseline
    for potential in optimized_tensor_ff.potentials:
        if potential.type in parameters_config:
            _update_handler(
                out[potential.type], potential, parameters_config[potential.type]
            )
    return out


# ============================================================
# Tensor force-field construction
# ============================================================


def build_tensor_ff(init_ff, smiles_iter):
    """Build ONE TensorForceField + per-SMILES topology dict over a set of SMILES.

    Doing this over the UNION of train + test SMILES lets us evaluate fitted
    params on the held-out set without re-converting interchanges.
    """
    smiles_list = sorted(set(smiles_iter))
    mols = [
        Molecule.from_mapped_smiles(s, allow_undefined_stereo=True) for s in smiles_list
    ]
    interchanges = [init_ff.create_interchange(m.to_topology()) for m in mols]
    tensor_ff, tensor_tops = smee.converters.convert_interchange(interchanges)
    return tensor_ff, dict(zip(smiles_list, tensor_tops))


# ============================================================
# MM single-point energy evaluation
# ============================================================


def mm_energies_at_coords(mol, coords_array, ff):
    """Single-point MM energies (kcal/mol) at each row of `coords_array`.

    NAGL charges via the FF's NAGLCharges handler -- we do NOT pass
    `charge_from_molecules`, since Sage 2.3.0 has the charge handler
    built in. Returns a (n_frames,) array of potential energies.
    """
    ix = Interchange.from_smirnoff(force_field=ff, topology=mol.to_topology())
    system = ix.to_openmm_system()
    # Cheap throwaway integrator; we never .step() it.
    integrator = openmm.VerletIntegrator(1.0 * ommunit.femtosecond)
    ctx = openmm.Context(system, integrator)
    out = np.empty(len(coords_array))
    for i, c in enumerate(coords_array):
        ctx.setPositions((c * ommunit.angstrom).in_units_of(ommunit.nanometer))
        out[i] = (
            ctx.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(ommunit.kilocalorie_per_mole)
        )
    del ctx, integrator
    return out


def evaluate_on_parquet(ff, scans_pq, test_ids):
    """Per-scan RMSE of `ff` vs the reference energies stored in `scans_pq`.

    Walks the parquet one scan-uuid at a time, computes MM energies along
    the angle grid, references both reference and MM curves to their
    respective minima (so we compare RELATIVE energies), and reports an
    RMSE per scan. Returns a list of dicts -- one per scan -- with the
    angles, reference & MM curves, and metadata for downstream plotting.
    """
    rows = []
    for tid, group in scans_pq.groupby("torsiondrive_id", sort=False):
        group = group.sort_values("scan_idx")
        smiles = group["smiles"].iloc[0]
        torsion_indices = [int(x) for x in group["torsion_indices"].iloc[0]]
        angles = group["angle"].to_numpy()
        # Reference (QM or MLIP) energies, shifted to zero at minimum.
        ref_e = group["energy"].to_numpy()
        ref_e = ref_e - ref_e.min()
        # pandas/pyarrow reads list<list<float>> as object-arrays-of-arrays;
        # stack each grid point to (n_atoms, 3), then the points to (n_pts, n_atoms, 3).
        coords = np.array([np.stack(c) for c in group["coords"]], dtype=np.float64)
        # Build the molecule and evaluate the FF at every angle.
        mol = Molecule.from_mapped_smiles(smiles, allow_undefined_stereo=True)
        e_mm = mm_energies_at_coords(mol, coords, ff)
        e_mm -= e_mm.min()  # match reference's zero
        rmse = float(np.sqrt(np.mean((e_mm - ref_e) ** 2)))
        rows.append(
            {
                "torsiondrive_id": tid,
                "smiles": smiles,
                "n_atoms": mol.n_atoms,
                "torsion_indices": torsion_indices,
                "is_test": tid in test_ids,
                "angles": angles,
                "qm": ref_e,
                "mm": e_mm,
                "rmse": rmse,
            }
        )
    return rows


# ============================================================
# GBSA implicit-solvent MD
# ============================================================

# MD constants shared by all three FFs so the comparison is apples-to-apples.
TEMPERATURE_K = 300.0
TIMESTEP_FS = 2.0  # 4 fs + GBSA -> occasional NaN; 2 fs is reliable
N_STEPS = 250_000  # 0.5 ns at 2 fs
WRITE_EVERY = 5_000  # -> 50 frames


def add_gbsa_obc2(system, mol):
    """Attach a GBSA-OBC implicit-solvent force, reading partial charges from `system`."""
    # NonbondedForce already has the partial charges from NAGL; reuse them
    # rather than re-deriving (single source of truth).
    nb = next(f for f in system.getForces() if isinstance(f, openmm.NonbondedForce))
    # Standard (approximate) OBC2 per-element radii [nm] and screening scales --
    # the usual mbondi-style values. Implicit solvent is only the cheap
    # conformational-fingerprint sampler here, not the quantitative target.
    radii = {"H": 0.12, "C": 0.17, "N": 0.155, "O": 0.15, "S": 0.18}
    scales = {"H": 0.85, "C": 0.72, "N": 0.79, "O": 0.85, "S": 0.96}
    gb = GBSAOBCForce()
    gb.setNonbondedMethod(GBSAOBCForce.NoCutoff)
    gb.setSolventDielectric(78.5)  # water at room T
    gb.setSoluteDielectric(1.0)
    for i, atom in enumerate(mol.atoms):
        q = nb.getParticleParameters(i)[0].value_in_unit(ommunit.elementary_charge)
        gb.addParticle(q, radii.get(atom.symbol, 0.15), scales.get(atom.symbol, 0.8))
    system.addForce(gb)


def run_md(mol, ff, label, art_dir, seed=42):
    """500-ps Langevin MD in GBSA implicit solvent, writing trajectory + topology.

    `art_dir` is the directory to write `combined_<label>_traj.dcd` and
    `combined_<label>_top.pdb` into. `seed` makes the run (approximately)
    reproducible: it fixes both the integrator's random stream and the
    initial Maxwell-Boltzmann velocities, so a participant re-running the
    notebook gets the same trajectory (modulo CPU-threading non-determinism).
    Vary `seed` across runs to generate independent replicas.
    """
    ix = Interchange.from_smirnoff(force_field=ff, topology=mol.to_topology())
    system = ix.to_openmm_system()  # vacuum
    topology = ix.to_openmm_topology()
    positions = ix.positions.to_openmm()
    add_gbsa_obc2(system, mol)  # add solvent

    integrator = LangevinMiddleIntegrator(
        TEMPERATURE_K * ommunit.kelvin,
        1.0 / ommunit.picosecond,
        TIMESTEP_FS * ommunit.femtosecond,
    )
    integrator.setRandomNumberSeed(seed)
    sim = app.Simulation(topology, system, integrator)
    sim.context.setPositions(positions)
    sim.minimizeEnergy(maxIterations=500)
    sim.context.setVelocitiesToTemperature(TEMPERATURE_K * ommunit.kelvin, seed)

    # Persist a PDB of the topology so MDAnalysis can read the DCD later.
    traj_path = art_dir / f"combined_{label}_traj.dcd"
    top_path = art_dir / f"combined_{label}_top.pdb"
    state = sim.context.getState(getPositions=True)
    with open(top_path, "w") as f:
        app.PDBFile.writeFile(topology, state.getPositions(), f)

    sim.reporters.append(app.DCDReporter(str(traj_path), WRITE_EVERY))
    desc = f"{label} ({N_STEPS * TIMESTEP_FS / 1000:.0f} ps)"
    for _ in tqdm.tqdm(range(N_STEPS // WRITE_EVERY), desc=desc, unit="frame"):
        sim.step(WRITE_EVERY)  # step in DCD-write chunks (250k single steps would crawl)
    print(f"  [{label}] -> {traj_path.name}")
    return top_path, traj_path


# ============================================================
# 2D depiction of the rotated dihedral
# ============================================================


def draw_dihedral(smiles, idxs, size=(320, 260)):
    """2D depiction with the dihedral atoms/bonds highlighted (returns RGB array).

    `idxs` are 0-based atom indices into the mapped-SMILES order. We build
    the RDKit mol via OpenFF's `from_mapped_smiles(...).to_rdkit()` so the
    order is preserved (a plain `Chem.MolFromSmiles` would drop explicit Hs
    and renumber, invalidating the indices). For drawing we suppress Hs and
    remap the (heavy) dihedral atoms to their H-suppressed indices.
    """
    full = Molecule.from_mapped_smiles(smiles, allow_undefined_stereo=True).to_rdkit()
    # full -> heavy-only index map (dihedral atoms are heavy in PEG fragments).
    heavy_map, h = {}, 0
    for a in range(full.GetNumAtoms()):
        if full.GetAtomWithIdx(a).GetAtomicNum() > 1:
            heavy_map[a] = h
            h += 1
    mol = Chem.RemoveHs(full)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)  # drop the map labels for a clean drawing
    rdDepictor.Compute2DCoords(mol)
    hi_atoms = [heavy_map[int(i)] for i in idxs]
    hi_bonds = []
    for a, b in zip(hi_atoms[:-1], hi_atoms[1:]):
        bond = mol.GetBondBetweenAtoms(a, b)
        if bond is not None:
            hi_bonds.append(bond.GetIdx())
    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=hi_atoms,
        highlightBonds=hi_bonds,
    )
    drawer.FinishDrawing()
    return mpimg.imread(io.BytesIO(drawer.GetDrawingText()), format="png")


# ============================================================
# Per-scan diagnostic grid for the cross-validation step
# ============================================================


def draw_torsions_by_id(scans_df, torsiondrive_ids):
    """Grid image of torsiondrives (heavy atoms) with each driven torsion highlighted.

    `scans_df` is a long-format scan DataFrame (columns ``torsiondrive_id``,
    ``smiles`` [mapped], ``torsion_indices``); `torsiondrive_ids` selects which
    drives to draw (e.g. the held-out split). One panel per id; the four driven
    atoms + their bonds are highlighted, remapped from the mapped (with-H) order
    to the H-suppressed drawing order.
    """
    mols, atom_lists, bond_lists, legends = [], [], [], []
    for tid in torsiondrive_ids:
        row = scans_df[scans_df["torsiondrive_id"] == tid].iloc[0]
        full = Molecule.from_mapped_smiles(row["smiles"], allow_undefined_stereo=True).to_rdkit()
        heavy = {a: h for h, a in enumerate(
            i for i in range(full.GetNumAtoms()) if full.GetAtomWithIdx(i).GetAtomicNum() > 1)}
        mol = Chem.RemoveHs(full)
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)                       # drop map labels for a clean drawing
        hi = [heavy[int(i)] for i in row["torsion_indices"]]
        bonds = [mol.GetBondBetweenAtoms(a, b).GetIdx() for a, b in zip(hi[:-1], hi[1:])]
        mols.append(mol); atom_lists.append(hi); bond_lists.append(bonds)
        legends.append(f"id {tid}")
    return Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(280, 220),
                                highlightAtomLists=atom_lists, highlightBondLists=bond_lists,
                                legends=legends)


def held_out_grid(rows_by_ff, test_label, ref_label, ff_palette, ref_color="black"):
    """Per-test-set figure overlaying every FF's predicted scan vs. the reference.

    `rows_by_ff` is a dict ``{ff_label: per_scan_rows}``; each value is the
    list returned by ``evaluate_on_parquet`` for that FF on the same parquet.
    The reference curve (``"qm"`` field; identical across FFs by
    construction) is drawn once per panel in `ref_color`; each FF's MM
    curve is overlaid in its colour from `ff_palette` ``{ff_label: hex}``.
    Per-FF RMSE is appended to the legend label so participants can read
    the numbers off without flipping to S7's table.

    One row per held-out scan; left panel = energy vs. dihedral, right
    panel = 2D mol depiction with the rotated O--C--C--O atoms highlighted.
    """
    ff_labels = list(rows_by_ff)
    # Index by uuid using the first FF as the canonical scan list (held-out
    # scans are the same set of uuids for every FF, since each FF evaluates
    # the same parquet).
    canonical = [r for r in rows_by_ff[ff_labels[0]] if r["is_test"]]
    n = len(canonical)
    if n == 0:
        print(f"[held_out_grid] no held-out scans for {test_label}; skipping.")
        return

    fig, axes = plt.subplots(
        n, 2, figsize=(10, 2.7 * n), squeeze=False,
        gridspec_kw={"width_ratios": [2, 1]},
        constrained_layout=True,
    )
    for i, ref_r in enumerate(canonical):
        ax_e, ax_m = axes[i]
        order = np.argsort(ref_r["angles"])
        a = ref_r["angles"][order]
        # Reference curve: same for every FF, plot once.
        ax_e.plot(a, ref_r["qm"][order], "-o", color=ref_color, ms=4, lw=2,
                  label=ref_label, zorder=10)
        # Overlay each FF's prediction.
        for ff_label in ff_labels:
            mm_row = next(r for r in rows_by_ff[ff_label] if r["torsiondrive_id"] == ref_r["torsiondrive_id"])
            ax_e.plot(
                a, mm_row["mm"][order],
                "-", color=ff_palette[ff_label], lw=1.4,
                label=f"{ff_label} (rmse={mm_row['rmse']:.2f})",
            )
        ax_e.set_title(f"{ref_r['torsiondrive_id'][:8]}  N={ref_r['n_atoms']}", fontsize=9)
        ax_e.set_ylabel("E / kcal·mol⁻¹", fontsize=9)
        ax_e.grid(alpha=0.3); ax_e.tick_params(labelsize=7)
        if i == n - 1:
            ax_e.set_xlabel("O–C–C–O dihedral / deg", fontsize=9)
        if i == 0:
            ax_e.legend(fontsize=7, loc="best")
        ax_m.imshow(draw_dihedral(ref_r["smiles"], ref_r["torsion_indices"]))
        ax_m.axis("off"); ax_m.set_title("rotated dihedral", fontsize=8)
    fig.suptitle(f"Held-out scans: {test_label}", fontsize=11)
    plt.show()


def plot_cross_benchmark(matrix, fit_ffs, test_sets, test_colors, width=0.38):
    """Grouped bar plot of the cross-benchmark matrix (S2.4).

    `matrix` is ``{(ff_label, test_label): mean_rmse}``; bars are grouped by
    force field along x and coloured by test set from `test_colors`
    ``{test_label: hex}``. `fit_ffs` / `test_sets` give the label order (their
    keys); the RMSE value is printed above each bar.
    """
    ff_labels = list(fit_ffs)
    test_labels = list(test_sets)
    x = np.arange(len(ff_labels))
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for j, t_label in enumerate(test_labels):
        vals = [matrix[(f, t_label)] for f in ff_labels]
        bars = ax.bar(x + (j - 0.5) * width, vals, width, label=t_label,
                      color=test_colors[t_label], alpha=0.85)
        for bx, v in zip(bars, vals):
            ax.text(bx.get_x() + bx.get_width() / 2, v + 0.01, f"{v:.2f}",
                    ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(ff_labels, rotation=15, ha="right")
    ax.set_ylabel("test-set mean RMSE / kcal·mol⁻¹")
    ax.set_title("Cross-benchmark: each fit vs both reference levels")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.show()


def scan_panel(ax, test_label, uuid, xval_cache, ff_labels, ff_palette,
               ylabel=True, legend=False):
    """Draw one held-out torsiondrive panel onto `ax`: every FF vs the reference.

    Pulls the per-scan rows for `uuid` from `xval_cache` ``{(ff_label,
    test_label): rows}`` (as built in S2.4), plots the black reference curve
    once, then overlays each FF in `ff_palette` ``{ff_label: hex}`` with its
    per-scan RMSE in the legend. `ylabel` / `legend` toggle the y-axis label and
    legend so a grid of panels only labels its edges.
    """
    rows = {ff: next(r for r in xval_cache[(ff, test_label)] if r["torsiondrive_id"] == uuid)
            for ff in ff_labels}
    order = np.argsort(rows[ff_labels[0]]["angles"])
    a = rows[ff_labels[0]]["angles"][order]
    ax.plot(a, rows[ff_labels[0]]["qm"][order], "-o", color="black", ms=3, lw=2,
            label="reference", zorder=10)
    for ff in ff_labels:
        ax.plot(a, rows[ff]["mm"][order], "-", color=ff_palette[ff], lw=1.4,
                label=f"{ff.split('(')[0].strip()} (rmse={rows[ff]['rmse']:.2f})")
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=7)
    if ylabel:
        ax.set_ylabel("E / kcal·mol⁻¹", fontsize=8)
    if legend:
        ax.legend(fontsize=6, loc="best")


# ============================================================
# Shared train/test split (single source of truth)
# ============================================================


def load_split(data_dir="data"):
    """Load the shared 5-torsiondrive held-out split written by 01_download_qca.py.

    Returns the parsed ``split.json`` dict. The split is keyed by torsiondrive
    record id (``test_uuids``) and also carries the mapped SMILES
    (``test_cmiles``), so the SAME molecules are held out for both the DFT
    (Fit 1) and Egret-1 (Fit 2) datasets. Split a descent dataset with, e.g.::

        split = load_split(DATA)
        test = set(split["test_cmiles"])
        train_ds = ds.filter(lambda r: r["smiles"] not in test)
        test_ds  = ds.filter(lambda r: r["smiles"] in test)
    """
    path = pathlib.Path(data_dir) / "split.json"
    return json.loads(path.read_text())


# ============================================================
# Explicit-solvent MD: reading, density, and Rg (MDAnalysis)
# ============================================================


def _col(columns, substr):
    """First column name containing `substr` (case-insensitive).

    OpenMM StateDataReporter / WaSP CSVs name columns with units, e.g.
    ``"Density (g/mL)"`` and a leading ``#"Step"`` -- substring match is more
    robust than an exact name. ``polars.read_csv`` parses both headers fine.
    """
    for c in columns:
        if substr.lower() in c.lower():
            return c
    raise KeyError(f"no column matching {substr!r} in {list(columns)}")


def density_from_state_csv(csv_path, equil_fraction=0.5):
    """Density trace from an OpenMM state-data CSV.

    Returns a dict with ``time_ps``, ``density`` (g/mL), ``volume`` (nm^3) arrays
    and the mean +/- std over the last ``equil_fraction`` of the run (the
    'equilibrated' estimate). Density equilibrates fast, so the tail average is
    a fair value; the full trace lets you SEE how fast it plateaus.
    """
    df = pd.read_csv(str(csv_path))
    density = df[_col(df.columns, "Density")].to_numpy()
    time_ps = df[_col(df.columns, "Time")].to_numpy()
    try:
        volume = df[_col(df.columns, "Volume")].to_numpy()
    except KeyError:
        volume = np.full_like(density, np.nan)
    tail = density[int((1.0 - equil_fraction) * len(density)):]
    return {
        "time_ps": time_ps,
        "density": density,
        "volume": volume,
        "mean": float(np.mean(tail)),
        "std": float(np.std(tail)),
    }


def block_average_se(x, n_blocks=5):
    """Mean and block-averaged standard error of a 1-D series.

    Splits `x` into `n_blocks` contiguous blocks, takes each block mean, and
    returns ``(mean, SE_of_block_means)`` -- a cheap autocorrelation-aware error
    bar for correlated MD observables like Rg (where adjacent frames are not
    independent, so the naive std/sqrt(N) underestimates the uncertainty).
    """
    x = np.asarray(x, dtype=float)
    if x.size < n_blocks:
        return float(x.mean()), float(x.std() / max(np.sqrt(x.size), 1.0))
    means = np.array([b.mean() for b in np.array_split(x, n_blocks)])
    return float(x.mean()), float(means.std(ddof=1) / np.sqrt(n_blocks))


def rg_per_frame_mda(top, traj, selection="resname MES OCC MEE", start_frame=0, stride=1):
    """Mass-weighted radius of gyration of `selection` per frame, in **nm**.

    Uses MDAnalysis' ``radius_of_gyration`` (returns Angstrom) on just the
    polymer atoms (``selection``), so the surrounding water is excluded. This is
    the FF-sensitive structural observable for the workshop. Divide-by-10
    converts Angstrom -> nm to match the WaSP reference.

    ``stride`` subsamples the trajectory (``traj[start_frame::stride]``) -- use it
    to keep long production trajectories (10k+ frames) fast, since Rg
    autocorrelates over ~ns so every frame is not independent anyway.
    """
    u = mda.Universe(str(top), str(traj))
    sel = u.select_atoms(selection)
    if sel.n_atoms == 0:
        raise ValueError(f"selection {selection!r} matched no atoms")
    rg = [sel.radius_of_gyration() for _ in u.trajectory[start_frame::stride]]
    return np.array(rg) / 10.0


# ============================================================
# Convergence diagnostics ("how long to simulate?")
# ============================================================


def integrated_autocorr_time(x):
    """Integrated autocorrelation time (in frames) of a 1-D series.

    tau = 1 + 2 * sum_{t>=1} rho(t), summed over the initial positive sequence
    (truncated at the first non-positive rho -- Geyer's IPS estimator). The
    autocorrelation rho is computed with statsmodels' FFT estimator rather than
    a hand-rolled O(n^2) loop. Multiply by the frame spacing for a physical
    correlation time; a well-sampled mean needs a run many tau long.
    """
    from statsmodels.tsa.stattools import acf

    x = np.asarray(x, dtype=float)
    if x.size < 2 or np.allclose(x, x[0]):
        return 1.0
    rho = acf(x, nlags=x.size - 1, fft=True)
    tau = 1.0
    for r in rho[1:]:
        if r <= 0:
            break
        tau += 2.0 * float(r)
    return tau
