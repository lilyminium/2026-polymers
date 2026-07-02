"""Boilerplate helpers for the combined PEG workshop notebook.

The notebook keeps the *teaching* code (parameter-config, loss terms,
training loop, cross-validation interpretation) inline; everything in this
module is mechanical plumbing the participant doesn't need to read line by
line. Imports here mirror what the notebook already has at the top.

Participant-facing API:
    - write_smirnoff
    - mm_energies_at_coords, evaluate_on_parquet
    - plot_cross_benchmark, scan_panel
    - block_average_se
    - draw_dihedral, draw_torsions_by_id

Sections:
    - Force-field round-tripping  (tensor params -> .offxml)
    - MM single-point energy evaluation
    - 2D depictions of the rotated dihedral + cross-validation panels
    - Block-averaged standard error for correlated MD observables
"""

import io
import warnings

import descent.train
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import smee
from openff.interchange import Interchange
from openff.interchange.drivers import get_openmm_energies
from openff.toolkit import ForceField, Molecule, Quantity
from openff.toolkit.typing.engines.smirnoff.parameters import ParameterHandler
from openff.units import unit as offunit
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import Draw

__all__ = [
    "block_average_se",
    "draw_dihedral",
    "draw_torsions_by_id",
    "evaluate_on_parquet",
    "mm_energies_at_coords",
    "plot_cross_benchmark",
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


def _update_handler(
    handler: ParameterHandler,
    potential: smee.TensorPotential,
    config: descent.train.ParameterConfig,
) -> None:
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
            col_name = name if key.mult is None else f"{name}{key.mult + 1}"
            try:
                setattr(param, col_name, Quantity(value, unit_))
            except (AttributeError, TypeError, ValueError):
                # Some attributes (e.g. fixed phase/periodicity) are
                # read-only on the SMIRNOFF side; skip them, but warn so a
                # genuinely-failed round-trip isn't silent.
                warnings.warn(
                    f"Could not write trained value back to '{col_name}' on "
                    f"{key.associated_handler} parameter {key.id!r}; skipping.",
                    stacklevel=2,
                )


def write_smirnoff(
    initial_ff: ForceField,
    optimized_tensor_ff: smee.TensorForceField,
    parameters_config: dict[str, descent.train.ParameterConfig],
) -> ForceField:
    """Round-trip trained params back into a fresh .offxml on top of the baseline."""
    out = ForceField(initial_ff.to_string())  # clone the baseline
    for potential in optimized_tensor_ff.potentials:
        if potential.type in parameters_config:
            _update_handler(
                out[potential.type], potential, parameters_config[potential.type]
            )
    return out


# ============================================================
# MM single-point energy evaluation
# ============================================================


def mm_energies_at_coords(mol: Molecule, coords_array: np.ndarray, ff: ForceField) -> np.ndarray:
    """Single-point MM energies (kcal/mol) at each row of `coords_array`.

    NAGL charges via the FF's NAGLCharges handler -- we do NOT pass
    `charge_from_molecules`, since Sage 2.3.0 has the charge handler
    built in. Returns a (n_frames,) array of potential energies.
    """
    interchange = Interchange.from_smirnoff(force_field=ff, topology=mol.to_topology())
    out = np.empty(len(coords_array))
    for frame_index, coords in enumerate(coords_array):
        interchange.positions = coords * offunit.angstrom
        out[frame_index] = get_openmm_energies(interchange).total_energy.m_as(
            "kilocalorie/mole"
        )
    return out


def evaluate_on_parquet(ff: ForceField, scans_pq: pd.DataFrame, test_ids: set[str]) -> list[dict]:
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
        # Reference (QM or MLIP) energies, shifted to zero at its minimum.
        ref_e = group["energy"].to_numpy()
        ref_min_index = int(np.argmin(ref_e))
        ref_e = ref_e - ref_e[ref_min_index]
        # pandas/pyarrow reads list<list<float>> as object-arrays-of-arrays;
        # stack each grid point to (n_atoms, 3), then the points to (n_pts, n_atoms, 3).
        coords = np.array([np.stack(c) for c in group["coords"]], dtype=np.float64)
        # Build the molecule and evaluate the FF at every angle.
        mol = Molecule.from_mapped_smiles(smiles, allow_undefined_stereo=True)
        e_mm = mm_energies_at_coords(mol, coords, ff)
        e_mm -= e_mm[ref_min_index]  # zero at the reference minimum grid point
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
# 2D depiction of the rotated dihedral
# ============================================================


def _highlighted_torsion_mol(smiles: str, idxs: list[int]) -> tuple[Chem.Mol, list[int], list[int]]:
    """RDKit mol plus heavy-atom torsion highlights from mapped-SMILES indices."""
    full = Molecule.from_mapped_smiles(smiles, allow_undefined_stereo=True).to_rdkit()
    for atom in full.GetAtoms():
        atom.SetAtomMapNum(0)
    for i in idxs:
        full.GetAtomWithIdx(int(i)).SetAtomMapNum(int(i) + 1)

    mol = Chem.RemoveHs(full)
    rdDepictor.Compute2DCoords(mol)
    mapped_atoms = {
        atom.GetAtomMapNum(): atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomMapNum()
    }
    hi_atoms = [mapped_atoms[int(i) + 1] for i in idxs]
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)

    hi_bonds = []
    for atom_1, atom_2 in zip(hi_atoms[:-1], hi_atoms[1:]):
        bond = mol.GetBondBetweenAtoms(atom_1, atom_2)
        if bond is not None:
            hi_bonds.append(bond.GetIdx())
    return mol, hi_atoms, hi_bonds


def draw_dihedral(smiles: str, idxs: list[int], size: tuple[int, int] = (320, 260)) -> np.ndarray:
    """2D depiction with the dihedral atoms/bonds highlighted (returns RGB array).

    `idxs` are 0-based atom indices into the mapped-SMILES order. We temporarily
    tag those atoms with RDKit atom-map numbers, suppress Hs, and recover the
    highlighted heavy atoms from those tags.
    """
    mol, hi_atoms, hi_bonds = _highlighted_torsion_mol(smiles, idxs)
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


def draw_torsions_by_id(scans_df: pd.DataFrame, torsiondrive_ids: list[str]):
    """Grid image of torsiondrives (heavy atoms) with each driven torsion highlighted.

    `scans_df` is a long-format scan DataFrame (columns ``torsiondrive_id``,
    ``smiles`` [mapped], ``torsion_indices``); `torsiondrive_ids` selects which
    drives to draw (e.g. the held-out split). One panel per id; the four driven
    atoms + their bonds are highlighted.
    """
    mols, atom_lists, bond_lists, legends = [], [], [], []
    for tid in torsiondrive_ids:
        row = scans_df[scans_df["torsiondrive_id"] == tid].iloc[0]
        mol, hi, bonds = _highlighted_torsion_mol(row["smiles"], row["torsion_indices"])
        mols.append(mol)
        atom_lists.append(hi)
        bond_lists.append(bonds)
        legends.append(f"id {tid}")
    return Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(280, 220),
                                highlightAtomLists=atom_lists, highlightBondLists=bond_lists,
                                legends=legends)


def plot_cross_benchmark(
    matrix: dict[tuple[str, str], float],
    fit_ffs: dict[str, ForceField],
    test_sets: dict,
    test_colors: dict[str, str],
    width: float = 0.38,
) -> None:
    """Grouped bar plot of the cross-benchmark matrix (S2.4).

    `matrix` is ``{(ff_label, test_label): mean_rmse}``; bars are grouped by
    force field along x and coloured by test set from `test_colors`
    ``{test_label: hex}``. `fit_ffs` / `test_sets` give the label order (their
    keys); the RMSE value is printed above each bar.
    """
    ff_labels = list(fit_ffs)
    test_labels = list(test_sets)
    x = np.arange(len(ff_labels))
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True, dpi=300)
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


def scan_panel(
    ax,
    test_label: str,
    uuid: str,
    xval_cache: dict[tuple[str, str], list[dict]],
    ff_labels: list[str],
    ff_palette: dict[str, str],
    ylabel: bool = True,
    legend: bool = False,
) -> None:
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
# Block-averaged standard error for correlated MD observables
# ============================================================


def block_average_se(x: np.ndarray, n_blocks: int = 5) -> tuple[float, float]:
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
