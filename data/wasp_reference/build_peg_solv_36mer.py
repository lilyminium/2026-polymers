#!/usr/bin/env python
"""Download the WaSP conf1 production PDB and convert it (with MDAnalysis) into the
residue-tiled peg_solv_36mer.pdb that openff-pablo reads.

Required for demo-ing the path where we take advantage of consistent residue
and atom names.

1. Download ``production_topology.pdb`` from the public shirtsgroup/WaSP_simulations
   repo (a single PEG chain CH3-O-(CH2CH2O)36-CH3 in 14,061 TIP3P waters).
2. Walk the PEG backbone using the PDB's own CONECT bonds, then use MDAnalysis to
   reorder + re-residue + rename the 261 PEG atoms into a regular tiling that reads
   CPL -> OCC x36 -> CPR with resids numbered from 1:
     * a start cap  CPL = CH3       (resid 1, 4 atoms)
     * 36 OCC monomers  O-CH2-CH2   (resids 2..37, 7 atoms each)
     * an end cap   CPR = O-CH3     (resid 38, 5 atoms)

    pixi run python data/wasp_reference/build_peg_solv_36mer.py
"""

import pathlib
import urllib.request

import MDAnalysis as mda

RAW_URL = (
    "https://raw.githubusercontent.com/shirtsgroup/WaSP_simulations/main/"
    "wasp_sims/peg_modified/Espaloma-AM1-BCC/conf1/production/production_topology.pdb"
)
HERE = pathlib.Path(__file__).resolve().parent  # data/wasp_reference/
RAW_PDB = HERE / "raw_peg.pdb"
OUT_PDB = HERE / "peg_solv_36mer.pdb"
N_MONOMERS = 36

# Atom order within each residue (residue-contiguous output so readers don't fragment).
ORDER_IN_RES = {
    "CPL": ["C2", "H21", "H22", "H23"],
    "OCC": ["O1", "C1", "H11", "H12", "C2", "H21", "H22"],
    "CPR": ["O1", "C2", "H21", "H22", "H23"],
}


def download():
    if RAW_PDB.exists():
        print(f"Using cached {RAW_PDB}  ({RAW_PDB.stat().st_size / 1e6:.1f} MB)")
        return
    RAW_PDB.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {RAW_URL}")
    urllib.request.urlretrieve(RAW_URL, RAW_PDB)
    print(f"  -> {RAW_PDB}  ({RAW_PDB.stat().st_size / 1e6:.1f} MB)")


def peg_residue_map(peg):
    """Walk the PEG backbone via the CONECT bonds -> {local_idx: (resname, resid, name)}."""
    elem = peg.elements
    g2l = {g: i for i, g in enumerate(peg.atoms.indices)}
    nbr = [[] for _ in range(peg.n_atoms)]
    for bond in peg.bonds:
        a, b = bond.atoms[0].index, bond.atoms[1].index
        if a in g2l and b in g2l:
            nbr[g2l[a]].append(g2l[b])
            nbr[g2l[b]].append(g2l[a])

    def heavy(i):
        return [j for j in nbr[i] if elem[j] != "H"]

    def hyd(i):
        return [j for j in nbr[i] if elem[j] == "H"]

    methyls = [
        i
        for i in range(peg.n_atoms)
        if elem[i] == "C" and len(hyd(i)) == 3 and len(heavy(i)) == 1
    ]
    assert len(methyls) == 2, f"expected 2 terminal methyls, got {len(methyls)}"

    seq, prev, cur = [methyls[0]], None, methyls[0]
    while True:
        nxt = [j for j in heavy(cur) if j != prev]
        if not nxt:
            break
        prev, cur = cur, nxt[0]
        seq.append(cur)
    # seq = [Cme, O, C, C, O, C, C, ..., O, Cme]: methyl, then OCC x36, then O, methyl.
    assert (
        len(seq) == 111 and seq[-1] in methyls
    ), f"unexpected backbone length {len(seq)}"

    res = {}

    def put(i, rn, ri, nm):
        res[i] = (rn, ri, nm)

    def put_h(c_idx, rn, ri, carbon):
        for k, h in enumerate(hyd(c_idx), start=1):
            put(h, rn, ri, f"H{carbon}{k}")

    # Start cap CPL = CH3 (resid 1; methyl named C2 so it carries the forward C2->O1 link).
    put(seq[0], "CPL", 1, "C2")
    put_h(seq[0], "CPL", 1, "2")
    # 36 OCC monomers = O-CH2-CH2 (resids 2..37); O1 links back, C2 links forward.
    for m in range(N_MONOMERS):
        o1, c1, c2 = seq[1 + 3 * m], seq[2 + 3 * m], seq[3 + 3 * m]
        rid = 2 + m
        put(o1, "OCC", rid, "O1")
        put(c1, "OCC", rid, "C1")
        put_h(c1, "OCC", rid, "1")
        put(c2, "OCC", rid, "C2")
        put_h(c2, "OCC", rid, "2")
    # End cap CPR = O-CH3 (resid 38; methyl named C2).
    put(seq[109], "CPR", 38, "O1")
    put(seq[110], "CPR", 38, "C2")
    put_h(seq[110], "CPR", 38, "2")
    assert len(res) == 261
    return res


def build_peg_pdb_lines(peg, res):
    """MDAnalysis reorder + re-residue + rename of the PEG -> its PDB ATOM lines."""
    order = sorted(
        range(peg.n_atoms),
        key=lambda i: (res[i][1], ORDER_IN_RES[res[i][0]].index(res[i][2])),
    )
    atom_resindex, resids, resnames = [], [], []
    last, ridx = None, -1
    for i in order:
        rn, ri = res[i][0], res[i][1]
        if ri != last:
            last, ridx = ri, ridx + 1
            resids.append(ri)
            resnames.append(rn)
        atom_resindex.append(ridx)

    pegU = mda.Universe.empty(
        peg.n_atoms,
        n_residues=len(resids),
        atom_resindex=atom_resindex,
        trajectory=True,
    )
    pegU.add_TopologyAttr("names", [res[i][2] for i in order])
    pegU.add_TopologyAttr("resnames", resnames)
    pegU.add_TopologyAttr("resids", resids)
    pegU.add_TopologyAttr("elements", [peg.elements[i] for i in order])
    pegU.add_TopologyAttr("chainIDs", ["A"] * peg.n_atoms)
    pegU.atoms.positions = peg.positions[order]

    # Remap the PEG's CONECT bonds into the new atom order and attach them, so the
    # written PDB carries CONECT records -- openff-pablo needs them to establish
    # the inter-residue links of these custom (non-CCD) residues.
    g2l = {g: i for i, g in enumerate(peg.atoms.indices)}
    new_pos = {old: new for new, old in enumerate(order)}
    bonds = sorted(
        {
            tuple(sorted((new_pos[g2l[b.atoms[0].index]], new_pos[g2l[b.atoms[1].index]])))
            for b in peg.bonds
            if b.atoms[0].index in g2l and b.atoms[1].index in g2l
        }
    )
    pegU.add_TopologyAttr("bonds", bonds)

    tmp = OUT_PDB.with_name("_peg_tmp.pdb")
    pegU.atoms.write(str(tmp), bonds="conect")
    text = tmp.read_text().splitlines()
    tmp.unlink()
    atom_lines = [ln for ln in text if ln.startswith(("ATOM", "HETATM"))]
    conect_lines = [ln for ln in text if ln.startswith("CONECT")]
    assert len(atom_lines) == 261, f"PEG block has {len(atom_lines)} atoms"
    assert conect_lines, "no CONECT records written for the PEG"
    return atom_lines, conect_lines


def water_lines_renamed(raw_text):
    """Original HOH lines, atoms renamed O1x/H1x/H2x -> O/H1/H2 (chains/resids kept)."""
    name_map = {"O1x": "O", "H1x": "H1", "H2x": "H2"}
    out = []
    for ln in raw_text.splitlines():
        if ln.startswith(("ATOM", "HETATM")) and ln[17:20].strip() == "HOH":
            nm = ln[12:16].strip()
            out.append(ln[:12] + f" {name_map.get(nm, nm):<3s}" + ln[16:])
    return out


def main():
    download()
    raw_text = RAW_PDB.read_text()
    cryst = [ln for ln in raw_text.splitlines() if ln.startswith("CRYST1")]

    u = mda.Universe(str(RAW_PDB))
    peg = u.select_atoms("resname peg")
    assert peg.n_atoms == 261, f"expected 261 PEG atoms, got {peg.n_atoms}"

    res = peg_residue_map(peg)
    peg_lines, peg_conect = build_peg_pdb_lines(peg, res)
    water_lines = water_lines_renamed(raw_text)

    out = [*cryst, *peg_lines, "TER", *water_lines, *peg_conect, "END"]
    OUT_PDB.write_text("\n".join(out) + "\n")
    print(
        f"Wrote {OUT_PDB}\n"
        f"  PEG: CPL + {N_MONOMERS} OCC + CPR ({len(peg_lines)} atoms); "
        f"{len(water_lines)} water atoms renamed to O/H1/H2"
    )


if __name__ == "__main__":
    main()
