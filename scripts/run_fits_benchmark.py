"""Run Fit 1 (DFT) + Fit 2 (Egret-1) and the cross-benchmark, headless.

Mirrors the notebook's fitting machinery exactly so the produced offxmls and the
cross-benchmark matrix match what the notebook computes. Used to (a) validate the
central scientific claim (each fit wins on its own reference) and (b) precompute
artifacts/peg_fitted_{dft,egret}.offxml.

    pixi run python -u scripts/run_fits_benchmark.py [n_epochs]
"""

import pathlib
import sys
import time

import numpy as np
import polars as pl
import torch
from datasets import load_from_disk
from openff.toolkit import ForceField

import descent.targets.energy
import descent.train

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from peg_workshop_utils import (
    build_tensor_ff,
    evaluate_on_parquet,
    load_split,
    write_smirnoff,
)

DATA = ROOT / "data"
ART = ROOT / "artifacts"
N_EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

init_ff = ForceField("openff-2.3.0.offxml")
CFG = {
    "ProperTorsions": descent.train.ParameterConfig(
        cols=["k"], scales={"k": 1.0e1}, limits={"k": [0.0, None]}
    )
}


def fit(train, tops_tr, test, tops_te, tensor_ff, label):
    trn = descent.train.Trainable(force_field=tensor_ff, parameters=CFG, attributes={})
    p = trn.to_values()
    e0, f0 = (x.detach() for x in _terms(trn.to_force_field(p), train, tops_tr))
    e0 = e0 if e0 > 1e-12 else torch.tensor(1.0)
    opt = torch.optim.Adam([p], lr=1e-3, amsgrad=True)
    t0 = time.time()
    for ep in range(N_EPOCHS):
        ff = trn.to_force_field(p)
        le, lf = _terms(ff, train, tops_tr)
        loss = 0.5 * (le / e0 + lf / f0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if ep % 200 == 0 or ep == N_EPOCHS - 1:
            lte, lfte = _terms(trn.to_force_field(p), test, tops_te)
            tl = (0.5 * (lte / e0 + lfte / f0)).item()
            print(f"  [{label}] ep {ep:4d}  train={loss.item():.4f}  test={tl:.4f}")
    out = ART / f"peg_fitted_{label}.offxml"
    write_smirnoff(init_ff, trn.to_force_field(p), CFG).to_file(str(out))
    print(f"  [{label}] {time.time()-t0:.0f}s -> {out.name}")
    return out


def _terms(ff, ds, tops):
    er, ep, fr, fp = descent.targets.energy.predict(ds, ff, tops, "mean")
    return ((ep - er) ** 2).mean(), ((fp - fr) ** 2).mean()


def main():
    split = load_split(DATA)
    test_uuids = set(split["test_uuids"])
    # DFT and Egret parquets share uuid->smiles, so the held-out SMILES set is
    # the same for both fits and the cross-benchmark: compute it once.
    pqs = {
        label: pl.read_parquet(next((DATA / sub).glob("*scans*.parquet")))
        for label, sub in [("dft", "qca_peg_td"), ("egret", "mlip_peg_td")]
    }
    test_smiles = set(pqs["dft"].filter(pl.col("uuid").is_in(test_uuids))["smiles"].unique())

    offxmls = {}
    for label, sub in [("dft", "qca_peg_td"), ("egret", "mlip_peg_td")]:
        ds = load_from_disk(str(DATA / sub / "descent_dataset"))
        train = ds.filter(lambda r: r["smiles"] not in test_smiles)
        test = ds.filter(lambda r: r["smiles"] in test_smiles)
        tff, tops = build_tensor_ff(init_ff, set(train["smiles"]) | set(test["smiles"]))
        offxmls[label] = fit(
            train,
            {s: tops[s] for s in train["smiles"]},
            test,
            {s: tops[s] for s in test["smiles"]},
            tff,
            label,
        )

    # Cross-benchmark
    ffs = {
        "baseline": init_ff,
        "dft": ForceField(str(offxmls["dft"])),
        "egret": ForceField(str(offxmls["egret"])),
    }
    print("\nCross-benchmark mean RMSE on 5 held-out torsiondrives (kcal/mol):")
    print(f"{'':<12s}{'DFT test':>12s}{'Egret test':>12s}")
    for fl, ff in ffs.items():
        row = []
        for tl in ("dft", "egret"):
            rows = evaluate_on_parquet(ff, pqs[tl], test_smiles)
            row.append(float(np.mean([r["rmse"] for r in rows if r["is_test"]])))
        print(f"{fl:<12s}{row[0]:>12.3f}{row[1]:>12.3f}")


if __name__ == "__main__":
    main()
