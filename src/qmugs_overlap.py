"""How many of our molecules already have precomputed quantum descriptors in QMugs?

QMugs contains ~665k drug-like molecules from ChEMBL with GFN2-xTB optimised
geometries and DFT (wB97X-D/def2-SVP) properties. Where a molecule matches, its
descriptors are a table lookup rather than a quantum chemistry calculation.

    # just compute and cache our InChIKeys
    python -m src.qmugs_overlap --datasets esol lipo bace bbbp

    # with the QMugs summary table downloaded locally
    python -m src.qmugs_overlap --datasets esol lipo --qmugs path/to/summary.csv

Matching is reported at two levels:
  exact     full InChIKey; same connectivity, stereochemistry and protonation
  skeleton  first block only (14 chars); same 2D connectivity, ignoring stereo,
            isotopes and salt/charge state

Skeleton matching is the useful number for descriptor transfer: a different
enantiomer or counter-ion has essentially the same HOMO/LUMO and dipole for these
purposes, while a different skeleton does not. Both are reported so the choice is
visible rather than buried.

QMugs is distributed by ETH Zurich (Isert et al., Sci Data 2022). The summary
table carrying the per-molecule properties is the only part needed here; the full
conformer archive is not. `openqdc` also exposes QMugs programmatically.
"""
from __future__ import annotations

import argparse
import os

import pandas as pd
from rdkit import Chem, RDLogger

from .data import DATASETS, load_dataframe

RDLogger.DisableLog("rdApp.*")

SMILES_COLS = ["smiles", "SMILES", "canonical_smiles", "Smiles", "smi"]


def inchikeys(smiles: list[str]) -> pd.DataFrame:
    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        key = None
        if mol is not None:
            try:
                key = Chem.MolToInchiKey(mol)
            except Exception:
                key = None
        rows.append(dict(smiles=smi, inchikey=key,
                         skeleton=key.split("-")[0] if key else None))
    return pd.DataFrame(rows)


def load_qmugs(path: str) -> pd.DataFrame:
    if path.endswith((".parquet", ".pq")):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    col = next((c for c in SMILES_COLS if c in df.columns), None)
    if col is None:
        raise SystemExit(
            f"no SMILES column found in {path}; columns are {list(df.columns)[:20]}")
    print(f"[qmugs] {len(df):,} rows, SMILES column '{col}'")

    if "inchikey" not in df.columns:
        print("[qmugs] computing InChIKeys (this is the slow part) ...")
        keys = inchikeys(df[col].astype(str).tolist())
        df = df.reset_index(drop=True)
        df["inchikey"] = keys["inchikey"]
    df["skeleton"] = df["inchikey"].astype(str).str.split("-").str[0]
    return df.dropna(subset=["inchikey"]).drop_duplicates(subset="inchikey")


def report(ds: str, ours: pd.DataFrame, qmugs: pd.DataFrame | None) -> dict:
    n = len(ours)
    valid = ours["inchikey"].notna().sum()
    row = dict(dataset=ds, n_molecules=n, inchikey_ok=int(valid))
    if qmugs is None:
        return row

    q_exact = set(qmugs["inchikey"])
    q_skel = set(qmugs["skeleton"])
    exact = ours["inchikey"].isin(q_exact).sum()
    skel = ours["skeleton"].isin(q_skel).sum()
    row.update(
        exact_match=int(exact), exact_pct=round(100 * exact / n, 1),
        skeleton_match=int(skel), skeleton_pct=round(100 * skel / n, 1),
    )
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["esol", "lipo", "bace", "bbbp"])
    ap.add_argument("--qmugs", default=None,
                    help="QMugs summary table (.csv or .parquet) with a SMILES column")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    qmugs = load_qmugs(args.qmugs) if args.qmugs else None
    if qmugs is None:
        print("[qmugs] no --qmugs table given; caching our InChIKeys only")

    summary = []
    for ds in args.datasets:
        if ds not in DATASETS:
            print(f"[skip] unknown dataset {ds}")
            continue
        df = load_dataframe(DATASETS[ds])
        ours = inchikeys(df["smiles"].tolist())
        ours["y"] = df["y"].values
        ours.to_csv(f"{args.outdir}/inchikeys_{ds}.csv", index=False)

        if qmugs is not None:
            # join descriptors on skeleton, keeping the first QMugs entry per skeleton
            q = qmugs.drop_duplicates(subset="skeleton")
            merged = ours.merge(q.drop(columns=["inchikey"]), on="skeleton",
                                how="left", suffixes=("", "_qmugs"))
            merged.to_csv(f"{args.outdir}/qmugs_join_{ds}.csv", index=False)

        summary.append(report(ds, ours, qmugs))

    tab = pd.DataFrame(summary)
    tab.to_csv(f"{args.outdir}/qmugs_overlap.csv", index=False)
    print("\n=== QMugs overlap ===")
    print(tab.to_string(index=False))

    if qmugs is not None and "skeleton_pct" in tab:
        print("\nInterpretation: datasets with high skeleton overlap can take quantum")
        print("descriptors by lookup. Datasets with low overlap need them computed,")
        print("e.g. GFN2-xTB on an RDKit conformer.")


if __name__ == "__main__":
    main()
