"""How many of our molecules already have precomputed quantum descriptors in QMugs?

QMugs contains ~665k drug-like molecules from ChEMBL (~2M conformers) with
GFN2-xTB optimised geometries and properties at GFN2-xTB and DFT
(wB97X-D/def2-SVP) levels. Where a molecule matches, its descriptors are a table
lookup rather than a quantum chemistry calculation.

    python -m src.qmugs_overlap --peek --qmugs summary.csv
    python -m src.qmugs_overlap --datasets esol lipo bace bbbp --qmugs summary.csv

Matching is reported at two levels:
  exact     full InChIKey; same connectivity, stereochemistry and protonation
  skeleton  first block only (14 chars); same 2D connectivity, ignoring stereo,
            isotopes and salt/charge state

Skeleton matching is the useful number for descriptor transfer: a different
enantiomer or counter-ion has essentially the same HOMO/LUMO and polarizability,
while a different skeleton does not. Both are reported so the choice is visible.

The published summary.csv is ~1.9 GB, one row per conformer, and carries no
InChIKey column. Generating 665k InChIKeys is the expensive step, so molecules
are first screened on heavy-atom and heteroatom counts, which QMugs provides as
integers and which are definitionally unambiguous. Only survivors are parsed.

QMugs is distributed by ETH Zurich, https://doi.org/10.3929/ethz-b-000482129
(Isert et al., Sci Data 9, 273, 2022).
"""
from __future__ import annotations

import argparse
import os

import pandas as pd
from rdkit import Chem, RDLogger

from .data import DATASETS, load_dataframe

RDLogger.DisableLog("rdApp.*")

SMILES_COLS = ["smiles", "SMILES", "canonical_smiles", "Smiles", "smi"]
SHAPE_COLS = ("heavy_atoms", "heteroatoms")


# --------------------------------------------------------------------------- #
# Molecule keys
# --------------------------------------------------------------------------- #

def _inchikey_one(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


def _keys_parallel(smiles: list[str], workers: int | None = None) -> list:
    """InChIKeys via a process pool; RDKit parsing dominates, so this scales."""
    workers = workers or max(1, (os.cpu_count() or 2))
    if workers == 1 or len(smiles) < 5000:
        return [_inchikey_one(s) for s in smiles]
    import multiprocessing as mp
    with mp.Pool(workers) as pool:
        return list(pool.imap(_inchikey_one, smiles, chunksize=500))


def shape(smi: str) -> tuple[int, int] | None:
    """(heavy atoms, heteroatoms) — the cheap screen, matching QMugs' definitions."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    heavy = mol.GetNumHeavyAtoms()
    hetero = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (1, 6))
    return heavy, hetero


def inchikeys(smiles: list[str], workers: int | None = None) -> pd.DataFrame:
    keys = _keys_parallel(smiles, workers)
    shapes = [shape(s) for s in smiles]
    return pd.DataFrame(dict(
        smiles=smiles,
        inchikey=keys,
        skeleton=[k.split("-")[0] if k else None for k in keys],
        heavy_atoms=[s[0] if s else None for s in shapes],
        heteroatoms=[s[1] if s else None for s in shapes],
    ))


# --------------------------------------------------------------------------- #
# QMugs
# --------------------------------------------------------------------------- #

def peek_columns(path: str, n: int = 3) -> pd.DataFrame:
    if path.endswith((".parquet", ".pq")):
        return pd.read_parquet(path).head(n)
    return pd.read_csv(path, nrows=n)


def load_qmugs(path: str, shape_filter: set | None = None,
               max_cols: int = 80, workers: int | None = None) -> pd.DataFrame:
    if not os.path.exists(path):
        raise SystemExit(
            f"'{path}' not found.\n"
            "QMugs is hosted by ETH Zurich at https://doi.org/10.3929/ethz-b-000482129 .\n"
            "Only the per-molecule property table (summary.csv) is needed here; the\n"
            "structure, spectra and wavefunction tarballs are far larger and unused.\n"
            "Run without --qmugs to cache our InChIKeys and skip the overlap check."
        )

    sample = pd.read_csv(path, nrows=2000) if not path.endswith((".parquet", ".pq")) \
        else pd.read_parquet(path).head(2000)

    smi_col = next((c for c in SMILES_COLS if c in sample.columns), None)
    if smi_col is None:
        raise SystemExit(f"no SMILES column; columns are {list(sample.columns)}")

    id_cols = [c for c in sample.columns
               if any(t in c.lower() for t in ("inchikey", "inchi_key", "chembl"))]
    num_cols = list(sample.select_dtypes("number").columns)[:max_cols]
    keep = list(dict.fromkeys([smi_col] + id_cols + num_cols))
    can_screen = shape_filter is not None and all(c in keep for c in SHAPE_COLS)
    print(f"[qmugs] SMILES '{smi_col}', ids {id_cols}, {len(num_cols)} numeric columns")
    print(f"[qmugs] heavy/heteroatom screen: {'on' if can_screen else 'off'}")

    if path.endswith((".parquet", ".pq")):
        df = pd.read_parquet(path, columns=keep).drop_duplicates(subset=smi_col)
    else:
        parts, seen, total = [], set(), 0
        for i, chunk in enumerate(pd.read_csv(path, usecols=keep, chunksize=250_000)):
            total += len(chunk)
            chunk = chunk.drop_duplicates(subset=smi_col)
            chunk = chunk[~chunk[smi_col].isin(seen)]
            seen.update(chunk[smi_col].tolist())
            if can_screen:
                pairs = list(zip(chunk[SHAPE_COLS[0]], chunk[SHAPE_COLS[1]]))
                chunk = chunk[[p in shape_filter for p in pairs]]
            parts.append(chunk)
            print(f"[qmugs]   rows {total:,} -> {sum(len(p) for p in parts):,} candidates")
        df = pd.concat(parts, ignore_index=True)

    print(f"[qmugs] {len(df):,} molecules to key")

    ik_col = next((c for c in df.columns if "inchikey" in c.lower()), None)
    if ik_col:
        print(f"[qmugs] using existing InChIKey column '{ik_col}'")
        df["inchikey"] = df[ik_col].astype(str)
    else:
        print(f"[qmugs] generating InChIKeys ({workers or os.cpu_count()} workers)")
        df["inchikey"] = _keys_parallel(df[smi_col].astype(str).tolist(), workers)

    df = df[df["inchikey"].notna()].copy()
    df["skeleton"] = df["inchikey"].astype(str).str.split("-").str[0]
    return df.drop_duplicates(subset="inchikey")


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["esol", "lipo", "bace", "bbbp"])
    ap.add_argument("--qmugs", default=None,
                    help="QMugs summary table (.csv or .parquet)")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--no-screen", action="store_true",
                    help="disable the heavy/heteroatom prefilter")
    ap.add_argument("--peek", action="store_true",
                    help="print the QMugs columns and exit")
    args = ap.parse_args()

    if args.peek:
        if not args.qmugs:
            raise SystemExit("--peek needs --qmugs")
        s = peek_columns(args.qmugs)
        print(f"{len(s.columns)} columns:\n" + "\n".join(f"  {c}" for c in s.columns))
        return

    os.makedirs(args.outdir, exist_ok=True)

    # Our molecules first: they define the screen.
    ours = {}
    for ds in args.datasets:
        if ds not in DATASETS:
            print(f"[skip] unknown dataset {ds}")
            continue
        df = load_dataframe(DATASETS[ds])
        keys = inchikeys(df["smiles"].tolist(), args.workers)
        keys["y"] = df["y"].values
        keys.to_csv(f"{args.outdir}/inchikeys_{ds}.csv", index=False)
        ours[ds] = keys
        print(f"[{ds}] {len(keys)} molecules, {keys.inchikey.notna().sum()} keyed")

    shape_filter = None
    if not args.no_screen:
        shape_filter = {
            (int(h), int(x))
            for k in ours.values()
            for h, x in zip(k.heavy_atoms.dropna(), k.heteroatoms.dropna())
        }
        print(f"\n[screen] {len(shape_filter)} distinct (heavy, hetero) combinations")

    qmugs = load_qmugs(args.qmugs, shape_filter, workers=args.workers) if args.qmugs else None
    if qmugs is None:
        print("\n[qmugs] no --qmugs table given; InChIKeys cached, overlap skipped")
        return

    q_exact, q_skel = set(qmugs["inchikey"]), set(qmugs["skeleton"])
    q_one = qmugs.drop_duplicates(subset="skeleton").drop(columns=["inchikey"])

    summary = []
    for ds, keys in ours.items():
        n = len(keys)
        exact = int(keys["inchikey"].isin(q_exact).sum())
        skel = int(keys["skeleton"].isin(q_skel).sum())
        merged = keys.merge(q_one, on="skeleton", how="left", suffixes=("", "_qmugs"))
        merged.to_csv(f"{args.outdir}/qmugs_join_{ds}.csv", index=False)
        summary.append(dict(dataset=ds, n_molecules=n,
                            exact_match=exact, exact_pct=round(100 * exact / n, 1),
                            skeleton_match=skel, skeleton_pct=round(100 * skel / n, 1)))

    tab = pd.DataFrame(summary)
    tab.to_csv(f"{args.outdir}/qmugs_overlap.csv", index=False)
    print("\n=== QMugs overlap ===")
    print(tab.to_string(index=False))
    print("\nDatasets with high skeleton overlap can take quantum descriptors by lookup.")
    print("Low overlap means computing them, e.g. GFN2-xTB on an RDKit conformer.")


if __name__ == "__main__":
    main()
