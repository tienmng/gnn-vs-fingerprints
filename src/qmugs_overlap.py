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


def _inchikey_one(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


def _keys_parallel(smiles: list[str], workers: int | None = None) -> list[str | None]:
    """InChIKeys with a process pool. RDKit parsing dominates, so this scales."""
    workers = workers or max(1, (os.cpu_count() or 2))
    if workers == 1 or len(smiles) < 5000:
        return [_inchikey_one(s) for s in smiles]
    import multiprocessing as mp
    with mp.Pool(workers) as pool:
        return list(pool.imap(_inchikey_one, smiles, chunksize=500))


def peek_columns(path: str, n: int = 2000) -> pd.DataFrame:
    """Read a small sample to discover columns and dtypes without loading 2 GB."""
    if path.endswith((".parquet", ".pq")):
        return pd.read_parquet(path).head(n)
    return pd.read_csv(path, nrows=n)


def load_qmugs(path: str, max_cols: int = 60, workers: int | None = None) -> pd.DataFrame:
    """Stream the QMugs summary table, keeping one row per unique molecule.

    The published summary.csv is ~1.9 GB with one row per conformer (~2M rows).
    Loading it whole exhausts a free Colab session, so it is read in chunks,
    restricted to the SMILES column plus numeric property columns, and collapsed
    to unique molecules before any InChIKey work happens.
    """
    if not os.path.exists(path):
        raise SystemExit(
            f"'{path}' not found.\n"
            "QMugs is hosted by ETH Zurich at https://doi.org/10.3929/ethz-b-000482129 .\n"
            "Only the per-molecule property table (summary.csv) is needed here; the\n"
            "structure, spectra and wavefunction tarballs are far larger and unused.\n"
            "Run without --qmugs to cache our InChIKeys and skip the overlap check."
        )

    sample = peek_columns(path)
    print(f"[qmugs] columns ({len(sample.columns)}): {list(sample.columns)}")

    smi_col = next((c for c in SMILES_COLS if c in sample.columns), None)
    if smi_col is None:
        raise SystemExit(f"no SMILES column found; columns are {list(sample.columns)}")

    id_cols = [c for c in sample.columns
               if any(t in c.lower() for t in ("inchikey", "inchi_key", "chembl"))]
    num_cols = [c for c in sample.select_dtypes("number").columns][:max_cols]
    keep = list(dict.fromkeys([smi_col] + id_cols + num_cols))
    print(f"[qmugs] keeping SMILES '{smi_col}', ids {id_cols}, "
          f"{len(num_cols)} numeric columns")

    if path.endswith((".parquet", ".pq")):
        df = pd.read_parquet(path, columns=keep)
    else:
        parts, seen = [], set()
        for i, chunk in enumerate(pd.read_csv(path, usecols=keep, chunksize=250_000)):
            chunk = chunk[~chunk[smi_col].isin(seen)]
            chunk = chunk.drop_duplicates(subset=smi_col)
            seen.update(chunk[smi_col].tolist())
            parts.append(chunk)
            print(f"[qmugs]   chunk {i + 1}: {len(seen):,} unique molecules so far")
        df = pd.concat(parts, ignore_index=True)
    print(f"[qmugs] {len(df):,} unique molecules")

    ik_col = next((c for c in df.columns if "inchikey" in c.lower()
                   or "inchi_key" in c.lower()), None)
    if ik_col:
        print(f"[qmugs] using existing InChIKey column '{ik_col}'")
        df["inchikey"] = df[ik_col].astype(str)
    else:
        print(f"[qmugs] computing {len(df):,} InChIKeys in parallel "
              f"({workers or os.cpu_count()} workers); this is the slow step")
        df["inchikey"] = _keys_parallel(df[smi_col].astype(str).tolist(), workers)

    df = df[df["inchikey"].notna() & (df["inchikey"].astype(str) != "nan")].copy()
    df["skeleton"] = df["inchikey"].astype(str).str.split("-").str[0]
    return df.drop_duplicates(subset="inchikey")


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
    ap.add_argument("--workers", type=int, default=None,
                    help="processes for InChIKey generation (default: all cores)")
    ap.add_argument("--peek", action="store_true",
                    help="print the QMugs columns and exit")
    args = ap.parse_args()

    if args.peek:
        if not args.qmugs:
            raise SystemExit("--peek needs --qmugs")
        print(peek_columns(args.qmugs).head(3).to_string())
        return

    os.makedirs(args.outdir, exist_ok=True)
    qmugs = load_qmugs(args.qmugs, workers=args.workers) if args.qmugs else None
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
