"""The experiment grid: {baseline, gnn} x {random, scaffold} x {3 seeds}.

    python -m src.run_all --dataset esol --seeds 0 1 2

Writes results/runs_<dataset>.csv and results/preds_<dataset>.csv.
Everything downstream (figure, error analysis, README table) reads those files.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

from . import baseline
from .data import DATASETS, build_graphs, load_dataframe
from .splits import get_split, split_report
from .train import metrics, train_gnn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="esol", choices=list(DATASETS))
    ap.add_argument("--csv", default=None, help="local CSV instead of downloading")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--splits", nargs="+", default=["random", "scaffold"])
    ap.add_argument("--models", nargs="+", default=["baseline", "gnn"])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    spec = DATASETS[args.dataset]
    os.makedirs(args.outdir, exist_ok=True)

    df = load_dataframe(spec, csv=args.csv)
    smiles = df["smiles"].tolist()
    y = df["y"].to_numpy(dtype=float)

    print("[run] featurising graphs ...")
    graphs = build_graphs(df)
    assert len(graphs) == len(df), "featurisation dropped molecules; check data.py"

    print("[run] featurising fingerprints ...")
    X = baseline.featurize(smiles)

    rows, pred_rows = [], []
    for split_kind in args.splits:
        for seed in args.seeds:
            tr, va, te = get_split(split_kind, smiles, seed=seed)
            rep = split_report(smiles, tr, va, te)
            print(f"\n[{spec.name} | {split_kind} | seed {seed}] {rep}")
            if split_kind == "scaffold":
                assert rep["train_test_scaffold_overlap"] == 0, "scaffold leak!"

            for model_name in args.models:
                t0 = time.time()
                if model_name == "baseline":
                    p = baseline.fit_predict(
                        X[tr], y[tr], X[va], y[va], X[te], task=spec.task, seed=seed
                    )
                    m = metrics(y[te], p, spec.task)
                    te_smiles, y_te = [smiles[i] for i in te], y[te]
                else:
                    m, p, y_te, te_smiles = train_gnn(
                        graphs, tr, va, te, task=spec.task, seed=seed, epochs=args.epochs
                    )

                dt = time.time() - t0
                rows.append(
                    dict(dataset=spec.name, model=model_name, split=split_kind,
                         seed=seed, seconds=round(dt, 1), **m)
                )
                print(f"    {model_name:9s} {m}  ({dt:.0f}s)")

                pred_rows.append(pd.DataFrame(dict(
                    dataset=spec.name, model=model_name, split=split_kind, seed=seed,
                    smiles=te_smiles, y_true=np.asarray(y_te).ravel(),
                    y_pred=np.asarray(p).ravel(),
                )))

    runs = pd.DataFrame(rows)
    runs.to_csv(f"{args.outdir}/runs_{spec.name}.csv", index=False)
    pd.concat(pred_rows).to_csv(f"{args.outdir}/preds_{spec.name}.csv", index=False)

    key = "rmse" if spec.task == "regression" else "roc_auc"
    summary = (
        runs.groupby(["model", "split"])[key]
        .agg(["mean", "std", "count"])
        .round(3)
        .reset_index()
    )
    print("\n================ SUMMARY ================")
    print(summary.to_string(index=False))
    summary.to_csv(f"{args.outdir}/summary_{spec.name}.csv", index=False)
    print(f"\nwrote {args.outdir}/runs_{spec.name}.csv")


if __name__ == "__main__":
    main()
