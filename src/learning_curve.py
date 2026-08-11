"""Learning curve: held-out error vs. training-set size under a scaffold split.

    python -m src.learning_curve --dataset lipo --seeds 0 1 2

For each seed the scaffold split is computed once, so validation and test sets are
fixed while the training set is subsampled. Subsets are nested (each larger size
contains the smaller ones), which removes one source of between-point variance.

Writes results/curve_<dataset>.csv and results/curve_<dataset>.png
"""
from __future__ import annotations

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import baseline
from .data import DATASETS, build_graphs, load_dataframe
from .splits import get_split
from .train import metrics, primary, train_gnn

COLORS = {"baseline": "#5b8ff9", "gnn": "#3fbf9f"}
GRID = [100, 250, 500, 1000, 2000, 4000, 8000, 16000]


def sizes_for(n_train: int, grid=GRID) -> list[int]:
    sizes = [s for s in grid if s < n_train]
    sizes.append(n_train)
    return sizes


def plot_curve(df: pd.DataFrame, key: str, ds: str, out: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for model in sorted(df.model.unique()):
        d = df[df.model == model].groupby("n_train")[key].agg(["mean", "std"]).reset_index()
        sd = d["std"].fillna(0.0)
        ax.plot(d.n_train, d["mean"], "o-", lw=2, color=COLORS.get(model, "#888"),
                label=model, markersize=5)
        ax.fill_between(d.n_train, d["mean"] - sd, d["mean"] + sd,
                        color=COLORS.get(model, "#888"), alpha=0.16, linewidth=0)
    ax.set_xscale("log")
    ax.set_xlabel("training molecules (log scale)")
    ax.set_ylabel(f"test {key.upper()}")
    ax.set_title(f"{ds.upper()} — learning curve, scaffold split (mean ± s.d.)",
                 fontweight="bold")
    ax.grid(alpha=0.25, which="both")
    ax.set_axisbelow(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lipo", choices=list(DATASETS))
    ap.add_argument("--csv", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--models", nargs="+", default=["baseline", "gnn"])
    ap.add_argument("--sizes", type=int, nargs="+", default=None,
                    help="explicit training sizes; defaults to a log grid")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    spec = DATASETS[args.dataset]
    os.makedirs(args.outdir, exist_ok=True)
    key = primary(spec.task)

    df = load_dataframe(spec, csv=args.csv)
    smiles = df["smiles"].tolist()
    y = df["y"].to_numpy(dtype=float)
    graphs = build_graphs(df)
    X = baseline.featurize(smiles)

    rows = []
    for seed in args.seeds:
        tr, va, te = get_split("scaffold", smiles, seed=seed)
        perm = np.random.default_rng(seed).permutation(tr)
        sizes = args.sizes or sizes_for(len(tr))
        sizes = [s for s in sizes if s <= len(tr)]
        print(f"\n[seed {seed}] train pool {len(tr)}, sizes {sizes}")

        for n in sizes:
            sub = perm[:n].tolist()
            for model_name in args.models:
                t0 = time.time()
                if model_name == "baseline":
                    p = baseline.fit_predict(
                        X[sub], y[sub], X[va], y[va], X[te], task=spec.task, seed=seed
                    )
                    m = metrics(y[te], p, spec.task)
                else:
                    m, _, _, _ = train_gnn(
                        graphs, sub, va, te, task=spec.task, seed=seed,
                        epochs=args.epochs,
                    )
                rows.append(dict(dataset=spec.name, model=model_name, n_train=n,
                                 seed=seed, seconds=round(time.time() - t0, 1), **m))
                print(f"  n={n:6d}  {model_name:9s} {key}={m[key]:.4f}"
                      f"  ({time.time() - t0:.0f}s)")

    out = pd.DataFrame(rows)
    out.to_csv(f"{args.outdir}/curve_{spec.name}.csv", index=False)
    plot_curve(out, key, spec.name, f"{args.outdir}/curve_{spec.name}.png")

    print("\n================ LEARNING CURVE ================")
    pivot = out.pivot_table(index="n_train", columns="model", values=key, aggfunc="mean")
    pivot["gnn - baseline"] = pivot.get("gnn") - pivot.get("baseline")
    print(pivot.round(4).to_string())
    print(f"\nwrote {args.outdir}/curve_{spec.name}.csv and .png")


if __name__ == "__main__":
    main()
