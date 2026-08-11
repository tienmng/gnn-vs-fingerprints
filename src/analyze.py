"""Figures and error analysis. Run after src.run_all.

    python -m src.analyze --dataset esol

Produces:
  results/fig_<ds>.png          RMSE / ROC-AUC by model and split
  results/parity_<ds>.png       predicted vs measured, scaffold split
  results/worst20_<ds>.csv      worst test predictions with descriptors
  results/worst20_<ds>.png      those molecules drawn
  results/error_vs_sim_<ds>.png error vs Tanimoto similarity to training set
  results/table_<ds>.md         summary table in markdown
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Draw, rdFingerprintGenerator

from .data import DATASETS

GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
COLORS = {"baseline": "#5b8ff9", "gnn": "#3fbf9f"}


def headline_figure(runs: pd.DataFrame, key: str, ds: str, out: str) -> None:
    splits = ["random", "scaffold"]
    models = ["baseline", "gnn"]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.2), sharey=True)
    for ax, sp in zip(axes, splits):
        sub = runs[runs["split"] == sp]
        means = [sub[sub.model == m][key].mean() for m in models]
        stds = [sub[sub.model == m][key].std(ddof=1) for m in models]
        bars = ax.bar(
            ["Morgan+desc\nLightGBM", "GINE\nMPNN"], means, yerr=stds, capsize=6,
            color=[COLORS[m] for m in models], width=0.55,
            error_kw=dict(ecolor="#444", lw=1.2),
        )
        for b, mval in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, mval, f"{mval:.3f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_title(f"{sp} split", fontsize=12)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    axes[0].set_ylabel(key.upper())
    fig.suptitle(f"{ds.upper()} — {key.upper()} (mean ± s.d. over seeds)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def parity_figure(preds: pd.DataFrame, ds: str, out: str) -> None:
    sub = preds[(preds["split"] == "scaffold") & (preds["seed"] == preds["seed"].min())]
    models = sorted(sub["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(4.4 * len(models), 4.2))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, models):
        d = sub[sub.model == m]
        ax.scatter(d.y_true, d.y_pred, s=14, alpha=0.55, color=COLORS.get(m, "#888"),
                   edgecolors="none")
        lims = [min(d.y_true.min(), d.y_pred.min()), max(d.y_true.max(), d.y_pred.max())]
        ax.plot(lims, lims, "k--", lw=1, alpha=0.6)
        rmse = float(np.sqrt(((d.y_true - d.y_pred) ** 2).mean()))
        ax.set_title(f"{m}  (RMSE {rmse:.3f})")
        ax.set_xlabel("measured")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("predicted")
    fig.suptitle(f"{ds.upper()} — scaffold split, held-out test set", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def worst_predictions(preds: pd.DataFrame, ds: str, outdir: str, n: int = 20) -> pd.DataFrame:
    sub = preds[(preds.split == "scaffold") & (preds.model == "gnn")].copy()
    if sub.empty:
        sub = preds[preds.split == "scaffold"].copy()
    sub["abs_err"] = (sub.y_true - sub.y_pred).abs()
    sub = sub.groupby("smiles", as_index=False).agg(
        y_true=("y_true", "mean"), y_pred=("y_pred", "mean"), abs_err=("abs_err", "mean")
    )
    worst = sub.sort_values("abs_err", ascending=False).head(n).reset_index(drop=True)

    for name, fn in [("MolWt", Descriptors.MolWt), ("MolLogP", Descriptors.MolLogP),
                     ("TPSA", Descriptors.TPSA), ("RotB", Descriptors.NumRotatableBonds),
                     ("Rings", Descriptors.RingCount)]:
        worst[name] = [round(fn(Chem.MolFromSmiles(s)), 2) for s in worst.smiles]

    worst.to_csv(f"{outdir}/worst20_{ds}.csv", index=False)

    mols = [Chem.MolFromSmiles(s) for s in worst.smiles]
    legends = [f"err {e:.2f}\nobs {t:.2f} / pred {p:.2f}"
               for e, t, p in zip(worst.abs_err, worst.y_true, worst.y_pred)]
    path = _save_grid(mols, legends, f"{outdir}/worst20_{ds}")
    print(f"[analyze] wrote {path}")
    return worst


def _save_grid(mols, legends, stem: str) -> str:
    """MolsToGridImage returns a PIL image, PNG bytes or an SVG string depending on
    RDKit version and execution context. Handle all three."""
    kw = dict(molsPerRow=5, subImgSize=(230, 190), legends=legends)
    try:
        img = Draw.MolsToGridImage(mols, returnPNG=False, **kw)
        if hasattr(img, "save"):                       # PIL.Image
            img.save(stem + ".png")
            return stem + ".png"
        data = img.data if hasattr(img, "data") else img
        if isinstance(data, (bytes, bytearray)):
            open(stem + ".png", "wb").write(data)
            return stem + ".png"
        open(stem + ".svg", "w").write(data)
        return stem + ".svg"
    except Exception:
        svg = Draw.MolsToGridImage(mols, useSVG=True, **kw)
        svg = svg.data if hasattr(svg, "data") else svg
        open(stem + ".svg", "w").write(svg)
        return stem + ".svg"


def error_vs_similarity(preds: pd.DataFrame, all_smiles: list[str], ds: str, out: str) -> pd.DataFrame:
    """Applicability domain: error binned by similarity to the nearest training molecule."""
    sub = preds[(preds.split == "scaffold")].copy()
    seed = sub.seed.min()
    sub = sub[sub.seed == seed]
    test_smiles = set(sub.smiles)
    train_smiles = [s for s in all_smiles if s not in test_smiles]

    fps_tr = [GEN.GetFingerprint(Chem.MolFromSmiles(s)) for s in train_smiles]
    sims = {}
    for s in test_smiles:
        fp = GEN.GetFingerprint(Chem.MolFromSmiles(s))
        sims[s] = max(DataStructs.BulkTanimotoSimilarity(fp, fps_tr))

    sub["max_sim"] = sub.smiles.map(sims)
    sub["abs_err"] = (sub.y_true - sub.y_pred).abs()
    bins = [0, 0.3, 0.4, 0.5, 0.7, 1.01]
    sub["bin"] = pd.cut(sub.max_sim, bins)
    tab = sub.groupby(["model", "bin"], observed=True)["abs_err"].agg(["mean", "count"]).reset_index()

    fig, ax = plt.subplots(figsize=(7, 4.2))
    labels = [str(b) for b in tab["bin"].cat.categories] if hasattr(tab["bin"], "cat") \
        else sorted(set(map(str, tab["bin"])))
    for m in sorted(sub.model.unique()):
        d = tab[tab.model == m]
        ax.plot(range(len(d)), d["mean"], "o-", label=m, color=COLORS.get(m, "#888"), lw=2)
        ax.set_xticks(range(len(d)))
        ax.set_xticklabels([str(b) for b in d["bin"]], rotation=20)
    ax.set_xlabel("max Tanimoto similarity to a training molecule")
    ax.set_ylabel("mean |error|")
    ax.set_title(f"{ds.upper()} — applicability domain (scaffold split)", fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    tab.to_csv(out.replace(".png", ".csv"), index=False)
    return tab


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="esol")
    ap.add_argument("--csv", default=None, help="same local CSV you passed to run_all")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    ds = args.dataset
    spec = DATASETS[ds]
    key = "rmse" if spec.task == "regression" else "roc_auc"

    runs = pd.read_csv(f"{args.outdir}/runs_{ds}.csv")
    preds = pd.read_csv(f"{args.outdir}/preds_{ds}.csv")
    os.makedirs(args.outdir, exist_ok=True)

    headline_figure(runs, key, ds, f"{args.outdir}/fig_{ds}.png")
    if spec.task == "regression":
        parity_figure(preds, ds, f"{args.outdir}/parity_{ds}.png")
        worst = worst_predictions(preds, ds, args.outdir)
        print("\nWorst 20 test predictions (scaffold split):")
        print(worst[["smiles", "y_true", "y_pred", "abs_err", "MolWt", "MolLogP"]]
              .head(10).to_string(index=False))

    from .data import load_dataframe
    all_smiles = load_dataframe(spec, csv=args.csv).smiles.tolist()
    tab = error_vs_similarity(preds, all_smiles, ds, f"{args.outdir}/error_vs_sim_{ds}.png")
    print("\nError vs similarity to training set:")
    print(tab.to_string(index=False))

    # markdown summary table
    md = (runs.groupby(["model", "split"])[key]
          .agg(["mean", "std"]).round(3).reset_index())
    lines = [f"| model | split | {key} (mean ± s.d.) |", "|---|---|---|"]
    for _, r in md.iterrows():
        lines.append(f"| {r['model']} | {r['split']} | {r['mean']:.3f} ± {r['std']:.3f} |")
    table = "\n".join(lines)
    open(f"{args.outdir}/table_{ds}.md", "w").write(table + "\n")
    print("\n" + table)


if __name__ == "__main__":
    main()
