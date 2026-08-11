"""Cross-dataset summary. Run after src.run_all and src.learning_curve on several datasets.

    python -m src.summarize

Reads every results/runs_*.csv and results/curve_*.csv present and writes:
  results/fig_all_curves.png   learning curves, one panel per dataset
  results/fig_all_splits.png   split effect vs. model effect, all datasets
  results/summary_all.md       combined table

Metrics differ in direction (RMSE lower is better, ROC-AUC higher), so effects are
reported as signed quantities in which positive always means "worse" for the split
effect and "GNN better" for the model effect, plus a percentage of the baseline
scaffold score for cross-dataset comparability.
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import DATASETS

COLORS = {"baseline": "#5b8ff9", "gnn": "#3fbf9f"}
ORDER = ["esol", "lipo", "bace", "bbbp"]


def _md(df: pd.DataFrame) -> str:
    """Markdown table; pandas.to_markdown needs tabulate, so fall back if absent."""
    try:
        return df.to_markdown(index=False)
    except ImportError:
        head = "| " + " | ".join(map(str, df.columns)) + " |"
        rule = "|" + "|".join(["---"] * len(df.columns)) + "|"
        body = ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
        return "\n".join([head, rule, *body])


def _key(ds: str) -> str:
    return "rmse" if DATASETS[ds].task == "regression" else "roc_auc"


def _lower_is_better(ds: str) -> bool:
    return DATASETS[ds].task == "regression"


def found(outdir: str, pattern: str) -> dict[str, pd.DataFrame]:
    out = {}
    for path in sorted(glob.glob(f"{outdir}/{pattern}")):
        m = re.search(r"_([a-z0-9]+)\.csv$", os.path.basename(path))
        if m and m.group(1) in DATASETS:
            out[m.group(1)] = pd.read_csv(path)
    return {d: out[d] for d in ORDER if d in out} | {d: v for d, v in out.items() if d not in ORDER}


def curves_figure(curves: dict[str, pd.DataFrame], out: str) -> None:
    n = len(curves)
    ncol = min(2, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.0 * ncol, 4.1 * nrow), squeeze=False)
    for ax, (ds, df) in zip(axes.ravel(), curves.items()):
        key = _key(ds)
        for model in sorted(df.model.unique()):
            d = df[df.model == model].groupby("n_train")[key].agg(["mean", "std"]).reset_index()
            sd = d["std"].fillna(0.0)
            ax.plot(d.n_train, d["mean"], "o-", lw=2, ms=4.5,
                    color=COLORS.get(model, "#888"), label=model)
            ax.fill_between(d.n_train, d["mean"] - sd, d["mean"] + sd,
                            color=COLORS.get(model, "#888"), alpha=0.15, linewidth=0)
        # mark the crossover, if one exists
        piv = df.pivot_table(index="n_train", columns="model", values=key, aggfunc="mean")
        if {"baseline", "gnn"}.issubset(piv.columns):
            diff = piv["gnn"] - piv["baseline"]
            if not _lower_is_better(ds):
                diff = -diff
            sign = np.sign(diff.values)
            flip = np.where(np.diff(sign) != 0)[0]
            if len(flip):
                i = flip[0]
                x = float(np.sqrt(piv.index[i] * piv.index[i + 1]))
                ax.axvline(x, color="#999", ls=":", lw=1.4)
                ax.annotate(f"crossover\n~{piv.index[i]}–{piv.index[i+1]}", xy=(x, 0.92),
                            xycoords=("data", "axes fraction"), fontsize=8.5,
                            ha="center", color="#666")
        ax.set_xscale("log")
        task_short = "regression" if _lower_is_better(ds) else "classification"
        ax.set_title(f"{ds.upper()} — {task_short}", fontweight="bold", fontsize=11)
        ax.set_xlabel("training molecules")
        ax.set_ylabel(f"test {key.upper()}")
        ax.grid(alpha=0.25, which="both")
        ax.set_axisbelow(True)
        ax.legend(fontsize=9)
    for ax in axes.ravel()[len(curves):]:
        ax.axis("off")
    fig.suptitle("Learning curves under scaffold splits (mean ± s.d. over seeds)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def effects_table(runs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for ds, df in runs.items():
        key = _key(ds)
        piv = df.pivot_table(index="model", columns="split", values=key, aggfunc="mean")
        sd = df[df.split == "scaffold"].groupby("model")[key].std(ddof=1)
        if not {"random", "scaffold"}.issubset(piv.columns):
            continue
        lower = _lower_is_better(ds)
        base_scaf = piv.loc["baseline", "scaffold"]
        # positive = scaffold split is harder
        split_b = (piv.loc["baseline", "scaffold"] - piv.loc["baseline", "random"]) * (1 if lower else -1)
        split_g = (piv.loc["gnn", "scaffold"] - piv.loc["gnn", "random"]) * (1 if lower else -1)
        # positive = GNN better than baseline, on the scaffold split
        model_e = (base_scaf - piv.loc["gnn", "scaffold"]) * (1 if lower else -1)
        rows.append(dict(
            dataset=ds, task=DATASETS[ds].task, metric=key.upper(),
            baseline_scaffold=round(base_scaf, 3),
            gnn_scaffold=round(piv.loc["gnn", "scaffold"], 3),
            split_effect=round(float(np.mean([split_b, split_g])), 3),
            model_effect=round(model_e, 3),
            split_pct=round(100 * float(np.mean([split_b, split_g])) / abs(base_scaf), 1),
            model_pct=round(100 * model_e / abs(base_scaf), 1),
            seed_sd_scaffold=round(float(sd.mean()), 3),
        ))
    return pd.DataFrame(rows)


def effects_figure(tab: pd.DataFrame, out: str) -> None:
    x = np.arange(len(tab))
    w = 0.36
    fig, ax = plt.subplots(figsize=(1.9 * len(tab) + 3.2, 4.9))
    ax.bar(x - w / 2, tab.split_pct, w, label="split effect (random → scaffold)",
           color="#e0884f")
    ax.bar(x + w / 2, tab.model_pct, w, label="model effect (baseline → GNN)",
           color=COLORS["gnn"])
    for xi, (s, m) in enumerate(zip(tab.split_pct, tab.model_pct)):
        ax.text(xi - w / 2, s, f"{s:+.0f}%", ha="center",
                va="bottom" if s >= 0 else "top", fontsize=9, fontweight="bold")
        ax.text(xi + w / 2, m, f"{m:+.0f}%", ha="center",
                va="bottom" if m >= 0 else "top", fontsize=9, fontweight="bold")
    ax.axhline(0, color="#333", lw=1)
    short = {"regression": "reg", "classification": "clf"}
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d.upper()}\n{short.get(t, t)}" for d, t in zip(tab.dataset, tab.task)])
    ax.set_ylabel("% of baseline scaffold score")
    ax.set_title("Evaluation protocol vs. model architecture", fontweight="bold", fontsize=12)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2,
              frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def crossovers(curves: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for ds, df in curves.items():
        key = _key(ds)
        lower = _lower_is_better(ds)
        piv = df.pivot_table(index="n_train", columns="model", values=key, aggfunc="mean")
        if not {"baseline", "gnn"}.issubset(piv.columns):
            continue
        diff = (piv["baseline"] - piv["gnn"]) * (1 if lower else -1)   # positive = GNN better
        wins = diff > 0
        if wins.all():
            note = f"GNN ahead at every size (from {piv.index.min()})"
        elif not wins.any():
            note = "GNN never ahead"
        else:
            first = wins.idxmax()
            prev = piv.index[max(0, list(piv.index).index(first) - 1)]
            note = f"crossover between {prev} and {first}"
        rows.append(dict(dataset=ds, metric=key.upper(), n_max=int(piv.index.max()),
                         gnn_advantage_at_max=round(float(diff.iloc[-1]), 3), crossover=note))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    runs = found(args.outdir, "runs_*.csv")
    curves = found(args.outdir, "curve_*.csv")
    print(f"[summarize] runs: {list(runs)}  curves: {list(curves)}")

    md = ["# Cross-dataset summary\n"]

    if runs:
        tab = effects_table(runs)
        if not tab.empty:
            effects_figure(tab, f"{args.outdir}/fig_all_splits.png")
            print("\n=== split effect vs model effect ===")
            print(tab.to_string(index=False))
            md += ["## Split effect vs. model effect\n",
                   "Positive split effect means the scaffold split is harder. Positive model",
                   "effect means the GNN is better. Percentages are of the baseline scaffold",
                   "score.\n",
                   _md(tab), "\n"]

    if curves:
        curves_figure(curves, f"{args.outdir}/fig_all_curves.png")
        cx = crossovers(curves)
        print("\n=== learning-curve crossovers ===")
        print(cx.to_string(index=False))
        md += ["## Learning-curve crossovers\n", _md(cx), "\n"]

    open(f"{args.outdir}/summary_all.md", "w").write("\n".join(md) + "\n")
    print(f"\nwrote {args.outdir}/summary_all.md")


if __name__ == "__main__":
    main()
