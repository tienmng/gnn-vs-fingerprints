"""Cross-dataset summary. Run after src.run_all and src.learning_curve on several datasets.

    python -m src.summarize

Reads every results/runs_*.csv and results/curve_*.csv present and writes:
  results/fig_all_curves.png   learning curves, one panel per dataset
  results/fig_all_splits.png   split effect vs. model effect, all datasets
  results/summary_all.md       combined tables

All comparisons are paired: within a dataset, the two models see the same split,
the same seed and the same training subset, so the difference is taken per pair
before averaging. The paired standard deviation is substantially smaller than the
standard deviation of either model alone, and an unpaired comparison of the means
can change sign on noise.

Metrics differ in direction (RMSE lower is better, ROC-AUC higher). Effects are
signed so that positive always means "the scaffold split is harder" for the split
effect and "the GNN is better" for the model effect.
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


def _sign(ds: str) -> int:
    """+1 when lower is better, so that (a - b) * sign means 'b beats a'."""
    return 1 if DATASETS[ds].task == "regression" else -1


def found(outdir: str, pattern: str) -> dict[str, pd.DataFrame]:
    out = {}
    for path in sorted(glob.glob(f"{outdir}/{pattern}")):
        m = re.search(r"_([a-z0-9]+)\.csv$", os.path.basename(path))
        if m and m.group(1) in DATASETS:
            out[m.group(1)] = pd.read_csv(path)
    ordered = {d: out[d] for d in ORDER if d in out}
    ordered.update({d: v for d, v in out.items() if d not in ORDER})
    return ordered


# --------------------------------------------------------------------------- #
# Paired statistics
# --------------------------------------------------------------------------- #

def paired_split_effect(runs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per dataset: scaffold minus random, paired within (model, seed)."""
    rows = []
    for ds, df in runs.items():
        key, s = _key(ds), _sign(ds)
        if not {"random", "scaffold"}.issubset(set(df["split"])):
            continue
        w = df.pivot_table(index=["model", "seed"], columns="split", values=key).reset_index()
        eff = (w["scaffold"] - w["random"]) * s
        base = df[(df.split == "scaffold") & (df.model == "baseline")][key].mean()
        rows.append(dict(
            dataset=ds, metric=key.upper(), baseline_scaffold=round(base, 3),
            split_effect=round(float(eff.mean()), 3),
            paired_sd=round(float(eff.std(ddof=1)), 3),
            pairs_scaffold_harder=f"{int((eff > 0).sum())}/{len(eff)}",
            pct_of_baseline=round(100 * float(eff.mean()) / abs(base), 1),
        ))
    return pd.DataFrame(rows)


def paired_model_effect(runs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per dataset: GNN minus baseline on the scaffold split, paired within seed."""
    rows = []
    for ds, df in runs.items():
        key, s = _key(ds), _sign(ds)
        d = df[df.split == "scaffold"]
        w = d.pivot_table(index="seed", columns="model", values=key).reset_index()
        if not {"baseline", "gnn"}.issubset(w.columns):
            continue
        eff = (w["baseline"] - w["gnn"]) * s
        base = w["baseline"].mean()
        rows.append(dict(
            dataset=ds, metric=key.upper(),
            model_effect=round(float(eff.mean()), 3),
            paired_sd=round(float(eff.std(ddof=1)), 3),
            seeds_gnn_wins=f"{int((eff > 0).sum())}/{len(eff)}",
            pct_of_baseline=round(100 * float(eff.mean()) / abs(base), 1),
        ))
    return pd.DataFrame(rows)


def paired_curve(df: pd.DataFrame, ds: str) -> pd.DataFrame:
    """Per training size: GNN minus baseline, paired within seed."""
    key, s = _key(ds), _sign(ds)
    w = df.pivot_table(index=["n_train", "seed"], columns="model", values=key).reset_index()
    if not {"baseline", "gnn"}.issubset(w.columns):
        return pd.DataFrame()
    w["eff"] = (w["baseline"] - w["gnn"]) * s
    g = w.groupby("n_train")["eff"].agg(
        mean="mean", sd=lambda x: x.std(ddof=1),
        wins=lambda x: int((x > 0).sum()), n="count").reset_index()
    g.insert(0, "dataset", ds)
    return g


def crossover(g: pd.DataFrame) -> tuple[str, float | None]:
    """First training size at which the GNN leads on both the paired mean and a
    majority of seeds, and holds at least half the seeds at every larger size.
    Returns (description, x for the marker).

    Requiring both criteria matters. A mean that changes sign is not sufficient:
    it can cross zero while only one seed in three favours the GNN. A seed
    majority alone is not sufficient either: seeds can split 2-1 for the GNN
    while the mean still favours the baseline because the one loss is large.
    """
    if g.empty:
        return "no data", None
    maj = (g["wins"] > g["n"] / 2) & (g["mean"] > 0)
    half = g["wins"] >= g["n"] / 2
    for i in range(len(g)):
        if maj.iloc[i] and half.iloc[i:].all():
            if i == 0:
                return f"GNN ahead from the smallest size tested ({int(g.n_train.iloc[0])})", None
            lo, hi = int(g.n_train.iloc[i - 1]), int(g.n_train.iloc[i])
            return f"crossover between {lo} and {hi}", float(np.sqrt(lo * hi))
    best = g.loc[g["wins"].idxmax()]
    return (f"no seed-consistent crossover (best {int(best.wins)}/{int(best.n)} "
            f"at n={int(best.n_train)})"), None


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

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
        desc, x = crossover(paired_curve(df, ds))
        if x is not None:
            ax.axvline(x, color="#999", ls=":", lw=1.4)
            ax.annotate(desc.replace("crossover between", "crossover\n"), xy=(x, 0.90),
                        xycoords=("data", "axes fraction"), fontsize=8.5, ha="center",
                        color="#666")
        else:
            ax.annotate(desc, xy=(0.5, 0.05), xycoords="axes fraction", fontsize=8.5,
                        ha="center", color="#999")
        task = "regression" if _sign(ds) > 0 else "classification"
        ax.set_xscale("log")
        ax.set_title(f"{ds.upper()} — {task}", fontweight="bold", fontsize=11)
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


def effects_figure(split: pd.DataFrame, model: pd.DataFrame, out: str) -> None:
    tab = split.merge(model, on="dataset", suffixes=("_split", "_model"))
    x = np.arange(len(tab))
    w = 0.36
    fig, ax = plt.subplots(figsize=(1.9 * len(tab) + 3.2, 5.0))
    ax.bar(x - w / 2, tab.pct_of_baseline_split, w, color="#e0884f",
           label="split effect (random → scaffold)")
    ax.bar(x + w / 2, tab.pct_of_baseline_model, w, color=COLORS["gnn"],
           label="model effect (baseline → GNN)")
    for xi, (a, b, sa, sb) in enumerate(zip(tab.pct_of_baseline_split,
                                            tab.pct_of_baseline_model,
                                            tab.pairs_scaffold_harder,
                                            tab.seeds_gnn_wins)):
        ax.text(xi - w / 2, a, f"{a:+.0f}%\n{sa}", ha="center",
                va="bottom" if a >= 0 else "top", fontsize=8.5, fontweight="bold")
        ax.text(xi + w / 2, b, f"{b:+.0f}%\n{sb}", ha="center",
                va="bottom" if b >= 0 else "top", fontsize=8.5, fontweight="bold")
    ax.axhline(0, color="#333", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d.upper()}\n{'reg' if _sign(d) > 0 else 'clf'}" for d in tab.dataset])
    ax.set_ylabel("% of baseline scaffold score")
    ax.set_title("Evaluation protocol vs. model architecture (paired)",
                 fontweight="bold", fontsize=12)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2,
              frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    runs = found(args.outdir, "runs_*.csv")
    curves = found(args.outdir, "curve_*.csv")
    print(f"[summarize] runs: {list(runs)}  curves: {list(curves)}")

    md = ["# Cross-dataset summary\n",
          "All effects are paired within seed (and within model, for the split effect).",
          "Positive split effect means the scaffold split is harder; positive model",
          "effect means the GNN is better.\n"]

    if runs:
        split = paired_split_effect(runs)
        model = paired_model_effect(runs)
        if not split.empty and not model.empty:
            effects_figure(split, model, f"{args.outdir}/fig_all_splits.png")
        print("\n=== split effect (paired) ===")
        print(split.to_string(index=False))
        print("\n=== model effect at full training size (paired) ===")
        print(model.to_string(index=False))
        md += ["## Split effect\n", _md(split), "\n",
               "## Model effect at full training size\n", _md(model), "\n"]

    if curves:
        curves_figure(curves, f"{args.outdir}/fig_all_curves.png")
        per_n, cx = [], []
        for ds, df in curves.items():
            g = paired_curve(df, ds)
            per_n.append(g)
            desc, _ = crossover(g)
            cx.append(dict(dataset=ds, metric=_key(ds).upper(),
                           n_max=int(df.n_train.max()),
                           advantage_at_max=round(float(g["mean"].iloc[-1]), 3),
                           wins_at_max=f"{int(g.wins.iloc[-1])}/{int(g.n.iloc[-1])}",
                           crossover=desc))
        per_n = pd.concat(per_n, ignore_index=True).round(4)
        cx = pd.DataFrame(cx)
        per_n.to_csv(f"{args.outdir}/paired_curve_all.csv", index=False)
        print("\n=== learning-curve crossovers (majority of seeds) ===")
        print(cx.to_string(index=False))
        print("\n=== paired GNN advantage by training size ===")
        print(per_n.to_string(index=False))
        md += ["## Learning-curve crossovers\n", _md(cx), "\n",
               "## Paired GNN advantage by training size\n", _md(per_n), "\n"]

    open(f"{args.outdir}/summary_all.md", "w").write("\n".join(md) + "\n")
    print(f"\nwrote {args.outdir}/summary_all.md")


if __name__ == "__main__":
    main()
