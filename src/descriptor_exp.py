"""Do quantum descriptors add anything over fingerprints?

    python -m src.qmugs_overlap --datasets lipo --qmugs summary.csv   # writes the join
    python -m src.descriptor_exp --dataset lipo

Restricted to molecules with a QMugs match, because coverage is not missing at
random: QMugs holds the ChEMBL-like molecules and omits solvents and
agrochemicals. Adding descriptors to the covered subset and comparing against a
score computed on the full dataset would confound the descriptor effect with a
subset effect. Every model here is therefore fitted and scored on exactly the
same molecules, with the scaffold split recomputed on that subset.

Purely classical columns in the QMugs table (mw, ring count, H-bond counts,
rotatable bonds) are excluded: the fingerprint baseline already carries RDKit
equivalents, and including them would measure duplication rather than added
quantum information.

Feature families are ablated separately so the answer is "which physics helped",
not just "more columns helped".
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import baseline
from .data import DATASETS
from .splits import get_split, split_report
from .train import metrics, primary

# Columns that are classical 2D descriptors, not quantum output.
CLASSICAL = {
    "mw", "atoms", "heavy_atoms", "heteroatoms", "rotatable_bonds", "stereocenters",
    "rings", "hbond_acceptors", "hbond_donors", "significant_negative_wavenumbers",
    "nonunique_smiles", "y",
    # convergence diagnostics, not descriptors
    "mace_fmax_final", "mace_n_conf_ok",
}

FAMILIES: dict[str, tuple[str, ...]] = {
    # frontier orbitals: reactivity, charge transfer
    "electronic": ("HOMO_ENERGY", "LUMO_ENERGY", "HOMO_LUMO_GAP", "FERMI_LEVEL"),
    # permanent charge distribution: polarity, hydrogen bonding
    "multipole": ("DIPOLE_", "QUADRUPOLE_"),
    # induced response: dispersion, London forces, packing
    "response": ("POLARIZABILITY_MOLECULAR", "DISPERSION_COEFFICIENT_MOLECULAR"),
    # vibrational thermodynamics: flexibility, entropy of fusion proxy
    "thermo": ("ENTHALPY_", "ENTROPY_", "HEAT_CAPACITY_", "TOTAL_FREE_ENERGY",
               "TOTAL_ENTHALPY"),
    # extensive energies; largely proxies for molecular size
    "energy": ("TOTAL_ENERGY", "ATOMIC_ENERGY", "FORMATION_ENERGY", "XC_ENERGY",
               "NUCLEAR_REPULSION_ENERGY", "ONE_ELECTRON_ENERGY",
               "TWO_ELECTRON_ENERGY"),
    # moments of inertia: shape and compactness
    "rotational": ("ROT_CONSTANT_",),
    # foundation-potential energetics (src/mace_desc.py)
    "mace_energy": ("mace_e_min", "mace_strain"),
    # conformer ensemble spread: flexibility, entropy of fusion proxy
    "mace_flex": ("mace_conf_spread", "mace_conf_std"),
    # 3D shape from the relaxed geometry
    "mace_shape": ("mace_rg", "mace_pmi", "mace_asphericity"),
}

PREFIXES = ("GFN2_", "DFT_", "mace_")


def quantum_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.select_dtypes("number").columns
            if c not in CLASSICAL and c.startswith(PREFIXES)]


def family_of(col: str) -> str:
    for name, tokens in FAMILIES.items():
        if any(t in col for t in tokens):
            return name
    return "other"


def build_feature_sets(qcols: list[str]) -> dict[str, list[str]]:
    sets = {"none": [], "all": qcols}
    by_fam: dict[str, list[str]] = {}
    for c in qcols:
        by_fam.setdefault(family_of(c), []).append(c)
    for fam, cols in sorted(by_fam.items()):
        sets[fam] = cols
    return sets


def quantum_only(X_fp, Q, y, tr, va, te, task, seed):
    """Quantum descriptors with no fingerprints.

    Without this condition a null result is ambiguous: descriptors that add
    nothing on top of a fingerprint may still be individually informative and
    merely redundant. Comparing the quantum block alone against the fingerprint
    block alone separates 'uninformative' from 'already covered'.
    """
    return baseline.fit_predict(Q[tr], y[tr], Q[va], y[va], Q[te],
                                task=task, seed=seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lipo", choices=list(DATASETS))
    ap.add_argument("--join", default=None,
                    help="results/qmugs_join_<dataset>.csv from src.qmugs_overlap")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--split", default="scaffold", choices=["scaffold", "random"])
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    spec = DATASETS[args.dataset]
    key = primary(spec.task)
    path = args.join or f"{args.outdir}/qmugs_join_{spec.name}.csv"
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found; run src.qmugs_overlap --qmugs first")

    df = pd.read_csv(path)
    qcols = quantum_columns(df)
    if not qcols:
        raise SystemExit(f"no GFN2_/DFT_ columns in {path}")

    covered = df.dropna(subset=qcols, how="any").reset_index(drop=True)
    print(f"[{spec.name}] {len(df)} molecules, {len(covered)} with complete QMugs "
          f"descriptors ({100 * len(covered) / len(df):.1f}%)")
    print(f"[{spec.name}] {len(qcols)} quantum columns in "
          f"{len({family_of(c) for c in qcols})} families")
    if len(covered) < 200:
        raise SystemExit("too few covered molecules for a meaningful comparison")

    smiles = covered["smiles"].tolist()
    y = covered["y"].to_numpy(dtype=float)
    X_fp = baseline.featurize(smiles)
    Q = covered[qcols].to_numpy(dtype=float)
    Q = np.nan_to_num(Q, nan=0.0, posinf=0.0, neginf=0.0)

    sets = build_feature_sets(qcols)
    idx_of = {c: i for i, c in enumerate(qcols)}

    rows = []
    for seed in args.seeds:
        tr, va, te = get_split(args.split, smiles, seed=seed)
        rep = split_report(smiles, tr, va, te)
        print(f"\n[seed {seed}] {rep}")
        if args.split == "scaffold":
            assert rep["train_test_scaffold_overlap"] == 0

        for name, cols in sets.items():
            if cols:
                sub = Q[:, [idx_of[c] for c in cols]]
                X = np.hstack([X_fp, sub])
            else:
                X = X_fp
            p = baseline.fit_predict(X[tr], y[tr], X[va], y[va], X[te],
                                     task=spec.task, seed=seed)
            m = metrics(y[te], p, spec.task)
            rows.append(dict(dataset=spec.name, features=name, n_extra=len(cols),
                             seed=seed, **m))
            print(f"    {name:12s} (+{len(cols):2d} cols)  {key}={m[key]:.4f}")

        p = quantum_only(X_fp, Q, y, tr, va, te, spec.task, seed)
        m = metrics(y[te], p, spec.task)
        rows.append(dict(dataset=spec.name, features="qm_only",
                         n_extra=len(qcols), seed=seed, **m))
        print(f"    {'qm_only':12s} ({len(qcols):3d} cols, no fingerprints)  "
              f"{key}={m[key]:.4f}")

        # Floor: predict the training mean (or base rate). Without it, the
        # quantum-only score has no reference and "weak" cannot be quantified.
        const = (np.full(len(te), y[tr].mean()) if spec.task == "regression"
                 else np.full(len(te), float(np.mean(y[tr]))))
        m = metrics(y[te], const, spec.task) if spec.task == "regression" else \
            {"roc_auc": 0.5, "pr_auc": float(np.mean(y[te]))}
        rows.append(dict(dataset=spec.name, features="mean_only",
                         n_extra=0, seed=seed, **m))
        print(f"    {'mean_only':12s} (constant prediction)  {key}={m[key]:.4f}")

    runs = pd.DataFrame(rows)
    runs.to_csv(f"{args.outdir}/desc_{spec.name}.csv", index=False)

    # paired against the fingerprint-only model, within seed
    piv = runs.pivot_table(index="seed", columns="features", values=key)
    sign = 1 if spec.task == "regression" else -1
    delta = (piv["none"].values[:, None] - piv.values) * sign   # + = added features help
    delta = pd.DataFrame(delta, index=piv.index, columns=piv.columns).drop(columns="none")

    n_cols = {name: len(cols) for name, cols in sets.items()}
    n_cols["qm_only"] = len(qcols)
    tab = pd.DataFrame({
        "features": delta.columns,
        "n_extra": [n_cols.get(c, 0) for c in delta.columns],
        "delta_mean": delta.mean().round(4).values,
        "delta_sd": delta.std(ddof=1).round(4).values,
        "seeds_improved": [f"{int((delta[c] > 0).sum())}/{len(delta)}" for c in delta.columns],
    }).sort_values("delta_mean", ascending=False)

    print(f"\n=== paired change vs fingerprints alone ({key}, + = better) ===")
    print(f"fingerprint-only {key}: {piv['none'].mean():.4f} "
          f"± {piv['none'].std(ddof=1):.4f}")
    print(tab.to_string(index=False))
    tab.to_csv(f"{args.outdir}/desc_summary_{spec.name}.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.6, 0.52 * len(tab) + 2.4))
    colors = ["#3fbf9f" if v > 0 else "#e0884f" for v in tab.delta_mean]
    ax.barh(tab.features, tab.delta_mean, xerr=tab.delta_sd, color=colors,
            error_kw=dict(ecolor="#555", lw=1.1), height=0.62)
    for i, (v, s) in enumerate(zip(tab.delta_mean, tab.seeds_improved)):
        ax.text(v, i, f"  {v:+.3f}  {s}", va="center",
                ha="left" if v >= 0 else "right", fontsize=9)
    ax.axvline(0, color="#333", lw=1)
    ax.invert_yaxis()
    ax.set_xlabel(f"change in {key.upper()} vs fingerprints alone (+ = better)")
    ax.set_title(f"{spec.name.upper()} — quantum descriptors added to the baseline\n"
                 f"{len(covered)} QMugs-covered molecules, {args.split} split, "
                 f"{len(args.seeds)} seeds, paired",
                 fontweight="bold", fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/fig_desc_{spec.name}.png", dpi=170)
    plt.close(fig)
    print(f"\nwrote {args.outdir}/fig_desc_{spec.name}.png")


if __name__ == "__main__":
    main()
