"""Geometry and conformational-energy descriptors from a foundation interatomic potential.

    python -m src.mace_desc --dataset esol --n-conformers 5
    python -m src.descriptor_exp --dataset esol --join results/mace_join_esol.csv

MACE-OFF is a machine-learned interatomic potential trained on SPICE at
wB97M-D3(BJ)/def2-TZVPPD. It returns energies and forces, not wavefunctions, so
it yields no orbital energies, multipoles or solvation terms. What it does give,
at a cost that scales to thousands of molecules, is relaxed geometry and the
relative energies of a conformer ensemble.

Descriptors produced per molecule:
  mace_e_min          lowest relaxed energy per atom
  mace_strain         embedded minus relaxed energy per atom, lowest conformer
  mace_conf_spread    max minus min relaxed energy across conformers
  mace_conf_std       standard deviation of relaxed conformer energies
  mace_fmax_final     largest residual force after relaxation (convergence check)
  mace_rg             radius of gyration of the lowest-energy conformer
  mace_pmi1/2/3       principal moments of inertia
  mace_asphericity    shape anisotropy from the PMI ratios
  mace_n_conf_ok      conformers that embedded and relaxed successfully

The ensemble spread terms are the reason for running this on solubility. Entropy
of fusion depends on molecular flexibility, and flexibility is not recoverable
from a 2D graph; the conformer energy spread is a computable proxy for it.

MACE-OFF weights are distributed under the Academic Software License
(https://github.com/ACEsuit/mace-off), which permits academic but not commercial
use. Requires `pip install mace-torch ase`.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors3D

from .data import DATASETS, load_dataframe

RDLogger.DisableLog("rdApp.*")

FEATURES = [
    "mace_e_min", "mace_strain", "mace_conf_spread", "mace_conf_std",
    "mace_fmax_final", "mace_rg", "mace_pmi1", "mace_pmi2", "mace_pmi3",
    "mace_asphericity", "mace_n_conf_ok",
]


def get_calculator(model: str = "medium", device: str | None = None, dtype: str = "float64"):
    """MACE-OFF as an ASE calculator. Import is deferred so the module loads
    without mace-torch installed."""
    import torch
    from mace.calculators import mace_off

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[mace] loading MACE-OFF '{model}' on {device}")
    return mace_off(model=model, device=device, default_dtype=dtype)


def embed_conformers(smiles: str, n_conf: int = 5, seed: int = 42):
    """ETKDG conformers with MMFF cleanup, returned as ASE Atoms.

    Two ETKDG settings silently destroy the ensemble and are avoided here.

    randomSeed=0 returns n identical conformers. Measured on n-octanol with
    numConfs=4: seed 0 gives pairwise RMSD 0.000 A, seed 42 gives 1.8-6.0 A.
    Any non-zero seed is fine, so 0 is remapped rather than accepted.

    pruneRmsThresh collapses the ensemble to a single conformer even for
    flexible chains. Pruning is unnecessary here because the ensemble is
    reduced by energy after relaxation, so duplicates cost time but not
    correctness.

    Both failures produce zero-variance spread descriptors rather than an
    error, so `describe` reports n_conf_ok and the spread terms explicitly.
    """
    from ase import Atoms

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = seed if seed != 0 else 42
    params.useSmallRingTorsions = True
    ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conf, params=params)
    if len(ids) == 0:
        return []
    try:
        AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=500)
    except Exception:
        pass

    numbers = [a.GetAtomicNum() for a in mol.GetAtoms()]
    out = []
    for cid in ids:
        pos = mol.GetConformer(cid).GetPositions()
        out.append(Atoms(numbers=numbers, positions=pos))
    return out


def relax(atoms, calc, fmax: float = 0.05, steps: int = 200):
    """Local relaxation. Returns (relaxed_total_eV, initial_total_eV, fmax, atoms).

    Energies are totals, not per-atom. Conformer energy differences and strain
    are properties of the whole molecule; dividing them by atom count shrinks
    them by an order of magnitude and mixes a size effect into a flexibility
    measure. Only the absolute energy is size-normalised, downstream.
    """
    from ase.optimize import LBFGS

    atoms = atoms.copy()
    atoms.calc = calc
    e0 = atoms.get_potential_energy()
    opt = LBFGS(atoms, logfile=None)
    opt.run(fmax=fmax, steps=steps)
    e1 = atoms.get_potential_energy()
    f = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
    return e1, e0, f, atoms


def shape_descriptors(atoms) -> dict:
    """Radius of gyration, principal moments and asphericity from coordinates."""
    pos = atoms.get_positions()
    masses = atoms.get_masses()
    com = (pos * masses[:, None]).sum(0) / masses.sum()
    d = pos - com
    rg = float(np.sqrt((masses * (d ** 2).sum(1)).sum() / masses.sum()))

    inertia = np.zeros((3, 3))
    for m, r in zip(masses, d):
        inertia += m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    pmi = np.sort(np.linalg.eigvalsh(inertia))
    asph = float(pmi[2] - 0.5 * (pmi[0] + pmi[1])) / (pmi[2] + 1e-9)
    return dict(mace_rg=rg, mace_pmi1=float(pmi[0]), mace_pmi2=float(pmi[1]),
                mace_pmi3=float(pmi[2]), mace_asphericity=asph)


def describe(smiles: str, calc, n_conf: int = 5, seed: int = 42) -> dict:
    confs = embed_conformers(smiles, n_conf, seed)
    if not confs:
        return {k: np.nan for k in FEATURES}

    energies, initials, forces, relaxed = [], [], [], []
    for at in confs:
        try:
            e1, e0, f, out = relax(at, calc)
        except Exception:
            continue
        energies.append(e1)
        initials.append(e0)
        forces.append(f)
        relaxed.append(out)

    if not energies:
        return {k: np.nan for k in FEATURES}

    energies = np.array(energies)          # total eV
    best = int(np.argmin(energies))
    n_atoms = len(relaxed[best])
    row = dict(
        mace_e_min=float(energies[best]) / n_atoms,        # size-normalised
        mace_strain=float(initials[best] - energies[best]),  # total eV
        mace_conf_spread=float(energies.max() - energies.min()),
        mace_conf_std=float(energies.std(ddof=1)) if len(energies) > 1 else 0.0,
        mace_fmax_final=float(forces[best]),
        mace_n_conf_ok=float(len(energies)),
    )
    row.update(shape_descriptors(relaxed[best]))
    return row


def debug_molecule(smiles: str, calc, n_conf: int = 5) -> None:
    """Per-conformer energies and pairwise RMSD for one molecule.

    Distinguishes 'the ensemble collapsed to one geometry' from 'distinct
    geometries with indistinguishable energies'. A zero spread means very
    different things in the two cases.
    """
    confs = embed_conformers(smiles, n_conf)
    print(f"\n{smiles}: {len(confs)} conformers embedded, "
          f"{len(confs[0]) if confs else 0} atoms")
    if not confs:
        return

    rows = []
    for i, at in enumerate(confs):
        e1, e0, f, out = relax(at, calc)
        rows.append((i, e0, e1, f, out))
        print(f"  conf {i}: initial {e0:14.6f} eV   relaxed {e1:14.6f} eV   "
              f"fmax {f:.4f}")

    energies = np.array([r[2] for r in rows])
    print(f"  spread {energies.max() - energies.min():.6f} eV   "
          f"std {energies.std(ddof=1):.6f} eV")

    n_unique = len(np.unique(np.round(energies, 6)))
    print(f"  {n_unique} distinct relaxed energies out of {len(energies)} conformers")

    print("  pairwise heavy-atom RMSD after optimal superposition (A):")
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            print(f"    {i}-{j}: {heavy_atom_rmsd(rows[i][4], rows[j][4]):.3f}")


def heavy_atom_rmsd(a, b) -> float:
    """RMSD over heavy atoms after Kabsch superposition.

    Without alignment, RMSD counts rigid-body rotation as conformational
    difference: methane, which has a single conformation, otherwise reports
    up to 1.1 A between identical structures.
    """
    keep = a.get_atomic_numbers() != 1
    P = a.get_positions()[keep]
    Q = b.get_positions()[keep]
    if len(P) < 2:
        return 0.0
    P = P - P.mean(0)
    Q = Q - Q.mean(0)
    V, _, W = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(V @ W))
    U = V @ np.diag([1.0, 1.0, d]) @ W
    return float(np.sqrt((((P @ U) - Q) ** 2).sum(1).mean()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="esol", choices=list(DATASETS))
    ap.add_argument("--model", default="medium", help="MACE-OFF size: small|medium|large")
    ap.add_argument("--n-conformers", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None, help="first N molecules only")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    ap.add_argument("--debug", nargs="+", default=None,
                    help="SMILES to diagnose per-conformer, then exit")
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--restart", action="store_true",
                    help="ignore an existing output file and recompute")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    if args.debug:
        calc = get_calculator(args.model, args.device, args.dtype)
        for smi in args.debug:
            debug_molecule(smi, calc, args.n_conformers)
        return

    spec = DATASETS[args.dataset]
    os.makedirs(args.outdir, exist_ok=True)
    df = load_dataframe(spec)
    if args.limit:
        df = df.head(args.limit)

    path = f"{args.outdir}/mace_join_{spec.name}.csv"

    # Resume: a full dataset takes long enough that losing a session to a
    # timeout is a realistic failure, so partial results are written
    # periodically and reused.
    rows, done = [], set()
    if os.path.exists(path) and not args.restart:
        prev = pd.read_csv(path)
        rows = prev.to_dict("records")
        done = set(prev["smiles"])
        print(f"[mace] resuming from {path}: {len(done)} molecules already done")

    todo = [(s, y) for s, y in zip(df["smiles"], df["y"]) if s not in done]
    if not todo:
        print("[mace] nothing to do; delete the file or pass --restart to recompute")
    else:
        calc = get_calculator(args.model, args.device, args.dtype)
        t0 = time.time()
        for i, (smi, y) in enumerate(todo):
            rows.append(dict(smiles=smi, y=y, **describe(smi, calc, args.n_conformers)))
            if (i + 1) % args.checkpoint_every == 0:
                pd.DataFrame(rows).to_csv(path, index=False)
                rate = (time.time() - t0) / (i + 1)
                left = rate * (len(todo) - i - 1)
                print(f"[mace] {i + 1}/{len(todo)}  {rate:.2f}s/mol  "
                      f"~{left / 60:.1f} min left  (checkpointed)")

    out = pd.DataFrame(rows)
    out.to_csv(path, index=False)

    ok = out[FEATURES].notna().all(axis=1).sum()
    print(f"\n[mace] {ok}/{len(out)} molecules with complete descriptors "
          f"({100 * ok / len(out):.1f}%)")
    print(f"[mace] median residual force {out.mace_fmax_final.median():.4f} eV/A")
    unconverged = (out.mace_fmax_final > 0.05).sum()
    print(f"[mace] {unconverged} conformers above the 0.05 eV/A force threshold")
    print(out[FEATURES].describe().round(3).to_string())
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
