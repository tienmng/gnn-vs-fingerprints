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


def get_calculator(model: str = "medium", device: str | None = None, dtype: str = "float32"):
    """MACE-OFF as an ASE calculator. Import is deferred so the module loads
    without mace-torch installed."""
    import torch
    from mace.calculators import mace_off

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[mace] loading MACE-OFF '{model}' on {device}")
    return mace_off(model=model, device=device, default_dtype=dtype)


def embed_conformers(smiles: str, n_conf: int = 5, seed: int = 0):
    """ETKDG conformers with MMFF cleanup, returned as ASE Atoms."""
    from ase import Atoms

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useSmallRingTorsions = True
    # No pruneRmsThresh: on this RDKit build it collapses an ETKDG ensemble to a
    # single conformer even for flexible chains, which would silently zero the
    # spread descriptors. Duplicates are harmless here because the ensemble is
    # reduced by energy after relaxation.
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
    """Local relaxation. Returns (energy_per_atom, initial_energy_per_atom, fmax)."""
    from ase.optimize import LBFGS

    atoms = atoms.copy()
    atoms.calc = calc
    e0 = atoms.get_potential_energy() / len(atoms)
    opt = LBFGS(atoms, logfile=None)
    opt.run(fmax=fmax, steps=steps)
    e1 = atoms.get_potential_energy() / len(atoms)
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


def describe(smiles: str, calc, n_conf: int = 5, seed: int = 0) -> dict:
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

    energies = np.array(energies)
    best = int(np.argmin(energies))
    row = dict(
        mace_e_min=float(energies[best]),
        mace_strain=float(initials[best] - energies[best]),
        mace_conf_spread=float(energies.max() - energies.min()),
        mace_conf_std=float(energies.std(ddof=1)) if len(energies) > 1 else 0.0,
        mace_fmax_final=float(forces[best]),
        mace_n_conf_ok=float(len(energies)),
    )
    row.update(shape_descriptors(relaxed[best]))
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="esol", choices=list(DATASETS))
    ap.add_argument("--model", default="medium", help="MACE-OFF size: small|medium|large")
    ap.add_argument("--n-conformers", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None, help="first N molecules only")
    ap.add_argument("--device", default=None)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    spec = DATASETS[args.dataset]
    os.makedirs(args.outdir, exist_ok=True)
    df = load_dataframe(spec)
    if args.limit:
        df = df.head(args.limit)

    calc = get_calculator(args.model, args.device)

    rows, t0 = [], time.time()
    for i, (smi, y) in enumerate(zip(df["smiles"], df["y"])):
        rows.append(dict(smiles=smi, y=y, **describe(smi, calc, args.n_conformers)))
        if (i + 1) % 50 == 0:
            rate = (time.time() - t0) / (i + 1)
            left = rate * (len(df) - i - 1)
            print(f"[mace] {i + 1}/{len(df)}  {rate:.2f}s/mol  ~{left / 60:.1f} min left")

    out = pd.DataFrame(rows)
    path = f"{args.outdir}/mace_join_{spec.name}.csv"
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
