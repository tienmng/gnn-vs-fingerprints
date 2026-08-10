"""Train / val / test splitters.

The whole point of the project lives in this file. A random split on molecular
data leaks near-duplicate analogues between train and test and inflates every
metric. A scaffold split forces the model to generalise to unseen chemotypes,
which is what actually happens when a project moves to a new series.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def random_split(
    smiles: list[str], frac=(0.8, 0.1, 0.1), seed: int = 0
) -> tuple[list[int], list[int], list[int]]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(smiles))
    n_tr = int(frac[0] * len(idx))
    n_va = int(frac[1] * len(idx))
    return (
        idx[:n_tr].tolist(),
        idx[n_tr : n_tr + n_va].tolist(),
        idx[n_tr + n_va :].tolist(),
    )


def murcko_scaffold(smiles: str, include_chirality: bool = False) -> str:
    """Bemis-Murcko scaffold SMILES. Empty string for acyclic molecules."""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(
            smiles=smiles, includeChirality=include_chirality
        )
    except Exception:
        return ""


def scaffold_split(
    smiles: list[str], frac=(0.8, 0.1, 0.1), seed: int = 0, balanced: bool = True
) -> tuple[list[int], list[int], list[int]]:
    """Group molecules by Bemis-Murcko scaffold; no scaffold spans two splits.

    balanced=True reproduces the Chemprop convention: scaffold groups larger
    than half a split go straight to train, the remainder are shuffled with
    `seed` and greedily packed. This gives you seed-to-seed variation (so error
    bars mean something) while keeping the split genuinely hard.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for i, smi in enumerate(smiles):
        groups[murcko_scaffold(smi)].append(i)

    n = len(smiles)
    n_tr, n_va = int(frac[0] * n), int(frac[1] * n)
    sets = list(groups.values())

    if balanced:
        rng = np.random.default_rng(seed)
        big = [s for s in sets if len(s) > n_va / 2]
        small = [s for s in sets if len(s) <= n_va / 2]
        rng.shuffle(small)
        sets = big + small
    else:
        sets = sorted(sets, key=len, reverse=True)

    train, val, test = [], [], []
    for group in sets:
        if len(train) + len(group) <= n_tr:
            train += group
        elif len(val) + len(group) <= n_va:
            val += group
        else:
            test += group
    return train, val, test


def get_split(kind: str, smiles: list[str], seed: int = 0, frac=(0.8, 0.1, 0.1)):
    if kind == "random":
        return random_split(smiles, frac, seed)
    if kind == "scaffold":
        return scaffold_split(smiles, frac, seed)
    raise ValueError(f"unknown split kind: {kind}")


def split_report(smiles: list[str], train, val, test) -> dict:
    """Sanity metrics you should print for every split. Scaffold overlap must be 0."""
    sc = [murcko_scaffold(s) for s in smiles]
    s_tr = {sc[i] for i in train}
    s_te = {sc[i] for i in test}
    return {
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "n_scaffolds_total": len(set(sc)),
        "train_test_scaffold_overlap": len(s_tr & s_te),
    }


if __name__ == "__main__":
    smis = ["CCO", "c1ccccc1C", "c1ccccc1CC", "c1ccncc1", "CCCC", "c1ccccc1CCC"]
    tr, va, te = scaffold_split(smis, seed=0)
    print(tr, va, te)
    print(split_report(smis, tr, va, te))
