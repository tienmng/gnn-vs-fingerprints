"""Dataset loading, cleaning, and graph featurisation.

Deliberately does NOT use torch_geometric.datasets.MoleculeNet: writing your own
featuriser is ~60 lines, makes the atom/bond features explicit and defensible in
an interview, and lets the same code run on any CSV of SMILES + label.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from torch_geometric.data import Data

RDLogger.DisableLog("rdApp.*")

# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #

S3 = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    url: str
    smiles_col: str
    target_col: str
    task: str  # "regression" | "classification"
    units: str


DATASETS = {
    "esol": DatasetSpec(
        name="esol",
        url=f"{S3}/delaney-processed.csv",
        smiles_col="smiles",
        target_col="measured log solubility in mols per litre",
        task="regression",
        units="log mol/L",
    ),
    "lipo": DatasetSpec(
        name="lipo",
        url=f"{S3}/Lipophilicity.csv",
        smiles_col="smiles",
        target_col="exp",
        task="regression",
        units="logD",
    ),
    "bbbp": DatasetSpec(
        name="bbbp",
        url=f"{S3}/BBBP.csv",
        smiles_col="smiles",
        target_col="p_np",
        task="classification",
        units="P(permeable)",
    ),
    "bace": DatasetSpec(
        name="bace",
        url=f"{S3}/bace.csv",
        smiles_col="mol",
        target_col="Class",
        task="classification",
        units="P(active)",
    ),
}


def load_dataframe(spec: DatasetSpec, cache_dir: str = "data", csv: str | None = None) -> pd.DataFrame:
    """Download (once) and clean. Returns a df with columns ['smiles', 'y']."""
    if csv is not None:
        path = csv
    else:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"{spec.name}.csv")
        if not os.path.exists(path):
            print(f"[data] downloading {spec.url}")
            pd.read_csv(spec.url).to_csv(path, index=False)

    df = pd.read_csv(path)
    df = df[[spec.smiles_col, spec.target_col]].rename(
        columns={spec.smiles_col: "smiles", spec.target_col: "y"}
    )

    n0 = len(df)
    # Canonicalise, drop molecules RDKit cannot parse, drop duplicates and NaNs.
    canon = []
    for smi in df["smiles"]:
        mol = Chem.MolFromSmiles(smi)
        canon.append(Chem.MolToSmiles(mol) if mol is not None else None)
    df["smiles"] = canon
    df = df.dropna(subset=["smiles", "y"])
    df = df[df["smiles"].str.len() > 0]
    df = df.drop_duplicates(subset="smiles", keep="first").reset_index(drop=True)
    print(f"[data] {spec.name}: {n0} rows -> {len(df)} clean unique molecules")
    return df


# --------------------------------------------------------------------------- #
# Featurisation
# --------------------------------------------------------------------------- #

ATOMS = ["B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"]
DEGREES = [0, 1, 2, 3, 4, 5]
CHARGES = [-2, -1, 0, 1, 2]
NUM_HS = [0, 1, 2, 3, 4]
HYBRIDS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
CHIRAL = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
]
BONDS = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
STEREO = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
]


def _one_hot(value, allowable) -> list[float]:
    """One-hot with an explicit 'other' bucket at the end."""
    vec = [0.0] * (len(allowable) + 1)
    try:
        vec[allowable.index(value)] = 1.0
    except ValueError:
        vec[-1] = 1.0
    return vec


def atom_features(atom: Chem.Atom) -> list[float]:
    return (
        _one_hot(atom.GetSymbol(), ATOMS)
        + _one_hot(atom.GetTotalDegree(), DEGREES)
        + _one_hot(atom.GetFormalCharge(), CHARGES)
        + _one_hot(atom.GetTotalNumHs(), NUM_HS)
        + _one_hot(atom.GetHybridization(), HYBRIDS)
        + _one_hot(atom.GetChiralTag(), CHIRAL)
        + [
            float(atom.GetIsAromatic()),
            float(atom.IsInRing()),
            atom.GetMass() * 0.01,  # scaled so it sits near the one-hots
        ]
    )


def bond_features(bond: Chem.Bond) -> list[float]:
    return (
        _one_hot(bond.GetBondType(), BONDS)
        + _one_hot(bond.GetStereo(), STEREO)
        + [float(bond.GetIsConjugated()), float(bond.IsInRing())]
    )


ATOM_DIM = len(atom_features(Chem.MolFromSmiles("C").GetAtomWithIdx(0)))
BOND_DIM = len(bond_features(Chem.MolFromSmiles("CC").GetBondWithIdx(0)))


def mol_to_graph(smiles: str, y: float) -> Data | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    src, dst, eattr = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        f = bond_features(bond)
        src += [i, j]          # both directions: message passing needs them
        dst += [j, i]
        eattr += [f, f]

    if len(src) == 0:  # single-atom molecules, e.g. "C" or "[Xe]"
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, BOND_DIM), dtype=torch.float)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(eattr, dtype=torch.float)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([[float(y)]], dtype=torch.float),
        smiles=smiles,
    )


def build_graphs(df: pd.DataFrame) -> list[Data]:
    graphs = []
    for smi, y in zip(df["smiles"], df["y"]):
        g = mol_to_graph(smi, y)
        if g is not None:
            graphs.append(g)
    return graphs


if __name__ == "__main__":
    print(f"atom feature dim = {ATOM_DIM}, bond feature dim = {BOND_DIM}")
    g = mol_to_graph("CC(=O)Oc1ccccc1C(=O)O", -1.0)
    print(g)
