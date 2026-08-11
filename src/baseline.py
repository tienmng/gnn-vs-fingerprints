"""Baseline model: Morgan fingerprints + RDKit descriptors -> LightGBM."""
from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator

_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

DESCRIPTORS = [
    "MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "RingCount", "NumAromaticRings", "FractionCSP3",
    "HeavyAtomCount", "NHOHCount", "NOCount", "MolMR", "BertzCT",
    "LabuteASA", "BalabanJ",
]
_DESC_FN = {name: fn for name, fn in Descriptors.descList if name in DESCRIPTORS}


def featurize(smiles: list[str]) -> np.ndarray:
    """[n, 2048 + n_desc] float32 matrix."""
    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        fp = np.zeros(2048, dtype=np.float32)
        desc = np.zeros(len(DESCRIPTORS), dtype=np.float32)
        if mol is not None:
            arr = _GEN.GetFingerprintAsNumPy(mol)
            fp[:] = arr.astype(np.float32)
            for k, name in enumerate(DESCRIPTORS):
                fn = _DESC_FN.get(name)
                try:
                    desc[k] = fn(mol) if fn is not None else 0.0
                except Exception:
                    desc[k] = 0.0
        rows.append(np.concatenate([fp, desc]))
    X = np.vstack(rows)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def fit_predict(
    X_tr, y_tr, X_va, y_va, X_te, task: str = "regression", seed: int = 0
) -> np.ndarray:
    """Train LightGBM with early stopping on the validation set; return test preds."""
    import lightgbm as lgb

    params = dict(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=10,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.5,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    Model = lgb.LGBMRegressor if task == "regression" else lgb.LGBMClassifier
    model = Model(**params)
    common = dict(
        eval_metric="l2" if task == "regression" else "auc",
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    try:  # lightgbm >= 4.7 renamed the validation arguments
        model.fit(X_tr, y_tr, eval_X=[X_va], eval_y=[y_va], **common)
    except TypeError:
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], **common)
    if task == "regression":
        return model.predict(X_te)
    return model.predict_proba(X_te)[:, 1]
