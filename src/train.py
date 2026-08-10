"""Training loop + metrics for a single (model, split, seed) configuration."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score, mean_absolute_error, mean_squared_error,
    r2_score, roc_auc_score,
)
from torch_geometric.loader import DataLoader

from .data import ATOM_DIM, BOND_DIM
from .model import MPNN


def metrics(y_true, y_pred, task: str) -> dict:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if task == "regression":
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }
    return {
        "roc_auc": float(roc_auc_score(y_true, y_pred)),
        "pr_auc": float(average_precision_score(y_true, y_pred)),
    }


def primary(task: str) -> str:
    return "rmse" if task == "regression" else "roc_auc"


def better(a: float, b: float, task: str) -> bool:
    """Is `a` better than `b`?"""
    return a < b if task == "regression" else a > b


@torch.no_grad()
def _predict(model, loader, device, mu=0.0, sigma=1.0, task="regression"):
    model.eval()
    preds, trues = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch).cpu().numpy().ravel()
        if task == "regression":
            out = out * sigma + mu
        else:
            out = 1.0 / (1.0 + np.exp(-out))
        preds.append(out)
        trues.append(batch.y.cpu().numpy().ravel())
    return np.concatenate(preds), np.concatenate(trues)


def train_gnn(
    graphs,
    train_idx,
    val_idx,
    test_idx,
    task: str = "regression",
    seed: int = 0,
    epochs: int = 150,
    patience: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden: int = 128,
    layers: int = 4,
    device: str | None = None,
    verbose: bool = False,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    tr = [graphs[i] for i in train_idx]
    va = [graphs[i] for i in val_idx]
    te = [graphs[i] for i in test_idx]

    # Standardise the target using TRAIN statistics only. Using the full dataset
    # here is a classic and invisible leak.
    if task == "regression":
        y_tr = np.array([float(g.y) for g in tr])
        mu, sigma = float(y_tr.mean()), float(y_tr.std() + 1e-8)
    else:
        mu, sigma = 0.0, 1.0

    dl_tr = DataLoader(tr, batch_size=batch_size, shuffle=True, drop_last=len(tr) > batch_size)
    dl_va = DataLoader(va, batch_size=256)
    dl_te = DataLoader(te, batch_size=256)

    model = MPNN(ATOM_DIM, BOND_DIM, hidden=hidden, layers=layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=max(1, epochs * len(dl_tr)), pct_start=0.1
    )
    loss_fn = nn.MSELoss() if task == "regression" else nn.BCEWithLogitsLoss()

    best = float("inf") if task == "regression" else -float("inf")
    best_state, bad = None, 0

    for epoch in range(epochs):
        model.train()
        for batch in dl_tr:
            batch = batch.to(device)
            target = (batch.y - mu) / sigma if task == "regression" else batch.y
            opt.zero_grad()
            loss = loss_fn(model(batch), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()

        p, t = _predict(model, dl_va, device, mu, sigma, task)
        score = metrics(t, p, task)[primary(task)]
        if better(score, best, task):
            best, bad = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose and epoch % 10 == 0:
            print(f"  epoch {epoch:3d}  val {primary(task)}={score:.4f}  best={best:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    p_te, y_te = _predict(model, dl_te, device, mu, sigma, task)
    return metrics(y_te, p_te, task), p_te, y_te, [g.smiles for g in te]
