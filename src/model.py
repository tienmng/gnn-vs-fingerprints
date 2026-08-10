"""A GINE message-passing network with edge features and residual connections.

Kept deliberately small and readable: ~50 lines is enough to be competitive on
MoleculeNet-scale data, and anything bigger just overfits 1k molecules.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_add_pool, global_max_pool, global_mean_pool


class MPNN(nn.Module):
    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        hidden: int = 128,
        layers: int = 4,
        dropout: float = 0.15,
        out_dim: int = 1,
    ):
        super().__init__()
        self.atom_emb = nn.Linear(atom_dim, hidden)
        self.bond_emb = nn.Linear(bond_dim, hidden)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, 2 * hidden),
                nn.ReLU(),
                nn.Linear(2 * hidden, hidden),
            )
            self.convs.append(GINEConv(mlp, edge_dim=hidden, train_eps=True))
            self.norms.append(nn.BatchNorm1d(hidden))

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(3 * hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, data):
        x = self.atom_emb(data.x)
        e = self.bond_emb(data.edge_attr)

        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, data.edge_index, e)
            h = norm(h)
            h = torch.relu(h)
            x = x + self.dropout(h)          # residual: helps past ~3 layers

        g = torch.cat(
            [
                global_mean_pool(x, data.batch),
                global_max_pool(x, data.batch),
                global_add_pool(x, data.batch),   # size-sensitive, matters for solubility
            ],
            dim=-1,
        )
        return self.head(g)
