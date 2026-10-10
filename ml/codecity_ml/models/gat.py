"""Heterogeneous GATv2 for per-file bug-risk prediction.

Differences from graphsage.py:
  * Attention: each edge gets a learned weight per head, so a file can
    weight some neighbors above others instead of averaging them equally.
  * Edge weights: co-change and authorship edges carry a 1-d edge_attr
    (log commit count, from dataset.py). GATv2Conv uses it in the attention
    score. Import edges have no weight, so they use no edge features.
  * Skip connection: GAT has no separate "self" weight like SAGEConv, so a
    per-layer linear skip keeps a node's own features (and gives files with
    no neighbors a non-zero embedding). Self-loops are disabled because
    author<->file edges are bipartite and cannot have them.

Returns raw logits (pair with BCEWithLogitsLoss), same as graphsage.py.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATv2Conv, HeteroConv

EDGE_TYPES: tuple[tuple[str, str, str], ...] = (
    ("file", "imports", "file"),
    ("file", "cochange", "file"),
    ("file", "authored_by", "author"),
    ("author", "authored", "file"),
)

# Edge types that carry a 1-d edge_attr in dataset.py.
WEIGHTED_EDGE_TYPES = {
    ("file", "cochange", "file"),
    ("file", "authored_by", "author"),
    ("author", "authored", "file"),
}


class HeteroGAT(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        heads: int = 4,
        edge_types: tuple[tuple[str, str, str], ...] = EDGE_TYPES,
    ) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.skips = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                HeteroConv(
                    {
                        et: GATv2Conv(
                            (-1, -1),
                            hidden_dim // heads,
                            heads=heads,  # concat=True -> output is hidden_dim
                            edge_dim=1 if et in WEIGHTED_EDGE_TYPES else None,
                            add_self_loops=False,
                            dropout=dropout,  # attention dropout
                        )
                        for et in edge_types
                    },
                    aggr="sum",
                )
            )
            self.skips.append(
                nn.ModuleDict({"file": nn.LazyLinear(hidden_dim), "author": nn.LazyLinear(hidden_dim)})
            )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data: HeteroData) -> torch.Tensor:
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        edge_attr_dict = data.edge_attr_dict  # only edge types that have it

        for conv, skip in zip(self.convs, self.skips):
            out = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            # Node types that received no messages are missing from `out`;
            # the skip path still covers them.
            x_dict = {
                k: F.relu(skip[k](x_dict[k]) + out.get(k, 0.0))
                for k in x_dict
            }
            x_dict = {
                k: F.dropout(v, p=self.dropout, training=self.training)
                for k, v in x_dict.items()
            }

        return self.head(x_dict["file"]).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, data: HeteroData) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(self.forward(data))