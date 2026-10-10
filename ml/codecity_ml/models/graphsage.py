"""Heterogeneous GraphSAGE for per-file bug-risk prediction.

Each layer runs one SAGEConv per edge type (imports, cochange, authored_by,
authored) and sums the results per destination node type. After
`num_layers` layers, a file's embedding contains information from files up
to `num_layers` hops away (imports, co-changed files, files sharing an
author). An MLP head maps file embeddings to one logit per file.

Returns raw logits, not probabilities. Pair with BCEWithLogitsLoss
(numerically stabler than sigmoid + BCE); use predict_proba() for scores.

Note: SAGEConv ignores edge_attr, so the log-commit-count edge weights from
dataset.py are not used here. Weighted variants (GATv2Conv with edge_dim,
or HGT) come in later files.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv

# Must match the edge types dataset.py creates.
EDGE_TYPES: tuple[tuple[str, str, str], ...] = (
    ("file", "imports", "file"),
    ("file", "cochange", "file"),
    ("file", "authored_by", "author"),
    ("author", "authored", "file"),
)


class HeteroGraphSAGE(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        edge_types: tuple[tuple[str, str, str], ...] = EDGE_TYPES,
    ) -> None:
        super().__init__()
        self.dropout = dropout

        # (-1, -1) = "infer input sizes on first forward". File and author
        # features have different widths, and lazy init avoids hardcoding
        # them. Run one forward pass before counting parameters or
        # building an optimizer.
        self.convs = nn.ModuleList(
            [
                HeteroConv(
                    {et: SAGEConv((-1, -1), hidden_dim) for et in edge_types},
                    aggr="sum",
                )
                for _ in range(num_layers)
            ]
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

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}
            x_dict = {
                k: F.dropout(v, p=self.dropout, training=self.training)
                for k, v in x_dict.items()
            }

        return self.head(x_dict["file"]).squeeze(-1)  # [num_files]

    @torch.no_grad()
    def predict_proba(self, data: HeteroData) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(self.forward(data))