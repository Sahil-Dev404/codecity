"""Model name -> constructor, so training and ablation code can pick a
model from a config string instead of importing a specific class."""

from __future__ import annotations

from torch import nn

from codecity_ml.models.gat import HeteroGAT
from codecity_ml.models.graphsage import HeteroGraphSAGE

MODELS: dict[str, type[nn.Module]] = {
    "sage": HeteroGraphSAGE,
    "gat": HeteroGAT,
}


def build_model(name: str, hidden_dim: int, num_layers: int, dropout: float) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from {sorted(MODELS)}")
    return MODELS[name](hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)