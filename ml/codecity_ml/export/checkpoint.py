"""Save and load a trained model as one self-contained checkpoint.

Contains everything needed to score a new graph: weights, architecture
hyperparameters, the feature scaler, the feature order, and metadata about
what the model was trained and tested on.

Loading needs one dummy forward pass first, because the layers use lazy
input sizes (-1): their weight shapes don't exist until they see data.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch_geometric.data import HeteroData

from codecity_ml.evaluation.metrics import EvalResult
from codecity_ml.graph.dataset import FeatureScaler
from codecity_ml.graph.features import FEATURE_NAMES
from codecity_ml.models.registry import build_model

CHECKPOINT_VERSION = 1
AUTHOR_FEATURES = 3  # width of author features built in dataset.py


def _dummy_graph() -> HeteroData:
    """Tiny graph with every node and edge type present, used to materialize
    the lazy layers before loading weights."""
    g = HeteroData()
    g["file"].x = torch.zeros(2, len(FEATURE_NAMES))
    g["author"].x = torch.zeros(1, AUTHOR_FEATURES)
    g["file", "imports", "file"].edge_index = torch.tensor([[0], [1]])
    g["file", "cochange", "file"].edge_index = torch.tensor([[0, 1], [1, 0]])
    g["file", "cochange", "file"].edge_attr = torch.ones(2, 1)
    g["file", "authored_by", "author"].edge_index = torch.tensor([[0], [0]])
    g["file", "authored_by", "author"].edge_attr = torch.ones(1, 1)
    g["author", "authored", "file"].edge_index = torch.tensor([[0], [0]])
    g["author", "authored", "file"].edge_attr = torch.ones(1, 1)
    return g


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    scaler: FeatureScaler,
    model_name: str,
    hparams: dict,
    split_mode: str,
    train_repos: list[str],
    test_repos: list[str],
    metrics: dict[str, EvalResult],
    per_repo_mean: dict[str, float],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": CHECKPOINT_VERSION,
            "model_name": model_name,
            "hparams": hparams,  # hidden_dim, num_layers, dropout
            "feature_names": list(FEATURE_NAMES),
            "scaler_mean": scaler.mean,
            "scaler_std": scaler.std,
            "state_dict": model.state_dict(),
            "split_mode": split_mode,
            "train_repos": train_repos,
            "test_repos": test_repos,
            "metrics": {k: asdict(v) for k, v in metrics.items()},
            "per_repo_mean_pr_auc": per_repo_mean,
        },
        path,
    )
    return path


def load_checkpoint(path: Path) -> tuple[nn.Module, FeatureScaler, dict]:
    """Return (model in eval mode, scaler, metadata dict)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)

    if ckpt["version"] != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported checkpoint version {ckpt['version']}")
    if ckpt["feature_names"] != list(FEATURE_NAMES):
        raise ValueError(
            "Checkpoint was trained with a different feature set/order than the "
            "current code. Retrain, or restore the matching graph/features.py."
        )

    hp = ckpt["hparams"]
    model = build_model(ckpt["model_name"], hp["hidden_dim"], hp["num_layers"], hp["dropout"])
    model.eval()
    with torch.no_grad():
        model(_dummy_graph())  # materialize lazy layers
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    scaler = FeatureScaler(mean=ckpt["scaler_mean"], std=ckpt["scaler_std"])
    meta = {k: v for k, v in ckpt.items() if k not in ("state_dict", "scaler_mean", "scaler_std")}
    return model, scaler, meta