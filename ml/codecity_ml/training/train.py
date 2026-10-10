"""Train a heterogeneous GNN and compare it against XGBoost.

Two split modes:

cross_project (default when there are >= 5 repos):
    Whole repos are held out as TEST. The model never sees them, which is
    the situation a pasted URL puts it in. Among the remaining repos,
    snapshots are split by DATE into [ fit | purge | val ]; val is used
    only for early stopping.

temporal (fallback for 1-4 repos):
    [ fit | purge | val ] [ purge ] [ test ], all by date.

Either way the scaler is fit on `fit` only, and the test set is scored
once per model. Results are reported pooled and per held-out repo (mean of
per-repo PR-AUC), because a pooled number is dominated by large repos.

Known remaining caveat: in cross_project mode, held-out repos can overlap in
calendar time with training repos. That is a much weaker leak than
same-repo leakage, but it is not zero. For a strict version, also drop test
snapshots earlier than the end of the training period.

Run:  python -m codecity_ml.training.train                (sage and gat)
      python -m codecity_ml.training.train gat            (one model)
      python -m codecity_ml.training.train --temporal     (force temporal)
"""

from __future__ import annotations

import argparse
import copy
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch_geometric.data import HeteroData
from xgboost import XGBClassifier

from codecity_ml.evaluation.metrics import EvalResult, evaluate, pr_auc
from codecity_ml.graph.dataset import FeatureScaler, load_dataset, temporal_split
from codecity_ml.models.registry import build_model

MIN_REPOS_FOR_CROSS_PROJECT = 5


@dataclass
class TrainConfig:
    model_name: str = "sage"  # "sage" | "gat"
    split_mode: str = "cross_project"  # "cross_project" | "temporal"
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.3
    lr: float = 0.005
    weight_decay: float = 1e-4
    max_epochs: int = 100
    patience: int = 15
    test_fraction: float = 0.2  # fraction of REPOS (cross_project) or snapshots (temporal)
    val_fraction: float = 0.2  # fraction of training snapshots, by date
    purge_days: int = 180  # match the label window_days
    seed: int = 42


@dataclass
class TrainResult:
    model: nn.Module
    scaler: FeatureScaler
    split_mode: str = ""
    test_repos: list[str] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    best_epoch: int = 0
    gnn_test: EvalResult | None = None
    xgb_test: EvalResult | None = None
    gnn_per_repo: dict[str, float] = field(default_factory=dict)
    xgb_per_repo: dict[str, float] = field(default_factory=dict)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def repo_slugs(graphs: list[HeteroData]) -> list[str]:
    return sorted({g.repo_slug for g in graphs})


def split_by_repo(
    graphs: list[HeteroData], test_fraction: float, seed: int
) -> tuple[list[HeteroData], list[HeteroData]]:
    """Hold out whole repos. Deterministic for a given seed."""
    slugs = repo_slugs(graphs)
    if len(slugs) < 2:
        raise ValueError("cross_project split needs at least 2 repos.")
    rng = random.Random(seed)
    rng.shuffle(slugs)
    n_test = max(1, round(len(slugs) * test_fraction))
    test_slugs = set(slugs[:n_test])
    train_all = [g for g in graphs if g.repo_slug not in test_slugs]
    test = [g for g in graphs if g.repo_slug in test_slugs]
    return train_all, test


def _split_fit_val(
    graphs: list[HeteroData], val_fraction: float, purge_days: int
) -> tuple[list[HeteroData], list[HeteroData]]:
    graphs = sorted(graphs, key=lambda g: g.snapshot_ts)
    n_val = max(1, int(len(graphs) * val_fraction))
    val = graphs[-n_val:]
    val_start = val[0].snapshot_ts
    fit = [g for g in graphs[:-n_val] if g.snapshot_ts < val_start - purge_days * 86400]
    return fit, val


def _pos_weight(graphs: list[HeteroData]) -> torch.Tensor:
    y = torch.cat([g["file"].y for g in graphs]).float()
    n_pos = y.sum().clamp_min(1.0)
    return (len(y) - n_pos) / n_pos


@torch.no_grad()
def _score(model: nn.Module, graphs: list[HeteroData]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenated (labels, probabilities) across graphs."""
    ys, ps = [], []
    for g in graphs:
        ys.append(g["file"].y.numpy())
        ps.append(model.predict_proba(g).numpy())
    return np.concatenate(ys), np.concatenate(ps)


def _per_repo_pr_auc(graphs: list[HeteroData], probs: list[np.ndarray]) -> dict[str, float]:
    """PR-AUC per repo, pooling that repo's snapshots. Repos with no
    positives (or no negatives) are skipped: PR-AUC is undefined there."""
    ys: dict[str, list[np.ndarray]] = defaultdict(list)
    ps: dict[str, list[np.ndarray]] = defaultdict(list)
    for g, p in zip(graphs, probs):
        ys[g.repo_slug].append(g["file"].y.numpy())
        ps[g.repo_slug].append(p)

    out: dict[str, float] = {}
    for slug in ys:
        y, p = np.concatenate(ys[slug]), np.concatenate(ps[slug])
        if 0 < y.sum() < len(y):
            out[slug] = pr_auc(y, p)
    return out


def _xgboost_on_same_split(
    fit: list[HeteroData], test: list[HeteroData]
) -> tuple[EvalResult, dict[str, float]]:
    """XGBoost on the identical split and identical (transformed, scaled)
    file features, with no graph structure."""
    X_fit = torch.cat([g["file"].x for g in fit]).numpy()
    y_fit = torch.cat([g["file"].y for g in fit]).numpy()
    X_test = torch.cat([g["file"].x for g in test]).numpy()
    y_test = torch.cat([g["file"].y for g in test]).numpy()

    n_pos = max(1, int(y_fit.sum()))
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=(len(y_fit) - n_pos) / n_pos,
        eval_metric="aucpr",
        random_state=42,
    )
    model.fit(X_fit, y_fit)
    probs = model.predict_proba(X_test)[:, 1]

    sizes = np.cumsum([g["file"].x.shape[0] for g in test])[:-1]
    per_repo = _per_repo_pr_auc(test, np.split(probs, sizes))
    return evaluate("XGBoost (same split)", y_test, probs), per_repo


def train(graphs: list[HeteroData], cfg: TrainConfig | None = None) -> TrainResult:
    cfg = cfg or TrainConfig()
    set_seed(cfg.seed)

    # Clone: the scaler modifies features in place, and the caller may train
    # several models on the same loaded graphs.
    graphs = [g.clone() for g in graphs]

    if cfg.split_mode == "cross_project":
        train_all, test = split_by_repo(graphs, cfg.test_fraction, cfg.seed)
    elif cfg.split_mode == "temporal":
        train_all, test = temporal_split(
            graphs, test_fraction=cfg.test_fraction, purge_days=cfg.purge_days
        )
    else:
        raise ValueError(f"Unknown split_mode '{cfg.split_mode}'")

    fit, val = _split_fit_val(train_all, cfg.val_fraction, cfg.purge_days)
    if not fit or not val or not test:
        raise ValueError(
            f"Too few snapshots after splitting (fit={len(fit)}, val={len(val)}, "
            f"test={len(test)}). Build more repos or lower purge_days."
        )

    scaler = FeatureScaler.fit(fit)
    for subset in (fit, val, test):
        scaler.apply(subset)

    model = build_model(cfg.model_name, cfg.hidden_dim, cfg.num_layers, cfg.dropout)
    model.eval()
    with torch.no_grad():
        model(fit[0])  # initializes lazy layers BEFORE the optimizer sees them

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=_pos_weight(fit))

    result = TrainResult(
        model=model,
        scaler=scaler,
        split_mode=cfg.split_mode,
        test_repos=repo_slugs(test),
    )
    best_val, best_state, stale = -1.0, None, 0

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        order = list(range(len(fit)))
        random.shuffle(order)
        total_loss = 0.0
        for i in order:
            g = fit[i]
            optimizer.zero_grad()
            loss = loss_fn(model(g), g["file"].y.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        y_val, p_val = _score(model, val)
        val_prauc = pr_auc(y_val, p_val)
        result.history.append(
            {"epoch": epoch, "train_loss": total_loss / len(fit), "val_pr_auc": val_prauc}
        )

        if val_prauc > best_val:
            best_val, stale = val_prauc, 0
            best_state = copy.deepcopy(model.state_dict())
            result.best_epoch = epoch
        else:
            stale += 1
            if stale >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    with torch.no_grad():
        test_probs = [model.predict_proba(g).numpy() for g in test]
    y_test = np.concatenate([g["file"].y.numpy() for g in test])
    result.gnn_test = evaluate(f"Hetero{cfg.model_name.upper()}", y_test, np.concatenate(test_probs))
    result.gnn_per_repo = _per_repo_pr_auc(test, test_probs)
    result.xgb_test, result.xgb_per_repo = _xgboost_on_same_split(fit, test)
    return result


def _mean(d: dict[str, float]) -> float:
    return sum(d.values()) / len(d) if d else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", default=["sage", "gat"])
    ap.add_argument("--temporal", action="store_true", help="force the temporal split")
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    args = ap.parse_args()

    graphs = load_dataset(args.data)
    n_repos = len(repo_slugs(graphs))
    mode = "cross_project"
    if args.temporal or n_repos < MIN_REPOS_FOR_CROSS_PROJECT:
        mode = "temporal"
        if not args.temporal:
            print(f"Only {n_repos} repos (< {MIN_REPOS_FOR_CROSS_PROJECT}): using temporal split.")
    print(f"snapshots: {len(graphs)} | repos: {n_repos} | split: {mode}")

    results = []
    for name in args.models:
        res = train(graphs, TrainConfig(model_name=name, split_mode=mode))
        results.append(res)
        print(f"\n[{name}] best epoch: {res.best_epoch}")
        for row in res.history[:: max(1, len(res.history) // 6)]:
            print(
                f"  epoch {row['epoch']:3d}  loss {row['train_loss']:.4f}  "
                f"val PR-AUC {row['val_pr_auc']:.3f}"
            )

    first = results[0]
    print(f"\n=== Test results ({mode}; identical split for every model) ===")
    if mode == "cross_project":
        print(f"held-out repos ({len(first.test_repos)}): {', '.join(first.test_repos)}")
    base_rate = first.xgb_test.n_positive / first.xgb_test.n_examples
    print(f"test examples: {first.xgb_test.n_examples} | positive rate: {base_rate:.1%}")
    print("\nPooled:")
    print(" ", first.xgb_test.summary())
    for res in results:
        print(" ", res.gnn_test.summary())

    print("\nMean per-repo PR-AUC (each held-out repo counts equally):")
    print(f"  XGBoost : {_mean(first.xgb_per_repo):.3f}  ({len(first.xgb_per_repo)} repos)")
    for res, name in zip(results, args.models):
        wins = sum(
            res.gnn_per_repo[s] > first.xgb_per_repo[s]
            for s in res.gnn_per_repo
            if s in first.xgb_per_repo
        )
        print(
            f"  {name:8s}: {_mean(res.gnn_per_repo):.3f}  "
            f"(beats XGBoost on {wins}/{len(res.gnn_per_repo)} repos)"
        )


if __name__ == "__main__":
    main()