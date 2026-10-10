"""Ablations: which features and which parts of the graph actually matter.

Two runners:

1. run_feature_group_ablation (XGBoost, tabular): drop a feature GROUP and
   measure the change. Unchanged from File 23.

2. run_graph_ablation (GNN): train the same model with parts of the graph
   removed, on the same cross-project split, repeated over several seeds.
   The seed controls which repos are held out, so each seed is a different
   test set; variants are compared WITHIN a seed (identical split), and the
   delta vs the full model is averaged across seeds.

   Variants: full | no_cochange | no_imports | no_authors | no_graph | layers_1 | layers_3
   "no_graph" removes every edge, leaving a per-file MLP on the same
   features. It is the control that answers "does any graph structure help?"

Caveats: SAGE ignores edge weights; GAT uses them. Compare variants within a
model, not across models. Small differences (< ~0.02 PR-AUC) are noise.

Run:
    python -m codecity_ml.evaluation.ablations features
    python -m codecity_ml.evaluation.ablations graph --model sage --seeds 42 43 44
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from codecity_ml.evaluation.metrics import EvalResult, evaluate
from codecity_ml.graph.features import FEATURE_NAMES
from codecity_ml.models.baselines import temporal_train_test_split

# ─────────────────────────────────────────────────────────────
# 1. Feature-group ablation (XGBoost, tabular)
# ─────────────────────────────────────────────────────────────

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "complexity": (
        "lines_of_code",
        "max_cyclomatic_complexity",
        "avg_cyclomatic_complexity",
        "function_count",
    ),
    "age_recency": ("file_age_days", "days_since_last_change"),
    "churn": ("total_commit_count", "recent_commit_count", "total_churn", "recent_churn"),
    "authorship": ("distinct_author_count", "top_author_share"),
    "cochange": ("cochange_neighbor_count",),
    "bug_history": ("past_bugfix_count",),
}


@dataclass(frozen=True)
class AblationRun:
    variant_name: str
    dropped_group: str | None  # None for the "all features" baseline
    features_used: tuple[str, ...]
    result: EvalResult


def _train_and_score(
    train: pd.DataFrame, test: pd.DataFrame, feature_cols: tuple[str, ...], variant_name: str
) -> EvalResult:
    X_train = train[list(feature_cols)].to_numpy(dtype="float32")
    y_train = train["label"].to_numpy()
    X_test = test[list(feature_cols)].to_numpy(dtype="float32")
    y_test = test["label"].to_numpy()

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return evaluate(variant_name, y_test, model.predict_proba(X_test)[:, 1])


def run_feature_group_ablation(df: pd.DataFrame) -> list[AblationRun]:
    """One fixed temporal split for every variant, so score differences are
    attributable to the features. (This tabular split has no purge gap; use
    the GNN ablation below for headline numbers.)"""
    train, test = temporal_train_test_split(df)
    all_features = tuple(FEATURE_NAMES)
    runs = [
        AblationRun(
            "all_features", None, all_features,
            _train_and_score(train, test, all_features, "all_features"),
        )
    ]
    for group_name, group_cols in FEATURE_GROUPS.items():
        remaining = tuple(f for f in FEATURE_NAMES if f not in group_cols)
        name = f"without_{group_name}"
        runs.append(
            AblationRun(name, group_name, remaining, _train_and_score(train, test, remaining, name))
        )
    return runs


def ablation_summary_table(runs: list[AblationRun]) -> pd.DataFrame:
    """One row per variant with PR-AUC delta vs the full-feature baseline,
    most negative delta (most important group) first."""
    baseline = next(r.result.pr_auc for r in runs if r.dropped_group is None)
    rows = [
        {
            "variant": r.variant_name,
            "dropped_group": r.dropped_group or "-",
            "pr_auc": round(r.result.pr_auc, 4),
            "pr_auc_delta": round(r.result.pr_auc - baseline, 4),
            "recall_at_10pct": round(r.result.recall_at_10pct, 4),
            "effort_at_recall_80pct": round(r.result.effort_at_recall_80pct, 4),
        }
        for r in runs
    ]
    return pd.DataFrame(rows).sort_values("pr_auc_delta").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# 2. Graph-structure ablation (GNN)
# ─────────────────────────────────────────────────────────────

IMPORTS = ("file", "imports", "file")
COCHANGE = ("file", "cochange", "file")
AUTHORED_BY = ("file", "authored_by", "author")
AUTHORED = ("author", "authored", "file")


@dataclass(frozen=True)
class GraphVariant:
    name: str
    drop: tuple[tuple[str, str, str], ...] = ()
    num_layers: int | None = None  # None = use TrainConfig default


GRAPH_VARIANTS: tuple[GraphVariant, ...] = (
    GraphVariant("full"),
    GraphVariant("no_cochange", drop=(COCHANGE,)),
    GraphVariant("no_imports", drop=(IMPORTS,)),
    GraphVariant("no_authors", drop=(AUTHORED_BY, AUTHORED)),
    GraphVariant("no_graph", drop=(IMPORTS, COCHANGE, AUTHORED_BY, AUTHORED)),
    GraphVariant("layers_1", num_layers=1),
    GraphVariant("layers_3", num_layers=3),
)


def drop_edge_types(graphs: list, drop: tuple[tuple[str, str, str], ...]) -> list:
    """Return copies of `graphs` with the given edge types emptied. The edge
    type stays present (with zero edges) so the model's per-edge-type layers
    still find the keys they expect."""
    import torch

    out = []
    for g in graphs:
        g = g.clone()
        for et in drop:
            store = g[et]
            store.edge_index = torch.empty((2, 0), dtype=torch.long)
            if "edge_attr" in store:
                store.edge_attr = torch.empty((0, 1), dtype=torch.float32)
        out.append(g)
    return out


def run_graph_ablation(
    graphs: list,
    model_name: str = "sage",
    seeds: tuple[int, ...] = (42, 43, 44),
    variants: tuple[GraphVariant, ...] = GRAPH_VARIANTS,
) -> pd.DataFrame:
    """Train every variant under every seed and summarize.

    Within one seed, every variant uses the same held-out repos (the split is
    a function of the seed only), so the per-seed delta vs 'full' isolates
    the effect of the graph change. We then report mean +/- std of that
    delta across seeds.
    """
    from codecity_ml.training.train import TrainConfig, _mean, train

    rows = []
    for seed in seeds:
        for v in variants:
            cfg = TrainConfig(model_name=model_name, seed=seed, split_mode="cross_project")
            if v.num_layers is not None:
                cfg.num_layers = v.num_layers
            res = train(drop_edge_types(graphs, v.drop), cfg)
            rows.append(
                {
                    "seed": seed,
                    "variant": v.name,
                    "per_repo_pr_auc": _mean(res.gnn_per_repo),
                    "pooled_pr_auc": res.gnn_test.pr_auc,
                    "recall_at_10pct": res.gnn_test.recall_at_10pct,
                    "xgb_per_repo_pr_auc": _mean(res.xgb_per_repo),
                }
            )
            print(f"  seed {seed} | {v.name:12s} per-repo PR-AUC {rows[-1]['per_repo_pr_auc']:.3f}")

    df = pd.DataFrame(rows)
    full = df[df.variant == "full"].set_index("seed")["per_repo_pr_auc"]
    df["delta_vs_full"] = df.apply(lambda r: r.per_repo_pr_auc - full[r.seed], axis=1)

    summary = (
        df.groupby("variant")
        .agg(
            per_repo_pr_auc=("per_repo_pr_auc", "mean"),
            delta_mean=("delta_vs_full", "mean"),
            delta_std=("delta_vs_full", "std"),
            pooled_pr_auc=("pooled_pr_auc", "mean"),
            recall_at_10pct=("recall_at_10pct", "mean"),
        )
        .round(4)
    )
    summary["xgboost_per_repo_pr_auc"] = round(df["xgb_per_repo_pr_auc"].mean(), 4)
    order = [v.name for v in variants]
    return summary.loc[order]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["features", "graph"])
    ap.add_argument("--model", default="sage", choices=["sage", "gat"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    args = ap.parse_args()

    if args.mode == "features":
        from codecity_ml.models.baselines import load_tabular_dataset

        runs = run_feature_group_ablation(load_tabular_dataset(args.data))
        print(ablation_summary_table(runs).to_string(index=False))
    else:
        from codecity_ml.graph.dataset import load_dataset

        graphs = load_dataset(args.data)
        table = run_graph_ablation(graphs, args.model, tuple(args.seeds))
        print(f"\n=== Graph ablation ({args.model}, seeds {args.seeds}) ===")
        print(table.to_string())


if __name__ == "__main__":
    main()