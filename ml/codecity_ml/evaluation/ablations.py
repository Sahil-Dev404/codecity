"""Ablation runner: systematically drop feature groups and compare results.

Right now (pre-GNN) this answers "which engineered features actually
matter," using XGBoost as the probe model since it's the stronger baseline
(established in baselines.py). Once models/graphsage.py exists, this file
gains a second runner that ablates graph structure itself (edge types,
layer count) — the pattern (train variant, evaluate, collect, compare) is
identical, just with a GNN instead of XGBoost as the model under test.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from xgboost import XGBClassifier

from codecity_ml.evaluation.metrics import EvalResult, evaluate
from codecity_ml.graph.features import FEATURE_NAMES
from codecity_ml.models.baselines import temporal_train_test_split

# Feature groups mirror graph/features.py's FEATURE_NAMES, bucketed by what
# kind of signal they represent. Used to ask "how much does THIS category
# contribute," not just "how much does one single column contribute."
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "complexity": (
        "lines_of_code",
        "max_cyclomatic_complexity",
        "avg_cyclomatic_complexity",
        "function_count",
    ),
    "age_recency": (
        "file_age_days",
        "days_since_last_change",
    ),
    "churn": (
        "total_commit_count",
        "recent_commit_count",
        "total_churn",
        "recent_churn",
    ),
    "authorship": (
        "distinct_author_count",
        "top_author_share",
    ),
    "cochange": (
        "cochange_neighbor_count",
    ),
    "bug_history": (
        "past_bugfix_count",
    ),
}


@dataclass(frozen=True)
class AblationRun:
    """One ablation variant's result, with enough metadata to build a
    comparison table afterward."""

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
    scores = model.predict_proba(X_test)[:, 1]

    return evaluate(variant_name, y_test, scores)


def run_feature_group_ablation(df: pd.DataFrame) -> list[AblationRun]:
    """Train XGBoost once with ALL features (the reference point), then once
    per feature group with that group's columns removed. A group whose
    removal causes a big PR-AUC drop is an important signal; a group whose
    removal barely matters might not be worth the engineering cost in a
    production version of this pipeline.

    Uses one fixed temporal split across all variants, so differences in
    score are attributable to the features, not to a different train/test
    boundary per run.
    """
    train, test = temporal_train_test_split(df)
    runs: list[AblationRun] = []

    all_features = tuple(FEATURE_NAMES)
    baseline = _train_and_score(train, test, all_features, "all_features")
    runs.append(
        AblationRun(
            variant_name="all_features",
            dropped_group=None,
            features_used=all_features,
            result=baseline,
        )
    )

    for group_name, group_cols in FEATURE_GROUPS.items():
        remaining = tuple(f for f in FEATURE_NAMES if f not in group_cols)
        variant_name = f"without_{group_name}"
        result = _train_and_score(train, test, remaining, variant_name)
        runs.append(
            AblationRun(
                variant_name=variant_name,
                dropped_group=group_name,
                features_used=remaining,
                result=result,
            )
        )

    return runs


def ablation_summary_table(runs: list[AblationRun]) -> pd.DataFrame:
    """Turn a list of AblationRuns into a sorted, readable comparison table:
    one row per variant, PR-AUC delta from the full-feature baseline, so
    the biggest contributors are obvious at a glance."""
    baseline_prauc = next(r.result.pr_auc for r in runs if r.dropped_group is None)

    rows = [
        {
            "variant": r.variant_name,
            "dropped_group": r.dropped_group or "-",
            "pr_auc": round(r.result.pr_auc, 4),
            "pr_auc_delta": round(r.result.pr_auc - baseline_prauc, 4),
            "recall_at_10pct": round(r.result.recall_at_10pct, 4),
            "effort_at_recall_80pct": round(r.result.effort_at_recall_80pct, 4),
        }
        for r in runs
    ]

    df = pd.DataFrame(rows)
    # Most negative delta first = the group whose removal hurt the most =
    # the most important group, at the top.
    return df.sort_values("pr_auc_delta").reset_index(drop=True)