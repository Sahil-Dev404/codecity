"""Non-graph baselines: logistic regression and XGBoost.

These exist to answer one question honestly: does the graph structure
actually help? They're trained on the exact same per-file features the GNN
will use (graph/features.py's FEATURE_NAMES), just without any message
passing — each file is treated as an independent row.

If a GNN can't beat these, that's a real, reportable result, not something
to hide. See evaluation/ablations.py for the head-to-head comparison.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from codecity_ml.graph.features import FEATURE_NAMES


@dataclass(frozen=True)
class TabularRow:
    """One (file, snapshot) training example, flattened — no graph structure."""

    repo_slug: str
    file_path: str
    snapshot_at: str  # ISO string, kept as string for easy DataFrame sorting
    label: int


def load_tabular_dataset(cache_dir: Path) -> pd.DataFrame:
    """Load every cached RepoGraph JSON (built by graph/builder.py) under
    `cache_dir` and flatten them into one DataFrame: one row per (file,
    snapshot), columns = FEATURE_NAMES + label + repo_slug + snapshot_at.

    This throws away all edge information on purpose — that's the point of
    a baseline. graph/dataset.py (for the GNN) will read the same cached
    JSON files but keep the edges.
    """
    rows: list[dict] = []

    for path in sorted(cache_dir.glob("*.json")):
        data = json.loads(path.read_text())
        repo_slug = data["repo_slug"]
        snapshot_at = data["snapshot_at"]
        features_by_path = data["features"]
        labels_by_path = data["labels"]

        for file_path, feat_dict in features_by_path.items():
            label_entry = labels_by_path.get(file_path)
            if label_entry is None:
                continue  # shouldn't happen if builder.py's invariants hold

            row = {name: feat_dict[name] for name in FEATURE_NAMES}
            row["label"] = int(label_entry["is_buggy_future"])
            row["repo_slug"] = repo_slug
            row["snapshot_at"] = snapshot_at
            row["file_path"] = file_path
            rows.append(row)

    if not rows:
        raise ValueError(
            f"No cached graphs found under {cache_dir}. Run graph building first."
        )

    df = pd.DataFrame(rows)
    df["snapshot_at"] = pd.to_datetime(df["snapshot_at"])
    return df.sort_values("snapshot_at").reset_index(drop=True)


def temporal_train_test_split(
    df: pd.DataFrame, *, test_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by TIME, not randomly — the last `test_fraction` of snapshots
    (by date) become the test set. A random split would leak information:
    the model could see a file's near-future snapshot during training and
    "predict" a snapshot from its past, which is not a fair test of
    generalizing to the future. This matches the project's evaluation plan
    exactly (temporal split, called out from the start).
    """
    cutoff_idx = int(len(df) * (1 - test_fraction))
    cutoff_date = df.iloc[cutoff_idx]["snapshot_at"]

    train = df[df["snapshot_at"] < cutoff_date].reset_index(drop=True)
    test = df[df["snapshot_at"] >= cutoff_date].reset_index(drop=True)
    return train, test


def _xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = df[list(FEATURE_NAMES)].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int32)
    return X, y


def train_logistic_regression(train_df: pd.DataFrame) -> LogisticRegression:
    """Logistic regression with class-balanced weights — the positive
    class (buggy-future files) is rare (we saw this directly when testing
    labels.py), so an unweighted model would just learn to always predict
    'not buggy' and still score high on raw accuracy."""
    X_train, y_train = _xy(train_df)
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(train_df: pd.DataFrame) -> XGBClassifier:
    """XGBoost with a positive-class weight computed from the training
    set's actual imbalance ratio — typically the stronger of the two
    baselines in the defect-prediction literature, which is exactly why
    it's the one the GNN most needs to beat."""
    X_train, y_train = _xy(train_df)

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",  # area under precision-recall curve
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def predict_proba(model, df: pd.DataFrame) -> np.ndarray:
    """Probability of the positive class (buggy-future), for either model
    type — both expose the same sklearn-style interface."""
    X, _ = _xy(df)
    return model.predict_proba(X)[:, 1]