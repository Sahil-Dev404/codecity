"""Shared evaluation metrics for defect prediction.

Every model — baselines today, GNNs later — gets scored through this one
module, so results are directly comparable across models and across
ablations. Accuracy is deliberately absent: with heavy class imbalance
(confirmed when testing labels.py), a model predicting "never buggy" would
score highly on accuracy while being useless. PR-AUC and effort-aware
metrics are what the project plan called for instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


@dataclass(frozen=True)
class EvalResult:
    """One model's scores on one evaluation set, with enough context to
    compare runs later (ablations.py will collect many of these)."""

    model_name: str
    n_examples: int
    n_positive: int
    pr_auc: float
    recall_at_10pct: float
    recall_at_20pct: float
    precision_at_10pct: float
    effort_at_recall_80pct: float

    def summary(self) -> str:
        return (
            f"{self.model_name}: PR-AUC={self.pr_auc:.3f} "
            f"Recall@10%={self.recall_at_10pct:.3f} "
            f"Recall@20%={self.recall_at_20pct:.3f} "
            f"Effort@80%Recall={self.effort_at_recall_80pct:.3f}"
        )


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the precision-recall curve. Preferred over ROC-AUC here
    because ROC-AUC is optimistic under heavy class imbalance — PR-AUC
    focuses on how well the model ranks the rare positive class, which is
    exactly what matters for "which files should I inspect first."""
    return float(average_precision_score(y_true, y_score))


def recall_at_top_k_percent(y_true: np.ndarray, y_score: np.ndarray, k_percent: float) -> float:
    """If a developer only had time to inspect the top k% riskiest files
    (by predicted score), what fraction of ALL actually-buggy files would
    they catch? This is the practical, 'would this tool actually help a
    team triage their backlog' question — more actionable than a raw
    PR-AUC number on its own."""
    n = len(y_score)
    k = max(1, int(np.ceil(n * k_percent / 100)))

    order = np.argsort(-y_score)  # descending by predicted risk
    top_k_true = y_true[order[:k]]

    total_positive = y_true.sum()
    if total_positive == 0:
        return 0.0
    return float(top_k_true.sum() / total_positive)


def precision_at_top_k_percent(y_true: np.ndarray, y_score: np.ndarray, k_percent: float) -> float:
    """Of the top k% flagged files, what fraction are actually buggy? This
    is the 'how much of the inspector's time is wasted on false alarms'
    question — recall and precision at the same cutoff tell a fuller story
    together than either alone."""
    n = len(y_score)
    k = max(1, int(np.ceil(n * k_percent / 100)))

    order = np.argsort(-y_score)
    top_k_true = y_true[order[:k]]
    return float(top_k_true.sum() / k)


def effort_at_recall(y_true: np.ndarray, y_score: np.ndarray, target_recall: float) -> float:
    """What fraction of ALL files would a developer need to inspect (in
    risk order) to catch `target_recall` of the actually-buggy ones?
    Lower is better — it's the direct 'effort saved' story for a resume
    bullet: 'catches 80% of bugs by inspecting only effort_at_recall_80pct
    of the codebase', compared to 100% effort with no model at all.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns recall in descending-threshold order,
    # i.e. recall values generally increase as we move through the array.
    idx = np.searchsorted(recall[::-1], target_recall)
    if idx >= len(recall):
        return 1.0  # never reaches target_recall even inspecting everything

    # Convert back to "how many files, in risk order, does this threshold
    # correspond to" by re-deriving it directly rather than trusting the
    # curve's implicit ordering, which is safer and easier to verify.
    n = len(y_score)
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    cumulative_catch = np.cumsum(y_sorted)
    total_positive = y_true.sum()
    if total_positive == 0:
        return 1.0

    target_catches = np.ceil(target_recall * total_positive)
    reached = np.searchsorted(cumulative_catch, target_catches) + 1
    return float(min(reached, n) / n)


def evaluate(model_name: str, y_true: np.ndarray, y_score: np.ndarray) -> EvalResult:
    """Run the full metric suite and package it into one EvalResult."""
    return EvalResult(
        model_name=model_name,
        n_examples=len(y_true),
        n_positive=int(y_true.sum()),
        pr_auc=pr_auc(y_true, y_score),
        recall_at_10pct=recall_at_top_k_percent(y_true, y_score, 10),
        recall_at_20pct=recall_at_top_k_percent(y_true, y_score, 20),
        precision_at_10pct=precision_at_top_k_percent(y_true, y_score, 10),
        effort_at_recall_80pct=effort_at_recall(y_true, y_score, 0.80),
    )