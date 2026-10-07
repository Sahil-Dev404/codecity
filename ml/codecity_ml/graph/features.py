"""Turn mined/parsed data into fixed-length numeric feature vectors per file.

Mirrors labels.py's snapshot design deliberately: every feature here is
computed using only commits up to and including `snapshot_at`, never later
ones, or the model would be trained on information it couldn't have known
at prediction time (leakage).

Known simplification: FileMetrics (size/complexity) are computed from the
CURRENT checkout, not reconstructed at each historical snapshot (that would
mean checking out every snapshot's revision and re-running lizard, which is
expensive). This means older snapshots use slightly future-dated complexity
numbers. Documented here and in the model card as a limitation.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime, timedelta

from codecity_ml.graph.edges import CoChangeEdge
from codecity_ml.mining.history import CommitRecord
from codecity_ml.parsing.metrics import FileMetrics

RECENT_CHURN_WINDOW_DAYS = 90

# Fixed order — this is the actual column order fed to the model.
# graph/dataset.py and any inference code must agree on this order.
FEATURE_NAMES: tuple[str, ...] = (
    "lines_of_code",
    "max_cyclomatic_complexity",
    "avg_cyclomatic_complexity",
    "function_count",
    "file_age_days",
    "days_since_last_change",
    "total_commit_count",
    "recent_commit_count",
    "total_churn",
    "recent_churn",
    "distinct_author_count",
    "top_author_share",
    "cochange_neighbor_count",
    "past_bugfix_count",
)


@dataclass(frozen=True)
class FileFeatures:
    """One file's feature vector, at one snapshot in time."""

    file_path: str
    snapshot_at: datetime
    lines_of_code: float
    max_cyclomatic_complexity: float
    avg_cyclomatic_complexity: float
    function_count: float
    file_age_days: float
    days_since_last_change: float
    total_commit_count: float
    recent_commit_count: float
    total_churn: float
    recent_churn: float
    distinct_author_count: float
    top_author_share: float
    cochange_neighbor_count: float
    past_bugfix_count: float

    def to_vector(self) -> list[float]:
        """Return values in FEATURE_NAMES order, ready for a tensor."""
        return [getattr(self, name) for name in FEATURE_NAMES]


@dataclass
class _Accum:
    """Mutable running totals for one file, built in a single commit pass."""

    first_seen: datetime | None = None
    last_changed: datetime | None = None
    total_commit_count: int = 0
    recent_commit_count: int = 0
    total_churn: int = 0
    recent_churn: int = 0
    authors: dict[str, int] | None = None
    past_bugfix_count: int = 0


def _accumulate_commit_stats(
    records: list[CommitRecord], snapshot_at: datetime
) -> dict[str, _Accum]:
    """Single pass over commits up to snapshot_at, building per-file running
    totals. Reused for every file, same 'compute once' principle as
    labels.py's _future_bugfix_dates_by_file."""
    recent_cutoff = snapshot_at - timedelta(days=RECENT_CHURN_WINDOW_DAYS)
    stats: dict[str, _Accum] = {}

    for commit in records:
        if commit.authored_at > snapshot_at:
            break  # records are chronological; nothing after this matters
        if commit.is_merge:
            continue

        for mf in commit.modified_files:
            acc = stats.setdefault(mf.path, _Accum(authors={}))
            if acc.first_seen is None:
                acc.first_seen = commit.authored_at
            acc.last_changed = commit.authored_at

            churn = mf.lines_added + mf.lines_deleted
            acc.total_commit_count += 1
            acc.total_churn += churn
            acc.authors[commit.author_email] = acc.authors.get(commit.author_email, 0) + 1

            if commit.authored_at >= recent_cutoff:
                acc.recent_commit_count += 1
                acc.recent_churn += churn

            if commit.is_bugfix:
                acc.past_bugfix_count += 1

    return stats


def build_file_features(
    file_paths: set[str],
    records: list[CommitRecord],
    snapshot_at: datetime,
    metrics_by_path: dict[str, FileMetrics],
    cochange_edges: list[CoChangeEdge],
) -> dict[str, FileFeatures]:
    """Build a FileFeatures row for every path in `file_paths`, as of
    `snapshot_at`. Files with no metrics (e.g. non-source files that still
    appear in commit history) are skipped."""
    commit_stats = _accumulate_commit_stats(records, snapshot_at)

    cochange_count: dict[str, int] = {}
    for edge in cochange_edges:
        cochange_count[edge.file_a] = cochange_count.get(edge.file_a, 0) + 1
        cochange_count[edge.file_b] = cochange_count.get(edge.file_b, 0) + 1

    results: dict[str, FileFeatures] = {}

    for path in file_paths:
        metrics = metrics_by_path.get(path)
        if metrics is None:
            continue  # not a parseable source file (e.g. a .md or .json)

        acc = commit_stats.get(path)

        if acc is None or acc.first_seen is None:
            file_age_days = 0.0
            days_since_last_change = 0.0
            author_count = 0
            top_share = 0.0
        else:
            file_age_days = (snapshot_at - acc.first_seen).days
            days_since_last_change = (snapshot_at - acc.last_changed).days
            author_count = len(acc.authors)
            top_commits = max(acc.authors.values()) if acc.authors else 0
            top_share = top_commits / acc.total_commit_count if acc.total_commit_count else 0.0

        results[path] = FileFeatures(
            file_path=path,
            snapshot_at=snapshot_at,
            lines_of_code=float(metrics.lines_of_code),
            max_cyclomatic_complexity=float(metrics.max_cyclomatic_complexity),
            avg_cyclomatic_complexity=metrics.avg_cyclomatic_complexity,
            function_count=float(metrics.function_count),
            file_age_days=float(file_age_days),
            days_since_last_change=float(days_since_last_change),
            total_commit_count=float(acc.total_commit_count if acc else 0),
            recent_commit_count=float(acc.recent_commit_count if acc else 0),
            total_churn=float(acc.total_churn if acc else 0),
            recent_churn=float(acc.recent_churn if acc else 0),
            distinct_author_count=float(author_count),
            top_author_share=top_share,
            cochange_neighbor_count=float(cochange_count.get(path, 0)),
            past_bugfix_count=float(acc.past_bugfix_count if acc else 0),
        )

    return results