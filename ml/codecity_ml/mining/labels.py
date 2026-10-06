"""Build per-file, per-snapshot training labels from commit history.

Answers the core prediction question: given the repo as of date T, was file
F touched by a bug-fix commit within the next `window_days`?

Note this does NOT use szz.py's bug-introducing commits — SZZ answers a
different question (which earlier commit caused the bug) and is used as a
feature/explainability signal later (graph/features.py), not for this label.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta

from codecity_ml.mining.history import CommitRecord, bugfix_commits

DEFAULT_WINDOW_DAYS = 180  # "next 6 months"
DEFAULT_STRIDE_DAYS = 30  # one snapshot per month
MIN_HISTORY_DAYS = 90  # need this much history before the first snapshot


@dataclass(frozen=True)
class FileLabel:
    """One training example: was `file_path` touched by a bug-fix commit
    within `window_days` after `snapshot_at`?"""

    file_path: str
    snapshot_at: datetime
    window_days: int
    is_buggy_future: bool
    bugfix_shas: tuple[str, ...]  # evidence: which future commits caused a positive label


def _future_bugfix_dates_by_file(
    records: list[CommitRecord],
) -> dict[str, list[tuple[datetime, str]]]:
    """Map file_path -> sorted [(bugfix_commit_date, sha), ...].

    Built once from all bug-fix commits, then reused across every snapshot
    so we don't re-scan commit history per file per snapshot.
    """
    by_file: dict[str, list[tuple[datetime, str]]] = defaultdict(list)
    for commit in bugfix_commits(records):
        for mf in commit.modified_files:
            by_file[mf.path].append((commit.authored_at, commit.sha))

    for dates in by_file.values():
        dates.sort(key=lambda pair: pair[0])

    return by_file


def generate_snapshot_dates(
    records: list[CommitRecord],
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
    stride_days: int = DEFAULT_STRIDE_DAYS,
) -> list[datetime]:
    """Pick evenly spaced dates to generate training examples at.

    Skips the first MIN_HISTORY_DAYS (too little history for decent
    churn/author features) and the last `window_days` (too close to "now"
    to know the future — there's no ground truth there yet).
    """
    if not records:
        return []

    first = min(r.authored_at for r in records)
    last = max(r.authored_at for r in records)

    start = first + timedelta(days=MIN_HISTORY_DAYS)
    end = last - timedelta(days=window_days)

    snapshots = []
    current = start
    while current <= end:
        snapshots.append(current)
        current += timedelta(days=stride_days)

    return snapshots


def files_as_of(records: list[CommitRecord], snapshot_at: datetime) -> set[str]:
    """The set of files touched at least once by `snapshot_at`.

    `records` must be in ascending chronological order (as history.py
    produces them), so we can stop early once we pass the snapshot date.

    Known simplification: file deletions aren't tracked yet (history.py's
    ModifiedFile doesn't currently record change_type), so a deleted file
    stays in this set. Noted in the model card as a limitation.
    """
    existing: set[str] = set()

    for commit in records:
        if commit.authored_at > snapshot_at:
            break
        for mf in commit.modified_files:
            existing.add(mf.path)
            if mf.old_path and mf.old_path != mf.path:
                existing.discard(mf.old_path)  # treat rename as the file moving

    return existing


def label_files_at_snapshot(
    records: list[CommitRecord],
    snapshot_at: datetime,
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
    future_by_file: dict[str, list[tuple[datetime, str]]] | None = None,
) -> list[FileLabel]:
    """Produce one FileLabel per file that existed as of `snapshot_at`.

    Pass a precomputed `future_by_file` (from `_future_bugfix_dates_by_file`)
    when labeling many snapshots, to avoid redundant work.
    """
    if future_by_file is None:
        future_by_file = _future_bugfix_dates_by_file(records)

    window_end = snapshot_at + timedelta(days=window_days)
    candidate_files = files_as_of(records, snapshot_at)

    labels: list[FileLabel] = []
    for path in sorted(candidate_files):
        future_hits = [
            (date, sha)
            for date, sha in future_by_file.get(path, [])
            if snapshot_at < date <= window_end
        ]
        labels.append(
            FileLabel(
                file_path=path,
                snapshot_at=snapshot_at,
                window_days=window_days,
                is_buggy_future=bool(future_hits),
                bugfix_shas=tuple(sha for _, sha in future_hits),
            )
        )

    return labels


def label_all_snapshots(
    records: list[CommitRecord],
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
    stride_days: int = DEFAULT_STRIDE_DAYS,
) -> list[FileLabel]:
    """Generate snapshot dates and label every file at each one — the full
    training-label set for one repo."""
    future_by_file = _future_bugfix_dates_by_file(records)
    all_labels: list[FileLabel] = []

    for snapshot_at in generate_snapshot_dates(
        records, window_days=window_days, stride_days=stride_days
    ):
        all_labels.extend(
            label_files_at_snapshot(
                records, snapshot_at, window_days=window_days, future_by_file=future_by_file
            )
        )

    return all_labels