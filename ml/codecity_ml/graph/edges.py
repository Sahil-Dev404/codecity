"""Non-import graph edges: co-change (files that change together) and
authorship (file <-> author).

Both builders take `as_of`: only commits authored at or before that moment
are used. This is what makes a snapshot's edges leak-free; without it, an
old snapshot would contain edges derived from its own future.

Import edges come from parsing/imports.py and are structural (what the code
references). These edges come from mining/history.py and are behavioral
(how developers actually work with the code). In defect-prediction
literature, co-change is often the strongest single signal, because it
captures hidden/logical coupling that imports miss.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations

from codecity_ml.mining.history import CommitRecord


@dataclass(frozen=True)
class CoChangeEdge:
    """Two files modified together in `commit_count` commits (undirected)."""

    file_a: str
    file_b: str
    commit_count: int


@dataclass(frozen=True)
class AuthorshipEdge:
    """One author's contribution to one file."""

    file_path: str
    author_email: str
    commit_count: int
    lines_changed: int  # added + deleted, summed across their commits to this file


# Commits touching more files than this are excluded from co-change: a mass
# rename or formatting sweep is noise that would create a near-complete graph.
_MAX_FILES_PER_COMMIT_FOR_COCHANGE = 20


def build_cochange_edges(
    records: list[CommitRecord],
    *,
    min_shared_commits: int = 2,
    as_of: datetime | None = None,
) -> list[CoChangeEdge]:
    """Count how often each pair of files is modified in the same commit,
    using only commits authored at or before `as_of` (None = all history).

    `records` must be chronological (as history.py produces them), so we can
    stop as soon as we pass `as_of`. Merge commits and mega-commits are
    excluded. min_shared_commits drops one-off coincidental pairings.
    """
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)

    for commit in records:
        if as_of is not None and commit.authored_at > as_of:
            break
        if commit.is_merge:
            continue

        files = sorted({mf.path for mf in commit.modified_files})
        if len(files) < 2 or len(files) > _MAX_FILES_PER_COMMIT_FOR_COCHANGE:
            continue

        for file_a, file_b in combinations(files, 2):
            pair_counts[(file_a, file_b)] += 1

    return [
        CoChangeEdge(file_a=a, file_b=b, commit_count=count)
        for (a, b), count in pair_counts.items()
        if count >= min_shared_commits
    ]


def build_authorship_edges(
    records: list[CommitRecord], *, as_of: datetime | None = None
) -> list[AuthorshipEdge]:
    """Aggregate each author's commit count and lines changed per file across
    non-merge commits authored at or before `as_of` (None = all history)."""
    stats: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])  # [commits, lines]

    for commit in records:
        if as_of is not None and commit.authored_at > as_of:
            break
        if commit.is_merge:
            continue
        for mf in commit.modified_files:
            key = (mf.path, commit.author_email)
            stats[key][0] += 1
            stats[key][1] += mf.lines_added + mf.lines_deleted

    return [
        AuthorshipEdge(
            file_path=path,
            author_email=email,
            commit_count=counts[0],
            lines_changed=counts[1],
        )
        for (path, email), counts in stats.items()
    ]