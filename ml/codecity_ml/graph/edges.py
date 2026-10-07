"""Non-import graph edges: co-change (files that change together) and
authorship (file <-> author).

Import edges come from parsing/imports.py and are structural (what the code
references). These edges come from mining/history.py and are behavioral
(how developers actually work with the code) — in defect-prediction
literature, co-change is often the single strongest signal, stronger than
static structure, because it captures hidden/logical coupling that imports
miss entirely.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

from codecity_ml.mining.history import CommitRecord


@dataclass(frozen=True)
class CoChangeEdge:
    """Two files that were modified together in `commit_count` commits.
    Undirected: (file_a, file_b) is the same relationship as (file_b, file_a).
    """

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


# Commits touching more files than this are excluded from co-change —
# a mass rename or formatting sweep across 300 files isn't meaningful
# coupling, it's noise that would create a near-complete graph.
_MAX_FILES_PER_COMMIT_FOR_COCHANGE = 20


def build_cochange_edges(
    records: list[CommitRecord], *, min_shared_commits: int = 2
) -> list[CoChangeEdge]:
    """Count how often each pair of files is modified in the same commit.

    Merge commits are excluded (they reflect integration, not a developer's
    actual intent to change those files together). Commits touching very
    many files are also excluded, for the same noise reason.

    min_shared_commits filters out one-off coincidental pairings, keeping
    only pairs that co-change repeatedly — a real sign of coupling.
    """
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)

    for commit in records:
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


def build_authorship_edges(records: list[CommitRecord]) -> list[AuthorshipEdge]:
    """Aggregate each author's commit count and total lines changed per file,
    across all non-merge commits."""
    stats: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])  # [commits, lines]

    for commit in records:
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