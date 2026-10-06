"""Walk a cloned repo's commit history and extract structured commit records.

This is the one place that iterates git history with PyDriller. Downstream
modules (labels.py for SZZ, graph/edges.py for co-change) consume the
CommitRecord list this produces instead of re-walking history themselves.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pydriller import Repository

from codecity_ml.config import settings

# Heuristic for "this commit fixes a bug" — used for labeling until SZZ
# (labels.py) refines it to the specific commit that introduced the bug.
_BUGFIX_MESSAGE_RE = re.compile(
    r"\b(fix|fixes|fixed|bug|bugfix|patch|hotfix|resolve[sd]?)\b",
    re.IGNORECASE,
)

# Skip files that aren't source code, so churn/co-change stats aren't
# polluted by lockfiles, build output or vendored assets.
_IGNORED_SUFFIXES = {
    ".lock", ".min.js", ".map", ".svg", ".png", ".jpg", ".jpeg", ".gif",
    ".ico", ".woff", ".woff2", ".ttf", ".pyc", ".so", ".dll",
}
_IGNORED_DIR_PARTS = {"node_modules", "dist", "build", ".git", "__pycache__", "vendor"}


@dataclass(frozen=True)
class ModifiedFile:
    """One file touched by one commit."""

    path: str  # repo-relative path, forward slashes, as of this commit
    old_path: str | None  # previous path, if this commit renamed the file
    lines_added: int
    lines_deleted: int


@dataclass(frozen=True)
class CommitRecord:
    """One commit, with the data later stages need."""

    sha: str
    author_email: str
    author_name: str
    authored_at: datetime
    message: str
    is_merge: bool
    is_bugfix: bool
    modified_files: tuple[ModifiedFile, ...] = field(default_factory=tuple)


def _is_relevant_path(path: str) -> bool:
    p = Path(path)
    if p.suffix.lower() in _IGNORED_SUFFIXES:
        return False
    if _IGNORED_DIR_PARTS & set(p.parts):
        return False
    return True


def _looks_like_bugfix(message: str) -> bool:
    return bool(_BUGFIX_MESSAGE_RE.search(message.splitlines()[0] if message else ""))


def mine_history(local_path: Path) -> list[CommitRecord]:
    """Walk `local_path`'s git history and return CommitRecords, newest last.

    Respects settings.history_years (how far back to look) and
    settings.max_commits (a hard cap so a huge repo can't stall the
    pipeline). Merge commits are included but flagged via is_merge, so
    downstream code can decide whether to count them toward co-change
    (usually: no, since merges don't reflect a developer's intent).
    """
    since = datetime.now(timezone.utc) - timedelta(days=365 * settings.history_years)

    records: list[CommitRecord] = []
    for commit in Repository(str(local_path), since=since).traverse_commits():
        if len(records) >= settings.max_commits:
            break

        modified = tuple(
            ModifiedFile(
                path=mf.new_path or mf.old_path,
                old_path=mf.old_path if mf.old_path != mf.new_path else None,
                lines_added=mf.added_lines,
                lines_deleted=mf.deleted_lines,
            )
            for mf in commit.modified_files
            if (mf.new_path or mf.old_path) and _is_relevant_path(mf.new_path or mf.old_path)
        )

        if not modified:
            continue  # nothing relevant changed (e.g. only a lockfile)

        records.append(
            CommitRecord(
                sha=commit.hash,
                author_email=(commit.author.email or "").lower(),
                author_name=commit.author.name or "unknown",
                authored_at=commit.author_date,
                message=commit.msg,
                is_merge=commit.merge,
                is_bugfix=_looks_like_bugfix(commit.msg),
                modified_files=modified,
            )
        )

    return records


def bugfix_commits(records: list[CommitRecord]) -> list[CommitRecord]:
    """Convenience filter: just the commits that look like bug fixes."""
    return [r for r in records if r.is_bugfix and not r.is_merge]