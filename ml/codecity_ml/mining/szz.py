"""Simplified SZZ algorithm: traces each bug-fix commit back to the commit(s)
that introduced the bug, by blaming the lines the fix deleted or changed.

This turns history.py's message-based `is_bugfix` heuristic into a precise,
per-file label: a file is "bug-introducing at commit X" if a later bug-fix
commit's changed lines blame back to X. labels.py consumes this output to
build the actual training targets.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from codecity_ml.mining.history import CommitRecord, bugfix_commits


@dataclass(frozen=True)
class BugIntroducingCommit:
    """One (bug-fix commit, file) pair traced back to the commit that
    introduced the bug in that file."""

    bugfix_sha: str
    file_path: str
    inducing_sha: str
    inducing_authored_at: datetime


def _deleted_line_numbers(local_path: Path, bugfix_sha: str, file_path: str) -> list[int]:
    """Return the line numbers, as they existed in the PARENT version of the
    file, that `bugfix_sha` deleted or changed for `file_path`.

    Parses `git diff --unified=0` output directly rather than using
    PyDriller's parsed diff, since we need exact old-file line numbers to
    feed into `git blame`.
    """
    result = subprocess.run(
        ["git", "diff", f"{bugfix_sha}^", bugfix_sha, "--unified=0", "--", file_path],
        cwd=local_path,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []

    deleted_lines: list[int] = []
    old_line = 0
    for line in result.stdout.splitlines():
        if line.startswith("@@"):
            # Hunk header: "@@ -old_start,old_count +new_start,new_count @@"
            old_part = line.split("@@")[1].strip().split(" ")[0]
            old_line = int(old_part.lstrip("-").split(",")[0])
        elif line.startswith("-") and not line.startswith("---"):
            deleted_lines.append(old_line)
            old_line += 1
        elif line.startswith("+") and not line.startswith("+++"):
            continue  # added lines don't exist in the old file
        elif not line.startswith(("+++", "---")):
            old_line += 1

    return deleted_lines


def _blame_lines(
    local_path: Path, parent_sha: str, file_path: str, line_numbers: list[int]
) -> set[str]:
    """git blame the given lines at `parent_sha`, returning the distinct
    commit SHAs that last touched each one — the bug-introducing candidates."""
    shas: set[str] = set()
    for ln in line_numbers:
        result = subprocess.run(
            ["git", "blame", "-l", f"-L{ln},{ln}", parent_sha, "--", file_path],
            cwd=local_path,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 or not result.stdout:
            continue
        sha = result.stdout.split(" ", 1)[0].lstrip("^")
        shas.add(sha)

    return shas


def trace_bug_introducing_commits(
    local_path: Path, records: list[CommitRecord]
) -> list[BugIntroducingCommit]:
    """Run simplified SZZ over every bug-fix commit in `records`.

    For each bug-fix commit and each file it touched: find the lines it
    deleted/changed, blame the parent revision for those lines, and record
    the resulting commit(s) as bug-introducing.

    This is slow — one `git blame` call per changed line — so it's meant to
    run once during offline graph building (and get cached), never on a
    live request.
    """
    by_sha = {r.sha: r for r in records}
    results: list[BugIntroducingCommit] = []

    for fix in bugfix_commits(records):
        for mf in fix.modified_files:
            deleted_lines = _deleted_line_numbers(local_path, fix.sha, mf.path)
            inducing_shas = _blame_lines(local_path, f"{fix.sha}^", mf.path, deleted_lines)

            for inducing_sha in inducing_shas:
                if inducing_sha == fix.sha:
                    continue  # blame landed back on the fix itself; skip

                inducing_record = by_sha.get(inducing_sha)
                authored_at = (
                    inducing_record.authored_at if inducing_record else fix.authored_at
                )
                results.append(
                    BugIntroducingCommit(
                        bugfix_sha=fix.sha,
                        file_path=mf.path,
                        inducing_sha=inducing_sha,
                        inducing_authored_at=authored_at,
                    )
                )

    return results