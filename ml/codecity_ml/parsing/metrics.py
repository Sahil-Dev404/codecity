"""Per-file size and complexity metrics via lizard.

Lizard supports Python, JavaScript and TypeScript (and many more languages)
with one consistent interface, which is why it was chosen over Python-only
tools like radon. These metrics become node features in graph/features.py
and drive the city's building heights/footprints in the frontend.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lizard

# Only analyze source files we actually care about. Mirrors history.py's
# filtering so metrics and commit data describe the same set of files.
_SOURCE_SUFFIXES = {".py", ".js", ".jsx", ".ts", ".tsx"}
_IGNORED_DIR_PARTS = {"node_modules", "dist", "build", ".git", "__pycache__", "vendor", "tests", "test"}


@dataclass(frozen=True)
class FunctionMetrics:
    """One function/method within a file."""

    name: str
    start_line: int
    end_line: int
    cyclomatic_complexity: int
    parameter_count: int
    length_lines: int


@dataclass(frozen=True)
class FileMetrics:
    """Aggregated metrics for one source file."""

    path: str  # repo-relative, forward slashes
    language: str
    lines_of_code: int  # non-comment, non-blank (lizard's nloc)
    total_lines: int  # physical line count, incl. comments/blank
    function_count: int
    max_cyclomatic_complexity: int  # the single most complex function
    avg_cyclomatic_complexity: float
    functions: tuple[FunctionMetrics, ...]


def _is_relevant(path: Path) -> bool:
    if path.suffix.lower() not in _SOURCE_SUFFIXES:
        return False
    return not (_IGNORED_DIR_PARTS & set(path.parts))


def analyze_file(path: Path, repo_root: Path) -> FileMetrics | None:
    """Analyze one file. Returns None for empty or unparseable files rather
    than raising, since a single malformed file shouldn't crash a whole-repo
    analysis run."""
    try:
        result = lizard.analyze_file(str(path))
    except Exception:
        return None

    if result is None or result.nloc == 0:
        return None

    functions = tuple(
        FunctionMetrics(
            name=fn.name,
            start_line=fn.start_line,
            end_line=fn.end_line,
            cyclomatic_complexity=fn.cyclomatic_complexity,
            parameter_count=len(fn.parameters),
            length_lines=fn.end_line - fn.start_line + 1,
        )
        for fn in result.function_list
    )

    complexities = [fn.cyclomatic_complexity for fn in functions] or [1]

    return FileMetrics(
        path=path.relative_to(repo_root).as_posix(),
        language=result.language if hasattr(result, "language") else path.suffix.lstrip("."),
        lines_of_code=result.nloc,
        total_lines=result.token_count if hasattr(result, "token_count") else result.nloc,
        function_count=len(functions),
        max_cyclomatic_complexity=max(complexities),
        avg_cyclomatic_complexity=sum(complexities) / len(complexities),
        functions=functions,
    )


def analyze_repo(repo_root: Path) -> list[FileMetrics]:
    """Walk every relevant source file under `repo_root` and return its
    FileMetrics. Files that fail to parse are silently skipped (see
    analyze_file), so a single bad file never aborts the whole repo."""
    results: list[FileMetrics] = []

    for path in repo_root.rglob("*"):
        if not path.is_file() or not _is_relevant(path):
            continue
        metrics = analyze_file(path, repo_root)
        if metrics is not None:
            results.append(metrics)

    return results