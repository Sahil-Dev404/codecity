"""Build ONE graph for a repo "as of now", for live scoring.

Training builds many historical snapshots with forward-looking labels.
Inference needs a single snapshot at the repo's most recent commit. There is
no future to label, so labels are all negative placeholders and are never
used. Everything else goes through the same code path as training, so
features and edges mean exactly what the model saw during training.
"""

from __future__ import annotations

from pathlib import Path

from codecity_ml.graph.builder import (
    RepoGraph,
    _extract_all_imports,
    build_repo_graph_at_snapshot,
)
from codecity_ml.mining.history import mine_history
from codecity_ml.parsing.metrics import analyze_repo


def build_inference_graph(repo_slug: str, repo_root: Path, *, window_days: int = 180) -> RepoGraph:
    records = mine_history(repo_root)
    if not records:
        raise ValueError("No usable commits found in the configured history window.")

    metrics_by_path = {m.path: m for m in analyze_repo(repo_root)}
    if not metrics_by_path:
        raise ValueError("No supported source files (.py/.js/.jsx/.ts/.tsx) found.")

    source_paths = [repo_root / Path(p) for p in metrics_by_path]
    import_edges = _extract_all_imports(repo_root, source_paths, set(metrics_by_path))

    snapshot_at = max(r.authored_at for r in records)
    graph = build_repo_graph_at_snapshot(
        repo_slug, records, metrics_by_path, import_edges, snapshot_at, window_days=window_days
    )
    if not graph.nodes:
        raise ValueError("No files with both commit history and parseable source.")
    return graph