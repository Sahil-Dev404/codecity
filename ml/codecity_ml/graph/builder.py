"""Assemble everything mined and parsed into one RepoGraph per snapshot.

This is the convergence point: mining/ (history, SZZ, labels), parsing/
(metrics, imports) and graph/edges.py (co-change, authorship) all feed in
here. The output is a plain, serializable snapshot — graph/dataset.py turns
it into PyG tensors; this file does not depend on torch at all, so it stays
usable even without the `gnn` extra installed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from codecity_ml.graph.edges import (
    AuthorshipEdge,
    CoChangeEdge,
    build_authorship_edges,
    build_cochange_edges,
)
from codecity_ml.graph.features import FEATURE_NAMES, FileFeatures, build_file_features
from codecity_ml.mining.history import CommitRecord, mine_history
from codecity_ml.mining.labels import FileLabel, label_files_at_snapshot
from codecity_ml.parsing.imports import ImportEdge, extract_imports, resolve_imports
from codecity_ml.parsing.metrics import FileMetrics, analyze_repo


@dataclass(frozen=True)
class RepoGraph:
    """One complete, self-contained snapshot of a repo's graph.

    `nodes` is the ordered list of file paths — every edge and feature
    below refers to files by this same path string, so this list also
    defines the canonical node ordering that dataset.py will turn into
    tensor indices.
    """

    repo_slug: str
    snapshot_at: datetime
    nodes: tuple[str, ...]
    features: dict[str, FileFeatures]
    labels: dict[str, FileLabel]
    import_edges: tuple[ImportEdge, ...]
    cochange_edges: tuple[CoChangeEdge, ...]
    authorship_edges: tuple[AuthorshipEdge, ...]

    def to_json_dict(self) -> dict:
        """Plain-dict form for caching to disk (see save_repo_graph)."""
        return {
            "repo_slug": self.repo_slug,
            "snapshot_at": self.snapshot_at.isoformat(),
            "feature_names": list(FEATURE_NAMES),
            "nodes": list(self.nodes),
            "features": {path: asdict(f) for path, f in self.features.items()},
            "labels": {path: asdict(l) for path, l in self.labels.items()},
            "import_edges": [asdict(e) for e in self.import_edges],
            "cochange_edges": [asdict(e) for e in self.cochange_edges],
            "authorship_edges": [asdict(e) for e in self.authorship_edges],
        }


def _extract_all_imports(
    repo_root: Path, source_paths: list[Path], known_files: set[str]
) -> list[ImportEdge]:
    raw = []
    for path in source_paths:
        raw.extend(extract_imports(path, repo_root))
    return resolve_imports(raw, known_files)


def build_repo_graph_at_snapshot(
    repo_slug: str,
    repo_root: Path,
    records: list[CommitRecord],
    metrics_by_path: dict[str, FileMetrics],
    import_edges: list[ImportEdge],
    cochange_edges: list[CoChangeEdge],
    authorship_edges: list[AuthorshipEdge],
    snapshot_at: datetime,
    *,
    window_days: int,
) -> RepoGraph:
    """Build one RepoGraph for a single snapshot date.

    Takes the already-computed repo-wide artifacts (metrics, import edges,
    co-change, authorship — none of which are snapshot-dependent) and
    combines them with the snapshot-dependent pieces (features, labels,
    which files existed yet) computed fresh for this date.
    """
    from codecity_ml.mining.labels import files_as_of

    files_at_snap = files_as_of(records, snapshot_at)

    features = build_file_features(
        files_at_snap, records, snapshot_at, metrics_by_path, cochange_edges
    )
    # Only files that got a feature vector (i.e. had parseable metrics)
    # become graph nodes — a node with no features can't be trained on.
    nodes = tuple(sorted(features.keys()))
    node_set = set(nodes)

    labels_list = label_files_at_snapshot(records, snapshot_at, window_days=window_days)
    labels = {label.file_path: label for label in labels_list if label.file_path in node_set}

    filtered_imports = tuple(
        e for e in import_edges if e.source_file in node_set and e.target_file in node_set
    )
    filtered_cochange = tuple(
        e for e in cochange_edges if e.file_a in node_set and e.file_b in node_set
    )
    filtered_authorship = tuple(e for e in authorship_edges if e.file_path in node_set)

    return RepoGraph(
        repo_slug=repo_slug,
        snapshot_at=snapshot_at,
        nodes=nodes,
        features=features,
        labels=labels,
        import_edges=filtered_imports,
        cochange_edges=filtered_cochange,
        authorship_edges=filtered_authorship,
    )


def build_all_snapshots(
    repo_slug: str,
    repo_root: Path,
    *,
    window_days: int,
    stride_days: int,
) -> list[RepoGraph]:
    """Full pipeline for one repo: mine history once, parse once, derive
    edges once, then build one RepoGraph per snapshot date, reusing all of
    the repo-wide (non-snapshot-dependent) work.

    This ordering — expensive repo-wide steps once, cheap snapshot-specific
    steps repeated — is what makes labeling dozens of snapshots per repo
    tractable instead of redoing mining/parsing/SZZ for each one.
    """
    from codecity_ml.mining.labels import generate_snapshot_dates

    records = mine_history(repo_root)
    metrics = analyze_repo(repo_root)
    metrics_by_path = {m.path: m for m in metrics}

    source_paths = [repo_root / Path(p) for p in metrics_by_path]
    known_files = set(metrics_by_path.keys())
    import_edges = _extract_all_imports(repo_root, source_paths, known_files)

    cochange_edges = build_cochange_edges(records)
    authorship_edges = build_authorship_edges(records)

    snapshots = generate_snapshot_dates(records, window_days=window_days, stride_days=stride_days)

    return [
        build_repo_graph_at_snapshot(
            repo_slug,
            repo_root,
            records,
            metrics_by_path,
            import_edges,
            cochange_edges,
            authorship_edges,
            snap,
            window_days=window_days,
        )
        for snap in snapshots
    ]


def save_repo_graph(graph: RepoGraph, out_dir: Path) -> Path:
    """Cache one RepoGraph to disk as JSON, keyed by repo + snapshot date.
    This is what lets graph building run once per repo and be reused across
    training runs/ablations without re-mining and re-parsing every time."""
    out_dir.mkdir(parents=True, exist_ok=True)
    date_str = graph.snapshot_at.strftime("%Y-%m-%d")
    out_path = out_dir / f"{graph.repo_slug}__{date_str}.json"
    out_path.write_text(json.dumps(graph.to_json_dict(), indent=2))
    return out_path