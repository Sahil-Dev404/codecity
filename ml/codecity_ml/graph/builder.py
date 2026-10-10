"""Assemble everything mined and parsed into one RepoGraph per snapshot.

Convergence point: mining/ (history, labels), parsing/ (metrics, imports)
and graph/edges.py (co-change, authorship) all feed in here. The output is
a plain, serializable snapshot with no torch dependency; graph/dataset.py
turns it into PyG tensors.

Leakage status per component (be precise about this in the model card):
  * node features ........ snapshot-aware (commits up to snapshot_at)
  * co-change edges ...... snapshot-aware (as_of=snapshot_at)
  * authorship edges ..... snapshot-aware (as_of=snapshot_at)
  * labels ............... look FORWARD by design (that is the target)
  * import edges ......... from the CURRENT checkout (not historical)
  * size/complexity ...... from the CURRENT checkout (not historical)
The last two are documented approximations: reconstructing them per snapshot
would require checking out and re-parsing every historical revision.
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
from codecity_ml.mining.labels import (
    FileLabel,
    files_as_of,
    generate_snapshot_dates,
    label_files_at_snapshot,
)
from codecity_ml.parsing.imports import ImportEdge, extract_imports, resolve_imports
from codecity_ml.parsing.metrics import FileMetrics, analyze_repo


@dataclass(frozen=True)
class RepoGraph:
    """One complete snapshot of a repo's graph. `nodes` is the canonical
    node ordering; every edge and feature refers to files by these paths."""

    repo_slug: str
    snapshot_at: datetime
    nodes: tuple[str, ...]
    features: dict[str, FileFeatures]
    labels: dict[str, FileLabel]
    import_edges: tuple[ImportEdge, ...]
    cochange_edges: tuple[CoChangeEdge, ...]
    authorship_edges: tuple[AuthorshipEdge, ...]

    def to_json_dict(self) -> dict:
        return {
            "repo_slug": self.repo_slug,
            "snapshot_at": self.snapshot_at.isoformat(),
            "feature_names": list(FEATURE_NAMES),
            "nodes": list(self.nodes),
            "features": {path: asdict(f) for path, f in self.features.items()},
            "labels": {path: asdict(lab) for path, lab in self.labels.items()},
            "import_edges": [asdict(e) for e in self.import_edges],
            "cochange_edges": [asdict(e) for e in self.cochange_edges],
            "authorship_edges": [asdict(e) for e in self.authorship_edges],
        }


def _json_default(obj):
    """asdict() leaves datetimes inside nested dicts; json can't encode them."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def _extract_all_imports(
    repo_root: Path, source_paths: list[Path], known_files: set[str]
) -> list[ImportEdge]:
    raw = []
    for path in source_paths:
        raw.extend(extract_imports(path, repo_root))
    return resolve_imports(raw, known_files)


def build_repo_graph_at_snapshot(
    repo_slug: str,
    records: list[CommitRecord],
    metrics_by_path: dict[str, FileMetrics],
    import_edges: list[ImportEdge],
    snapshot_at: datetime,
    *,
    window_days: int,
) -> RepoGraph:
    """Build one RepoGraph for a single snapshot date.

    Import edges and metrics are repo-wide inputs (current checkout).
    Co-change and authorship edges are recomputed here with
    as_of=snapshot_at, so nothing after the snapshot date leaks in.
    """
    cochange_edges = build_cochange_edges(records, as_of=snapshot_at)
    authorship_edges = build_authorship_edges(records, as_of=snapshot_at)

    files_at_snap = files_as_of(records, snapshot_at)

    features = build_file_features(
        files_at_snap, records, snapshot_at, metrics_by_path, cochange_edges
    )
    # Only files with a feature vector become nodes.
    nodes = tuple(sorted(features.keys()))
    node_set = set(nodes)

    labels_list = label_files_at_snapshot(records, snapshot_at, window_days=window_days)
    labels = {lab.file_path: lab for lab in labels_list if lab.file_path in node_set}

    return RepoGraph(
        repo_slug=repo_slug,
        snapshot_at=snapshot_at,
        nodes=nodes,
        features=features,
        labels=labels,
        import_edges=tuple(
            e for e in import_edges if e.source_file in node_set and e.target_file in node_set
        ),
        cochange_edges=tuple(
            e for e in cochange_edges if e.file_a in node_set and e.file_b in node_set
        ),
        authorship_edges=tuple(e for e in authorship_edges if e.file_path in node_set),
    )


def build_all_snapshots(
    repo_slug: str,
    repo_root: Path,
    *,
    window_days: int,
    stride_days: int,
) -> list[RepoGraph]:
    """Full pipeline for one repo: mine history, parse and extract imports
    once, then build one RepoGraph per snapshot date."""
    records = mine_history(repo_root)
    metrics_by_path = {m.path: m for m in analyze_repo(repo_root)}

    source_paths = [repo_root / Path(p) for p in metrics_by_path]
    import_edges = _extract_all_imports(repo_root, source_paths, set(metrics_by_path))

    snapshots = generate_snapshot_dates(records, window_days=window_days, stride_days=stride_days)

    return [
        build_repo_graph_at_snapshot(
            repo_slug,
            records,
            metrics_by_path,
            import_edges,
            snap,
            window_days=window_days,
        )
        for snap in snapshots
    ]


def save_repo_graph(graph: RepoGraph, out_dir: Path) -> Path:
    """Cache one RepoGraph as JSON, keyed by repo + snapshot date."""
    out_dir.mkdir(parents=True, exist_ok=True)
    date_str = graph.snapshot_at.strftime("%Y-%m-%d")
    out_path = out_dir / f"{graph.repo_slug}__{date_str}.json"
    out_path.write_text(json.dumps(graph.to_json_dict(), indent=2, default=_json_default))
    return out_path