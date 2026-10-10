"""Clone -> graph -> score -> CityData. Synchronous and blocking (git,
parsing and torch are all CPU/IO bound), so it must run in a worker or a
thread, never directly in an async request handler.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

import numpy as np
import torch
from torch import nn

from codecity_ml.graph.dataset import FeatureScaler, graph_from_dict
from codecity_ml.graph.inference import build_inference_graph
from codecity_ml.mining.clone import clone_repo

_LANG_BY_SUFFIX = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
}

# Share of files in each risk bucket, riskiest first. See _risk_levels.
_BUCKETS = (("critical", 0.05), ("high", 0.15), ("medium", 0.35))


def _risk_levels(scores: np.ndarray) -> list[str]:
    """Rank-based buckets: top 5% critical, next 10% high, next 20% medium,
    the rest low.

    The model is trained with a positive-class weight, which inflates its
    outputs, so the raw scores are good for RANKING but are not calibrated
    probabilities. Fixed cut-offs like 0.75 would flag most of a repo.
    Percentile buckets keep the colors meaningful for any repo.
    """
    n = len(scores)
    order = np.argsort(-scores)
    levels = ["low"] * n
    for rank, idx in enumerate(order):
        frac = (rank + 1) / n
        for name, upper in _BUCKETS:
            if frac <= upper:
                levels[idx] = name
                break
    return levels


def analyze_url(
    url: str,
    model: nn.Module,
    scaler: FeatureScaler,
    meta: dict,
    on_stage: Callable[[str], None] | None = None,
) -> dict:
    """Run the full pipeline for one repo URL and return CityData as a dict
    (camelCase keys, matching frontend/src/types/city.ts).

    `on_stage` is called with "clone", "parse", "graph", "predict" as each
    phase begins. "parse" covers history mining and source parsing, which
    both happen inside build_inference_graph.
    """
    stage = on_stage or (lambda _name: None)

    stage("clone")
    cloned = clone_repo(url)

    stage("parse")
    repo_graph = build_inference_graph(cloned.slug, cloned.local_path)

    stage("graph")
    g = graph_from_dict(repo_graph.to_json_dict())
    if g is None:
        raise ValueError("Graph has no files.")
    scaler.apply([g])

    stage("predict")
    with torch.no_grad():
        scores = model.predict_proba(g).numpy()
    paths: list[str] = g["file"].paths
    levels = _risk_levels(scores)

    nodes = []
    for path, score, level in zip(paths, scores, levels):
        f = repo_graph.features[path]
        suffix = "." + path.rsplit(".", 1)[-1].lower() if "." in path else ""
        nodes.append(
            {
                "id": path,
                "path": path,
                "language": _LANG_BY_SUFFIX.get(suffix, "unknown"),
                "linesOfCode": int(f.lines_of_code),
                "complexity": int(f.max_cyclomatic_complexity),
                "riskScore": round(float(score), 4),
                "riskLevel": level,
                "authorCount": int(f.distinct_author_count),
                "lastChangedDaysAgo": int(f.days_since_last_change),
            }
        )

    edges = [
        {"source": e.source_file, "target": e.target_file, "kind": "import", "weight": 1}
        for e in repo_graph.import_edges
    ] + [
        {"source": e.file_a, "target": e.file_b, "kind": "cochange", "weight": e.commit_count}
        for e in repo_graph.cochange_edges
    ]

    by_dir: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        by_dir[path.rsplit("/", 1)[0] if "/" in path else ""].append(path)
    districts = [
        {"id": d, "name": d.rsplit("/", 1)[-1] if d else "(root)", "nodeIds": ids}
        for d, ids in sorted(by_dir.items())
    ]

    gnn_metrics = (meta.get("metrics") or {}).get("gnn") or {}
    return {
        "repoSlug": cloned.slug,
        "repoUrl": cloned.url,
        "snapshotAt": repo_graph.snapshot_at.isoformat(),
        "nodes": nodes,
        "edges": edges,
        "districts": districts,
        "modelMetrics": {
            "prAuc": round(gnn_metrics.get("pr_auc", 0.0), 4),
            "recallAt10Pct": round(gnn_metrics.get("recall_at_10pct", 0.0), 4),
        },
    }