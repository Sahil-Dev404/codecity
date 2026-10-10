"""Convert cached RepoGraph JSON (graph/builder.py) into PyG HeteroData.

Node types:  "file" (features + label), "author" (small activity features)
Edge types:  file-imports-file, file-cochange-file (both directions),
             file-authored_by-author and its reverse.

One HeteroData per snapshot. Also provides the temporal split and the
feature scaler, so training code never has to reinvent either.

Leakage status is documented in graph/builder.py. Edges and node features
are snapshot-aware; import edges and size/complexity come from the current
checkout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import HeteroData

from codecity_ml.graph.features import FEATURE_NAMES

# Features that are already bounded ratios; everything else is a heavy-
# tailed count/size and gets log1p-compressed.
_RATIO_FEATURES = {"top_author_share"}


def _transform_features(raw: np.ndarray) -> np.ndarray:
    out = raw.astype(np.float32).copy()
    for j, name in enumerate(FEATURE_NAMES):
        if name not in _RATIO_FEATURES:
            out[:, j] = np.log1p(np.maximum(out[:, j], 0.0))
    return out


def _edge_tensor(pairs: list[tuple[int, int]]) -> torch.Tensor:
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


def graph_from_dict(data: dict) -> HeteroData | None:
    """Build HeteroData from a RepoGraph dict (RepoGraph.to_json_dict() or a
    loaded cache file). Returns None if the snapshot has no featurized files.
    Missing labels become 0, which is what live inference has."""
    paths: list[str] = data["nodes"]
    if not paths:
        return None

    file_index = {p: i for i, p in enumerate(paths)}

    raw = np.array(
        [[data["features"][p][name] for name in FEATURE_NAMES] for p in paths],
        dtype=np.float32,
    )
    y = np.array(
        [int(data["labels"].get(p, {}).get("is_buggy_future", 0)) for p in paths],
        dtype=np.int64,
    )

    g = HeteroData()
    g["file"].x = torch.from_numpy(_transform_features(raw))
    g["file"].y = torch.from_numpy(y)
    g["file"].paths = paths

    # ── import edges (directed: source imports target) ──
    imports = [
        (file_index[e["source_file"]], file_index[e["target_file"]])
        for e in data["import_edges"]
        if e["source_file"] in file_index and e["target_file"] in file_index
    ]
    g["file", "imports", "file"].edge_index = _edge_tensor(imports)

    # ── co-change edges (undirected: store both directions) ──
    co_pairs: list[tuple[int, int]] = []
    co_weights: list[float] = []
    for e in data["cochange_edges"]:
        a, b = e["file_a"], e["file_b"]
        if a in file_index and b in file_index:
            w = float(np.log1p(e["commit_count"]))
            co_pairs += [(file_index[a], file_index[b]), (file_index[b], file_index[a])]
            co_weights += [w, w]
    g["file", "cochange", "file"].edge_index = _edge_tensor(co_pairs)
    g["file", "cochange", "file"].edge_attr = torch.tensor(
        co_weights, dtype=torch.float32
    ).unsqueeze(1)

    # ── authors: nodes + file<->author edges ──
    auth_edges = [e for e in data["authorship_edges"] if e["file_path"] in file_index]
    emails = sorted({e["author_email"] for e in auth_edges})
    author_index = {m: i for i, m in enumerate(emails)}

    commits = np.zeros(len(emails), dtype=np.float32)
    lines = np.zeros(len(emails), dtype=np.float32)
    files_touched = np.zeros(len(emails), dtype=np.float32)
    fa_pairs: list[tuple[int, int]] = []
    fa_weights: list[float] = []
    for e in auth_edges:
        ai = author_index[e["author_email"]]
        commits[ai] += e["commit_count"]
        lines[ai] += e["lines_changed"]
        files_touched[ai] += 1
        fa_pairs.append((file_index[e["file_path"]], ai))
        fa_weights.append(float(np.log1p(e["commit_count"])))

    author_x = (
        np.log1p(np.stack([commits, lines, files_touched], axis=1))
        if emails
        else np.zeros((0, 3), dtype=np.float32)
    )
    g["author"].x = torch.from_numpy(author_x.astype(np.float32))

    g["file", "authored_by", "author"].edge_index = _edge_tensor(fa_pairs)
    g["file", "authored_by", "author"].edge_attr = torch.tensor(
        fa_weights, dtype=torch.float32
    ).unsqueeze(1)
    g["author", "authored", "file"].edge_index = _edge_tensor([(a, f) for f, a in fa_pairs])
    g["author", "authored", "file"].edge_attr = torch.tensor(
        fa_weights, dtype=torch.float32
    ).unsqueeze(1)

    g.repo_slug = data["repo_slug"]
    snapshot_at = data["snapshot_at"]
    if isinstance(snapshot_at, str):
        snapshot_at = datetime.fromisoformat(snapshot_at)
    g.snapshot_ts = snapshot_at.timestamp()
    return g


def load_graph(path: Path) -> HeteroData | None:
    """Load one cached snapshot file as HeteroData."""
    return graph_from_dict(json.loads(path.read_text()))


def load_dataset(cache_dir: Path) -> list[HeteroData]:
    """Load every cached snapshot under `cache_dir`, oldest first."""
    graphs = []
    for path in sorted(cache_dir.glob("*.json")):
        g = load_graph(path)
        if g is not None:
            graphs.append(g)
    if not graphs:
        raise ValueError(f"No usable cached graphs under {cache_dir}. Run graph building first.")
    return sorted(graphs, key=lambda g: g.snapshot_ts)


def temporal_split(
    graphs: list[HeteroData], *, test_fraction: float = 0.2, purge_days: int = 0
) -> tuple[list[HeteroData], list[HeteroData]]:
    """Split snapshots by DATE: the latest `test_fraction` become the test
    set. `purge_days` drops training snapshots within that many days of the
    cutoff. Labels look `window_days` ahead, so a training snapshot right
    before the cutoff has labels computed from commits inside the test
    period; setting purge_days=window_days (180) removes that overlap."""
    if len(graphs) < 2:
        raise ValueError("Need at least 2 snapshots to split.")
    stamps = sorted(g.snapshot_ts for g in graphs)
    cutoff = stamps[int(len(stamps) * (1 - test_fraction))]
    train = [g for g in graphs if g.snapshot_ts < cutoff - purge_days * 86400]
    test = [g for g in graphs if g.snapshot_ts >= cutoff]
    return train, test


@dataclass
class FeatureScaler:
    """Standardizes file features. Fit on TRAIN graphs only, then apply to
    train and test, so test statistics never influence preprocessing."""

    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def fit(cls, graphs: list[HeteroData]) -> FeatureScaler:
        x = torch.cat([g["file"].x for g in graphs], dim=0)
        return cls(mean=x.mean(dim=0), std=x.std(dim=0, unbiased=False).clamp_min(1e-6))

    def apply(self, graphs: list[HeteroData]) -> None:
        """Scales in place."""
        for g in graphs:
            g["file"].x = (g["file"].x - self.mean) / self.std