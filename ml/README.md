# codecity-ml

Mining, parsing, graph construction and GNN-based defect prediction for **CodeCity**.

Turns a git repository into a heterogeneous graph (files, authors, directories) and
predicts which files are likely to be touched by a bug fix in the near future.

## Install

```bash
pip install -e ".[dev]"        # core + dev tools (lint, test)
pip install -e ".[dev,gnn]"    # + PyTorch / PyTorch Geometric
```

## Package layout

| Module | Responsibility |
|---|---|
| `mining/` | Clone repos, walk commit history, trace bug-introducing commits (SZZ) |
| `parsing/` | tree-sitter based parsing: imports, complexity, per-language metrics |
| `graph/` | Build the heterogeneous graph and its node/edge features |
| `models/` | Baselines (logistic regression, XGBoost) and GNNs (GraphSAGE, GAT, HGT) |
| `training/` | Training loop, temporal/cross-project splits, imbalance-aware losses |
| `evaluation/` | PR-AUC, Recall@Top-K, ablations, error analysis |
| `explain/` | GNNExplainer and counterfactual "what-if" edits |
| `export/` | TorchScript / ONNX export for serving |

## CLI

```bash
codecity mine --repo <url>
codecity build-graph --repo-id <id>
codecity train --config configs/train_sage.yaml
codecity eval --run <run-id>
```

## Tests

```bash
pytest tests -q
```