"""Command-line interface.

    python -m codecity_ml.cli train --config ml/configs/train_sage.yaml
    python -m codecity_ml.cli eval ml/artifacts/model_sage.pt

(Also available as `codecity train ...` after `pip install -e ml`.)
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import torch
import typer
import yaml
from rich.console import Console

app = typer.Typer(add_completion=False, help="CodeCity ML tools")
console = Console()


@app.command()
def train(
    config: Path = typer.Option(Path("ml/configs/train_sage.yaml"), help="YAML training config"),
    data: Path = typer.Option(Path("data/processed"), help="Cached graph snapshots"),
    out: Path = typer.Option(Path("ml/artifacts"), help="Where to write the checkpoint"),
    temporal: bool = typer.Option(False, help="Force the temporal split"),
) -> None:
    """Train a model and save a checkpoint."""
    import codecity_ml.training.train as trainer
    from codecity_ml.export.checkpoint import save_checkpoint
    from codecity_ml.graph.dataset import load_dataset

    raw = yaml.safe_load(config.read_text()) or {}
    allowed = {f.name for f in fields(trainer.TrainConfig)}
    unknown = set(raw) - allowed
    if unknown:
        raise typer.BadParameter(f"Unknown keys in {config}: {sorted(unknown)}")

    graphs = load_dataset(data)
    slugs = trainer.repo_slugs(graphs)

    mode = raw.pop("split_mode", "auto")
    if temporal:
        mode = "temporal"
    if mode == "auto":
        mode = "cross_project" if len(slugs) >= trainer.MIN_REPOS_FOR_CROSS_PROJECT else "temporal"

    cfg = trainer.TrainConfig(**raw, split_mode=mode)
    console.print(f"snapshots: {len(graphs)} | repos: {len(slugs)} | split: {mode} | model: {cfg.model_name}")

    res = trainer.train(graphs, cfg)

    train_repos = [s for s in slugs if s not in res.test_repos] if mode == "cross_project" else slugs
    path = save_checkpoint(
        out / f"model_{cfg.model_name}.pt",
        model=res.model,
        scaler=res.scaler,
        model_name=cfg.model_name,
        hparams={"hidden_dim": cfg.hidden_dim, "num_layers": cfg.num_layers, "dropout": cfg.dropout},
        split_mode=mode,
        train_repos=train_repos,
        test_repos=res.test_repos,
        metrics={"gnn": res.gnn_test, "xgboost": res.xgb_test},
        per_repo_mean=trainer._mean(res.gnn_per_repo),
    )
    console.print(f"best epoch: {res.best_epoch}")
    console.print(res.xgb_test.summary())
    console.print(res.gnn_test.summary())
    console.print(f"[green]saved[/green] {path}")


@app.command("eval")
def eval_cmd(
    checkpoint: Path = typer.Argument(..., help="Path to a .pt checkpoint"),
    data: Path = typer.Option(Path("data/processed")),
    repo: list[str] | None = typer.Option(
        None, "--repo", help="Repo slug(s) to evaluate on. Default: the held-out test repos."
    ),
) -> None:
    """Score a saved checkpoint, by default on its held-out test repos."""
    import numpy as np

    from codecity_ml.evaluation.metrics import evaluate
    from codecity_ml.export.checkpoint import load_checkpoint
    from codecity_ml.graph.dataset import load_dataset

    model, scaler, meta = load_checkpoint(checkpoint)
    targets = set(repo or meta["test_repos"])
    if not targets:
        raise typer.BadParameter("Checkpoint has no held-out repos; pass --repo explicitly.")

    seen = targets & set(meta["train_repos"])
    if seen:
        console.print(f"[yellow]warning:[/yellow] trained on {sorted(seen)}; scores there are optimistic.")

    graphs = [g.clone() for g in load_dataset(data) if g.repo_slug in targets]
    if not graphs:
        raise typer.BadParameter(f"No cached snapshots for {sorted(targets)} under {data}")
    scaler.apply(graphs)

    with torch.no_grad():
        probs = np.concatenate([model.predict_proba(g).numpy() for g in graphs])
    y = np.concatenate([g["file"].y.numpy() for g in graphs])

    console.print(f"repos: {sorted(targets)} | snapshots: {len(graphs)} | examples: {len(y)}")
    console.print(evaluate(meta["model_name"], y, probs).summary())


if __name__ == "__main__":
    app()