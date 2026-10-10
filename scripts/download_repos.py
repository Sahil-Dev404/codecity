"""Batch-build cached graph snapshots for every repo in a list.

Run from the repo root (settings paths like ./data are relative to cwd):

    python scripts/download_repos.py --limit 3        # smoke test
    python scripts/download_repos.py --cleanup        # full run, free disk as you go

Resumable: a repo that already has snapshots in data/processed is skipped
(use --force to rebuild). One repo failing never stops the batch; failures
are listed at the end.
"""

from __future__ import annotations

import argparse
import shutil
import stat
import time
import traceback
from pathlib import Path

from codecity_ml.config import settings
from codecity_ml.graph.builder import build_all_snapshots, save_repo_graph
from codecity_ml.mining.clone import clone_repo, parse_repo_url


def read_repo_list(path: Path) -> list[str]:
    urls = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            urls.append(line)
    return urls


def _make_writable_and_retry(func, path, _exc_info) -> None:
    # Git marks pack files read-only; on Windows rmtree fails on them.
    Path(path).chmod(stat.S_IWRITE)
    func(path)


def remove_tree(path: Path) -> None:
    shutil.rmtree(path, onerror=_make_writable_and_retry)


def has_snapshots(slug: str) -> bool:
    return any(settings.processed_dir.glob(f"{slug}__*.json"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", type=Path, default=Path("ml/configs/repos_train.txt"))
    ap.add_argument("--window-days", type=int, default=180)
    ap.add_argument("--stride-days", type=int, default=30)
    ap.add_argument("--limit", type=int, default=None, help="only process the first N repos")
    ap.add_argument("--force", action="store_true", help="rebuild repos that already have snapshots")
    ap.add_argument("--cleanup", action="store_true", help="delete each clone after snapshots are saved")
    ap.add_argument("--clone-timeout", type=int, default=None, help="override CLONE_TIMEOUT_SECONDS")
    args = ap.parse_args()

    if args.clone_timeout:
        settings.clone_timeout_seconds = args.clone_timeout

    urls = read_repo_list(args.repos)
    if args.limit:
        urls = urls[: args.limit]

    done: list[tuple[str, int, float]] = []
    skipped: list[str] = []
    failed: list[tuple[str, str]] = []

    for i, url in enumerate(urls, 1):
        host, owner, name = parse_repo_url(url)
        slug = f"{host}__{owner}__{name}"
        print(f"\n[{i}/{len(urls)}] {slug}", flush=True)

        if has_snapshots(slug) and not args.force:
            print("  already built, skipping")
            skipped.append(slug)
            continue

        t0 = time.time()
        try:
            cloned = clone_repo(url)
            graphs = build_all_snapshots(
                slug,
                cloned.local_path,
                window_days=args.window_days,
                stride_days=args.stride_days,
            )
            if not graphs:
                raise RuntimeError("0 snapshots (too little history in the window)")
            for g in graphs:
                save_repo_graph(g, settings.processed_dir)
            elapsed = time.time() - t0
            print(f"  saved {len(graphs)} snapshots in {elapsed:.0f}s")
            done.append((slug, len(graphs), elapsed))

            if args.cleanup:
                remove_tree(cloned.local_path)
        except Exception as e:  # keep the batch going
            print(f"  FAILED: {e}")
            traceback.print_exc(limit=2)
            failed.append((slug, str(e)))

    print("\n=== Summary ===")
    print(f"built: {len(done)} | skipped: {len(skipped)} | failed: {len(failed)}")
    for slug, n, secs in done:
        print(f"  ok     {slug}: {n} snapshots, {secs:.0f}s")
    for slug, why in failed:
        print(f"  FAILED {slug}: {why}")


if __name__ == "__main__":
    main()