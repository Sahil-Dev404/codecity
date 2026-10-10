"""Runs analysis jobs on a small thread pool.

Threads are enough here: the heavy work is git subprocesses and torch ops
(which release the GIL). max_workers is small on purpose, since each job
clones a repo and builds a graph in memory.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from torch import nn

from app.core.security import status_for_error
from app.services.analysis_service import analyze_url
from app.services.job_store import Job, store
from codecity_ml.graph.dataset import FeatureScaler

log = logging.getLogger("codecity.jobs")
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="analysis")


def submit(job: Job, model: nn.Module, scaler: FeatureScaler, meta: dict) -> None:
    _executor.submit(_run, job.id, job.url, model, scaler, meta)


def _run(job_id: str, url: str, model: nn.Module, scaler: FeatureScaler, meta: dict) -> None:
    store.set_running(job_id)
    try:
        city = analyze_url(url, model, scaler, meta, on_stage=lambda s: store.set_stage(job_id, s))
        store.complete(job_id, city)
    except Exception as exc:  # noqa: BLE001 - every failure must end the job
        code, message = status_for_error(exc)
        if code == 500:
            log.exception("Job %s failed unexpectedly", job_id)  # full detail stays server-side
        store.fail(job_id, code, message)