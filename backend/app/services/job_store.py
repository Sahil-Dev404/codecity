"""In-memory job and result store. Thread-safe, because jobs are updated
from worker threads while request handlers read them.

Interface (create / get / set_stage / complete / fail / get_result) is what
the routes depend on. A Redis-backed version can replace this module later
without touching them. Limits of this version: state is lost on restart and
is not shared between server processes (run uvicorn with one worker).
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field

from app.core.config import settings

STAGES = ("clone", "parse", "graph", "predict")
_MAX_JOBS = 200  # oldest finished jobs are evicted beyond this


@dataclass
class Job:
    id: str
    url: str
    slug: str
    status: str = "queued"  # queued | running | done | failed
    stage: str | None = None
    stage_index: int = -1
    error: str | None = None
    error_status: int | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def to_public(self) -> dict:
        return {
            "jobId": self.id,
            "repoSlug": self.slug,
            "status": self.status,
            "stage": self.stage,
            "stageIndex": self.stage_index,
            "error": self.error,
            "errorStatus": self.error_status,
        }


class JobStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._results: dict[str, tuple[float, dict]] = {}  # slug -> (time, city)

    def create(self, url: str, slug: str, *, done: bool = False) -> Job:
        job = Job(id=uuid.uuid4().hex, url=url, slug=slug)
        if done:
            job.status = "done"
            job.stage, job.stage_index = STAGES[-1], len(STAGES) - 1
            job.finished_at = time.time()
        with self._lock:
            self._jobs[job.id] = job
            self._evict()
        return job

    def _evict(self) -> None:
        if len(self._jobs) <= _MAX_JOBS:
            return
        finished = sorted(
            (j for j in self._jobs.values() if j.status in ("done", "failed")),
            key=lambda j: j.created_at,
        )
        for j in finished[: len(self._jobs) - _MAX_JOBS]:
            del self._jobs[j.id]

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.to_public() if job else None

    def find_active(self, slug: str) -> Job | None:
        """A queued or running job for this repo, so double submits share one."""
        with self._lock:
            for j in self._jobs.values():
                if j.slug == slug and j.status in ("queued", "running"):
                    return j
        return None

    def set_running(self, job_id: str) -> None:
        with self._lock:
            self._jobs[job_id].status = "running"

    def set_stage(self, job_id: str, stage: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.stage, job.stage_index = stage, STAGES.index(stage)

    def complete(self, job_id: str, city: dict) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status, job.finished_at = "done", time.time()
            job.stage, job.stage_index = STAGES[-1], len(STAGES) - 1
            self._results[job.slug] = (time.time(), city)

    def fail(self, job_id: str, status_code: int, message: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status, job.finished_at = "failed", time.time()
            job.error, job.error_status = message, status_code

    def get_result(self, slug: str) -> dict | None:
        """Cached city for a repo, or None if absent or older than the TTL.
        Keyed by repo slug with a TTL (not by commit SHA), so a result can
        be up to cache_ttl_seconds behind the repo's HEAD."""
        with self._lock:
            entry = self._results.get(slug)
            if entry is None:
                return None
            saved_at, city = entry
            if time.time() - saved_at > settings.cache_ttl_seconds:
                del self._results[slug]
                return None
            return city


store = JobStore()