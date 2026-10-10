from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.core.security import validate_repo_url
from app.schemas.analyze import AnalyzeRequest, AnalyzeResponse
from app.services import job_runner
from app.services.job_store import store

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse, status_code=202)
async def analyze(body: AnalyzeRequest, request: Request) -> AnalyzeResponse:
    """Start (or reuse) an analysis job. Returns immediately with a job id."""
    host, owner, name = validate_repo_url(body.repo_url)  # 400 on bad URL / host
    slug = f"{host}__{owner}__{name}"

    if request.app.state.model is None:
        raise HTTPException(503, "No trained model is loaded. Run `tasks train` and restart.")

    if store.get_result(slug) is not None:  # fresh cached result: instant "done" job
        job = store.create(body.repo_url, slug, done=True)
    elif (active := store.find_active(slug)) is not None:  # same repo already running
        job = active
    else:
        job = store.create(body.repo_url, slug)
        job_runner.submit(job, request.app.state.model, request.app.state.scaler,
                          request.app.state.model_meta)

    return AnalyzeResponse(jobId=job.id, repoSlug=slug, status=job.status)