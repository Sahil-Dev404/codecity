from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.services.job_store import store

router = APIRouter()


@router.get("/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job.")
    return job


@router.get("/jobs/{job_id}/stream")
async def stream_job(job_id: str, request: Request) -> StreamingResponse:
    """Server-Sent Events: one message whenever the job's state changes,
    ending once it is done or failed."""
    if store.get(job_id) is None:
        raise HTTPException(404, "Unknown job.")

    async def events():
        last = None
        while True:
            if await request.is_disconnected():
                break
            job = store.get(job_id)
            if job is None:
                break
            payload = json.dumps(job)
            if payload != last:
                yield f"data: {payload}\n\n"
                last = payload
            if job["status"] in ("done", "failed"):
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )