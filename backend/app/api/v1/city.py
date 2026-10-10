from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException

from app.services.job_store import store

router = APIRouter()

_SLUG_RE = re.compile(r"^[\w.-]+$")  # e.g. github.com__pallets__click


@router.get("/repos/{slug}/city")
async def get_city(slug: str) -> dict:
    if not _SLUG_RE.match(slug):
        raise HTTPException(400, "Invalid repo id.")
    city = store.get_result(slug)
    if city is None:
        raise HTTPException(404, "No analysis for this repo yet. POST /api/v1/analyze first.")
    return city