from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> dict:
    """Liveness plus whether the model checkpoint loaded at startup."""
    meta = getattr(request.app.state, "model_meta", None)
    return {
        "status": "ok",
        "model_loaded": getattr(request.app.state, "model", None) is not None,
        "model_name": meta["model_name"] if meta else None,
        "split_mode": meta["split_mode"] if meta else None,
    }