"""FastAPI app factory.

The model checkpoint is loaded ONCE at startup (lifespan) and stored on
app.state, so requests never pay the load cost. A missing checkpoint does
not stop the server: it starts with model_loaded=false, so the API is still
usable for development before you have trained anything.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config import settings

logging.basicConfig(level=settings.log_level)
log = logging.getLogger("codecity.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    app.state.scaler = None
    app.state.model_meta = None

    if settings.model_path.exists():
        from codecity_ml.export.checkpoint import load_checkpoint

        model, scaler, meta = load_checkpoint(settings.model_path)
        app.state.model, app.state.scaler, app.state.model_meta = model, scaler, meta
        log.info("Loaded %s checkpoint from %s", meta["model_name"], settings.model_path)
    else:
        log.warning("No checkpoint at %s; starting without a model", settings.model_path)

    yield
    # Nothing to release yet (DB and Redis pools will close here later).


def create_app() -> FastAPI:
    app = FastAPI(title="CodeCity API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router)
    return app


app = create_app()