from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import analyze, city, health, jobs

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health.router, tags=["health"])
api_router.include_router(analyze.router, tags=["analyze"])
api_router.include_router(jobs.router, tags=["jobs"])
api_router.include_router(city.router, tags=["city"])
# files (explanations) and whatif routers get added here in later files.