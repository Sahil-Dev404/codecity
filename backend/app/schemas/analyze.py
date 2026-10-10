from __future__ import annotations

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    repo_url: str = Field(..., max_length=300, examples=["https://github.com/pallets/click"])


class AnalyzeResponse(BaseModel):
    jobId: str
    repoSlug: str
    status: str