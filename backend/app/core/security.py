"""Map the ml package's validation errors to HTTP responses.

The actual checks (https-only shape, host allowlist, size and time limits)
live in codecity_ml.mining.clone so the CLI and API can't diverge. This
module only decides which status code each failure becomes.
"""

from __future__ import annotations

from fastapi import HTTPException

from codecity_ml.mining.clone import (
    CloneTimeoutError,
    DisallowedHostError,
    InvalidRepoUrlError,
    RepoTooLargeError,
    parse_repo_url,
)


def validate_repo_url(url: str) -> tuple[str, str, str]:
    """Validate before any job is created. Returns (host, owner, name)."""
    try:
        return parse_repo_url(url)
    except (InvalidRepoUrlError, DisallowedHostError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


def status_for_error(exc: Exception) -> tuple[int, str]:
    """(http status, safe message) for a failure during analysis. Unknown
    errors get a generic message so internals never leak to the client."""
    if isinstance(exc, (InvalidRepoUrlError, DisallowedHostError)):
        return 400, str(exc)
    if isinstance(exc, RepoTooLargeError):
        return 413, str(exc)
    if isinstance(exc, CloneTimeoutError):
        return 504, str(exc)
    if isinstance(exc, ValueError):
        return 422, str(exc)  # e.g. no supported files, no history
    return 500, "Analysis failed."