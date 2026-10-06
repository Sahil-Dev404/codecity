"""Safe, size- and host-limited cloning of git repositories.

This is the only place in codecity_ml that touches the network. Every
caller (CLI, backend worker) should go through `clone_repo`, never call
git directly, so the safety checks here can't be bypassed.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from codecity_ml.config import settings

# Only https URLs matching this shape are accepted, e.g.
# https://github.com/psf/requests(.git)?
_REPO_URL_RE = re.compile(
    r"^https://(?P<host>[a-zA-Z0-9.-]+)/(?P<owner>[\w.-]+)/(?P<repo>[\w.-]+?)(\.git)?/?$"
)


class InvalidRepoUrlError(ValueError):
    """Raised when a URL doesn't match the expected https host/owner/repo shape."""


class DisallowedHostError(ValueError):
    """Raised when the URL's host isn't in settings.allowed_git_hosts."""


class RepoTooLargeError(RuntimeError):
    """Raised after cloning, if the result exceeds settings.max_repo_size_mb."""


class CloneTimeoutError(RuntimeError):
    """Raised when `git clone` exceeds settings.clone_timeout_seconds."""


@dataclass(frozen=True)
class ClonedRepo:
    """Where a cloned repo ended up, plus identifying info extracted from the URL."""

    url: str
    host: str
    owner: str
    name: str
    local_path: Path

    @property
    def slug(self) -> str:
        """Filesystem- and URL-safe identifier, e.g. 'github.com__psf__requests'."""
        return f"{self.host}__{self.owner}__{self.name}"


def parse_repo_url(url: str) -> tuple[str, str, str]:
    """Validate a repo URL and split it into (host, owner, repo_name).

    Raises InvalidRepoUrlError if the shape doesn't match, or
    DisallowedHostError if the host isn't in settings.allowed_git_hosts.
    This runs before any network call, so a malicious or malformed URL
    never reaches `git clone`.
    """
    match = _REPO_URL_RE.match(url.strip())
    if not match:
        raise InvalidRepoUrlError(
            f"'{url}' doesn't look like a valid https://host/owner/repo URL"
        )

    host = match.group("host").lower()
    if host not in settings.allowed_git_hosts:
        raise DisallowedHostError(
            f"Host '{host}' is not allowed. Allowed hosts: {settings.allowed_git_hosts}"
        )

    return host, match.group("owner"), match.group("repo")


def _dir_size_mb(path: Path) -> float:
    total_bytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total_bytes / (1024 * 1024)


def clone_repo(url: str, *, force: bool = False) -> ClonedRepo:
    """Clone `url` into settings.raw_dir, enforcing host, size and time limits.

    Uses a shallow, blobless clone (`--filter=blob:none`) so history metadata
    is available for mining without downloading every historical file
    version up front — large repos stay fast to clone.

    If `force` is False and the destination already exists, the existing
    clone is reused instead of re-cloning (handy for repeat demo runs).
    """
    host, owner, name = parse_repo_url(url)

    dest = settings.raw_dir / f"{host}__{owner}__{name}"

    if dest.exists():
        if force:
            shutil.rmtree(dest)
        else:
            return ClonedRepo(url=url, host=host, owner=owner, name=name, local_path=dest)

    dest.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "git",
        "clone",
        "--filter=blob:none",  # don't download file contents up front
        "--no-single-branch",  # keep all branches' history reachable
        url,
        str(dest),
    ]

    try:
        subprocess.run(
            cmd,
            check=True,
            timeout=settings.clone_timeout_seconds,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired as e:
        shutil.rmtree(dest, ignore_errors=True)
        raise CloneTimeoutError(
            f"Cloning {url} exceeded {settings.clone_timeout_seconds}s"
        ) from e
    except subprocess.CalledProcessError as e:
        shutil.rmtree(dest, ignore_errors=True)
        raise RuntimeError(f"git clone failed for {url}: {e.stderr.strip()}") from e

    size_mb = _dir_size_mb(dest)
    if size_mb > settings.max_repo_size_mb:
        shutil.rmtree(dest, ignore_errors=True)
        raise RepoTooLargeError(
            f"{url} is {size_mb:.0f}MB, exceeding the {settings.max_repo_size_mb}MB limit"
        )

    return ClonedRepo(url=url, host=host, owner=owner, name=name, local_path=dest)