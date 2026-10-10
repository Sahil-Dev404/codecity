"""Centralized, typed settings for codecity_ml.

Reads environment variables / a .env file (relative to the working
directory, so run from the repo root). Every other module imports
`settings` from here instead of reading os.environ.

List-valued settings are stored as plain comma-separated strings and
exposed as properties: pydantic-settings JSON-decodes list-typed env values
before any validator runs, so `ALLOWED_GIT_HOSTS=github.com` would crash.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # the shared .env also holds backend-only keys
    )

    # Safety limits for cloning/analyzing untrusted repos
    allowed_git_hosts_raw: str = Field(default="github.com", alias="ALLOWED_GIT_HOSTS")
    max_repo_size_mb: int = 500
    max_commits: int = 5000
    history_years: int = 3
    clone_timeout_seconds: int = 120

    # Paths
    data_dir: Path = Path("./data")

    # Logging
    log_level: str = "INFO"

    @property
    def allowed_git_hosts(self) -> list[str]:
        return [h.strip().lower() for h in self.allowed_git_hosts_raw.split(",") if h.strip()]

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def samples_dir(self) -> Path:
        return self.data_dir / "samples"


settings = Settings()