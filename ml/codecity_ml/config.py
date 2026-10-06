"""Centralized, typed settings for codecity_ml.

Reads from environment variables / a .env file. Every other module should
import `settings` from here rather than reading os.environ directly, so
there is exactly one source of truth for limits and paths.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore backend-only vars like DATABASE_URL
    )

    # ── Safety limits for cloning/analyzing untrusted repos ────
    allowed_git_hosts: list[str] = Field(default=["github.com"])
    max_repo_size_mb: int = 500
    max_commits: int = 5000
    history_years: int = 3
    clone_timeout_seconds: int = 120

    # ── Paths ───────────────────────────────────────────────
    data_dir: Path = Path("./data")

    # ── Logging ─────────────────────────────────────────────
    log_level: str = "INFO"

    @field_validator("allowed_git_hosts", mode="before")
    @classmethod
    def _split_csv(cls, v: str | list[str]) -> list[str]:
        """Allow ALLOWED_GIT_HOSTS=github.com,gitlab.com in .env (a plain
        comma-separated string) instead of requiring JSON list syntax."""
        if isinstance(v, str):
            return [host.strip() for host in v.split(",") if host.strip()]
        return v

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