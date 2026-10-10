"""Backend settings, read from the shared repo-root .env (run from the root).

List-valued settings are plain comma-separated strings exposed as
properties, for the same reason as codecity_ml/config.py.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ml-only keys live in the same .env
    )

    app_env: str = "development"
    log_level: str = "INFO"
    cors_origins_raw: str = Field(default="http://localhost:3000", alias="CORS_ORIGINS")

    database_url: str = "postgresql+asyncpg://codecity:change-me@localhost:5432/codecity"
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 86400
    rate_limit_per_minute: int = 10

    model_path: Path = Path("./ml/artifacts/model_sage.pt")

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.cors_origins_raw.split(",") if o.strip()]


settings = Settings()