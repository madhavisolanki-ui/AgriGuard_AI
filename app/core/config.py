"""Application configuration helpers."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings sourced from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="AgriGuard")
    app_version: str = Field(default="0.1.0")
    model_path: Path = Field(default=Path("models/model.pt"))
    debug: bool = Field(default=False)


settings = Settings()
