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
    app_description: str = Field(
        default=(
            "Crop health intelligence API for image-based disease triage "
            "and precision agriculture workflows."
        )
    )
    app_version: str = Field(default="0.1.0")
    api_prefix: str = Field(default="")
    model_path: Path = Field(default=Path("models/model.pt"))
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    docs_url: str = Field(default="/docs")
    redoc_url: str = Field(default="/redoc")
    openapi_url: str = Field(default="/openapi.json")
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:8501",
            "http://127.0.0.1:8501",
        ]
    )


settings = Settings()
