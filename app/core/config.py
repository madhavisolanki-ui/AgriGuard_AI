"""Application configuration helpers."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    """Runtime settings sourced from environment variables."""

    app_name: str = os.getenv("AGRIGUARD_APP_NAME", "AgriGuard")
    app_version: str = os.getenv("AGRIGUARD_APP_VERSION", "0.1.0")


settings = Settings()

