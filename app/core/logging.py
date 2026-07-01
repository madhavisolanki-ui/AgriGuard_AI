"""Logging helpers for AgriGuard."""

from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> None:
    """Configure application-wide logging with a concise production format."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format=(
            "%(asctime)s | %(levelname)s | %(name)s | "
            "%(message)s"
        ),
    )
    logging.getLogger("uvicorn.error").setLevel(numeric_level)
    logging.getLogger("uvicorn.access").setLevel(numeric_level)

