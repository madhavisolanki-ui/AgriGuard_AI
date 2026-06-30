"""FastAPI application entry point for AgriGuard."""

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import settings


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Crop Health and Precision Agriculture API for health checks "
            "and model-backed agronomic workflows."
        ),
    )
    app.include_router(router)
    return app


app = create_app()

