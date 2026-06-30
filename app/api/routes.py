"""HTTP route definitions for AgriGuard."""

from fastapi import APIRouter

from app.schemas.health import HealthResponse
from app.services.predictor import get_service_status


router = APIRouter()


@router.get("/", tags=["Root"])
def root() -> dict[str, str]:
    """Return a simple landing response for the API."""
    return {
        "message": "AgriGuard API is running",
        "docs": "/docs",
        "health": "/health",
    }


@router.get("/health", response_model=HealthResponse, tags=["Health"])
def health() -> HealthResponse:
    """Return application health and service status."""
    return HealthResponse(status="ok", service=get_service_status())

