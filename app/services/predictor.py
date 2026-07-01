"""Prediction service abstraction for AgriGuard."""

from app.services.inference_service import get_inference_service


def get_service_status() -> str:
    """Return the current inference service status."""
    try:
        get_inference_service()
    except Exception:
        return "unavailable"
    return "ready"
