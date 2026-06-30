"""Prediction service abstraction for AgriGuard."""


def get_service_status() -> str:
    """Return the current inference service status.

    This stub is intentionally simple for the initial scaffold and can later
    be expanded to validate model loading, GPU availability, and downstream
    dependencies.
    """

    return "ready"
