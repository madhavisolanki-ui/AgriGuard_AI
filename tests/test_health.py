"""Tests for the AgriGuard health endpoints."""

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root_endpoint() -> None:
    """Verify the root endpoint returns the expected landing payload."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "AgriGuard",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


def test_health_endpoint() -> None:
    """Verify the health endpoint reports a ready service."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "ready"}
