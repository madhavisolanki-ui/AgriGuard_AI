"""API tests for the AgriGuard FastAPI application."""

from __future__ import annotations

import base64
from typing import Generator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.routes import get_inference_service
from app.main import app
from app.services.inference_service import InferenceService


@pytest.fixture(autouse=True)
def clear_dependency_overrides() -> Generator[None, None, None]:
    """Ensure dependency overrides do not leak between tests."""
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


@pytest.fixture()
def client() -> TestClient:
    """Return a test client for the FastAPI application."""
    return TestClient(app)


def test_root_endpoint(client: TestClient) -> None:
    """Verify the root endpoint returns the expected landing payload."""
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "AgriGuard API is running",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


def test_predict_endpoint_uses_mocked_inference_service(client: TestClient) -> None:
    """Verify the prediction endpoint delegates to a mocked inference service."""
    fake_service = MagicMock(spec=InferenceService)
    expected_response = {
        "status": "success",
        "predicted_class": "healthy_crop",
        "confidence": 0.97,
        "probabilities": {
            "healthy_crop": 0.97,
            "leaf_blight": 0.02,
            "water_stress": 0.01,
        },
        "model_name": "dummy-agriculture-classifier",
        "processing_time_ms": 12.5,
    }
    fake_service.predict.return_value = expected_response
    app.dependency_overrides[get_inference_service] = lambda: fake_service

    payload = {
        "image_base64": base64.b64encode(b"fake image payload").decode("utf-8"),
        "filename": "sample.jpg",
        "content_type": "image/jpeg",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json() == expected_response
    fake_service.predict.assert_called_once_with(payload["image_base64"])
