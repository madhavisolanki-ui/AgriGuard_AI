"""HTTP route definitions for AgriGuard."""

from fastapi import APIRouter, Depends, status

from app.schemas.health import HealthResponse
from app.schemas.predict_schema import PredictionRequest, PredictionResponse
from app.schemas.system import RootResponse
from app.core.config import settings
from app.services.inference_service import (
    InferenceService,
    get_inference_service,
)
from app.services.predictor import get_service_status


router = APIRouter()


@router.get("/", response_model=RootResponse, tags=["Root"])
def root() -> RootResponse:
    """Return a concise application landing payload."""
    return RootResponse(
        version=settings.app_version,
        docs="/docs",
        health="/health",
        predict="/predict",
    )


@router.get("/health", response_model=HealthResponse, tags=["Health"])
def health() -> HealthResponse:
    """Return application health and service status."""
    return HealthResponse(status="ok", service=get_service_status())


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Predictions"],
)
def predict(
    request: PredictionRequest,
    service: InferenceService = Depends(get_inference_service),
) -> PredictionResponse:
    """Run image inference and return the model prediction."""
    return service.predict(
        request.image_base64,
        filename=request.filename,
        content_type=request.content_type,
    )
