"""HTTP route definitions for AgriGuard."""

from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.health import HealthResponse
from app.schemas.predict_schema import PredictionRequest, PredictionResponse
from app.services.inference_service import (
    InferenceError,
    InferenceService,
    get_inference_service,
)
from app.services.predictor import get_service_status


router = APIRouter()


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
    try:
        return service.predict(request.image_base64)
    except InferenceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
