"""Schemas for prediction requests and responses."""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request payload for image-based inference."""

    image_base64: str = Field(
        ...,
        description="Base64-encoded image payload used for inference.",
        examples=["iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB..."],
    )
    filename: str | None = Field(
        default=None,
        description="Optional original filename for tracing and observability.",
        examples=["leaf_sample.jpg"],
    )
    content_type: str = Field(
        default="image/jpeg",
        description="MIME type of the submitted image payload.",
        examples=["image/jpeg"],
    )


class PredictionResponse(BaseModel):
    """Response payload returned after inference completes."""

    status: str = Field(default="success", description="High-level request status.")
    predicted_class: str = Field(
        ...,
        description="Most likely prediction returned by the model.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score associated with the predicted class.",
    )
    probabilities: dict[str, float] = Field(
        ...,
        description="Class probability distribution produced by the model.",
    )
    model_name: str = Field(
        ...,
        description="Human-readable name of the inference model.",
    )
    processing_time_ms: float = Field(
        ...,
        ge=0.0,
        description="End-to-end processing time measured in milliseconds.",
    )

