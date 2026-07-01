"""Inference service for AgriGuard."""

from __future__ import annotations

import base64
import binascii
import logging
import time
from functools import lru_cache
from typing import Final

import torch
from torch import Tensor, nn
from torchvision.io import ImageReadMode, decode_image

from app.schemas.predict_schema import PredictionResponse


LOGGER = logging.getLogger(__name__)
MODEL_NAME: Final[str] = "dummy-agriculture-classifier"
CLASS_LABELS: Final[tuple[str, ...]] = (
    "healthy_crop",
    "leaf_blight",
    "water_stress",
)


class InferenceError(Exception):
    """Raised when inference cannot be completed safely."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        """Initialize an inference error with HTTP semantics."""
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class DummyAgricultureModel(nn.Module):
    """Deterministic placeholder model that scores basic image statistics."""

    def forward(self, image_tensor: Tensor) -> Tensor:
        """Compute class logits from an RGB image tensor."""
        if image_tensor.ndim != 3 or image_tensor.shape[0] != 3:
            raise ValueError("Expected an RGB tensor with shape [3, H, W].")

        normalized = image_tensor.float() / 255.0
        channel_means = normalized.mean(dim=(1, 2))
        channel_stds = normalized.std(dim=(1, 2), unbiased=False)
        brightness = normalized.mean()
        contrast = normalized.var(unbiased=False)

        logits = torch.stack(
            (
                brightness + channel_means[1] - channel_means[2],
                contrast + channel_stds.mean(),
                (1.0 - brightness) + channel_means[0] - channel_means[1],
            )
        )
        return logits


class InferenceService:
    """Encapsulates model loading, preprocessing, and prediction logic."""

    def __init__(self) -> None:
        """Load the underlying model and prepare the inference service."""
        self._model = self._load_model()
        LOGGER.info("InferenceService initialized with model '%s'.", MODEL_NAME)

    def _load_model(self) -> nn.Module:
        """Load the placeholder PyTorch model."""
        try:
            model = DummyAgricultureModel()
            model.eval()
            return model
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.exception("Failed to initialize the inference model.")
            raise InferenceError("Unable to initialize the inference model.") from exc

    def _decode_base64_image(self, image_base64: str) -> bytes:
        """Decode a base64-encoded image string into raw bytes."""
        if not image_base64 or not image_base64.strip():
            raise InferenceError("image_base64 must not be empty.", status_code=422)

        try:
            return base64.b64decode(image_base64, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise InferenceError(
                "image_base64 must contain valid base64-encoded data.",
                status_code=400,
            ) from exc

    def _decode_image_tensor(self, image_bytes: bytes) -> Tensor:
        """Convert raw image bytes into a PyTorch tensor."""
        try:
            encoded = torch.frombuffer(memoryview(image_bytes), dtype=torch.uint8).clone()
            return decode_image(encoded, mode=ImageReadMode.RGB)
        except Exception as exc:
            LOGGER.exception("Failed to decode image payload.")
            raise InferenceError(
                "The submitted image could not be decoded.",
                status_code=400,
            ) from exc

    def predict(self, image_base64: str) -> PredictionResponse:
        """Run inference for a base64-encoded image payload."""
        start_time = time.perf_counter()

        try:
            image_bytes = self._decode_base64_image(image_base64)
            image_tensor = self._decode_image_tensor(image_bytes)

            with torch.no_grad():
                logits = self._model(image_tensor)
                probabilities_tensor = torch.softmax(logits, dim=0)

            predicted_index = int(torch.argmax(probabilities_tensor).item())
            probabilities = {
                label: float(probabilities_tensor[index].item())
                for index, label in enumerate(CLASS_LABELS)
            }
            confidence = probabilities[CLASS_LABELS[predicted_index]]
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            LOGGER.info(
                "Prediction completed for model '%s' with class '%s'.",
                MODEL_NAME,
                CLASS_LABELS[predicted_index],
            )

            return PredictionResponse(
                predicted_class=CLASS_LABELS[predicted_index],
                confidence=confidence,
                probabilities=probabilities,
                model_name=MODEL_NAME,
                processing_time_ms=elapsed_ms,
            )
        except InferenceError:
            raise
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.exception("Unexpected inference failure.")
            raise InferenceError(
                "An unexpected error occurred while running inference.",
            ) from exc


@lru_cache(maxsize=1)
def get_inference_service() -> InferenceService:
    """Return a cached inference service instance."""
    return InferenceService()

