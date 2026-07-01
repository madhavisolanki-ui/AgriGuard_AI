"""Inference service for AgriGuard."""

from __future__ import annotations

import base64
import io
import binascii
from dataclasses import dataclass
import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import Final

import torch
from PIL import Image
from torch import Tensor, nn
from torchvision.io import ImageReadMode, decode_image

from app.core.config import settings
from app.schemas.predict_schema import PredictionResponse


LOGGER = logging.getLogger(__name__)
MODEL_NAME: Final[str] = "dummy-agriculture-classifier"
CLASS_LABELS: Final[tuple[str, ...]] = (
    "healthy_crop",
    "leaf_blight",
    "water_stress",
)


@dataclass(frozen=True, slots=True)
class InferenceContext:
    """Execution context used to enrich prediction logs."""

    filename: str | None = None
    content_type: str | None = None


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
    """Production-grade service handling ML inference lifecycle."""

    def __init__(self) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._load_model()
        
        # Preprocessing: Essential for MobileNetV3
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        LOGGER.info("InferenceService initialized on %s with model '%s'.", self._device, MODEL_NAME)

    def _load_model(self) -> nn.Module:
        """Load the placeholder PyTorch model."""
        model = DummyAgricultureModel().to(self._device)
        model_path = settings.model_path

        if model_path.exists():
            self._load_checkpoint(model, model_path)
        else:
            LOGGER.info(
                "Model checkpoint not found at '%s'; using built-in dummy model.",
                model_path,
            )

        model.eval()
        return model

    def _load_checkpoint(self, model: nn.Module, model_path: Path) -> None:
        """Load a checkpoint into the model with defensive validation."""
        try:
            checkpoint = torch.load(model_path, map_location=self._device)
            state_dict = self._extract_state_dict(checkpoint)
            missing_keys, unexpected_keys = model.load_state_dict(
                state_dict,
                strict=False,
            )
            if missing_keys:
                LOGGER.warning(
                    "Checkpoint at '%s' is missing keys: %s",
                    model_path,
                    sorted(missing_keys),
                )
            if unexpected_keys:
                LOGGER.warning(
                    "Checkpoint at '%s' has unexpected keys: %s",
                    model_path,
                    sorted(unexpected_keys),
                )
            LOGGER.info("Loaded model weights from '%s'.", model_path)
        except FileNotFoundError as exc:
            raise InferenceError(
                f"Model checkpoint not found at '{model_path}'.",
                status_code=500,
            ) from exc
        except (RuntimeError, TypeError, ValueError) as exc:
            LOGGER.exception("Invalid checkpoint format at '%s'.", model_path)
            raise InferenceError(
                f"Invalid model checkpoint at '{model_path}'.",
                status_code=500,
            ) from exc
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.exception("Failed to initialize the inference model.")
            raise InferenceError(
                "Unable to initialize the inference model.",
            ) from exc

    def _extract_state_dict(self, checkpoint: object) -> dict[str, Tensor]:
        """Extract a state dictionary from a checkpoint payload."""
        if isinstance(checkpoint, dict):
            candidate = checkpoint.get("state_dict", checkpoint)
            if isinstance(candidate, dict):
                return candidate

        raise ValueError("Checkpoint did not contain a valid state dictionary.")

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
        if not image_bytes:
            raise InferenceError("Uploaded image is empty.", status_code=422)

        try:
            encoded = torch.frombuffer(memoryview(image_bytes), dtype=torch.uint8).clone()
            return decode_image(encoded, mode=ImageReadMode.RGB)
        except RuntimeError as exc:
            LOGGER.exception("Failed to decode image payload.")
            raise InferenceError(
                "The submitted image could not be decoded.",
                status_code=400,
            ) from exc
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.exception("Unexpected failure while decoding image payload.")
            raise InferenceError(
                "The submitted image could not be decoded.",
                status_code=400,
            ) from exc

    def _build_context(
        self,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> InferenceContext:
        """Build a structured context object for logging."""
        return InferenceContext(filename=filename, content_type=content_type)

    def _log_prediction_start(self, context: InferenceContext) -> None:
        """Log the beginning of an inference request."""
        LOGGER.info(
            "Starting prediction request for model '%s' (filename=%s, content_type=%s).",
            MODEL_NAME,
            context.filename,
            context.content_type,
        )
        LOGGER.debug(
            "Prediction request metadata captured.",
            extra={
                "model_name": MODEL_NAME,
                "filename": context.filename,
                "content_type": context.content_type,
            },
        )

    def predict(
        self,
        image_base64: str,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> PredictionResponse:
        """Run inference with proper preprocessing and confidence thresholding."""
        start_time = time.perf_counter()
        context = self._build_context(filename=filename, content_type=content_type)
        self._log_prediction_start(context)

        try:
            img_bytes = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(img_bytes)).convert('RGB')

            input_tensor = self.preprocess(image).unsqueeze(0).to(self._device)
            
            with torch.no_grad():
                logits = self._model(input_tensor)
                probs = torch.nn.functional.softmax(logits, dim=1).squeeze()

                max_conf, predicted_idx = torch.max(probs, 0)
                threshold = 0.50

            predicted_class = CLASS_LABELS[predicted_idx.item()] if max_conf > threshold else "uncertain"
            
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
                predicted_class=predicted_class,
                confidence=float(max_conf),
                probabilities={label: float(probs[i]) for i, label in enumerate(CLASS_LABELS)},
                model_name=MODEL_NAME,
                processing_time_ms=elapsed_ms,
            )
        except InferenceError:
            raise
        except Exception as exc:
            LOGGER.error("Inference execution error: %s", exc)
            raise InferenceError("Prediction pipeline failed.") from exc


@lru_cache(maxsize=1)
def get_inference_service() -> InferenceService:
    """Return a cached inference service instance."""
    return InferenceService()
