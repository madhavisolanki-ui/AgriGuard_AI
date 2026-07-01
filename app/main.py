"""FastAPI application entry point for AgriGuard."""

from contextlib import asynccontextmanager
import logging
import time
from collections.abc import Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.config import settings
from app.core.logging import configure_logging
from app.services.inference_service import InferenceError

LOGGER = logging.getLogger(__name__)


configure_logging(settings.log_level)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Manage application startup and shutdown events."""
    LOGGER.info("Starting %s %s", settings.app_name, settings.app_version)
    yield
    LOGGER.info("Shutting down %s", settings.app_name)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
        description=settings.app_description,
        docs_url=settings.docs_url,
        redoc_url=settings.redoc_url,
        openapi_url=settings.openapi_url,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)

    @app.middleware("http")
    async def add_request_metrics(
        request: Request,
        call_next: Callable[[Request], Response],
    ) -> Response:
        """Attach simple request timing metadata to responses."""
        start_time = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        response.headers["X-Process-Time-ms"] = f"{elapsed_ms:.2f}"
        return response

    @app.exception_handler(InferenceError)
    async def inference_error_handler(
        request: Request,
        exc: InferenceError,
    ) -> JSONResponse:
        """Convert inference failures into consistent JSON responses."""
        LOGGER.warning(
            "Inference error on %s %s: %s",
            request.method,
            request.url.path,
            exc.message,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.message},
        )

    return app


app = create_app()
