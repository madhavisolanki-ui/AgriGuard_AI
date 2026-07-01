"""Schemas for system metadata responses."""

from pydantic import BaseModel, Field


class RootResponse(BaseModel):
    """Response payload returned by the root endpoint."""

    status: str = Field(default="ok", description="Overall API status.")
    service: str = Field(default="AgriGuard", description="Service name.")
    version: str = Field(..., description="Application version.")
    docs: str = Field(..., description="Link to interactive API docs.")
    health: str = Field(..., description="Health check endpoint path.")
    predict: str = Field(..., description="Prediction endpoint path.")

