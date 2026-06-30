"""Response schema for health-check endpoints."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Health response returned by the API."""

    status: str
    service: str

