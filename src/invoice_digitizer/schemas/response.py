"""Response bodies for the API."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from invoice_digitizer.schemas.detection import DetectionResult


def _utc_now() -> datetime:
    return datetime.now(UTC)


class ResponseStatus(StrEnum):
    """Outcome of a request."""

    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"


class ErrorDetail(BaseModel):
    """One problem, for example a file in a batch that could not be processed."""

    model_config = ConfigDict(frozen=True)

    code: str
    message: str
    field: str | None = Field(default=None, description="File name or field involved")
    details: dict[str, Any] | None = None


class InferenceResponse(BaseModel):
    """Result for a single uploaded image."""

    status: ResponseStatus = ResponseStatus.SUCCESS
    request_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=_utc_now)
    result: DetectionResult


class BatchJobStatus(StrEnum):
    """Outcome of a batch. Batches run synchronously, so there is no pending state."""

    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class BatchProcessResponse(BaseModel):
    """Results for a batch of uploaded images."""

    status: ResponseStatus = ResponseStatus.SUCCESS
    request_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=_utc_now)
    job_status: BatchJobStatus
    total_images: int = Field(..., ge=0)
    processed_images: int = Field(default=0, ge=0)
    failed_images: int = Field(default=0, ge=0)
    results: list[DetectionResult] = Field(default_factory=list)
    errors: list[ErrorDetail] = Field(default_factory=list)


class ModelConfigResponse(BaseModel):
    """Thresholds in force after an update."""

    message: str
    thresholds: dict[str, float | int]


class HealthStatus(StrEnum):
    """Health of the service or one component."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth(BaseModel):
    """Health of one component."""

    model_config = ConfigDict(frozen=True)

    name: str
    status: HealthStatus
    latency_ms: float | None = None
    message: str | None = None


class HealthResponse(BaseModel):
    """Overall health."""

    status: HealthStatus
    timestamp: datetime = Field(default_factory=_utc_now)
    version: str
    uptime_seconds: float = Field(..., ge=0)
    components: list[ComponentHealth] = Field(default_factory=list)

    @property
    def is_healthy(self) -> bool:
        """True when the service and every component are healthy."""
        return self.status == HealthStatus.HEALTHY and all(
            c.status == HealthStatus.HEALTHY for c in self.components
        )
