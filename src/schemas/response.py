"""Response schemas for API endpoints."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Generic, TypeVar
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from src.schemas.detection import DetectionResult

T = TypeVar("T")


class ResponseStatus(str, Enum):
    """API response status."""

    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"


class APIResponse(BaseModel, Generic[T]):
    """Generic API response wrapper."""

    model_config = ConfigDict(populate_by_name=True)

    status: ResponseStatus = Field(..., description="Response status")
    request_id: UUID = Field(default_factory=uuid4, description="Unique request ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    data: T | None = Field(default=None, description="Response payload")
    message: str | None = Field(default=None, description="Human-readable message")


class ErrorDetail(BaseModel):
    """Detailed error information."""

    model_config = ConfigDict(frozen=True)

    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    field: str | None = Field(default=None, description="Field that caused the error")
    details: dict[str, Any] | None = Field(default=None, description="Additional error details")


class ErrorResponse(BaseModel):
    """Error response schema."""

    model_config = ConfigDict(populate_by_name=True)

    status: ResponseStatus = Field(default=ResponseStatus.ERROR)
    request_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    error: ErrorDetail = Field(..., description="Error details")
    trace_id: str | None = Field(default=None, description="Trace ID for debugging")


class InferenceResponse(BaseModel):
    """Response schema for inference endpoint."""

    model_config = ConfigDict(populate_by_name=True)

    status: ResponseStatus = Field(default=ResponseStatus.SUCCESS)
    request_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    result: DetectionResult = Field(..., description="Detection results")
    visualization_url: str | None = Field(
        default=None, description="URL to annotated image if requested"
    )


class BatchJobStatus(str, Enum):
    """Batch job status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class BatchProcessResponse(BaseModel):
    """Response schema for batch processing."""

    model_config = ConfigDict(populate_by_name=True)

    status: ResponseStatus = Field(default=ResponseStatus.SUCCESS)
    request_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    job_id: UUID = Field(default_factory=uuid4, description="Batch job ID")
    job_status: BatchJobStatus = Field(..., description="Current job status")
    total_images: int = Field(..., ge=0, description="Total images in batch")
    processed_images: int = Field(default=0, ge=0, description="Successfully processed images")
    failed_images: int = Field(default=0, ge=0, description="Failed images")
    results: list[DetectionResult] | None = Field(
        default=None, description="Detection results (when completed)"
    )
    errors: list[ErrorDetail] | None = Field(
        default=None, description="Processing errors (if any)"
    )
    download_url: str | None = Field(
        default=None, description="URL to download results (when completed)"
    )

    @property
    def progress_percentage(self) -> float:
        """Calculate processing progress."""
        if self.total_images == 0:
            return 100.0
        return (self.processed_images + self.failed_images) / self.total_images * 100


class HealthStatus(str, Enum):
    """Health check status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth(BaseModel):
    """Health status for a single component."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Component name")
    status: HealthStatus = Field(..., description="Component health status")
    latency_ms: float | None = Field(default=None, description="Response latency")
    message: str | None = Field(default=None, description="Status message")


class HealthResponse(BaseModel):
    """Health check response."""

    model_config = ConfigDict(populate_by_name=True)

    status: HealthStatus = Field(..., description="Overall health status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(..., description="Application version")
    uptime_seconds: float = Field(..., ge=0, description="Application uptime")
    components: list[ComponentHealth] = Field(
        default_factory=list, description="Individual component health"
    )

    @property
    def is_healthy(self) -> bool:
        """Check if all components are healthy."""
        return self.status == HealthStatus.HEALTHY and all(
            c.status == HealthStatus.HEALTHY for c in self.components
        )
