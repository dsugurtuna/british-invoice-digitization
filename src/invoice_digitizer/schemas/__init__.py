"""Pydantic schemas for results, requests and responses."""

from invoice_digitizer.schemas.detection import (
    BoundingBox,
    DetectionResult,
    InvoiceField,
    InvoiceFieldType,
    ProcessingMetadata,
)
from invoice_digitizer.schemas.request import ModelConfigRequest
from invoice_digitizer.schemas.response import (
    BatchJobStatus,
    BatchProcessResponse,
    ComponentHealth,
    ErrorDetail,
    HealthResponse,
    HealthStatus,
    InferenceResponse,
    ModelConfigResponse,
    ResponseStatus,
)

__all__ = [
    "BatchJobStatus",
    "BatchProcessResponse",
    "BoundingBox",
    "ComponentHealth",
    "DetectionResult",
    "ErrorDetail",
    "HealthResponse",
    "HealthStatus",
    "InferenceResponse",
    "InvoiceField",
    "InvoiceFieldType",
    "ModelConfigRequest",
    "ModelConfigResponse",
    "ProcessingMetadata",
    "ResponseStatus",
]
