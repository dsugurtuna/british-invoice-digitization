"""Pydantic schemas for data validation and serialization."""

from invoice_digitizer.schemas.detection import (
    BoundingBox,
    DetectionResult,
    InvoiceField,
    ProcessingMetadata,
)
from invoice_digitizer.schemas.request import (
    BatchProcessRequest,
    InferenceRequest,
    ModelConfigRequest,
)
from invoice_digitizer.schemas.response import (
    APIResponse,
    BatchProcessResponse,
    ErrorResponse,
    HealthResponse,
    InferenceResponse,
)

__all__ = [
    # Detection schemas
    "BoundingBox",
    "DetectionResult",
    "InvoiceField",
    "ProcessingMetadata",
    # Request schemas
    "BatchProcessRequest",
    "InferenceRequest",
    "ModelConfigRequest",
    # Response schemas
    "APIResponse",
    "BatchProcessResponse",
    "ErrorResponse",
    "HealthResponse",
    "InferenceResponse",
]
