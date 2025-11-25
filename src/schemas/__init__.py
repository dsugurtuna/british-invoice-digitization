"""Pydantic schemas for data validation and serialization."""

from src.schemas.detection import (
    BoundingBox,
    DetectionResult,
    InvoiceField,
    ProcessingMetadata,
)
from src.schemas.request import (
    BatchProcessRequest,
    InferenceRequest,
    ModelConfigRequest,
)
from src.schemas.response import (
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
