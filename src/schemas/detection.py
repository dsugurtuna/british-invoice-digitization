"""Detection result schemas."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class InvoiceFieldType(str, Enum):
    """Invoice field types detected by the model."""

    INVOICE_DATE = "Invoice Date"
    INVOICE_NUMBER = "Invoice Number"
    VENDOR_NAME = "Vendor Name"
    TOTAL_AMOUNT = "Total Amount"
    VAT_AMOUNT = "VAT Amount"
    LINE_ITEM = "Line Item"


class BoundingBox(BaseModel):
    """Bounding box coordinates for a detected field.

    Coordinates are in XYXY format (top-left x, top-left y, bottom-right x, bottom-right y).
    All values are in pixels relative to the original image dimensions.
    """

    model_config = ConfigDict(frozen=True)

    x_min: float = Field(..., ge=0, description="Left edge X coordinate")
    y_min: float = Field(..., ge=0, description="Top edge Y coordinate")
    x_max: float = Field(..., ge=0, description="Right edge X coordinate")
    y_max: float = Field(..., ge=0, description="Bottom edge Y coordinate")

    @field_validator("x_max")
    @classmethod
    def x_max_greater_than_x_min(cls, v: float, info: Any) -> float:
        """Ensure x_max > x_min."""
        if "x_min" in info.data and v <= info.data["x_min"]:
            msg = "x_max must be greater than x_min"
            raise ValueError(msg)
        return v

    @field_validator("y_max")
    @classmethod
    def y_max_greater_than_y_min(cls, v: float, info: Any) -> float:
        """Ensure y_max > y_min."""
        if "y_min" in info.data and v <= info.data["y_min"]:
            msg = "y_max must be greater than y_min"
            raise ValueError(msg)
        return v

    @property
    def width(self) -> float:
        """Calculate bounding box width."""
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        """Calculate bounding box height."""
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        """Calculate bounding box area."""
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        """Calculate center point of bounding box."""
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)

    def to_xywh(self) -> tuple[float, float, float, float]:
        """Convert to XYWH format (center x, center y, width, height)."""
        cx, cy = self.center
        return (cx, cy, self.width, self.height)

    def to_xyxy(self) -> tuple[float, float, float, float]:
        """Return as XYXY tuple."""
        return (self.x_min, self.y_min, self.x_max, self.y_max)

    def iou(self, other: "BoundingBox") -> float:
        """Calculate Intersection over Union with another bounding box."""
        # Calculate intersection
        x_inter_min = max(self.x_min, other.x_min)
        y_inter_min = max(self.y_min, other.y_min)
        x_inter_max = min(self.x_max, other.x_max)
        y_inter_max = min(self.y_max, other.y_max)

        if x_inter_max <= x_inter_min or y_inter_max <= y_inter_min:
            return 0.0

        intersection = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
        union = self.area + other.area - intersection

        return intersection / union if union > 0 else 0.0


class InvoiceField(BaseModel):
    """A detected invoice field with its properties."""

    model_config = ConfigDict(populate_by_name=True)

    field_id: UUID = Field(default_factory=uuid4, description="Unique field identifier")
    label: InvoiceFieldType = Field(..., description="Type of invoice field detected")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    bounding_box: BoundingBox = Field(..., description="Field location in image")
    extracted_text: str | None = Field(
        default=None, description="OCR-extracted text from the field region"
    )
    ocr_confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="OCR extraction confidence"
    )

    @property
    def is_high_confidence(self) -> bool:
        """Check if detection has high confidence (>= 0.8)."""
        return self.confidence >= 0.8

    @property
    def needs_review(self) -> bool:
        """Check if detection needs manual review (confidence < 0.6)."""
        return self.confidence < 0.6


class ProcessingMetadata(BaseModel):
    """Metadata about the processing operation."""

    model_config = ConfigDict(frozen=True)

    request_id: UUID = Field(default_factory=uuid4, description="Unique request identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Processing timestamp")
    processing_time_ms: float = Field(..., ge=0, description="Total processing time in milliseconds")
    model_version: str = Field(..., description="Model version used for inference")
    device: str = Field(..., description="Compute device used (cpu/cuda)")
    image_width: int = Field(..., ge=1, description="Original image width in pixels")
    image_height: int = Field(..., ge=1, description="Original image height in pixels")
    image_source: str = Field(..., description="Image source identifier or filename")


class DetectionResult(BaseModel):
    """Complete detection result for an invoice image."""

    model_config = ConfigDict(populate_by_name=True)

    metadata: ProcessingMetadata = Field(..., description="Processing metadata")
    detections: list[InvoiceField] = Field(
        default_factory=list, description="List of detected fields"
    )
    raw_image_path: str | None = Field(
        default=None, description="Path to the processed image (if stored)"
    )
    annotated_image_path: str | None = Field(
        default=None, description="Path to annotated visualization"
    )

    @property
    def detection_count(self) -> int:
        """Get total number of detections."""
        return len(self.detections)

    @property
    def high_confidence_count(self) -> int:
        """Get count of high-confidence detections."""
        return sum(1 for d in self.detections if d.is_high_confidence)

    @property
    def fields_by_type(self) -> dict[InvoiceFieldType, list[InvoiceField]]:
        """Group detections by field type."""
        result: dict[InvoiceFieldType, list[InvoiceField]] = {}
        for detection in self.detections:
            if detection.label not in result:
                result[detection.label] = []
            result[detection.label].append(detection)
        return result

    def get_field(self, field_type: InvoiceFieldType) -> InvoiceField | None:
        """Get the highest confidence detection for a specific field type."""
        fields = [d for d in self.detections if d.label == field_type]
        if not fields:
            return None
        return max(fields, key=lambda x: x.confidence)

    def to_flat_dict(self) -> dict[str, Any]:
        """Convert to flat dictionary for easy export (CSV, etc.)."""
        result: dict[str, Any] = {
            "request_id": str(self.metadata.request_id),
            "timestamp": self.metadata.timestamp.isoformat(),
            "processing_time_ms": self.metadata.processing_time_ms,
            "detection_count": self.detection_count,
        }

        # Add best detection for each field type
        for field_type in InvoiceFieldType:
            field = self.get_field(field_type)
            key_prefix = field_type.value.lower().replace(" ", "_")
            if field:
                result[f"{key_prefix}_confidence"] = field.confidence
                result[f"{key_prefix}_text"] = field.extracted_text
            else:
                result[f"{key_prefix}_confidence"] = None
                result[f"{key_prefix}_text"] = None

        return result
