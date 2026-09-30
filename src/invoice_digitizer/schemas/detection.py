"""Detection result schemas."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

HIGH_CONFIDENCE = 0.8
REVIEW_BELOW = 0.6


class InvoiceFieldType(StrEnum):
    """The six field types the model is trained to locate."""

    INVOICE_DATE = "Invoice Date"
    INVOICE_NUMBER = "Invoice Number"
    VENDOR_NAME = "Vendor Name"
    TOTAL_AMOUNT = "Total Amount"
    VAT_AMOUNT = "VAT Amount"
    LINE_ITEM = "Line Item"


def _utc_now() -> datetime:
    return datetime.now(UTC)


class BoundingBox(BaseModel):
    """Box in pixel coordinates of the original image, XYXY order."""

    model_config = ConfigDict(frozen=True)

    x_min: float = Field(..., ge=0, description="Left edge")
    y_min: float = Field(..., ge=0, description="Top edge")
    x_max: float = Field(..., ge=0, description="Right edge")
    y_max: float = Field(..., ge=0, description="Bottom edge")

    @field_validator("x_max")
    @classmethod
    def _x_max_after_x_min(cls, value: float, info: ValidationInfo) -> float:
        if "x_min" in info.data and value <= info.data["x_min"]:
            raise ValueError("x_max must be greater than x_min")
        return value

    @field_validator("y_max")
    @classmethod
    def _y_max_after_y_min(cls, value: float, info: ValidationInfo) -> float:
        if "y_min" in info.data and value <= info.data["y_min"]:
            raise ValueError("y_max must be greater than y_min")
        return value

    @property
    def width(self) -> float:
        """Box width."""
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        """Box height."""
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        """Box area."""
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        """Centre point."""
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)

    def to_xywh(self) -> tuple[float, float, float, float]:
        """Centre x, centre y, width, height."""
        cx, cy = self.center
        return (cx, cy, self.width, self.height)

    def to_xyxy(self) -> tuple[float, float, float, float]:
        """x_min, y_min, x_max, y_max."""
        return (self.x_min, self.y_min, self.x_max, self.y_max)

    def iou(self, other: BoundingBox) -> float:
        """Intersection over union with another box."""
        x_left = max(self.x_min, other.x_min)
        y_top = max(self.y_min, other.y_min)
        x_right = min(self.x_max, other.x_max)
        y_bottom = min(self.y_max, other.y_max)
        if x_right <= x_left or y_bottom <= y_top:
            return 0.0
        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0


class InvoiceField(BaseModel):
    """One detected field."""

    model_config = ConfigDict(populate_by_name=True)

    field_id: UUID = Field(default_factory=uuid4)
    label: InvoiceFieldType
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detector confidence")
    bounding_box: BoundingBox
    extracted_text: str | None = Field(
        default=None,
        description="Reserved for an OCR step. This service does not run OCR, so it is null.",
    )
    ocr_confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @property
    def is_high_confidence(self) -> bool:
        """Confidence of at least 0.8."""
        return self.confidence >= HIGH_CONFIDENCE

    @property
    def needs_review(self) -> bool:
        """Confidence below 0.6: worth a human look.

        The 0.6 and 0.8 cut-offs are conventions, not calibrated values. Detector
        confidence is not a probability of being correct unless it has been calibrated
        on held-out data.
        """
        return self.confidence < REVIEW_BELOW


class ProcessingMetadata(BaseModel):
    """What was processed, how, and with which model."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    request_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=_utc_now)
    processing_time_ms: float = Field(..., ge=0)
    model_version: str = Field(..., description="Hub repo and weights fingerprint")
    device: str
    image_width: int = Field(..., ge=0, description="0 when the image could not be read")
    image_height: int = Field(..., ge=0, description="0 when the image could not be read")
    image_source: str = Field(..., description="File name, 'upload' or 'memory_buffer'")
    ignored_detections: int = Field(
        default=0,
        ge=0,
        description=(
            "Boxes the model returned that were dropped: unknown class names or boxes "
            "with no area after clipping to the image."
        ),
    )


class DetectionResult(BaseModel):
    """All fields found on one image, or the reason it failed."""

    model_config = ConfigDict(populate_by_name=True)

    metadata: ProcessingMetadata
    detections: list[InvoiceField] = Field(default_factory=list)
    error: str | None = Field(default=None, description="Set when processing failed")

    @property
    def succeeded(self) -> bool:
        """True when the image was processed, even if nothing was found."""
        return self.error is None

    @property
    def detection_count(self) -> int:
        """Number of detected fields."""
        return len(self.detections)

    @property
    def high_confidence_count(self) -> int:
        """Number of detections with confidence of at least 0.8."""
        return sum(1 for d in self.detections if d.is_high_confidence)

    @property
    def fields_by_type(self) -> dict[InvoiceFieldType, list[InvoiceField]]:
        """Detections grouped by field type."""
        grouped: dict[InvoiceFieldType, list[InvoiceField]] = {}
        for detection in self.detections:
            grouped.setdefault(detection.label, []).append(detection)
        return grouped

    def get_field(self, field_type: InvoiceFieldType) -> InvoiceField | None:
        """Highest-confidence detection of one field type, if any."""
        fields = [d for d in self.detections if d.label == field_type]
        return max(fields, key=lambda d: d.confidence) if fields else None

    def to_flat_dict(self) -> dict[str, Any]:
        """One flat row per image, for CSV export."""
        row: dict[str, Any] = {
            "request_id": str(self.metadata.request_id),
            "timestamp": self.metadata.timestamp.isoformat(),
            "processing_time_ms": self.metadata.processing_time_ms,
            "detection_count": self.detection_count,
            "error": self.error,
        }
        for field_type in InvoiceFieldType:
            best = self.get_field(field_type)
            prefix = field_type.value.lower().replace(" ", "_")
            row[f"{prefix}_confidence"] = best.confidence if best else None
            row[f"{prefix}_text"] = best.extracted_text if best else None
        return row
