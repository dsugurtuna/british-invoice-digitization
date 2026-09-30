"""Request schemas for API endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class InferenceRequest(BaseModel):
    """Request schema for single image inference."""

    model_config = ConfigDict(extra="forbid")

    confidence_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Override default confidence threshold",
    )
    iou_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Override default IoU threshold for NMS",
    )
    extract_text: bool = Field(
        default=True,
        description="Whether to run OCR on detected regions",
    )
    include_visualization: bool = Field(
        default=False,
        description="Include annotated image in response",
    )
    output_format: str = Field(
        default="json",
        description="Output format: json, csv, or xml",
    )

    @field_validator("output_format")
    @classmethod
    def validate_output_format(cls, v: str) -> str:
        """Validate output format."""
        allowed = {"json", "csv", "xml"}
        if v.lower() not in allowed:
            msg = f"output_format must be one of: {allowed}"
            raise ValueError(msg)
        return v.lower()


class BatchProcessRequest(BaseModel):
    """Request schema for batch processing."""

    model_config = ConfigDict(extra="forbid")

    confidence_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Override default confidence threshold",
    )
    iou_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Override default IoU threshold for NMS",
    )
    extract_text: bool = Field(
        default=True,
        description="Whether to run OCR on detected regions",
    )
    include_visualizations: bool = Field(
        default=False,
        description="Include annotated images in response",
    )
    output_format: str = Field(
        default="json",
        description="Output format: json, csv, or xml",
    )
    max_concurrent: int | None = Field(
        default=None,
        ge=1,
        le=16,
        description="Maximum concurrent processing tasks",
    )
    callback_url: str | None = Field(
        default=None,
        description="Webhook URL for completion notification",
    )

    @field_validator("output_format")
    @classmethod
    def validate_output_format(cls, v: str) -> str:
        """Validate output format."""
        allowed = {"json", "csv", "xml"}
        if v.lower() not in allowed:
            msg = f"output_format must be one of: {allowed}"
            raise ValueError(msg)
        return v.lower()


class ModelConfigRequest(BaseModel):
    """Request schema for updating model configuration."""

    model_config = ConfigDict(extra="forbid")

    confidence_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Detection confidence threshold",
    )
    iou_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="NMS IoU threshold",
    )
    image_size: int | None = Field(
        default=None,
        ge=32,
        le=1280,
        description="Inference image size",
    )
    max_detections: int | None = Field(
        default=None,
        ge=1,
        le=1000,
        description="Maximum detections per image",
    )
    half_precision: bool | None = Field(
        default=None,
        description="Use FP16 inference (requires CUDA)",
    )

    def get_updates(self) -> dict[str, Any]:
        """Get non-None fields as dictionary."""
        return {k: v for k, v in self.model_dump().items() if v is not None}
