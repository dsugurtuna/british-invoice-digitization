"""Request bodies for the API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ModelConfigRequest(BaseModel):
    """Runtime threshold changes. Only these three settings can change without a restart."""

    model_config = ConfigDict(extra="forbid")

    confidence_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    iou_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    max_detections: int | None = Field(default=None, ge=1, le=1000)

    def get_updates(self) -> dict[str, Any]:
        """The fields that were actually provided."""
        return self.model_dump(exclude_none=True)
