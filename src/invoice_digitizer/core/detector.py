"""Turn raw model boxes into validated invoice fields."""

from __future__ import annotations

import time

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog

from PIL import Image, ImageDraw

from invoice_digitizer.config.settings import get_settings
from invoice_digitizer.core.images import ImageSource, load_image
from invoice_digitizer.core.model_manager import ModelManager
from invoice_digitizer.schemas.detection import (
    BoundingBox,
    DetectionResult,
    InvoiceField,
    InvoiceFieldType,
    ProcessingMetadata,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from invoice_digitizer.config.settings import Settings
    from invoice_digitizer.core.yolov5_backend import RawDetection

logger = structlog.get_logger(__name__)

# Class-name spellings seen in YOLO label files, mapped to the canonical field type.
_ALIASES: dict[str, InvoiceFieldType] = {
    "invoice_date": InvoiceFieldType.INVOICE_DATE,
    "date": InvoiceFieldType.INVOICE_DATE,
    "invoice_number": InvoiceFieldType.INVOICE_NUMBER,
    "number": InvoiceFieldType.INVOICE_NUMBER,
    "vendor_name": InvoiceFieldType.VENDOR_NAME,
    "vendor": InvoiceFieldType.VENDOR_NAME,
    "total_amount": InvoiceFieldType.TOTAL_AMOUNT,
    "total": InvoiceFieldType.TOTAL_AMOUNT,
    "vat_amount": InvoiceFieldType.VAT_AMOUNT,
    "vat": InvoiceFieldType.VAT_AMOUNT,
    "line_item": InvoiceFieldType.LINE_ITEM,
    "item": InvoiceFieldType.LINE_ITEM,
}

_COLOURS: dict[InvoiceFieldType, tuple[int, int, int]] = {
    InvoiceFieldType.INVOICE_DATE: (0, 158, 115),
    InvoiceFieldType.INVOICE_NUMBER: (0, 114, 178),
    InvoiceFieldType.VENDOR_NAME: (213, 94, 0),
    InvoiceFieldType.TOTAL_AMOUNT: (204, 121, 167),
    InvoiceFieldType.VAT_AMOUNT: (230, 159, 0),
    InvoiceFieldType.LINE_ITEM: (86, 180, 233),
}


def map_label(class_name: str) -> InvoiceFieldType | None:
    """Map a model class name to a field type, or None if it is not an invoice field."""
    try:
        return InvoiceFieldType(class_name)
    except ValueError:
        pass
    normalised = class_name.strip().lower().replace(" ", "_").replace("-", "_")
    return _ALIASES.get(normalised)


def to_fields(raw: list[RawDetection], width: int, height: int) -> tuple[list[InvoiceField], int]:
    """Convert raw boxes to fields, clipped to the image, highest confidence first.

    Returns:
        The fields and how many raw boxes were dropped (unknown class, or no area left
        after clipping). The count is reported so a silent failure, such as serving a
        model trained on other classes, shows up in every response.
    """
    fields: list[InvoiceField] = []
    ignored = 0
    for box in raw:
        label = map_label(box.class_name)
        x_min, x_max = max(0.0, box.x_min), min(float(width), box.x_max)
        y_min, y_max = max(0.0, box.y_min), min(float(height), box.y_max)
        if label is None or x_max <= x_min or y_max <= y_min:
            ignored += 1
            continue
        fields.append(
            InvoiceField(
                label=label,
                confidence=min(max(box.confidence, 0.0), 1.0),
                bounding_box=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
            )
        )
    fields.sort(key=lambda f: f.confidence, reverse=True)
    return fields, ignored


class InvoiceFieldDetector:
    """Runs the model on one image and returns a validated ``DetectionResult``.

    Args:
        settings: Application settings. Defaults to ``get_settings()``.
        model_manager: Shared model owner. Pass one in so several detectors (or the
            API and a batch job) use a single model in memory.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        model_manager: ModelManager | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._model_manager = model_manager or ModelManager(self._settings.model)

    @property
    def model_manager(self) -> ModelManager:
        """The model owner used by this detector."""
        return self._model_manager

    def detect(
        self,
        image_source: ImageSource,
        confidence_threshold: float | None = None,
        iou_threshold: float | None = None,
        source_name: str | None = None,
    ) -> DetectionResult:
        """Detect invoice fields in one image.

        Args:
            image_source: File path, or an RGB (or grey-scale) uint8 array.
            confidence_threshold: Override for this call only.
            iou_threshold: Override for this call only.
            source_name: Label recorded in the metadata instead of the default.

        Raises:
            FileNotFoundError: The file does not exist.
            ImageValidationError: The input is not an acceptable image.
        """
        start = time.perf_counter()
        image, default_name = load_image(image_source, self._settings.preprocessing)
        height, width = image.shape[:2]

        raw, loaded = self._model_manager.predict(
            image, confidence=confidence_threshold, iou=iou_threshold
        )
        fields, ignored = to_fields(raw, width, height)

        metadata = ProcessingMetadata(
            processing_time_ms=(time.perf_counter() - start) * 1000,
            model_version=loaded.version,
            device=loaded.device,
            image_width=width,
            image_height=height,
            image_source=source_name or default_name,
            ignored_detections=ignored,
        )
        result = DetectionResult(metadata=metadata, detections=fields)
        logger.info(
            "detection_complete",
            detections=result.detection_count,
            ignored=ignored,
            processing_time_ms=round(metadata.processing_time_ms, 1),
        )
        return result

    def detect_batch(
        self,
        image_sources: list[ImageSource],
        confidence_threshold: float | None = None,
        iou_threshold: float | None = None,
    ) -> list[DetectionResult]:
        """Detect fields in several images, one after another.

        A failing image produces a result with ``error`` set instead of stopping the batch.
        """
        results: list[DetectionResult] = []
        for source in image_sources:
            try:
                results.append(self.detect(source, confidence_threshold, iou_threshold))
            except Exception as exc:  # one bad file must not sink the batch
                logger.warning("batch_item_failed", error_type=type(exc).__name__)
                results.append(self.error_result(source, exc))
        return results

    def error_result(self, source: ImageSource, error: BaseException | str) -> DetectionResult:
        """Build a result that records a failure without loading the model."""
        loaded = self._model_manager.info()
        name = "memory_buffer" if isinstance(source, np.ndarray) else Path(source).name
        metadata = ProcessingMetadata(
            processing_time_ms=0.0,
            model_version=loaded["version"] or "not loaded",
            device=loaded["device"] or "n/a",
            image_width=0,
            image_height=0,
            image_source=name,
        )
        return DetectionResult(metadata=metadata, detections=[], error=str(error))

    def visualize(
        self,
        image_source: ImageSource,
        result: DetectionResult,
        output_path: str | Path | None = None,
        show_confidence: bool = True,
        line_width: int = 3,
    ) -> NDArray[np.uint8]:
        """Draw the detected boxes on the image.

        Returns:
            The annotated image as an RGB array. Also saved when ``output_path`` is given.
        """
        image, _ = load_image(image_source, self._settings.preprocessing)
        canvas = Image.fromarray(image)
        draw = ImageDraw.Draw(canvas)
        for field in result.detections:
            colour = _COLOURS.get(field.label, (128, 128, 128))
            box = field.bounding_box.to_xyxy()
            draw.rectangle(box, outline=colour, width=line_width)
            label = field.label.value
            if show_confidence:
                label = f"{label} {field.confidence:.0%}"
            text_box = draw.textbbox((box[0], box[1]), label)
            text_height = text_box[3] - text_box[1]
            top = max(0.0, box[1] - text_height - 4)
            draw.rectangle(
                (box[0], top, box[0] + (text_box[2] - text_box[0]) + 4, top + text_height + 4),
                fill=colour,
            )
            draw.text((box[0] + 2, top + 2), label, fill=(255, 255, 255))
        if output_path is not None:
            canvas.save(output_path)
            logger.info("annotated_image_saved", path=str(output_path))
        return np.asarray(canvas, dtype=np.uint8)

    def get_model_info(self) -> dict[str, Any]:
        """Model description, without forcing a load."""
        return self._model_manager.info()
