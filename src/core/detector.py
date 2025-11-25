"""Invoice field detection using YOLOv5."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import structlog
import torch

from src.core.model_manager import ModelManager
from src.schemas.detection import (
    BoundingBox,
    DetectionResult,
    InvoiceField,
    InvoiceFieldType,
    ProcessingMetadata,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.config.settings import Settings

logger = structlog.get_logger(__name__)


class InvoiceFieldDetector:
    """High-performance invoice field detection using YOLOv5.

    This class provides the core detection functionality, wrapping the
    YOLOv5 model with invoice-specific preprocessing and postprocessing.

    Attributes:
        model_manager: Singleton model manager instance.
        settings: Application settings.

    Example:
        >>> detector = InvoiceFieldDetector()
        >>> result = detector.detect("invoice.jpg")
        >>> print(f"Found {result.detection_count} fields")
    """

    def __init__(self, settings: "Settings | None" = None) -> None:
        """Initialize the detector.

        Args:
            settings: Application settings. If None, loads from default config.
        """
        from src.config.settings import get_settings

        self._settings = settings or get_settings()
        self._model_manager = ModelManager(self._settings)

        logger.info(
            "InvoiceFieldDetector initialized",
            confidence_threshold=self._settings.model.confidence_threshold,
            device=str(self._model_manager.device),
        )

    @property
    def model(self) -> Any:
        """Get the underlying model."""
        return self._model_manager.get_model()

    @property
    def device(self) -> torch.device:
        """Get the compute device."""
        return self._model_manager.device

    def detect(
        self,
        image_source: str | Path | "NDArray[np.uint8]",
        confidence_threshold: float | None = None,
        iou_threshold: float | None = None,
    ) -> DetectionResult:
        """Detect invoice fields in an image.

        Args:
            image_source: Path to image file or numpy array (BGR or RGB).
            confidence_threshold: Override default confidence threshold.
            iou_threshold: Override default IoU threshold.

        Returns:
            DetectionResult with all detected fields and metadata.

        Raises:
            FileNotFoundError: If image file doesn't exist.
            ValueError: If image cannot be processed.
        """
        start_time = time.perf_counter()

        # Load and validate image
        image, image_path = self._load_image(image_source)
        height, width = image.shape[:2]

        logger.debug(
            "Processing image",
            source=image_path,
            dimensions=f"{width}x{height}",
        )

        # Update thresholds if specified
        if confidence_threshold is not None or iou_threshold is not None:
            self._model_manager.update_thresholds(confidence_threshold, iou_threshold)

        # Run inference
        with torch.no_grad():
            results = self.model(image, size=self._settings.model.image_size)

        # Parse results
        detections = self._parse_detections(results)

        # Calculate processing time
        processing_time_ms = (time.perf_counter() - start_time) * 1000

        # Build metadata
        metadata = ProcessingMetadata(
            processing_time_ms=processing_time_ms,
            model_version=self._model_manager.model_version,
            device=str(self.device),
            image_width=width,
            image_height=height,
            image_source=image_path,
        )

        result = DetectionResult(
            metadata=metadata,
            detections=detections,
        )

        logger.info(
            "Detection complete",
            detections=len(detections),
            processing_time_ms=f"{processing_time_ms:.1f}",
            high_confidence=result.high_confidence_count,
        )

        return result

    def detect_batch(
        self,
        image_sources: list[str | Path | "NDArray[np.uint8]"],
        confidence_threshold: float | None = None,
        iou_threshold: float | None = None,
    ) -> list[DetectionResult]:
        """Detect fields in multiple images.

        Args:
            image_sources: List of image paths or numpy arrays.
            confidence_threshold: Override default confidence threshold.
            iou_threshold: Override default IoU threshold.

        Returns:
            List of DetectionResult, one per image.
        """
        results = []
        for source in image_sources:
            try:
                result = self.detect(source, confidence_threshold, iou_threshold)
                results.append(result)
            except Exception as e:
                logger.error("Batch detection failed for image", source=str(source), error=str(e))
                # Create error result
                results.append(self._create_error_result(source, str(e)))
        return results

    def _load_image(
        self,
        image_source: str | Path | "NDArray[np.uint8]",
    ) -> tuple["NDArray[np.uint8]", str]:
        """Load and validate an image.

        Args:
            image_source: Path to image or numpy array.

        Returns:
            Tuple of (image array, source identifier).

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If image cannot be loaded.
        """
        if isinstance(image_source, np.ndarray):
            return image_source, "memory_buffer"

        path = Path(image_source)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        # Validate extension
        if path.suffix.lower() not in self._settings.preprocessing.supported_formats:
            raise ValueError(f"Unsupported image format: {path.suffix}")

        # Load image
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Failed to load image: {path}")

        # Check size limits
        height, width = image.shape[:2]
        max_dim = self._settings.preprocessing.max_image_dimension
        if width > max_dim or height > max_dim:
            logger.warning(
                "Image exceeds maximum dimension, will be resized",
                original=f"{width}x{height}",
                max_allowed=max_dim,
            )

        return image, str(path)

    def _parse_detections(self, results: Any) -> list[InvoiceField]:
        """Parse YOLOv5 results into InvoiceField objects.

        Args:
            results: Raw YOLOv5 inference results.

        Returns:
            List of InvoiceField detections.
        """
        detections = []

        # Get pandas dataframe from results
        df = results.pandas().xyxy[0]

        for _, row in df.iterrows():
            try:
                # Map class name to InvoiceFieldType
                label = self._map_label(row["name"])

                bounding_box = BoundingBox(
                    x_min=float(row["xmin"]),
                    y_min=float(row["ymin"]),
                    x_max=float(row["xmax"]),
                    y_max=float(row["ymax"]),
                )

                field = InvoiceField(
                    label=label,
                    confidence=float(row["confidence"]),
                    bounding_box=bounding_box,
                )

                detections.append(field)

            except (KeyError, ValueError) as e:
                logger.warning("Failed to parse detection", error=str(e), row=dict(row))
                continue

        # Sort by confidence descending
        detections.sort(key=lambda x: x.confidence, reverse=True)

        return detections

    def _map_label(self, class_name: str) -> InvoiceFieldType:
        """Map YOLOv5 class name to InvoiceFieldType.

        Args:
            class_name: Raw class name from model.

        Returns:
            Corresponding InvoiceFieldType enum value.

        Raises:
            ValueError: If class name is unknown.
        """
        # Handle exact matches
        try:
            return InvoiceFieldType(class_name)
        except ValueError:
            pass

        # Handle common variations
        name_mapping = {
            "invoice_date": InvoiceFieldType.INVOICE_DATE,
            "invoice_number": InvoiceFieldType.INVOICE_NUMBER,
            "vendor_name": InvoiceFieldType.VENDOR_NAME,
            "total_amount": InvoiceFieldType.TOTAL_AMOUNT,
            "vat_amount": InvoiceFieldType.VAT_AMOUNT,
            "line_item": InvoiceFieldType.LINE_ITEM,
            "date": InvoiceFieldType.INVOICE_DATE,
            "number": InvoiceFieldType.INVOICE_NUMBER,
            "vendor": InvoiceFieldType.VENDOR_NAME,
            "total": InvoiceFieldType.TOTAL_AMOUNT,
            "vat": InvoiceFieldType.VAT_AMOUNT,
            "item": InvoiceFieldType.LINE_ITEM,
        }

        normalized = class_name.lower().replace(" ", "_").replace("-", "_")
        if normalized in name_mapping:
            return name_mapping[normalized]

        raise ValueError(f"Unknown class name: {class_name}")

    def _create_error_result(
        self,
        source: str | Path | "NDArray[np.uint8]",
        error: str,
    ) -> DetectionResult:
        """Create an error result for failed processing.

        Args:
            source: Image source that failed.
            error: Error message.

        Returns:
            DetectionResult with empty detections and error info.
        """
        source_str = str(source) if not isinstance(source, np.ndarray) else "memory_buffer"

        metadata = ProcessingMetadata(
            processing_time_ms=0,
            model_version=self._model_manager.model_version,
            device=str(self.device),
            image_width=0,
            image_height=0,
            image_source=f"{source_str} (ERROR: {error})",
        )

        return DetectionResult(metadata=metadata, detections=[])

    def visualize(
        self,
        image_source: str | Path | "NDArray[np.uint8]",
        result: DetectionResult,
        output_path: str | Path | None = None,
        show_labels: bool = True,
        show_confidence: bool = True,
        line_thickness: int = 2,
    ) -> "NDArray[np.uint8]":
        """Draw detection boxes on an image.

        Args:
            image_source: Original image.
            result: Detection result to visualize.
            output_path: If provided, save annotated image to this path.
            show_labels: Whether to show field labels.
            show_confidence: Whether to show confidence scores.
            line_thickness: Bounding box line thickness.

        Returns:
            Annotated image as numpy array.
        """
        # Load image
        image, _ = self._load_image(image_source)
        annotated = image.copy()

        # Color mapping for different field types
        colors = {
            InvoiceFieldType.INVOICE_DATE: (0, 255, 0),      # Green
            InvoiceFieldType.INVOICE_NUMBER: (255, 0, 0),   # Blue
            InvoiceFieldType.VENDOR_NAME: (0, 0, 255),      # Red
            InvoiceFieldType.TOTAL_AMOUNT: (255, 255, 0),   # Cyan
            InvoiceFieldType.VAT_AMOUNT: (255, 0, 255),     # Magenta
            InvoiceFieldType.LINE_ITEM: (0, 255, 255),      # Yellow
        }

        for detection in result.detections:
            bbox = detection.bounding_box
            color = colors.get(detection.label, (128, 128, 128))

            # Draw bounding box
            pt1 = (int(bbox.x_min), int(bbox.y_min))
            pt2 = (int(bbox.x_max), int(bbox.y_max))
            cv2.rectangle(annotated, pt1, pt2, color, line_thickness)

            # Draw label
            if show_labels or show_confidence:
                label_parts = []
                if show_labels:
                    label_parts.append(detection.label.value)
                if show_confidence:
                    label_parts.append(f"{detection.confidence:.0%}")
                label = " ".join(label_parts)

                # Calculate label position and background
                font_scale = 0.5
                font_thickness = 1
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                )

                # Background rectangle
                cv2.rectangle(
                    annotated,
                    (pt1[0], pt1[1] - text_height - baseline - 5),
                    (pt1[0] + text_width + 5, pt1[1]),
                    color,
                    -1,
                )

                # Text
                cv2.putText(
                    annotated,
                    label,
                    (pt1[0] + 2, pt1[1] - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    font_thickness,
                )

        # Save if output path provided
        if output_path is not None:
            cv2.imwrite(str(output_path), annotated)
            logger.info("Saved annotated image", path=str(output_path))

        return annotated

    def get_model_info(self) -> dict[str, Any]:
        """Get model information.

        Returns:
            Dictionary with model metadata.
        """
        return self._model_manager.get_model_info()
