"""Main InvoiceDigitizer class - high-level API for invoice processing."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from src.core.detector import InvoiceFieldDetector
from src.schemas.detection import DetectionResult

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from src.config.settings import Settings

logger = structlog.get_logger(__name__)


class InvoiceDigitizer:
    """High-level API for invoice digitization.

    This class provides the main interface for processing invoices,
    combining field detection with optional OCR extraction.

    The class supports both synchronous and asynchronous processing,
    with built-in support for batch operations.

    Attributes:
        detector: The underlying field detector.
        settings: Application settings.

    Example:
        >>> digitizer = InvoiceDigitizer()
        >>> # Synchronous
        >>> result = digitizer.process("invoice.jpg")
        >>> # Asynchronous
        >>> result = await digitizer.process_async("invoice.jpg")
        >>> # Batch
        >>> results = await digitizer.process_batch(["inv1.jpg", "inv2.jpg"])
    """

    def __init__(
        self,
        settings: "Settings | None" = None,
        max_workers: int = 4,
    ) -> None:
        """Initialize the InvoiceDigitizer.

        Args:
            settings: Application settings. If None, loads from default config.
            max_workers: Maximum concurrent workers for batch processing.
        """
        from src.config.settings import get_settings

        self._settings = settings or get_settings()
        self._detector = InvoiceFieldDetector(self._settings)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._ocr_enabled = self._settings.ocr.enabled if hasattr(self._settings, 'ocr') else False

        logger.info(
            "InvoiceDigitizer initialized",
            max_workers=max_workers,
            ocr_enabled=self._ocr_enabled,
        )

    @property
    def detector(self) -> InvoiceFieldDetector:
        """Get the underlying detector."""
        return self._detector

    def process(
        self,
        image_source: str | Path | "NDArray[np.uint8]",
        confidence_threshold: float | None = None,
        extract_text: bool = True,
    ) -> DetectionResult:
        """Process a single invoice image.

        This is the main synchronous entry point for invoice processing.

        Args:
            image_source: Path to image file or numpy array.
            confidence_threshold: Override default confidence threshold.
            extract_text: Whether to run OCR on detected regions.

        Returns:
            DetectionResult with detected fields and metadata.

        Example:
            >>> result = digitizer.process("invoice.jpg")
            >>> for field in result.detections:
            ...     print(f"{field.label}: {field.confidence:.2%}")
        """
        # Run detection
        result = self._detector.detect(
            image_source,
            confidence_threshold=confidence_threshold,
        )

        # Optional OCR extraction
        if extract_text and self._ocr_enabled:
            result = self._extract_text(image_source, result)

        return result

    async def process_async(
        self,
        image_source: str | Path | "NDArray[np.uint8]",
        confidence_threshold: float | None = None,
        extract_text: bool = True,
    ) -> DetectionResult:
        """Process a single invoice image asynchronously.

        This is the main asynchronous entry point for invoice processing.
        Useful for web APIs and concurrent applications.

        Args:
            image_source: Path to image file or numpy array.
            confidence_threshold: Override default confidence threshold.
            extract_text: Whether to run OCR on detected regions.

        Returns:
            DetectionResult with detected fields and metadata.

        Example:
            >>> result = await digitizer.process_async("invoice.jpg")
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.process(image_source, confidence_threshold, extract_text),
        )

    async def process_batch(
        self,
        image_sources: list[str | Path],
        confidence_threshold: float | None = None,
        extract_text: bool = True,
        max_concurrent: int | None = None,
    ) -> list[DetectionResult]:
        """Process multiple invoice images concurrently.

        Args:
            image_sources: List of image paths.
            confidence_threshold: Override default confidence threshold.
            extract_text: Whether to run OCR on detected regions.
            max_concurrent: Maximum concurrent tasks. Defaults to executor workers.

        Returns:
            List of DetectionResult, one per image.

        Example:
            >>> results = await digitizer.process_batch([
            ...     "invoice1.jpg",
            ...     "invoice2.jpg",
            ...     "invoice3.jpg",
            ... ])
        """
        max_concurrent = max_concurrent or self._executor._max_workers

        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(source: str | Path) -> DetectionResult:
            async with semaphore:
                return await self.process_async(
                    source,
                    confidence_threshold=confidence_threshold,
                    extract_text=extract_text,
                )

        tasks = [process_with_semaphore(source) for source in image_sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "Batch processing failed",
                    source=str(image_sources[i]),
                    error=str(result),
                )
                # Create error result
                processed_results.append(
                    self._detector._create_error_result(image_sources[i], str(result))
                )
            else:
                processed_results.append(result)

        logger.info(
            "Batch processing complete",
            total=len(image_sources),
            successful=sum(1 for r in processed_results if r.detection_count > 0),
        )

        return processed_results

    def _extract_text(
        self,
        image_source: str | Path | "NDArray[np.uint8]",
        result: DetectionResult,
    ) -> DetectionResult:
        """Extract text from detected regions using OCR.

        Args:
            image_source: Original image.
            result: Detection result with bounding boxes.

        Returns:
            Updated DetectionResult with extracted text.
        """
        # OCR implementation would go here
        # For now, return result unchanged
        logger.debug("OCR extraction not implemented, skipping")
        return result

    def visualize(
        self,
        image_source: str | Path | "NDArray[np.uint8]",
        result: DetectionResult,
        output_path: str | Path | None = None,
    ) -> "NDArray[np.uint8]":
        """Visualize detection results on an image.

        Args:
            image_source: Original image.
            result: Detection result to visualize.
            output_path: If provided, save annotated image.

        Returns:
            Annotated image as numpy array.
        """
        return self._detector.visualize(image_source, result, output_path)

    def get_model_info(self) -> dict[str, Any]:
        """Get model information.

        Returns:
            Dictionary with model metadata.
        """
        return self._detector.get_model_info()

    def __enter__(self) -> "InvoiceDigitizer":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - cleanup resources."""
        self._executor.shutdown(wait=False)

    async def __aenter__(self) -> "InvoiceDigitizer":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        self._executor.shutdown(wait=False)
