"""High-level entry point: single images, async calls and concurrent batches."""

from __future__ import annotations

import asyncio
import functools

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Self

import structlog

from invoice_digitizer.config.settings import get_settings
from invoice_digitizer.core.detector import InvoiceFieldDetector

if TYPE_CHECKING:
    import numpy as np

    from numpy.typing import NDArray

    from invoice_digitizer.config.settings import Settings
    from invoice_digitizer.core.images import ImageSource
    from invoice_digitizer.core.model_manager import ModelManager
    from invoice_digitizer.schemas.detection import DetectionResult

logger = structlog.get_logger(__name__)


class InvoiceDigitizer:
    """Process invoice images synchronously, asynchronously or in batches.

    The async methods run detection in a thread pool so an event loop (the API) stays
    responsive while the model works. The pool does not make inference parallel:
    ``ModelManager`` runs one prediction at a time. Image decoding and result building
    do overlap. For more throughput, run more processes.

    Args:
        settings: Application settings. Defaults to ``get_settings()``.
        model_manager: Shared model owner. Created from the settings if omitted.
        max_workers: Threads available to the async methods.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        model_manager: ModelManager | None = None,
        max_workers: int = 4,
    ) -> None:
        self._settings = settings or get_settings()
        self._detector = InvoiceFieldDetector(self._settings, model_manager)
        self._max_workers = max_workers
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="invoice-digitizer"
        )

    @property
    def detector(self) -> InvoiceFieldDetector:
        """The underlying detector."""
        return self._detector

    def process(
        self,
        image_source: ImageSource,
        confidence_threshold: float | None = None,
        iou_threshold: float | None = None,
        source_name: str | None = None,
    ) -> DetectionResult:
        """Detect fields in one image and wait for the answer."""
        return self._detector.detect(
            image_source,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            source_name=source_name,
        )

    async def process_async(
        self,
        image_source: ImageSource,
        confidence_threshold: float | None = None,
        iou_threshold: float | None = None,
        source_name: str | None = None,
    ) -> DetectionResult:
        """Detect fields in one image without blocking the event loop."""
        loop = asyncio.get_running_loop()
        call = functools.partial(
            self.process, image_source, confidence_threshold, iou_threshold, source_name
        )
        return await loop.run_in_executor(self._executor, call)

    async def process_batch(
        self,
        image_sources: list[ImageSource],
        confidence_threshold: float | None = None,
        max_concurrent: int | None = None,
    ) -> list[DetectionResult]:
        """Process several images concurrently, keeping the input order.

        A failing image yields a result with ``error`` set; the rest still complete.
        """
        semaphore = asyncio.Semaphore(max_concurrent or self._max_workers)

        async def run_one(source: ImageSource) -> DetectionResult:
            async with semaphore:
                return await self.process_async(source, confidence_threshold)

        outcomes = await asyncio.gather(
            *(run_one(source) for source in image_sources), return_exceptions=True
        )

        results: list[DetectionResult] = []
        for source, outcome in zip(image_sources, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                logger.warning("batch_item_failed", error_type=type(outcome).__name__)
                results.append(self._detector.error_result(source, outcome))
            else:
                results.append(outcome)

        logger.info(
            "batch_complete",
            total=len(results),
            succeeded=sum(1 for r in results if r.succeeded),
        )
        return results

    def visualize(
        self,
        image_source: ImageSource,
        result: DetectionResult,
        output_path: str | Path | None = None,
    ) -> NDArray[np.uint8]:
        """Draw the detections on the image; see ``InvoiceFieldDetector.visualize``."""
        return self._detector.visualize(image_source, result, output_path)

    def get_model_info(self) -> dict[str, Any]:
        """Model description, without forcing a load."""
        return self._detector.get_model_info()

    def close(self) -> None:
        """Stop the worker threads."""
        self._executor.shutdown(wait=False, cancel_futures=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
