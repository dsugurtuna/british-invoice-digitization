"""High-level digitizer: sync, async and batch paths."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from invoice_digitizer.config.settings import Settings
from invoice_digitizer.core.digitizer import InvoiceDigitizer
from invoice_digitizer.core.model_manager import ModelManager
from invoice_digitizer.schemas.detection import InvoiceFieldType
from tests.conftest import FakeLoader, FakePredictor


@pytest.fixture
def digitizer(settings: Settings, model_manager: ModelManager) -> InvoiceDigitizer:
    with InvoiceDigitizer(settings, model_manager=model_manager, max_workers=2) as instance:
        yield instance


def test_process(digitizer: InvoiceDigitizer, invoice_image_path: Path) -> None:
    result = digitizer.process(invoice_image_path)
    assert result.get_field(InvoiceFieldType.TOTAL_AMOUNT) is not None


async def test_process_async(digitizer: InvoiceDigitizer, invoice_image_path: Path) -> None:
    result = await digitizer.process_async(invoice_image_path, confidence_threshold=0.9)
    assert result.detection_count == 1


async def test_batch_keeps_order_and_isolates_failures(
    digitizer: InvoiceDigitizer, invoice_image_path: Path, tmp_path: Path
) -> None:
    sources = [invoice_image_path, tmp_path / "missing.png", invoice_image_path]

    results = await digitizer.process_batch(sources, max_concurrent=2)

    assert [r.succeeded for r in results] == [True, False, True]
    assert results[1].metadata.image_source == "missing.png"


async def test_batch_survives_a_model_error(settings: Settings) -> None:
    broken = FakePredictor(error=RuntimeError("CUDA out of memory"))
    manager = ModelManager(settings.model, loader=FakeLoader(broken))
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    async with InvoiceDigitizer(settings, model_manager=manager) as digitizer:
        results = await digitizer.process_batch([image, image])

    assert [r.error for r in results] == ["CUDA out of memory"] * 2


def test_visualize_and_model_info(
    digitizer: InvoiceDigitizer, invoice_image_path: Path, tmp_path: Path
) -> None:
    result = digitizer.process(invoice_image_path)
    annotated = digitizer.visualize(invoice_image_path, result, tmp_path / "out.png")

    assert annotated.ndim == 3
    assert digitizer.get_model_info()["loaded"] is True
    assert digitizer.detector.model_manager.is_loaded
