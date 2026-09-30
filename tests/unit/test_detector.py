"""From raw model boxes to validated invoice fields."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from PIL import Image

from invoice_digitizer.config.settings import Settings
from invoice_digitizer.core.detector import InvoiceFieldDetector, map_label, to_fields
from invoice_digitizer.core.images import ImageValidationError
from invoice_digitizer.core.model_manager import ModelManager
from invoice_digitizer.core.yolov5_backend import RawDetection
from invoice_digitizer.schemas.detection import InvoiceFieldType
from tests.conftest import PAGE_HEIGHT, PAGE_WIDTH, FakeLoader, FakePredictor


@pytest.fixture
def detector(settings: Settings, model_manager: ModelManager) -> InvoiceFieldDetector:
    return InvoiceFieldDetector(settings, model_manager)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Invoice Date", InvoiceFieldType.INVOICE_DATE),
        ("invoice_number", InvoiceFieldType.INVOICE_NUMBER),
        ("Vendor-Name", InvoiceFieldType.VENDOR_NAME),
        ("TOTAL", InvoiceFieldType.TOTAL_AMOUNT),
        ("vat", InvoiceFieldType.VAT_AMOUNT),
        (" line item ", InvoiceFieldType.LINE_ITEM),
        ("person", None),
        ("", None),
    ],
)
def test_map_label(raw: str, expected: InvoiceFieldType | None) -> None:
    assert map_label(raw) == expected


def test_to_fields_clips_sorts_and_counts_ignored() -> None:
    raw = [
        RawDetection(-10, -5, 50, 40, 0.6, "Invoice Date"),  # clipped at the top-left
        RawDetection(10, 10, 90, 30, 0.95, "total"),
        RawDetection(10, 10, 90, 30, 0.99, "car"),  # not an invoice field
        RawDetection(120, 10, 150, 30, 0.8, "vat"),  # entirely off the right edge
        RawDetection(10, 10, 90, 30, 1.2, "Line Item"),  # confidence clamped to 1.0
    ]

    fields, ignored = to_fields(raw, width=100, height=50)

    assert ignored == 2
    assert [f.label for f in fields] == [
        InvoiceFieldType.LINE_ITEM,
        InvoiceFieldType.TOTAL_AMOUNT,
        InvoiceFieldType.INVOICE_DATE,
    ]
    assert fields[0].confidence == 1.0
    assert fields[-1].bounding_box.to_xyxy() == (0.0, 0.0, 50.0, 40.0)


def test_detect_on_a_file(
    detector: InvoiceFieldDetector, invoice_image_path: Path, fake_predictor: FakePredictor
) -> None:
    result = detector.detect(invoice_image_path)

    assert result.succeeded
    assert [f.label for f in result.detections] == [
        InvoiceFieldType.INVOICE_DATE,
        InvoiceFieldType.TOTAL_AMOUNT,
        InvoiceFieldType.VAT_AMOUNT,
    ]
    # "person" and the off-page box are reported, not silently dropped
    assert result.metadata.ignored_detections == 2
    assert result.metadata.image_width == PAGE_WIDTH
    assert result.metadata.image_height == PAGE_HEIGHT
    assert result.metadata.image_source == "synthetic_invoice.png"
    assert result.metadata.model_version.startswith("fake/yolov5:test")
    assert fake_predictor.calls[0]["shape"] == (PAGE_HEIGHT, PAGE_WIDTH, 3)


def test_detect_with_a_stricter_threshold(detector: InvoiceFieldDetector) -> None:
    image = np.zeros((PAGE_HEIGHT, PAGE_WIDTH, 3), dtype=np.uint8)
    result = detector.detect(image, confidence_threshold=0.9)

    assert [f.label for f in result.detections] == [InvoiceFieldType.INVOICE_DATE]
    assert result.metadata.image_source == "memory_buffer"


def test_detect_rejects_bad_input(detector: InvoiceFieldDetector, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        detector.detect(tmp_path / "missing.png")
    with pytest.raises(ImageValidationError):
        detector.detect(np.zeros((4, 4, 3), dtype=np.float64))


def test_batch_turns_failures_into_error_results(
    detector: InvoiceFieldDetector, invoice_image_path: Path, tmp_path: Path
) -> None:
    results = detector.detect_batch([invoice_image_path, tmp_path / "missing.png"])

    assert [r.succeeded for r in results] == [True, False]
    failed = results[1]
    assert "missing.png" in (failed.error or "")
    assert failed.metadata.image_width == 0  # used to raise: the schema required >= 1
    assert failed.metadata.image_source == "missing.png"


def test_error_result_does_not_load_the_model(settings: Settings, fake_loader: FakeLoader) -> None:
    detector = InvoiceFieldDetector(settings, ModelManager(settings.model, loader=fake_loader))

    result = detector.error_result(np.zeros((2, 2, 3), dtype=np.uint8), "boom")

    assert fake_loader.calls == 0
    assert result.error == "boom"
    assert result.metadata.model_version == "not loaded"


def test_visualize_draws_and_saves(
    detector: InvoiceFieldDetector, invoice_image_path: Path, tmp_path: Path
) -> None:
    result = detector.detect(invoice_image_path)
    output = tmp_path / "annotated.png"

    annotated = detector.visualize(invoice_image_path, result, output_path=output)

    original = np.asarray(Image.open(invoice_image_path).convert("RGB"))
    assert annotated.shape == original.shape
    assert not np.array_equal(annotated, original)
    assert output.is_file()


def test_default_settings_are_used_when_none_given(fake_loader: FakeLoader) -> None:
    detector = InvoiceFieldDetector(model_manager=ModelManager(Settings().model, fake_loader))
    assert detector.get_model_info()["loaded"] is False
