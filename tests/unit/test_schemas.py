"""Unit tests for detection schemas."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

import pytest

from src.schemas.detection import (
    BoundingBox,
    DetectionResult,
    InvoiceField,
    InvoiceFieldType,
    ProcessingMetadata,
)


class TestBoundingBox:
    """Tests for BoundingBox schema."""

    def test_create_valid_bbox(self) -> None:
        """Test creating a valid bounding box."""
        bbox = BoundingBox(x_min=10.0, y_min=20.0, x_max=100.0, y_max=200.0)

        assert bbox.x_min == 10.0
        assert bbox.y_min == 20.0
        assert bbox.x_max == 100.0
        assert bbox.y_max == 200.0

    def test_bbox_width_height(self) -> None:
        """Test width and height calculations."""
        bbox = BoundingBox(x_min=10.0, y_min=20.0, x_max=110.0, y_max=220.0)

        assert bbox.width == 100.0
        assert bbox.height == 200.0

    def test_bbox_area(self) -> None:
        """Test area calculation."""
        bbox = BoundingBox(x_min=0.0, y_min=0.0, x_max=10.0, y_max=10.0)

        assert bbox.area == 100.0

    def test_bbox_center(self) -> None:
        """Test center point calculation."""
        bbox = BoundingBox(x_min=0.0, y_min=0.0, x_max=100.0, y_max=100.0)

        center = bbox.center
        assert center == (50.0, 50.0)

    def test_bbox_to_xywh(self) -> None:
        """Test XYWH format conversion."""
        bbox = BoundingBox(x_min=10.0, y_min=20.0, x_max=110.0, y_max=120.0)

        xywh = bbox.to_xywh()
        assert xywh == (60.0, 70.0, 100.0, 100.0)  # (cx, cy, w, h)

    def test_bbox_to_xyxy(self) -> None:
        """Test XYXY format conversion."""
        bbox = BoundingBox(x_min=10.0, y_min=20.0, x_max=110.0, y_max=120.0)

        xyxy = bbox.to_xyxy()
        assert xyxy == (10.0, 20.0, 110.0, 120.0)

    def test_bbox_iou_full_overlap(self) -> None:
        """Test IoU with identical boxes."""
        bbox1 = BoundingBox(x_min=0.0, y_min=0.0, x_max=100.0, y_max=100.0)
        bbox2 = BoundingBox(x_min=0.0, y_min=0.0, x_max=100.0, y_max=100.0)

        assert bbox1.iou(bbox2) == 1.0

    def test_bbox_iou_no_overlap(self) -> None:
        """Test IoU with non-overlapping boxes."""
        bbox1 = BoundingBox(x_min=0.0, y_min=0.0, x_max=50.0, y_max=50.0)
        bbox2 = BoundingBox(x_min=100.0, y_min=100.0, x_max=150.0, y_max=150.0)

        assert bbox1.iou(bbox2) == 0.0

    def test_bbox_iou_partial_overlap(self) -> None:
        """Test IoU with partial overlap."""
        bbox1 = BoundingBox(x_min=0.0, y_min=0.0, x_max=100.0, y_max=100.0)
        bbox2 = BoundingBox(x_min=50.0, y_min=50.0, x_max=150.0, y_max=150.0)

        iou = bbox1.iou(bbox2)
        # Intersection: 50x50 = 2500
        # Union: 10000 + 10000 - 2500 = 17500
        # IoU: 2500/17500 ≈ 0.143
        assert 0.14 < iou < 0.15

    def test_bbox_immutable(self) -> None:
        """Test that bounding box is immutable."""
        bbox = BoundingBox(x_min=10.0, y_min=20.0, x_max=100.0, y_max=200.0)

        with pytest.raises(Exception):  # Pydantic raises ValidationError
            bbox.x_min = 50.0  # type: ignore


class TestInvoiceField:
    """Tests for InvoiceField schema."""

    def test_create_field(self) -> None:
        """Test creating an invoice field."""
        bbox = BoundingBox(x_min=10.0, y_min=20.0, x_max=100.0, y_max=200.0)
        field = InvoiceField(
            label=InvoiceFieldType.INVOICE_DATE,
            confidence=0.95,
            bounding_box=bbox,
        )

        assert field.label == InvoiceFieldType.INVOICE_DATE
        assert field.confidence == 0.95
        assert isinstance(field.field_id, UUID)

    def test_high_confidence_detection(self) -> None:
        """Test high confidence threshold."""
        bbox = BoundingBox(x_min=0.0, y_min=0.0, x_max=100.0, y_max=100.0)
        
        high_conf = InvoiceField(
            label=InvoiceFieldType.TOTAL_AMOUNT,
            confidence=0.85,
            bounding_box=bbox,
        )
        low_conf = InvoiceField(
            label=InvoiceFieldType.TOTAL_AMOUNT,
            confidence=0.75,
            bounding_box=bbox,
        )

        assert high_conf.is_high_confidence is True
        assert low_conf.is_high_confidence is False

    def test_needs_review(self) -> None:
        """Test review threshold."""
        bbox = BoundingBox(x_min=0.0, y_min=0.0, x_max=100.0, y_max=100.0)
        
        clear = InvoiceField(
            label=InvoiceFieldType.VENDOR_NAME,
            confidence=0.65,
            bounding_box=bbox,
        )
        unclear = InvoiceField(
            label=InvoiceFieldType.VENDOR_NAME,
            confidence=0.55,
            bounding_box=bbox,
        )

        assert clear.needs_review is False
        assert unclear.needs_review is True

    def test_field_with_ocr(self) -> None:
        """Test field with OCR text."""
        bbox = BoundingBox(x_min=0.0, y_min=0.0, x_max=100.0, y_max=100.0)
        field = InvoiceField(
            label=InvoiceFieldType.TOTAL_AMOUNT,
            confidence=0.9,
            bounding_box=bbox,
            extracted_text="£1,250.00",
            ocr_confidence=0.92,
        )

        assert field.extracted_text == "£1,250.00"
        assert field.ocr_confidence == 0.92


class TestDetectionResult:
    """Tests for DetectionResult schema."""

    @pytest.fixture
    def sample_detections(self) -> list[InvoiceField]:
        """Create sample detections for testing."""
        bbox = BoundingBox(x_min=0.0, y_min=0.0, x_max=100.0, y_max=100.0)
        return [
            InvoiceField(
                label=InvoiceFieldType.INVOICE_DATE,
                confidence=0.95,
                bounding_box=bbox,
            ),
            InvoiceField(
                label=InvoiceFieldType.TOTAL_AMOUNT,
                confidence=0.88,
                bounding_box=bbox,
            ),
            InvoiceField(
                label=InvoiceFieldType.VENDOR_NAME,
                confidence=0.55,
                bounding_box=bbox,
            ),
        ]

    @pytest.fixture
    def sample_metadata(self) -> ProcessingMetadata:
        """Create sample metadata."""
        return ProcessingMetadata(
            processing_time_ms=45.5,
            model_version="yolov5x_v7.0",
            device="cuda",
            image_width=1920,
            image_height=1080,
            image_source="test_invoice.jpg",
        )

    def test_detection_count(
        self,
        sample_detections: list[InvoiceField],
        sample_metadata: ProcessingMetadata,
    ) -> None:
        """Test detection counting."""
        result = DetectionResult(
            metadata=sample_metadata,
            detections=sample_detections,
        )

        assert result.detection_count == 3

    def test_high_confidence_count(
        self,
        sample_detections: list[InvoiceField],
        sample_metadata: ProcessingMetadata,
    ) -> None:
        """Test high confidence detection counting."""
        result = DetectionResult(
            metadata=sample_metadata,
            detections=sample_detections,
        )

        # 0.95 and 0.88 are >= 0.8
        assert result.high_confidence_count == 2

    def test_fields_by_type(
        self,
        sample_metadata: ProcessingMetadata,
    ) -> None:
        """Test grouping detections by type."""
        bbox = BoundingBox(x_min=0.0, y_min=0.0, x_max=100.0, y_max=100.0)
        detections = [
            InvoiceField(
                label=InvoiceFieldType.LINE_ITEM,
                confidence=0.9,
                bounding_box=bbox,
            ),
            InvoiceField(
                label=InvoiceFieldType.LINE_ITEM,
                confidence=0.85,
                bounding_box=bbox,
            ),
            InvoiceField(
                label=InvoiceFieldType.TOTAL_AMOUNT,
                confidence=0.95,
                bounding_box=bbox,
            ),
        ]

        result = DetectionResult(
            metadata=sample_metadata,
            detections=detections,
        )

        by_type = result.fields_by_type
        assert len(by_type[InvoiceFieldType.LINE_ITEM]) == 2
        assert len(by_type[InvoiceFieldType.TOTAL_AMOUNT]) == 1

    def test_get_field_best_confidence(
        self,
        sample_metadata: ProcessingMetadata,
    ) -> None:
        """Test getting best field by type."""
        bbox = BoundingBox(x_min=0.0, y_min=0.0, x_max=100.0, y_max=100.0)
        detections = [
            InvoiceField(
                label=InvoiceFieldType.TOTAL_AMOUNT,
                confidence=0.85,
                bounding_box=bbox,
                extracted_text="£500.00",
            ),
            InvoiceField(
                label=InvoiceFieldType.TOTAL_AMOUNT,
                confidence=0.95,
                bounding_box=bbox,
                extracted_text="£1,000.00",
            ),
        ]

        result = DetectionResult(
            metadata=sample_metadata,
            detections=detections,
        )

        best = result.get_field(InvoiceFieldType.TOTAL_AMOUNT)
        assert best is not None
        assert best.confidence == 0.95
        assert best.extracted_text == "£1,000.00"

    def test_get_field_not_found(
        self,
        sample_metadata: ProcessingMetadata,
    ) -> None:
        """Test getting non-existent field type."""
        result = DetectionResult(
            metadata=sample_metadata,
            detections=[],
        )

        field = result.get_field(InvoiceFieldType.INVOICE_DATE)
        assert field is None

    def test_to_flat_dict(
        self,
        sample_detections: list[InvoiceField],
        sample_metadata: ProcessingMetadata,
    ) -> None:
        """Test flat dictionary export."""
        result = DetectionResult(
            metadata=sample_metadata,
            detections=sample_detections,
        )

        flat = result.to_flat_dict()

        assert "request_id" in flat
        assert "timestamp" in flat
        assert "detection_count" in flat
        assert flat["detection_count"] == 3
        assert "invoice_date_confidence" in flat
        assert flat["invoice_date_confidence"] == 0.95


class TestInvoiceFieldType:
    """Tests for InvoiceFieldType enum."""

    def test_all_field_types(self) -> None:
        """Test all expected field types exist."""
        expected_types = [
            "Invoice Date",
            "Invoice Number",
            "Vendor Name",
            "Total Amount",
            "VAT Amount",
            "Line Item",
        ]

        for field_type in expected_types:
            assert InvoiceFieldType(field_type) is not None

    def test_enum_values(self) -> None:
        """Test enum value access."""
        assert InvoiceFieldType.INVOICE_DATE.value == "Invoice Date"
        assert InvoiceFieldType.VAT_AMOUNT.value == "VAT Amount"
