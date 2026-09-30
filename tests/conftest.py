"""Shared fixtures.

No test downloads model code or weights, and none needs a GPU or PyTorch. The model
is replaced by ``FakePredictor``, which returns fixed boxes and filters them by the
confidence threshold it is given, the same way YOLOv5 does. Test images are
synthetic pages drawn with Pillow; they contain no real invoice data.
"""

from __future__ import annotations

import io

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from PIL import Image, ImageDraw

from invoice_digitizer.config.settings import ENV_PREFIX, Settings, get_settings
from invoice_digitizer.core.model_manager import ModelManager
from invoice_digitizer.core.yolov5_backend import LoadedModel, RawDetection

PAGE_WIDTH, PAGE_HEIGHT = 800, 1000

# Boxes on the synthetic page. Two are meant to be ignored (an unknown class and a
# box entirely outside the page) and one sits below the default 0.4 threshold.
DEFAULT_RAW_DETECTIONS = [
    RawDetection(500, 60, 740, 100, 0.93, "Invoice Date"),
    RawDetection(500, 850, 740, 900, 0.88, "Total Amount"),
    RawDetection(500, 800, 740, 840, 0.55, "vat"),
    RawDetection(40, 40, 300, 120, 0.91, "person"),
    RawDetection(900, 1100, 950, 1200, 0.97, "Line Item"),
    RawDetection(40, 140, 300, 170, 0.35, "Invoice Number"),
]


class FakePredictor:
    """Stands in for YOLOv5 and records every call."""

    def __init__(
        self, detections: list[RawDetection] | None = None, error: Exception | None = None
    ) -> None:
        self.detections = list(DEFAULT_RAW_DETECTIONS if detections is None else detections)
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        image: np.ndarray,
        *,
        image_size: int,
        confidence: float,
        iou: float,
        max_detections: int,
    ) -> list[RawDetection]:
        self.calls.append(
            {
                "shape": image.shape,
                "dtype": image.dtype,
                "image_size": image_size,
                "confidence": confidence,
                "iou": iou,
                "max_detections": max_detections,
            }
        )
        if self.error is not None:
            raise self.error
        kept = [d for d in self.detections if d.confidence >= confidence]
        return sorted(kept, key=lambda d: d.confidence, reverse=True)[:max_detections]


class FakeLoader:
    """Builds ``LoadedModel`` objects around a ``FakePredictor`` and counts loads."""

    def __init__(self, predictor: FakePredictor) -> None:
        self.predictor = predictor
        self.calls = 0

    def __call__(self, _settings: Any) -> LoadedModel:
        self.calls += 1
        return LoadedModel(
            predictor=self.predictor,
            device="cpu",
            hub_repo="fake/yolov5:test",
            source=f"fake weights #{self.calls}",
            weights_sha256=f"{self.calls:064d}",
        )


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Stop real environment variables or cached settings leaking into tests."""
    import os

    for key in list(os.environ):
        if key.upper().startswith(ENV_PREFIX):
            monkeypatch.delenv(key)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings that point at a weights path that does not exist."""
    return Settings(
        model={"weights_path": str(tmp_path / "missing.pt")},
        api={"rate_limit_enabled": False},
    )


@pytest.fixture
def fake_predictor() -> FakePredictor:
    return FakePredictor()


@pytest.fixture
def fake_loader(fake_predictor: FakePredictor) -> FakeLoader:
    return FakeLoader(fake_predictor)


@pytest.fixture
def model_manager(settings: Settings, fake_loader: FakeLoader) -> ModelManager:
    return ModelManager(settings.model, loader=fake_loader)


def make_invoice_page(width: int = PAGE_WIDTH, height: int = PAGE_HEIGHT) -> Image.Image:
    """A synthetic, obviously fake invoice-like page."""
    page = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(page)
    draw.text((40, 40), "SYNTHETIC TEST PAGE - NOT A REAL INVOICE", fill="black")
    draw.rectangle((500, 60, 740, 100), outline="black")  # date
    draw.rectangle((40, 140, 300, 170), outline="black")  # number
    for row in range(5):  # line items
        top = 300 + row * 60
        draw.rectangle((40, top, 740, top + 40), outline="grey")
    draw.rectangle((500, 800, 740, 840), outline="black")  # VAT
    draw.rectangle((500, 850, 740, 900), outline="black")  # total
    return page


@pytest.fixture
def invoice_image_path(tmp_path: Path) -> Path:
    path = tmp_path / "synthetic_invoice.png"
    make_invoice_page().save(path)
    return path


@pytest.fixture
def invoice_png_bytes() -> bytes:
    buffer = io.BytesIO()
    make_invoice_page().save(buffer, format="PNG")
    return buffer.getvalue()
