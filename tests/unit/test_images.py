"""Image loading and validation."""

from __future__ import annotations

import io

from pathlib import Path

import numpy as np
import pytest

from PIL import Image

from invoice_digitizer.config.settings import PreprocessingSettings
from invoice_digitizer.core.images import (
    ImageValidationError,
    as_rgb_array,
    decode_image_bytes,
    load_image,
    load_image_file,
)

LIMITS = PreprocessingSettings()


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_file_is_loaded_as_rgb(tmp_path: Path) -> None:
    """Colour order matters: the model expects RGB. OpenCV would have returned BGR."""
    path = tmp_path / "red.png"
    Image.new("RGB", (4, 3), color=(255, 0, 0)).save(path)

    array = load_image_file(path, LIMITS)

    assert array.shape == (3, 4, 3)
    assert array.dtype == np.uint8
    assert tuple(array[0, 0]) == (255, 0, 0)


def test_rgba_and_greyscale_uploads_become_rgb() -> None:
    rgba = decode_image_bytes(_png_bytes(Image.new("RGBA", (5, 5), (0, 0, 255, 128))), LIMITS)
    grey = decode_image_bytes(_png_bytes(Image.new("L", (5, 5), 200)), LIMITS)

    assert rgba.shape == (5, 5, 3)
    assert tuple(rgba[0, 0]) == (0, 0, 255)
    assert grey.shape == (5, 5, 3)


def test_exif_orientation_is_applied() -> None:
    image = Image.new("RGB", (40, 20), "white")
    exif = image.getexif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise on display
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif)

    array = decode_image_bytes(buffer.getvalue(), LIMITS)

    assert array.shape[:2] == (40, 20)


def test_undecodable_bytes_are_rejected() -> None:
    with pytest.raises(ImageValidationError, match="could not be decoded"):
        decode_image_bytes(b"definitely not an image", LIMITS)


def test_oversized_image_is_rejected_before_decoding() -> None:
    limits = PreprocessingSettings(max_image_dimension=64)
    with pytest.raises(ImageValidationError, match="largest side"):
        decode_image_bytes(_png_bytes(Image.new("RGB", (65, 10))), limits)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_image_file(tmp_path / "nope.png", LIMITS)


def test_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "invoice.pdf"
    path.write_bytes(b"%PDF-1.7")
    with pytest.raises(ImageValidationError, match="Unsupported"):
        load_image_file(path, LIMITS)


def test_corrupt_file_with_image_extension(tmp_path: Path) -> None:
    path = tmp_path / "broken.png"
    path.write_bytes(b"not really a png")
    with pytest.raises(ImageValidationError, match="Could not decode"):
        load_image_file(path, LIMITS)


def test_array_inputs() -> None:
    grey = np.full((6, 8), 10, dtype=np.uint8)
    rgba = np.zeros((6, 8, 4), dtype=np.uint8)

    assert as_rgb_array(grey, LIMITS).shape == (6, 8, 3)
    assert as_rgb_array(rgba, LIMITS).shape == (6, 8, 3)

    with pytest.raises(ImageValidationError, match="uint8"):
        as_rgb_array(np.zeros((6, 8, 3), dtype=np.float32), LIMITS)
    with pytest.raises(ImageValidationError, match="H x W x 3"):
        as_rgb_array(np.zeros((6, 8, 2), dtype=np.uint8), LIMITS)


def test_load_image_reports_a_short_source_label(invoice_image_path: Path) -> None:
    _, label = load_image(invoice_image_path, LIMITS)
    _, buffer_label = load_image(np.zeros((4, 4, 3), dtype=np.uint8), LIMITS)

    assert label == "synthetic_invoice.png"  # file name only, not the full path
    assert buffer_label == "memory_buffer"
