"""Image loading and validation.

Everything is converted to an RGB ``uint8`` array because that is what the YOLOv5
wrapper expects for NumPy input. (The previous version read files with OpenCV,
which returns BGR, so file inputs reached the model with red and blue swapped.)
"""

from __future__ import annotations

import io

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from PIL import Image, ImageOps

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from invoice_digitizer.config.settings import PreprocessingSettings

ImageSource = str | Path | np.ndarray


class ImageValidationError(ValueError):
    """The input is not an image this service will process."""


def _check_dimensions(width: int, height: int, limits: PreprocessingSettings) -> None:
    largest = limits.max_image_dimension
    if width > largest or height > largest:
        raise ImageValidationError(
            f"Image is {width}x{height}; the largest side allowed is {largest} pixels."
        )


def _to_rgb_array(image: Image.Image, limits: PreprocessingSettings) -> NDArray[np.uint8]:
    # Check the size from the header before decoding pixels, so a huge image is
    # rejected without being loaded into memory.
    _check_dimensions(image.width, image.height, limits)
    oriented = ImageOps.exif_transpose(image) or image
    return np.asarray(oriented.convert("RGB"), dtype=np.uint8)


def decode_image_bytes(data: bytes, limits: PreprocessingSettings) -> NDArray[np.uint8]:
    """Decode an uploaded file into an RGB array."""
    try:
        with Image.open(io.BytesIO(data)) as image:
            return _to_rgb_array(image, limits)
    except ImageValidationError:
        raise
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ImageValidationError("The file could not be decoded as an image.") from exc


def load_image_file(path: str | Path, limits: PreprocessingSettings) -> NDArray[np.uint8]:
    """Read an image file from disk into an RGB array."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Image not found: {file_path}")
    if file_path.suffix.lower() not in limits.supported_formats:
        raise ImageValidationError(f"Unsupported image format: {file_path.suffix}")
    try:
        with Image.open(file_path) as image:
            return _to_rgb_array(image, limits)
    except ImageValidationError:
        raise
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ImageValidationError(f"Could not decode image: {file_path.name}") from exc


def as_rgb_array(array: np.ndarray, limits: PreprocessingSettings) -> NDArray[np.uint8]:
    """Validate an in-memory image and return it as H x W x 3 uint8 RGB.

    Grey-scale arrays are expanded to three channels and an alpha channel is dropped.
    Colour arrays are assumed to be RGB already.
    """
    if array.dtype != np.uint8:
        raise ImageValidationError(f"Expected a uint8 image array, got {array.dtype}.")
    if array.ndim == 2:  # grey-scale H x W
        array = np.stack([array] * 3, axis=-1)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ImageValidationError(f"Expected an H x W x 3 image array, got {array.shape}.")
    height, width = array.shape[:2]
    _check_dimensions(width, height, limits)
    return np.ascontiguousarray(array[:, :, :3])


def load_image(source: ImageSource, limits: PreprocessingSettings) -> tuple[NDArray[np.uint8], str]:
    """Load any supported source and return (RGB array, short source label)."""
    if isinstance(source, np.ndarray):
        return as_rgb_array(source, limits), "memory_buffer"
    return load_image_file(source, limits), Path(source).name
