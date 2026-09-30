"""Invoice field detection with YOLOv5, served through FastAPI.

The package finds where six invoice fields sit on a page image (date, number,
vendor, total, VAT, line items) and returns bounding boxes with confidence
scores. It does not read the text inside those boxes.

Example:
    >>> from invoice_digitizer import InvoiceDigitizer
    >>> digitizer = InvoiceDigitizer()          # needs trained weights, see README
    >>> result = digitizer.process("invoice.png")
    >>> [field.label for field in result.detections]
"""

from invoice_digitizer._version import __version__
from invoice_digitizer.config.settings import Settings, get_settings
from invoice_digitizer.core.detector import InvoiceFieldDetector
from invoice_digitizer.core.digitizer import InvoiceDigitizer
from invoice_digitizer.schemas.detection import DetectionResult, InvoiceField

__all__ = [
    "DetectionResult",
    "InvoiceDigitizer",
    "InvoiceField",
    "InvoiceFieldDetector",
    "Settings",
    "__version__",
    "get_settings",
]
