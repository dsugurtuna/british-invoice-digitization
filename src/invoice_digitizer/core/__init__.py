"""Detection pipeline: image loading, model lifecycle and result building."""

from invoice_digitizer.core.detector import InvoiceFieldDetector
from invoice_digitizer.core.digitizer import InvoiceDigitizer
from invoice_digitizer.core.model_manager import ModelManager

__all__ = ["InvoiceDigitizer", "InvoiceFieldDetector", "ModelManager"]
