"""
RoyalAudit Digitizer - Enterprise Invoice Extraction System
===========================================================

A production-grade ML pipeline for automated extraction of structured
financial data from unstructured British invoices using YOLOv5.

Modules:
    - core: Core ML inference engine and model management
    - api: FastAPI REST endpoints for inference
    - preprocessing: Image preprocessing and validation
    - schemas: Pydantic data models and validation
    - config: Configuration management
    - utils: Utility functions and helpers

Example:
    >>> from src import InvoiceDigitizer
    >>> digitizer = InvoiceDigitizer()
    >>> result = await digitizer.process_invoice("invoice.jpg")
    >>> print(result.detections)
"""

from src.core.digitizer import InvoiceDigitizer
from src.core.detector import InvoiceFieldDetector
from src.schemas.detection import DetectionResult, InvoiceField
from src.config.settings import Settings, get_settings

__all__ = [
    "InvoiceDigitizer",
    "InvoiceFieldDetector",
    "DetectionResult",
    "InvoiceField",
    "Settings",
    "get_settings",
]

__version__ = "2.0.0"
__author__ = "Ugur Tuna"
