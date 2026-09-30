"""API module - FastAPI REST endpoints."""

from invoice_digitizer.api.main import create_app, app
from invoice_digitizer.api.routes import router

__all__ = ["create_app", "app", "router"]
