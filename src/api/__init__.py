"""API module - FastAPI REST endpoints."""

from src.api.main import create_app, app
from src.api.routes import router

__all__ = ["create_app", "app", "router"]
