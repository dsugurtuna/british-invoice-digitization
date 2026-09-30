"""FastAPI application factory.

Run with ``uvicorn --factory invoice_digitizer.api.main:create_app``.
"""

from __future__ import annotations

import asyncio
import time

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from invoice_digitizer._version import __version__
from invoice_digitizer.api.middleware import RateLimitMiddleware, RequestContextMiddleware
from invoice_digitizer.api.routes import router
from invoice_digitizer.config.settings import Settings, get_settings
from invoice_digitizer.core.digitizer import InvoiceDigitizer
from invoice_digitizer.core.model_manager import ModelManager

logger = structlog.get_logger(__name__)

DESCRIPTION = """
Locates six invoice fields on a page image (invoice date, invoice number, vendor
name, total amount, VAT amount, line items) with a YOLOv5 detector and returns
bounding boxes with confidence scores.

* Text is not read: there is no OCR step.
* Accuracy depends entirely on the weights you load. No trained weights or
  evaluation results are published with this code.
* Endpoints under `admin` need the `X-Admin-Key` header and are disabled when no
  key is configured.
"""


def create_app(
    settings: Settings | None = None,
    model_manager: ModelManager | None = None,
) -> FastAPI:
    """Build the application.

    Everything the routes need is attached to ``app.state`` here, not in the lifespan
    handler, so the app is fully wired as soon as it exists.

    Args:
        settings: Application settings. Defaults to ``get_settings()``.
        model_manager: Model owner. Tests pass one with a fake loader.
    """
    settings = settings or get_settings()
    manager = model_manager or ModelManager(settings.model)
    digitizer = InvoiceDigitizer(settings, model_manager=manager)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if settings.model.preload:
            # Fail at start-up, not on the first request, if the model cannot load.
            await asyncio.to_thread(manager.load)
        yield
        digitizer.close()

    app = FastAPI(
        title="Invoice Field Detection API",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.model_manager = manager
    app.state.digitizer = digitizer
    app.state.started_at = time.time()

    # Starlette runs the last-added middleware first: CORS, then request context
    # (so rejected requests are still logged and counted), then the rate limit.
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    if settings.api.rate_limit_enabled:
        app.add_middleware(
            RateLimitMiddleware, requests_per_minute=settings.api.requests_per_minute
        )
    app.add_middleware(RequestContextMiddleware)
    if settings.api.cors_allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.api.cors_allow_origins,
            allow_methods=["GET", "POST", "PUT"],
            allow_headers=["*"],
        )

    app.include_router(router)

    if settings.monitoring.prometheus_enabled:

        @app.get("/metrics", include_in_schema=False)
        async def metrics() -> Response:
            return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    logger.info("app_created", environment=settings.environment.value)
    return app
