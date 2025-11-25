"""FastAPI application factory and configuration."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncGenerator

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from src.api.routes import router
from src.api.middleware import RequestLoggingMiddleware, RateLimitMiddleware
from src.config.settings import get_settings
from src.core.model_manager import ModelManager

if TYPE_CHECKING:
    from src.config.settings import Settings

logger = structlog.get_logger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"],
)
INFERENCE_COUNT = Counter(
    "inference_requests_total",
    "Total inference requests",
    ["status"],
)
INFERENCE_LATENCY = Histogram(
    "inference_duration_seconds",
    "Inference latency in seconds",
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler.

    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting RoyalAudit Digitizer API")
    settings = get_settings()

    # Pre-load the model
    logger.info("Pre-loading ML model...")
    start_time = time.time()
    model_manager = ModelManager(settings)
    _ = model_manager.model  # Force model loading
    load_time = time.time() - start_time
    logger.info("Model loaded", load_time_seconds=f"{load_time:.2f}")

    # Store in app state
    app.state.settings = settings
    app.state.model_manager = model_manager
    app.state.start_time = time.time()

    yield

    # Shutdown
    logger.info("Shutting down RoyalAudit Digitizer API")
    ModelManager.reset()


def create_app(settings: "Settings | None" = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Application settings. If None, loads from default config.

    Returns:
        Configured FastAPI application instance.
    """
    settings = settings or get_settings()

    app = FastAPI(
        title="RoyalAudit Digitizer API",
        description="""
## 🇬🇧 Enterprise Invoice Extraction System

A production-grade ML API for automated extraction of structured financial data
from unstructured British invoices using YOLOv5.

### Features
- **Real-time inference** - Process invoices in milliseconds
- **Batch processing** - Handle multiple documents asynchronously
- **High accuracy** - 98.5% mAP on British invoice datasets
- **Production ready** - Full observability, rate limiting, and error handling

### Authentication
API key authentication is available for production deployments.
Contact the administrator for access.

### Rate Limits
- Standard: 60 requests/minute
- Burst: 10 requests

### Support
For issues and feature requests, please contact the development team.
        """,
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Add middleware
    _configure_middleware(app, settings)

    # Include routers
    app.include_router(router)

    # Add metrics endpoint
    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        """Prometheus metrics endpoint."""
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )

    # Request timing middleware
    @app.middleware("http")
    async def add_timing_header(request: Request, call_next: Any) -> Response:
        """Add timing information to response headers."""
        start_time = time.perf_counter()
        response = await call_next(request)
        process_time = time.perf_counter() - start_time

        response.headers["X-Process-Time"] = f"{process_time:.4f}"
        response.headers["X-Request-ID"] = request.state.request_id if hasattr(request.state, 'request_id') else "unknown"

        # Record metrics
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code,
        ).inc()
        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=request.url.path,
        ).observe(process_time)

        return response

    logger.info(
        "FastAPI application created",
        environment=settings.environment.value,
        debug=settings.debug,
    )

    return app


def _configure_middleware(app: FastAPI, settings: "Settings") -> None:
    """Configure application middleware.

    Args:
        app: FastAPI application instance.
        settings: Application settings.
    """
    # CORS middleware
    if settings.api.cors_enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.api.cors_allow_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # GZip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Custom middleware
    app.add_middleware(RequestLoggingMiddleware)

    if settings.api.rate_limit_enabled:
        app.add_middleware(
            RateLimitMiddleware,
            requests_per_minute=settings.api.requests_per_minute,
        )


# Create default app instance
app = create_app()
