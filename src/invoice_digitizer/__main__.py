"""Command-line entry point: ``invoice-digitizer`` or ``python -m invoice_digitizer``.

Starts the API with host, port and log level taken from the settings.
"""

from __future__ import annotations

import logging

import structlog
import uvicorn

from invoice_digitizer.config.settings import Settings, get_settings


def configure_logging(level: str, json_output: bool) -> None:
    """Set up structlog: JSON lines in production, readable console output otherwise."""
    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer() if json_output else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelNamesMapping()[level]),
        cache_logger_on_first_use=True,
    )


def main(settings: Settings | None = None) -> None:
    """Run the API server."""
    settings = settings or get_settings()
    configure_logging(settings.log_level.value, json_output=settings.is_production)
    uvicorn.run(
        "invoice_digitizer.api.main:create_app",
        factory=True,
        host=settings.api.host,
        port=settings.api.port,
        log_level=settings.log_level.value.lower(),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
