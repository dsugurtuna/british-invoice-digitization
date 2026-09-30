"""The command-line entry point."""

from __future__ import annotations

import json

from collections.abc import Iterator
from typing import Any

import pytest
import structlog

from invoice_digitizer import __main__ as cli
from invoice_digitizer.config.settings import Settings


@pytest.fixture(autouse=True)
def restore_structlog() -> Iterator[None]:
    yield
    structlog.reset_defaults()


def test_main_starts_uvicorn_with_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    monkeypatch.setattr(cli.uvicorn, "run", lambda *a, **k: calls.append((a, k)))

    cli.main(Settings(api={"host": "192.0.2.10", "port": 9001}, log_level="WARNING"))

    ((args, kwargs),) = calls
    assert args == ("invoice_digitizer.api.main:create_app",)
    assert kwargs == {"factory": True, "host": "192.0.2.10", "port": 9001, "log_level": "warning"}


def test_production_logs_are_json_and_filtered(capsys: pytest.CaptureFixture[str]) -> None:
    cli.configure_logging("WARNING", json_output=True)
    logger = structlog.get_logger("test")

    logger.info("hidden")
    logger.warning("shown", detail=1)

    lines = [line for line in capsys.readouterr().out.splitlines() if line]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["event"] == "shown"
    assert record["level"] == "warning"
    assert record["detail"] == 1
