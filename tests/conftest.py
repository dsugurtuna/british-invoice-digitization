"""Test configuration and fixtures."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_data_dir(project_root: Path) -> Path:
    """Get test data directory."""
    path = project_root / "tests" / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests."""
    from src.core.model_manager import ModelManager

    yield

    # Reset after test
    ModelManager.reset()


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables."""
    monkeypatch.setenv("ROYALAUDIT_ENVIRONMENT", "development")
    monkeypatch.setenv("ROYALAUDIT_DEBUG", "true")


@pytest.fixture
def sample_invoice_image(test_data_dir: Path) -> Path:
    """Create a sample invoice image for testing."""
    from PIL import Image, ImageDraw

    img_path = test_data_dir / "sample_invoice.jpg"

    if not img_path.exists():
        # Create a simple test image
        img = Image.new("RGB", (800, 1000), color="white")
        draw = ImageDraw.Draw(img)

        # Add some text-like rectangles
        draw.rectangle([50, 50, 200, 80], outline="black")
        draw.rectangle([50, 100, 300, 130], outline="black")
        draw.rectangle([500, 800, 700, 850], outline="black")

        img.save(img_path, "JPEG")

    return img_path
