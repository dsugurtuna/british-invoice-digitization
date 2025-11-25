"""Integration tests for API endpoints."""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np

from src.api.main import create_app
from src.config.settings import Settings


@pytest.fixture
def settings() -> Settings:
    """Create test settings."""
    return Settings(
        environment="development",
        debug=True,
    )


@pytest.fixture
def mock_model_manager():
    """Create mock model manager."""
    with patch("src.api.main.ModelManager") as mock:
        manager = MagicMock()
        manager.model = MagicMock()
        manager.device = "cpu"
        manager.model_version = "test_v1.0"
        manager.get_model_info.return_value = {
            "architecture": "yolov5x",
            "version": "test_v1.0",
            "device": "cpu",
            "classes": ["Invoice Date", "Invoice Number", "Vendor Name", "Total Amount", "VAT Amount", "Line Item"],
        }
        mock.return_value = manager
        yield manager


@pytest.fixture
def client(settings: Settings, mock_model_manager) -> TestClient:
    """Create test client."""
    app = create_app(settings)
    app.state.model_manager = mock_model_manager
    app.state.start_time = 0
    return TestClient(app)


@pytest.fixture
def sample_image() -> bytes:
    """Create a sample test image."""
    img = Image.new("RGB", (640, 480), color="white")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer.getvalue()


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_check(self, client: TestClient) -> None:
        """Test main health endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "uptime_seconds" in data
        assert "components" in data

    def test_readiness_check(self, client: TestClient) -> None:
        """Test readiness probe."""
        response = client.get("/health/ready")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"

    def test_liveness_check(self, client: TestClient) -> None:
        """Test liveness probe."""
        response = client.get("/health/live")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"


class TestModelEndpoints:
    """Tests for model management endpoints."""

    def test_get_model_info(self, client: TestClient) -> None:
        """Test model info endpoint."""
        response = client.get("/api/v1/model/info")

        assert response.status_code == 200
        data = response.json()
        assert "architecture" in data
        assert "version" in data
        assert "classes" in data

    def test_list_classes(self, client: TestClient) -> None:
        """Test classes listing endpoint."""
        response = client.get("/api/v1/classes")

        assert response.status_code == 200
        data = response.json()
        assert "classes" in data
        assert len(data["classes"]) == 6
        assert "Invoice Date" in data["classes"]
        assert "Total Amount" in data["classes"]

    def test_update_model_config(self, client: TestClient) -> None:
        """Test model config update."""
        response = client.put(
            "/api/v1/model/config",
            json={"confidence_threshold": 0.5},
        )

        assert response.status_code == 200
        data = response.json()
        assert "message" in data

    def test_update_model_config_no_updates(self, client: TestClient) -> None:
        """Test model config update with no changes."""
        response = client.put(
            "/api/v1/model/config",
            json={},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "No updates provided"


class TestInferenceEndpoints:
    """Tests for inference endpoints."""

    def test_inference_invalid_file_type(self, client: TestClient) -> None:
        """Test inference with invalid file type."""
        response = client.post(
            "/api/v1/inference",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )

        assert response.status_code == 415

    def test_inference_file_too_large(
        self,
        client: TestClient,
        settings: Settings,
    ) -> None:
        """Test inference with oversized file."""
        # Create a file that exceeds the limit
        max_size = settings.preprocessing.max_file_size_mb * 1024 * 1024
        large_content = b"x" * (max_size + 1000)

        response = client.post(
            "/api/v1/inference",
            files={"file": ("large.jpg", large_content, "image/jpeg")},
        )

        assert response.status_code == 413


class TestDocsEndpoints:
    """Tests for documentation endpoints."""

    def test_openapi_json(self, client: TestClient) -> None:
        """Test OpenAPI JSON endpoint."""
        response = client.get("/openapi.json")

        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert data["info"]["title"] == "RoyalAudit Digitizer API"

    def test_docs_endpoint(self, client: TestClient) -> None:
        """Test Swagger docs endpoint."""
        response = client.get("/docs")

        assert response.status_code == 200

    def test_redoc_endpoint(self, client: TestClient) -> None:
        """Test ReDoc endpoint."""
        response = client.get("/redoc")

        assert response.status_code == 200
