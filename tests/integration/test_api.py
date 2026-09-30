"""API tests through FastAPI's TestClient, with a fake model behind the real pipeline."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from fastapi.testclient import TestClient

from invoice_digitizer import __version__
from invoice_digitizer.api.main import create_app
from invoice_digitizer.config.settings import Settings
from invoice_digitizer.core.model_manager import ModelManager
from tests.conftest import FakeLoader, FakePredictor

ADMIN_KEY = "test-admin-key"


@pytest.fixture
def api_settings(tmp_path: Path) -> Settings:
    """Test settings: small upload limit, admin key set, no rate limit."""
    return Settings(
        environment="development",
        debug=True,
        model={"weights_path": str(tmp_path / "missing.pt")},
        preprocessing={"max_file_size_mb": 1},
        api={"rate_limit_enabled": False, "admin_api_key": ADMIN_KEY, "max_batch_files": 3},
    )


def _client(settings: Settings, loader: FakeLoader) -> Iterator[TestClient]:
    app = create_app(settings, ModelManager(settings.model, loader=loader))
    with TestClient(app) as client:  # runs the lifespan, so the model preloads
        yield client


@pytest.fixture
def client(api_settings: Settings, fake_loader: FakeLoader) -> Iterator[TestClient]:
    yield from _client(api_settings, fake_loader)


def _upload(content: bytes, name: str = "page.png", content_type: str = "image/png") -> dict:
    return {"file": (name, content, content_type)}


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_check(self, client: TestClient) -> None:
        """Test main health endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == __version__
        assert "uptime_seconds" in data
        assert data["components"][0]["name"] == "ml_model"

    def test_readiness_check(self, client: TestClient) -> None:
        """Test readiness probe."""
        response = client.get("/health/ready")

        assert response.status_code == 200
        assert response.json()["status"] == "ready"

    def test_liveness_check(self, client: TestClient) -> None:
        """Test liveness probe."""
        response = client.get("/health/live")

        assert response.status_code == 200
        assert response.json()["status"] == "alive"

    def test_not_ready_until_the_model_is_loaded(
        self, api_settings: Settings, fake_loader: FakeLoader
    ) -> None:
        settings = api_settings.model_copy(
            update={"model": api_settings.model.model_copy(update={"preload": False})}
        )
        for client in _client(settings, fake_loader):
            assert client.get("/health/ready").status_code == 503
            assert client.get("/health").json()["status"] == "degraded"
            assert fake_loader.calls == 0

    def test_startup_fails_when_the_model_cannot_load(self, api_settings: Settings) -> None:
        def broken_loader(_: object) -> None:
            raise FileNotFoundError("No trained weights")

        app = create_app(api_settings, ModelManager(api_settings.model, loader=broken_loader))
        with pytest.raises(FileNotFoundError), TestClient(app):
            pass


class TestModelEndpoints:
    """Tests for model management endpoints."""

    def test_get_model_info(self, client: TestClient) -> None:
        """Test model info endpoint."""
        response = client.get("/api/v1/model/info")

        assert response.status_code == 200
        data = response.json()
        assert data["loaded"] is True
        assert data["hub_repo"] == "ultralytics/yolov5:v7.0"
        assert data["version"].startswith("fake/yolov5:test")
        assert "classes" in data

    def test_list_classes(self, client: TestClient) -> None:
        """Test classes listing endpoint."""
        response = client.get("/api/v1/classes")

        assert response.status_code == 200
        data = response.json()
        assert len(data["classes"]) == 6
        assert "Invoice Date" in data["classes"]
        assert "Total Amount" in data["classes"]

    def test_update_model_config(self, client: TestClient) -> None:
        """Test model config update (needs the admin key)."""
        response = client.put(
            "/api/v1/model/config",
            json={"confidence_threshold": 0.5},
            headers={"X-Admin-Key": ADMIN_KEY},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Thresholds updated"
        assert data["thresholds"]["confidence_threshold"] == 0.5
        assert data["thresholds"]["max_detections"] == 100

    def test_update_model_config_no_updates(self, client: TestClient) -> None:
        """Test model config update with no changes."""
        response = client.put("/api/v1/model/config", json={}, headers={"X-Admin-Key": ADMIN_KEY})

        assert response.status_code == 200
        assert response.json()["message"] == "No updates provided"

    def test_unsupported_config_fields_are_rejected(self, client: TestClient) -> None:
        """image_size and half_precision cannot change at runtime, so they are refused."""
        response = client.put(
            "/api/v1/model/config", json={"image_size": 320}, headers={"X-Admin-Key": ADMIN_KEY}
        )
        assert response.status_code == 422

    @pytest.mark.parametrize("headers", [{}, {"X-Admin-Key": "wrong"}])
    def test_admin_endpoints_need_the_key(self, client: TestClient, headers: dict) -> None:
        assert client.put("/api/v1/model/config", json={}, headers=headers).status_code == 401
        assert client.post("/api/v1/model/reload", headers=headers).status_code == 401

    def test_admin_endpoints_are_off_without_a_configured_key(
        self, api_settings: Settings, fake_loader: FakeLoader
    ) -> None:
        settings = api_settings.model_copy(
            update={"api": api_settings.api.model_copy(update={"admin_api_key": None})}
        )
        for client in _client(settings, fake_loader):
            response = client.post("/api/v1/model/reload", headers={"X-Admin-Key": ""})
            assert response.status_code == 403
            assert "disabled" in response.json()["detail"]

    def test_reload(self, client: TestClient, fake_loader: FakeLoader) -> None:
        response = client.post("/api/v1/model/reload", headers={"X-Admin-Key": ADMIN_KEY})

        assert response.status_code == 200
        assert response.json()["model"]["source"] == "fake weights #2"
        assert fake_loader.calls == 2

    def test_failed_reload_reports_and_keeps_serving(
        self, api_settings: Settings, fake_predictor: FakePredictor
    ) -> None:
        class OneShotLoader(FakeLoader):
            def __call__(self, settings: object):  # type: ignore[override]
                if self.calls >= 1:
                    raise RuntimeError("bad weights")
                return super().__call__(settings)

        for client in _client(api_settings, OneShotLoader(fake_predictor)):
            response = client.post("/api/v1/model/reload", headers={"X-Admin-Key": ADMIN_KEY})
            assert response.status_code == 500
            assert "previous model is still serving" in response.json()["detail"]
            assert client.get("/health/ready").status_code == 200


class TestInferenceEndpoints:
    """Tests for inference endpoints."""

    def test_inference(
        self, client: TestClient, invoice_png_bytes: bytes, fake_predictor: FakePredictor
    ) -> None:
        response = client.post("/api/v1/inference", files=_upload(invoice_png_bytes))

        assert response.status_code == 200
        body = response.json()
        assert body["request_id"] == response.headers["X-Request-ID"]
        labels = [d["label"] for d in body["result"]["detections"]]
        assert labels == ["Invoice Date", "Total Amount", "VAT Amount"]
        assert body["result"]["metadata"]["ignored_detections"] == 2
        assert body["result"]["metadata"]["image_source"] == "upload"
        assert all(d["extracted_text"] is None for d in body["result"]["detections"])
        assert fake_predictor.calls[0]["shape"] == (1000, 800, 3)

    def test_per_request_threshold_does_not_leak(
        self, client: TestClient, invoice_png_bytes: bytes
    ) -> None:
        strict = client.post(
            "/api/v1/inference?confidence_threshold=0.9", files=_upload(invoice_png_bytes)
        )
        default = client.post("/api/v1/inference", files=_upload(invoice_png_bytes))

        assert len(strict.json()["result"]["detections"]) == 1
        assert len(default.json()["result"]["detections"]) == 3

    def test_inference_invalid_file_type(self, client: TestClient) -> None:
        """Test inference with invalid file type."""
        response = client.post("/api/v1/inference", files=_upload(b"hello", "a.txt", "text/plain"))

        assert response.status_code == 415

    def test_inference_file_too_large(self, client: TestClient, api_settings: Settings) -> None:
        """Test inference with oversized file."""
        max_size = api_settings.preprocessing.max_file_size_mb * 1024 * 1024
        large_content = b"x" * (max_size + 1000)

        response = client.post("/api/v1/inference", files=_upload(large_content, "big.jpg"))

        assert response.status_code == 413

    def test_inference_undecodable_image(self, client: TestClient) -> None:
        response = client.post("/api/v1/inference", files=_upload(b"not a png"))

        assert response.status_code == 400
        assert "could not be decoded" in response.json()["detail"]

    def test_model_failure_is_a_500_without_internals(
        self, api_settings: Settings, invoice_png_bytes: bytes
    ) -> None:
        broken = FakeLoader(FakePredictor(error=RuntimeError("secret internal detail")))
        for client in _client(api_settings, broken):
            response = client.post("/api/v1/inference", files=_upload(invoice_png_bytes))
            assert response.status_code == 500
            assert "secret internal detail" not in response.text
            assert response.headers["X-Request-ID"] in response.json()["detail"]

    def test_batch_partial(self, client: TestClient, invoice_png_bytes: bytes) -> None:
        files = [
            ("files", ("one.png", invoice_png_bytes, "image/png")),
            ("files", ("notes.txt", b"hello", "text/plain")),
            ("files", ("two.png", invoice_png_bytes, "image/png")),
        ]

        response = client.post("/api/v1/inference/batch", files=files)

        assert response.status_code == 200
        body = response.json()
        assert body["job_status"] == "partial"
        assert (body["total_images"], body["processed_images"], body["failed_images"]) == (3, 2, 1)
        assert body["errors"][0]["field"] == "notes.txt"

    def test_batch_all_failed(self, client: TestClient) -> None:
        files = [("files", ("a.txt", b"x", "text/plain"))]
        body = client.post("/api/v1/inference/batch", files=files).json()
        assert body["job_status"] == "failed"
        assert body["status"] == "error"

    def test_batch_inference_error_is_reported_per_file(
        self, api_settings: Settings, invoice_png_bytes: bytes
    ) -> None:
        broken = FakeLoader(FakePredictor(error=RuntimeError("boom")))
        files = [("files", ("one.png", invoice_png_bytes, "image/png"))]
        for client in _client(api_settings, broken):
            body = client.post("/api/v1/inference/batch", files=files).json()
            assert body["job_status"] == "failed"
            assert body["errors"][0] == {
                "code": "PROCESSING_ERROR",
                "message": "Inference failed.",
                "field": "one.png",
                "details": None,
            }

    def test_batch_size_limit(self, client: TestClient, invoice_png_bytes: bytes) -> None:
        files = [("files", (f"{i}.png", invoice_png_bytes, "image/png")) for i in range(4)]
        response = client.post("/api/v1/inference/batch", files=files)
        assert response.status_code == 400


class TestCrossCutting:
    """Rate limiting, metrics and request IDs."""

    def test_rate_limit(self, api_settings: Settings, fake_loader: FakeLoader) -> None:
        settings = api_settings.model_copy(
            update={
                "api": api_settings.api.model_copy(
                    update={"rate_limit_enabled": True, "requests_per_minute": 2}
                )
            }
        )
        for client in _client(settings, fake_loader):
            codes = [client.get("/api/v1/classes").status_code for _ in range(3)]
            assert codes == [200, 200, 429]
            assert client.get("/health/live").status_code == 200  # health is exempt

    def test_metrics_use_route_templates(
        self, client: TestClient, invoice_png_bytes: bytes
    ) -> None:
        client.post("/api/v1/inference", files=_upload(invoice_png_bytes))
        client.get("/no/such/path")

        text = client.get("/metrics").text

        assert 'route="/api/v1/inference"' in text
        assert 'route="unmatched"' in text
        assert "inference_images_total" in text

    def test_metrics_can_be_switched_off(
        self, api_settings: Settings, fake_loader: FakeLoader
    ) -> None:
        settings = api_settings.model_copy(
            update={
                "monitoring": api_settings.monitoring.model_copy(
                    update={"prometheus_enabled": False}
                )
            }
        )
        for client in _client(settings, fake_loader):
            assert client.get("/metrics").status_code == 404


class TestDocsEndpoints:
    """Tests for documentation endpoints."""

    def test_openapi_json(self, client: TestClient) -> None:
        """Test OpenAPI JSON endpoint."""
        response = client.get("/openapi.json")

        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert data["info"]["title"] == "Invoice Field Detection API"
        assert data["info"]["version"] == __version__

    def test_docs_endpoint(self, client: TestClient) -> None:
        """Test Swagger docs endpoint."""
        response = client.get("/docs")

        assert response.status_code == 200

    def test_redoc_endpoint(self, client: TestClient) -> None:
        """Test ReDoc endpoint."""
        response = client.get("/redoc")

        assert response.status_code == 200
