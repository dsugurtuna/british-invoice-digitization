"""Unit tests for configuration settings."""

from __future__ import annotations

from pathlib import Path

import pytest

from pydantic import ValidationError

from invoice_digitizer import __version__
from invoice_digitizer.config.settings import (
    CONFIG_FILE_ENV,
    DeviceType,
    Environment,
    LogLevel,
    ModelSettings,
    Settings,
    get_settings,
    reload_settings,
)


class TestEnvironment:
    """Tests for Environment enum."""

    def test_all_environments(self) -> None:
        """Test all environments exist."""
        assert Environment.DEVELOPMENT == "development"
        assert Environment.STAGING == "staging"
        assert Environment.PRODUCTION == "production"


class TestDeviceType:
    """Tests for DeviceType enum."""

    def test_all_device_types(self) -> None:
        """Test all device types exist."""
        assert DeviceType.AUTO == "auto"
        assert DeviceType.CPU == "cpu"
        assert DeviceType.CUDA == "cuda"
        assert DeviceType.MPS == "mps"


class TestModelSettings:
    """Tests for ModelSettings."""

    def test_default_values(self) -> None:
        """Test default model settings."""
        settings = ModelSettings()

        assert settings.hub_repo == "ultralytics/yolov5:v7.0"
        assert settings.confidence_threshold == 0.4
        assert settings.iou_threshold == 0.45
        assert settings.max_detections == 100
        assert settings.image_size == 640
        assert settings.device == DeviceType.AUTO
        assert settings.allow_pretrained_fallback is False
        assert len(settings.classes) == 6

    def test_custom_values(self) -> None:
        """Test custom model settings."""
        settings = ModelSettings(
            architecture="yolov5m",
            confidence_threshold=0.5,
            iou_threshold=0.5,
            image_size=1280,
        )

        assert settings.architecture == "yolov5m"
        assert settings.confidence_threshold == 0.5
        assert settings.iou_threshold == 0.5
        assert settings.image_size == 1280

    def test_confidence_bounds(self) -> None:
        """Test confidence threshold bounds."""
        ModelSettings(confidence_threshold=0.0)
        ModelSettings(confidence_threshold=1.0)
        ModelSettings(confidence_threshold=0.5)

        with pytest.raises(ValidationError):
            ModelSettings(confidence_threshold=-0.1)

        with pytest.raises(ValidationError):
            ModelSettings(confidence_threshold=1.1)

    def test_hub_source_is_restricted(self) -> None:
        with pytest.raises(ValidationError):
            ModelSettings(hub_source="somewhere-else")


class TestSettings:
    """Tests for main Settings class."""

    def test_default_settings(self) -> None:
        """Test default settings creation."""
        settings = Settings()

        assert settings.app_name == "British Invoice Digitisation"
        assert settings.version == __version__
        assert settings.environment == Environment.DEVELOPMENT
        assert settings.debug is False
        assert settings.log_level == LogLevel.INFO

    def test_is_production(self) -> None:
        """Test production check."""
        dev_settings = Settings(environment=Environment.DEVELOPMENT)
        prod_settings = Settings(environment=Environment.PRODUCTION)

        assert dev_settings.is_production is False
        assert prod_settings.is_production is True

    def test_is_development(self) -> None:
        """Test development check."""
        dev_settings = Settings(environment=Environment.DEVELOPMENT)
        prod_settings = Settings(environment=Environment.PRODUCTION)

        assert dev_settings.is_development is True
        assert prod_settings.is_development is False

    def test_environment_is_case_insensitive(self) -> None:
        assert Settings(environment="Production").environment == Environment.PRODUCTION

    def test_nested_settings(self) -> None:
        """Test nested settings access."""
        settings = Settings()

        assert settings.model.hub_repo == "ultralytics/yolov5:v7.0"
        assert settings.preprocessing.max_file_size_mb == 50
        assert settings.api.port == 8000
        assert settings.monitoring.prometheus_enabled is True

    def test_secure_defaults(self) -> None:
        """Bind to localhost, CORS off and admin endpoints off unless configured."""
        settings = Settings()

        assert settings.api.host == "127.0.0.1"
        assert settings.api.cors_allow_origins == []
        assert settings.api.admin_api_key is None

    def test_no_filesystem_side_effects(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Building settings must not create directories (the old version did)."""
        monkeypatch.chdir(tmp_path)
        Settings()
        assert list(tmp_path.iterdir()) == []


class TestSettingsSources:
    """Environment variables and the optional YAML file."""

    def test_env_var_overrides_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("INVOICE_DIGITIZER_MODEL__CONFIDENCE_THRESHOLD", "0.65")
        monkeypatch.setenv("INVOICE_DIGITIZER_API__ADMIN_API_KEY", "s3cret")

        settings = Settings()

        assert settings.model.confidence_threshold == 0.65
        assert settings.api.admin_api_key is not None
        assert settings.api.admin_api_key.get_secret_value() == "s3cret"
        assert "s3cret" not in repr(settings)

    def test_unprefixed_env_vars_are_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Generic names such as PORT or DEVICE must not change the configuration."""
        monkeypatch.setenv("PORT", "1234")
        monkeypatch.setenv("DEVICE", "cuda")

        settings = Settings()

        assert settings.api.port == 8000
        assert settings.model.device == DeviceType.AUTO

    def test_yaml_file_is_loaded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        config = tmp_path / "config.yaml"
        config.write_text("model:\n  image_size: 1024\napi:\n  port: 9000\n", encoding="utf-8")
        monkeypatch.setenv(CONFIG_FILE_ENV, str(config))

        settings = Settings()

        assert settings.model.image_size == 1024
        assert settings.api.port == 9000
        assert settings.model.confidence_threshold == 0.4  # untouched default

    def test_env_var_beats_yaml(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        config = tmp_path / "config.yaml"
        config.write_text("model:\n  image_size: 1024\n  iou_threshold: 0.3\n", encoding="utf-8")
        monkeypatch.setenv(CONFIG_FILE_ENV, str(config))
        monkeypatch.setenv("INVOICE_DIGITIZER_MODEL__IMAGE_SIZE", "512")

        settings = Settings()

        assert settings.model.image_size == 512
        assert settings.model.iou_threshold == 0.3

    def test_shipped_example_config_is_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        example = Path(__file__).parents[2] / "config" / "default.yaml"
        monkeypatch.setenv(CONFIG_FILE_ENV, str(example))

        settings = Settings()

        assert settings.model.weights_path == "models/invoice_fields.pt"
        assert settings.model.classes == ModelSettings().classes


class TestGetSettings:
    """Tests for settings singleton."""

    def test_singleton(self) -> None:
        """Test that get_settings returns same instance."""
        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2

    def test_reload_settings(self) -> None:
        """Reload returns a freshly built object."""
        settings1 = get_settings()
        settings2 = reload_settings()

        assert settings2 is not settings1
        assert settings2 == settings1
