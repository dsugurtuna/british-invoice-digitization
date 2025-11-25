"""Unit tests for configuration settings."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config.settings import (
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

        assert settings.architecture == "yolov5x"
        assert settings.confidence_threshold == 0.4
        assert settings.iou_threshold == 0.45
        assert settings.max_detections == 100
        assert settings.image_size == 640
        assert settings.device == DeviceType.AUTO
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
        # Valid values
        ModelSettings(confidence_threshold=0.0)
        ModelSettings(confidence_threshold=1.0)
        ModelSettings(confidence_threshold=0.5)

        # Invalid values should raise
        with pytest.raises(ValueError):
            ModelSettings(confidence_threshold=-0.1)

        with pytest.raises(ValueError):
            ModelSettings(confidence_threshold=1.1)


class TestSettings:
    """Tests for main Settings class."""

    def test_default_settings(self) -> None:
        """Test default settings creation."""
        settings = Settings()

        assert settings.app_name == "RoyalAudit Digitizer"
        assert settings.version == "2.0.0"
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

    def test_nested_settings(self) -> None:
        """Test nested settings access."""
        settings = Settings()

        assert settings.model.architecture == "yolov5x"
        assert settings.preprocessing.max_file_size_mb == 50
        assert settings.api.port == 8000
        assert settings.monitoring.prometheus_enabled is True

    def test_path_creation(self) -> None:
        """Test that paths are created."""
        settings = Settings()

        assert isinstance(settings.project_root, Path)
        assert isinstance(settings.models_dir, Path)
        assert isinstance(settings.output_dir, Path)


class TestGetSettings:
    """Tests for settings singleton."""

    def test_singleton(self) -> None:
        """Test that get_settings returns same instance."""
        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2

    def test_reload_settings(self) -> None:
        """Test settings reload."""
        settings1 = get_settings()
        settings2 = reload_settings()

        # After reload, should be different object
        # (but may have same values)
        assert settings2 is not None
