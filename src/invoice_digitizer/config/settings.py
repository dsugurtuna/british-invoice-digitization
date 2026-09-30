"""Configuration management using Pydantic Settings."""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Application environment."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class DeviceType(str, Enum):
    """Supported compute devices."""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class LogLevel(str, Enum):
    """Logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ModelSettings(BaseSettings):
    """Model configuration settings."""

    architecture: str = "yolov5x"
    weights_path: str = "models/royal_audit_v2_best.pt"
    fallback_weights: str = "yolov5x.pt"
    confidence_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    max_detections: int = Field(default=100, ge=1)
    image_size: int = Field(default=640, ge=32)
    device: DeviceType = DeviceType.AUTO
    half_precision: bool = True
    classes: list[str] = Field(
        default_factory=lambda: [
            "Invoice Date",
            "Invoice Number",
            "Vendor Name",
            "Total Amount",
            "VAT Amount",
            "Line Item",
        ]
    )


class PreprocessingSettings(BaseSettings):
    """Image preprocessing settings."""

    auto_orient: bool = True
    normalize: bool = True
    supported_formats: list[str] = Field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp", ".pdf"]
    )
    max_file_size_mb: int = Field(default=50, ge=1)
    max_image_dimension: int = Field(default=4096, ge=32)
    enhance_enabled: bool = True
    denoise: bool = True
    sharpen: bool = False
    contrast_adjustment: float = Field(default=1.1, ge=0.5, le=2.0)


class APISettings(BaseSettings):
    """API server settings."""

    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=4, ge=1)
    reload: bool = False
    max_upload_size_mb: int = Field(default=100, ge=1)
    request_timeout_seconds: int = Field(default=300, ge=1)
    rate_limit_enabled: bool = True
    requests_per_minute: int = Field(default=60, ge=1)
    cors_enabled: bool = True
    cors_allow_origins: list[str] = Field(default_factory=lambda: ["*"])


class MonitoringSettings(BaseSettings):
    """Monitoring and observability settings."""

    prometheus_enabled: bool = True
    prometheus_port: int = Field(default=9090, ge=1, le=65535)
    prometheus_path: str = "/metrics"
    log_format: str = "json"
    include_request_id: bool = True
    health_enabled: bool = True
    health_path: str = "/health"


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_prefix="ROYALAUDIT_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    # Application settings
    app_name: str = "RoyalAudit Digitizer"
    version: str = "2.0.0"
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    log_level: LogLevel = LogLevel.INFO

    # Nested settings
    model: ModelSettings = Field(default_factory=ModelSettings)
    preprocessing: PreprocessingSettings = Field(default_factory=PreprocessingSettings)
    api: APISettings = Field(default_factory=APISettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)

    # Paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)
    config_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "config")
    models_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "models")
    data_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "data")
    output_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "output")

    @model_validator(mode="after")
    def validate_paths(self) -> "Settings":
        """Ensure required directories exist."""
        for path_attr in ["models_dir", "output_dir"]:
            path = getattr(self, path_attr)
            path.mkdir(parents=True, exist_ok=True)
        return self

    @field_validator("environment", mode="before")
    @classmethod
    def validate_environment(cls, v: Any) -> Environment:
        """Validate environment setting."""
        if isinstance(v, str):
            return Environment(v.lower())
        return v

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == Environment.PRODUCTION

    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == Environment.DEVELOPMENT

    @classmethod
    def from_yaml(cls, config_path: str | Path | None = None) -> "Settings":
        """Load settings from a YAML configuration file.

        Args:
            config_path: Path to YAML config file. If None, uses default.yaml.

        Returns:
            Settings instance populated from YAML and environment variables.
        """
        # Determine config path
        if config_path is None:
            config_dir = Path(__file__).parent.parent.parent / "config"
            env = os.getenv("ROYALAUDIT_ENVIRONMENT", "development").lower()
            config_path = config_dir / f"{env}.yaml"
            if not config_path.exists():
                config_path = config_dir / "default.yaml"

        config_path = Path(config_path)

        # Load YAML if exists
        yaml_config: dict[str, Any] = {}
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                yaml_config = yaml.safe_load(f) or {}

        # Flatten nested config for pydantic
        flat_config = cls._flatten_config(yaml_config)

        return cls(**flat_config)

    @staticmethod
    def _flatten_config(config: dict[str, Any], parent_key: str = "") -> dict[str, Any]:
        """Flatten nested configuration dictionary."""
        items: list[tuple[str, Any]] = []
        for key, value in config.items():
            new_key = f"{parent_key}_{key}" if parent_key else key
            if isinstance(value, dict) and not any(
                isinstance(v, dict) for v in value.values()
            ):
                # This is a leaf dict, use it as-is
                items.append((key, value))
            elif isinstance(value, dict):
                items.extend(Settings._flatten_config(value, key).items())
            else:
                items.append((new_key, value))
        return dict(items)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Returns:
        Singleton Settings instance.
    """
    return Settings.from_yaml()


def reload_settings() -> Settings:
    """Reload settings (clears cache).

    Returns:
        Fresh Settings instance.
    """
    get_settings.cache_clear()
    return get_settings()
