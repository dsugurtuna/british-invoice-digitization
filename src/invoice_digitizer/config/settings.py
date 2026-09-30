"""Application settings.

Values come from three places, highest priority first:

1. Keyword arguments passed to ``Settings(...)`` (used by tests).
2. Environment variables prefixed ``INVOICE_DIGITIZER_``. Nested fields use a double
   underscore, for example ``INVOICE_DIGITIZER_MODEL__CONFIDENCE_THRESHOLD=0.5``.
3. An optional YAML file named by ``INVOICE_DIGITIZER_CONFIG_FILE``.

Anything not set falls back to the defaults below.
"""

from __future__ import annotations

import os

from enum import StrEnum
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from invoice_digitizer._version import __version__

ENV_PREFIX = "INVOICE_DIGITIZER_"
CONFIG_FILE_ENV = f"{ENV_PREFIX}CONFIG_FILE"

DEFAULT_CLASSES = [
    "Invoice Date",
    "Invoice Number",
    "Vendor Name",
    "Total Amount",
    "VAT Amount",
    "Line Item",
]


class Environment(StrEnum):
    """Deployment environment."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class DeviceType(StrEnum):
    """Compute device for inference."""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class LogLevel(StrEnum):
    """Logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# Nested sections are plain BaseModels, not BaseSettings. A nested BaseSettings
# reads environment variables on its own, without the prefix, so an unrelated
# variable such as PORT or DEVICE would silently change the configuration.


class ModelSettings(BaseModel):
    """How the detection model is found, loaded and run."""

    hub_repo: str = Field(
        default="ultralytics/yolov5:v7.0",
        description="torch.hub repo pinned to a release tag, or a local directory.",
    )
    hub_source: str = Field(default="github", pattern="^(github|local)$")
    weights_path: str = Field(
        default="models/invoice_fields.pt",
        description="Trained YOLOv5 weights. Not shipped with this repository.",
    )
    allow_pretrained_fallback: bool = Field(
        default=False,
        description=(
            "If the weights file is missing, load generic COCO-pretrained weights instead. "
            "Only useful to smoke-test the serving path: COCO classes are not invoice "
            "fields, so every detection is reported as ignored."
        ),
    )
    architecture: str = Field(default="yolov5s", description="Used only by the fallback.")
    confidence_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    max_detections: int = Field(default=100, ge=1)
    image_size: int = Field(default=640, ge=32)
    device: DeviceType = DeviceType.AUTO
    half_precision: bool = Field(default=False, description="FP16 inference, CUDA only.")
    preload: bool = Field(default=True, description="Load the model when the API starts.")
    classes: list[str] = Field(default_factory=lambda: list(DEFAULT_CLASSES))


class PreprocessingSettings(BaseModel):
    """Input validation limits."""

    supported_formats: list[str] = Field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"]
    )
    max_file_size_mb: int = Field(default=50, ge=1)
    max_image_dimension: int = Field(default=8192, ge=32)


class APISettings(BaseModel):
    """HTTP server behaviour."""

    host: str = Field(default="127.0.0.1", description="Use 0.0.0.0 only inside a container.")
    port: int = Field(default=8000, ge=1, le=65535)
    rate_limit_enabled: bool = True
    requests_per_minute: int = Field(default=60, ge=1)
    cors_allow_origins: list[str] = Field(
        default_factory=list, description="Empty means CORS is off."
    )
    max_batch_files: int = Field(default=20, ge=1)
    admin_api_key: SecretStr | None = Field(
        default=None,
        description=(
            "Required in the X-Admin-Key header for endpoints that change the running "
            "service (threshold updates, model reload). Unset means those endpoints are off."
        ),
    )


class MonitoringSettings(BaseModel):
    """Observability switches."""

    prometheus_enabled: bool = Field(default=True, description="Serve /metrics.")


class Settings(BaseSettings):
    """Top-level application settings."""

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "British Invoice Digitisation"
    version: str = __version__
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    log_level: LogLevel = LogLevel.INFO

    model: ModelSettings = Field(default_factory=ModelSettings)
    preprocessing: PreprocessingSettings = Field(default_factory=PreprocessingSettings)
    api: APISettings = Field(default_factory=APISettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)

    @field_validator("environment", mode="before")
    @classmethod
    def _lowercase_environment(cls, value: Any) -> Any:
        return value.lower() if isinstance(value, str) else value

    @property
    def is_production(self) -> bool:
        """True when running in production."""
        return self.environment == Environment.PRODUCTION

    @property
    def is_development(self) -> bool:
        """True when running in development."""
        return self.environment == Environment.DEVELOPMENT

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,  # noqa: ARG003 - fixed signature
        file_secret_settings: PydanticBaseSettingsSource,  # noqa: ARG003
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Order the sources so environment variables override the YAML file."""
        sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings]
        config_file = os.environ.get(CONFIG_FILE_ENV)
        if config_file:
            sources.append(YamlConfigSettingsSource(settings_cls, yaml_file=config_file))
        return tuple(sources)


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings, built once."""
    return Settings()


def reload_settings() -> Settings:
    """Clear the cache and build the settings again."""
    get_settings.cache_clear()
    return get_settings()
