"""Model management and lifecycle handling."""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
import torch

if TYPE_CHECKING:
    from src.config.settings import Settings

logger = structlog.get_logger(__name__)


class ModelManager:
    """Thread-safe singleton manager for YOLOv5 model lifecycle.

    Handles model loading, device management, and provides a single point
    of access to the inference model across the application.

    This implements the Singleton pattern to ensure only one model instance
    exists in memory, preventing GPU memory issues and ensuring consistency.

    Attributes:
        model: The loaded YOLOv5 model instance.
        device: The compute device (CPU/CUDA/MPS).
        model_hash: MD5 hash of the loaded weights for versioning.

    Example:
        >>> manager = ModelManager()
        >>> model = manager.get_model()
        >>> results = model(image)
    """

    _instance: "ModelManager | None" = None
    _lock: threading.Lock = threading.Lock()
    _initialized: bool = False

    def __new__(cls, settings: "Settings | None" = None) -> "ModelManager":
        """Create or return the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, settings: "Settings | None" = None) -> None:
        """Initialize the ModelManager.

        Args:
            settings: Application settings. If None, loads from default config.
        """
        # Prevent re-initialization
        if ModelManager._initialized:
            return

        with ModelManager._lock:
            if ModelManager._initialized:
                return

            # Import here to avoid circular imports
            from src.config.settings import get_settings

            self._settings = settings or get_settings()
            self._model: Any = None
            self._device: torch.device | None = None
            self._model_hash: str | None = None
            self._model_version: str = "unknown"

            logger.info(
                "ModelManager initialized",
                model_path=self._settings.model.weights_path,
                architecture=self._settings.model.architecture,
            )

            ModelManager._initialized = True

    @property
    def device(self) -> torch.device:
        """Get the compute device, initializing if necessary."""
        if self._device is None:
            self._device = self._resolve_device()
        return self._device

    @property
    def model(self) -> Any:
        """Get the model, loading if necessary."""
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def model_hash(self) -> str:
        """Get the model weights hash."""
        if self._model_hash is None:
            self._model_hash = "not_loaded"
        return self._model_hash

    @property
    def model_version(self) -> str:
        """Get the model version string."""
        return self._model_version

    def _resolve_device(self) -> torch.device:
        """Determine the optimal compute device.

        Returns:
            torch.device configured for the best available hardware.
        """
        device_setting = self._settings.model.device.value

        if device_setting == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(
                    "Auto-detected CUDA device",
                    gpu_name=torch.cuda.get_device_name(0),
                    gpu_memory=f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB",
                )
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
                logger.info("Auto-detected Apple MPS device")
            else:
                device = torch.device("cpu")
                logger.info("Using CPU device")
        else:
            device = torch.device(device_setting)
            logger.info("Using configured device", device=device_setting)

        return device

    def _load_model(self) -> None:
        """Load the YOLOv5 model from weights.

        Raises:
            FileNotFoundError: If weights file doesn't exist and fallback fails.
            RuntimeError: If model loading fails.
        """
        weights_path = Path(self._settings.model.weights_path)

        # Check for custom weights, fall back to pretrained
        if not weights_path.exists():
            logger.warning(
                "Custom weights not found, using fallback",
                custom_path=str(weights_path),
                fallback=self._settings.model.fallback_weights,
            )
            weights_path = Path(self._settings.model.fallback_weights)

        logger.info("Loading YOLOv5 model", weights_path=str(weights_path))

        try:
            # Load model using torch.hub for YOLOv5
            # For production, you might use ultralytics YOLO directly
            if weights_path.exists():
                self._model = torch.hub.load(
                    "ultralytics/yolov5",
                    "custom",
                    path=str(weights_path),
                    force_reload=False,
                    trust_repo=True,
                )
                self._model_hash = self._compute_file_hash(weights_path)
            else:
                # Load pretrained model
                self._model = torch.hub.load(
                    "ultralytics/yolov5",
                    self._settings.model.architecture,
                    pretrained=True,
                    trust_repo=True,
                )
                self._model_hash = f"pretrained_{self._settings.model.architecture}"

            # Configure model
            self._model.to(self.device)
            self._model.conf = self._settings.model.confidence_threshold
            self._model.iou = self._settings.model.iou_threshold
            self._model.max_det = self._settings.model.max_detections

            # Enable half precision if configured and on CUDA
            if self._settings.model.half_precision and self.device.type == "cuda":
                self._model.half()
                logger.info("Enabled FP16 half-precision inference")

            # Set to eval mode
            self._model.eval()

            self._model_version = f"yolov5_{self._settings.model.architecture}_v7.0"

            logger.info(
                "Model loaded successfully",
                device=str(self.device),
                model_hash=self._model_hash[:12],
                classes=len(self._settings.model.classes),
            )

        except Exception as e:
            logger.exception("Failed to load model", error=str(e))
            raise RuntimeError(f"Model loading failed: {e}") from e

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute MD5 hash of a file for versioning.

        Args:
            file_path: Path to the file.

        Returns:
            MD5 hex digest string.
        """
        md5 = hashlib.md5()  # noqa: S324 - Used for fingerprinting, not security
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5.update(chunk)
        return md5.hexdigest()

    def get_model(self) -> Any:
        """Get the loaded model instance.

        Returns:
            The YOLOv5 model ready for inference.
        """
        return self.model

    def reload_model(self) -> None:
        """Force reload the model from disk.

        Useful for hot-reloading updated weights without restart.
        """
        logger.info("Reloading model...")
        self._model = None
        self._model_hash = None
        self._load_model()
        logger.info("Model reloaded successfully")

    def update_thresholds(
        self,
        confidence: float | None = None,
        iou: float | None = None,
    ) -> None:
        """Update model inference thresholds.

        Args:
            confidence: New confidence threshold (0-1).
            iou: New IoU threshold for NMS (0-1).
        """
        if self._model is not None:
            if confidence is not None:
                self._model.conf = confidence
                logger.info("Updated confidence threshold", value=confidence)
            if iou is not None:
                self._model.iou = iou
                logger.info("Updated IoU threshold", value=iou)

    def get_model_info(self) -> dict[str, Any]:
        """Get model information and statistics.

        Returns:
            Dictionary with model metadata.
        """
        info = {
            "architecture": self._settings.model.architecture,
            "version": self._model_version,
            "hash": self._model_hash,
            "device": str(self.device),
            "half_precision": self._settings.model.half_precision and self.device.type == "cuda",
            "confidence_threshold": self._settings.model.confidence_threshold,
            "iou_threshold": self._settings.model.iou_threshold,
            "classes": self._settings.model.classes,
            "num_classes": len(self._settings.model.classes),
        }

        # Add GPU memory info if available
        if self.device.type == "cuda":
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated(0) / 1e6
            info["gpu_memory_cached_mb"] = torch.cuda.memory_reserved(0) / 1e6

        return info

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance. Primarily for testing."""
        with cls._lock:
            if cls._instance is not None:
                if cls._instance._model is not None:
                    del cls._instance._model
                cls._instance = None
                cls._initialized = False
                logger.info("ModelManager reset")
