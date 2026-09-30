"""Model lifecycle for one process: load once, run safely, reload without a gap."""

from __future__ import annotations

import threading
import time

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import structlog

from invoice_digitizer.core.yolov5_backend import LoadedModel, RawDetection, load_yolov5

if TYPE_CHECKING:
    import numpy as np

    from numpy.typing import NDArray

    from invoice_digitizer.config.settings import ModelSettings

logger = structlog.get_logger(__name__)

Loader = Callable[["ModelSettings"], LoadedModel]


class ModelManager:
    """Owns the detection model for one process.

    Behaviour, and the reason for each part:

    * Loads lazily and exactly once, even when several threads ask at the same time,
      because loading YOLOv5 twice wastes memory and start-up time.
    * Serialises predictions with a lock, because the YOLOv5 wrapper keeps its
      thresholds as mutable attributes; two concurrent calls with different
      thresholds would otherwise race. This makes inference correct, not parallel.
    * Reloads by loading the new model first and then swapping the reference, so
      requests keep being served by the old model until the new one is ready. If the
      new load fails, the old model stays in place.
    * Threshold overrides live on the manager, not on the model, so a per-request
      override never leaks into later requests.

    The state is per process. With several worker processes, each has its own model
    and its own thresholds, and a reload only reaches the process that handled it.

    Args:
        settings: Model settings.
        loader: Builds a ``LoadedModel``. Defaults to YOLOv5 via torch.hub; tests
            inject a fake so no weights or GPU are needed.
    """

    def __init__(self, settings: ModelSettings, loader: Loader | None = None) -> None:
        self._settings = settings
        self._loader: Loader = loader or load_yolov5
        self._load_lock = threading.Lock()
        self._predict_lock = threading.Lock()
        self._loaded: LoadedModel | None = None
        self._confidence = settings.confidence_threshold
        self._iou = settings.iou_threshold
        self._max_detections = settings.max_detections

    @property
    def settings(self) -> ModelSettings:
        """The model settings this manager was built with."""
        return self._settings

    @property
    def is_loaded(self) -> bool:
        """True once a model is in memory."""
        return self._loaded is not None

    def load(self) -> LoadedModel:
        """Return the loaded model, loading it on first use."""
        loaded = self._loaded
        if loaded is not None:
            return loaded
        with self._load_lock:
            if self._loaded is None:
                self._loaded = self._timed_load()
            return self._loaded

    def reload(self) -> LoadedModel:
        """Load the model again from disk and swap it in once it is ready."""
        with self._load_lock:
            new_model = self._timed_load()
            self._loaded = new_model
        return new_model

    def predict(
        self,
        image: NDArray[np.uint8],
        *,
        confidence: float | None = None,
        iou: float | None = None,
    ) -> tuple[list[RawDetection], LoadedModel]:
        """Run the model on one RGB image.

        Args:
            image: H x W x 3 uint8 RGB array.
            confidence: Override the confidence threshold for this call only.
            iou: Override the NMS IoU threshold for this call only.

        Returns:
            The raw detections and the model that produced them.
        """
        loaded = self.load()
        with self._predict_lock:
            detections = loaded.predictor(
                image,
                image_size=self._settings.image_size,
                confidence=self._confidence if confidence is None else confidence,
                iou=self._iou if iou is None else iou,
                max_detections=self._max_detections,
            )
        return detections, loaded

    def update_thresholds(
        self,
        *,
        confidence: float | None = None,
        iou: float | None = None,
        max_detections: int | None = None,
    ) -> dict[str, float | int]:
        """Change the default thresholds used by later calls in this process."""
        with self._predict_lock:
            if confidence is not None:
                self._confidence = confidence
            if iou is not None:
                self._iou = iou
            if max_detections is not None:
                self._max_detections = max_detections
            current = self._thresholds()
        logger.info("thresholds_updated", **current)
        return current

    def thresholds(self) -> dict[str, float | int]:
        """The thresholds currently applied to calls without overrides."""
        with self._predict_lock:
            return self._thresholds()

    def info(self) -> dict[str, Any]:
        """Describe the model without forcing it to load."""
        loaded = self._loaded
        return {
            "loaded": loaded is not None,
            "hub_repo": self._settings.hub_repo,
            "weights_path": self._settings.weights_path,
            "source": loaded.source if loaded else None,
            "version": loaded.version if loaded else None,
            "weights_sha256": loaded.weights_sha256 if loaded else None,
            "device": loaded.device if loaded else None,
            "image_size": self._settings.image_size,
            "classes": list(self._settings.classes),
            **self.thresholds(),
        }

    def _thresholds(self) -> dict[str, float | int]:
        return {
            "confidence_threshold": self._confidence,
            "iou_threshold": self._iou,
            "max_detections": self._max_detections,
        }

    def _timed_load(self) -> LoadedModel:
        start = time.perf_counter()
        loaded = self._loader(self._settings)
        logger.info(
            "model_ready",
            version=loaded.version,
            device=loaded.device,
            load_seconds=round(time.perf_counter() - start, 3),
        )
        return loaded
