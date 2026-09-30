"""Loading and running YOLOv5 through ``torch.hub``.

This is the only module that touches PyTorch, and it imports torch inside
``load_yolov5``. The API, schemas and pipeline therefore install and test without
the ML stack; only a process that really loads a model needs ``.[yolov5]``.

What is and is not verified: the adapters here are unit-tested against fakes
(see tests/unit/test_yolov5_backend.py). Loading real YOLOv5 code and weights
needs network access to GitHub and is not exercised by CI.
"""

from __future__ import annotations

import hashlib

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import structlog

if TYPE_CHECKING:
    import numpy as np

    from numpy.typing import NDArray

    from invoice_digitizer.config.settings import ModelSettings

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RawDetection:
    """One box as the model reports it, before it is mapped to an invoice field."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float
    class_name: str


class Predictor(Protocol):
    """Runs a detection model on one RGB image (H x W x 3, uint8).

    Implementations do not have to be thread-safe: ``ModelManager`` serialises calls.
    """

    def __call__(
        self,
        image: NDArray[np.uint8],
        *,
        image_size: int,
        confidence: float,
        iou: float,
        max_detections: int,
    ) -> list[RawDetection]: ...


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """A ready-to-run model plus the facts needed to trace a result back to it."""

    predictor: Predictor
    device: str
    hub_repo: str
    source: str
    weights_sha256: str | None = None

    @property
    def version(self) -> str:
        """Short identifier written into every result's metadata."""
        if self.weights_sha256:
            return f"{self.hub_repo} weights sha256:{self.weights_sha256[:12]}"
        return f"{self.hub_repo} {self.source}"


def sha256_file(path: Path) -> str:
    """Fingerprint a weights file so a result can be traced to the exact file used."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_device(requested: str, torch: Any) -> str:
    """Turn the ``device`` setting into a concrete torch device string."""
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def class_name(names: Mapping[int, str] | Sequence[str], index: int) -> str:
    """Look up a class name; YOLOv5 has used both dicts and lists for ``names``."""
    try:
        return str(names[index])
    except (KeyError, IndexError):
        return str(index)


class YOLOv5Predictor:
    """Adapts a YOLOv5 ``AutoShape`` model to the ``Predictor`` protocol.

    AutoShape reads its thresholds from attributes on the model object, so every call
    sets them first. That shared mutable state is why ``ModelManager`` holds a lock
    around each call.
    """

    def __init__(self, model: Any, torch: Any) -> None:
        self._model = model
        self._torch = torch

    def __call__(
        self,
        image: NDArray[np.uint8],
        *,
        image_size: int,
        confidence: float,
        iou: float,
        max_detections: int,
    ) -> list[RawDetection]:
        self._model.conf = confidence
        self._model.iou = iou
        self._model.max_det = max_detections
        with self._torch.inference_mode():
            results = self._model(image, size=image_size)
        # results.xyxy[0]: one row per box, [x1, y1, x2, y2, confidence, class index]
        rows = results.xyxy[0].tolist()
        return [
            RawDetection(
                x_min=float(x1),
                y_min=float(y1),
                x_max=float(x2),
                y_max=float(y2),
                confidence=float(conf),
                class_name=class_name(results.names, int(cls)),
            )
            for x1, y1, x2, y2, conf, cls in rows
        ]


def load_yolov5(settings: ModelSettings, torch: Any | None = None) -> LoadedModel:
    """Load YOLOv5 through ``torch.hub`` using the repo pinned in the settings.

    Args:
        settings: Model settings.
        torch: The torch module. Tests pass a fake; normal use leaves it as None.

    Raises:
        FileNotFoundError: The weights file is missing and the pretrained fallback is off.
    """
    if torch is None:
        import torch as torch_module  # heavy import, only needed when a model loads

        torch = torch_module

    device = resolve_device(settings.device.value, torch)
    weights = Path(settings.weights_path)
    # trust_repo=True runs code downloaded from hub_repo. Pinning hub_repo to a release
    # tag (not a branch) keeps that code fixed between runs.
    hub_options: dict[str, Any] = {
        "source": settings.hub_source,
        "trust_repo": True,
        "device": device,
    }

    if weights.is_file():
        model = torch.hub.load(settings.hub_repo, "custom", path=str(weights), **hub_options)
        loaded_source = f"custom weights {weights.name}"
        sha = sha256_file(weights)
    elif settings.allow_pretrained_fallback:
        logger.warning(
            "weights_missing_using_pretrained_fallback",
            weights_path=str(weights),
            architecture=settings.architecture,
            note="COCO classes are not invoice fields; all detections will be ignored",
        )
        model = torch.hub.load(
            settings.hub_repo, settings.architecture, pretrained=True, **hub_options
        )
        loaded_source = f"pretrained {settings.architecture} (COCO classes)"
        sha = None
    else:
        raise FileNotFoundError(
            f"No trained weights at {weights}. Train them with "
            "notebooks/01_train_yolov5_invoices.ipynb, or set "
            "INVOICE_DIGITIZER_MODEL__ALLOW_PRETRAINED_FALLBACK=true to smoke-test the "
            "service with generic COCO weights (which cannot detect invoice fields)."
        )

    if settings.half_precision and device.startswith("cuda"):
        model.half()
    model.eval()

    logger.info("model_loaded", hub_repo=settings.hub_repo, source=loaded_source, device=device)
    return LoadedModel(
        predictor=YOLOv5Predictor(model, torch),
        device=device,
        hub_repo=settings.hub_repo,
        source=loaded_source,
        weights_sha256=sha,
    )
