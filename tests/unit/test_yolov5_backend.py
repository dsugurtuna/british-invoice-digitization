"""The torch.hub adapter, tested against a fake ``torch`` module.

These tests pin down how the real loader calls torch.hub (pinned repo, weights path,
device, trust flag) and how YOLOv5 output is converted. They do not prove that the
upstream YOLOv5 code runs with a given torch version; see the README.
"""

from __future__ import annotations

import contextlib
import hashlib

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from invoice_digitizer.config.settings import ModelSettings
from invoice_digitizer.core.yolov5_backend import (
    LoadedModel,
    RawDetection,
    YOLOv5Predictor,
    class_name,
    load_yolov5,
    resolve_device,
    sha256_file,
)


class FakeRows:
    def __init__(self, rows: list[list[float]]) -> None:
        self._rows = rows

    def tolist(self) -> list[list[float]]:
        return self._rows


class FakeAutoShape:
    """Mimics the parts of YOLOv5's AutoShape model that the adapter touches."""

    def __init__(self) -> None:
        self.conf = 0.25
        self.iou = 0.45
        self.max_det = 1000
        self.calls: list[dict[str, Any]] = []
        self.halved = False
        self.evaluated = False

    def __call__(self, image: np.ndarray, size: int) -> SimpleNamespace:
        self.calls.append({"size": size, "conf": self.conf, "iou": self.iou, "max": self.max_det})
        rows = [[10.0, 20.0, 110.0, 60.0, 0.9, 0.0], [5.0, 5.0, 50.0, 30.0, 0.6, 7.0]]
        return SimpleNamespace(xyxy=[FakeRows(rows)], names={0: "Invoice Date"})

    def half(self) -> FakeAutoShape:
        self.halved = True
        return self

    def eval(self) -> FakeAutoShape:
        self.evaluated = True
        return self


class FakeTorch:
    """Just enough of the torch API for the loader."""

    def __init__(self, cuda: bool = False, mps: bool = False) -> None:
        self.cuda = SimpleNamespace(is_available=lambda: cuda)
        self.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps))
        self.model = FakeAutoShape()
        self.hub_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.hub = SimpleNamespace(load=self._hub_load)
        self.inference_mode_entered = 0

    def _hub_load(self, *args: Any, **kwargs: Any) -> FakeAutoShape:
        self.hub_calls.append((args, kwargs))
        return self.model

    @contextlib.contextmanager
    def inference_mode(self) -> Any:
        self.inference_mode_entered += 1
        yield


@pytest.mark.parametrize(
    ("requested", "cuda", "mps", "expected"),
    [
        ("auto", True, True, "cuda"),
        ("auto", False, True, "mps"),
        ("auto", False, False, "cpu"),
        ("cpu", True, True, "cpu"),
        ("cuda:1", False, False, "cuda:1"),
    ],
)
def test_resolve_device(requested: str, cuda: bool, mps: bool, expected: str) -> None:
    assert resolve_device(requested, FakeTorch(cuda=cuda, mps=mps)) == expected


def test_class_name_handles_dicts_lists_and_gaps() -> None:
    assert class_name({0: "a", 1: "b"}, 1) == "b"
    assert class_name(["a", "b"], 0) == "a"
    assert class_name({0: "a"}, 5) == "5"


def test_sha256_file(tmp_path: Path) -> None:
    path = tmp_path / "w.pt"
    path.write_bytes(b"weights")
    assert sha256_file(path) == hashlib.sha256(b"weights").hexdigest()


def test_custom_weights_load_through_pinned_hub_repo(tmp_path: Path) -> None:
    weights = tmp_path / "invoice_fields.pt"
    weights.write_bytes(b"fake weights")
    torch = FakeTorch()

    loaded = load_yolov5(ModelSettings(weights_path=str(weights), device="cpu"), torch=torch)

    ((args, kwargs),) = torch.hub_calls
    assert args == ("ultralytics/yolov5:v7.0", "custom")
    assert kwargs == {
        "path": str(weights),
        "source": "github",
        "trust_repo": True,
        "device": "cpu",
    }
    assert loaded.device == "cpu"
    assert loaded.weights_sha256 == hashlib.sha256(b"fake weights").hexdigest()
    assert loaded.version.startswith("ultralytics/yolov5:v7.0 weights sha256:")
    assert torch.model.evaluated is True
    assert torch.model.halved is False


def test_missing_weights_fail_loudly_by_default(tmp_path: Path) -> None:
    torch = FakeTorch()
    with pytest.raises(FileNotFoundError, match="No trained weights"):
        load_yolov5(ModelSettings(weights_path=str(tmp_path / "missing.pt")), torch=torch)
    assert torch.hub_calls == []


def test_missing_weights_message_needs_no_torch(tmp_path: Path) -> None:
    """On a base install (no torch), the user sees the weights message, not an ImportError."""
    with pytest.raises(FileNotFoundError, match="No trained weights"):
        load_yolov5(ModelSettings(weights_path=str(tmp_path / "missing.pt")))


def test_pretrained_fallback_is_opt_in_and_labelled(tmp_path: Path) -> None:
    torch = FakeTorch(cuda=True)
    settings = ModelSettings(
        weights_path=str(tmp_path / "missing.pt"),
        allow_pretrained_fallback=True,
        architecture="yolov5n",
        half_precision=True,
    )

    loaded = load_yolov5(settings, torch=torch)

    ((args, kwargs),) = torch.hub_calls
    assert args == ("ultralytics/yolov5:v7.0", "yolov5n")
    assert kwargs["pretrained"] is True
    assert kwargs["device"] == "cuda"
    assert loaded.weights_sha256 is None
    assert "COCO" in loaded.source
    assert torch.model.halved is True  # FP16 only because the device is CUDA


def test_predictor_sets_thresholds_and_converts_rows() -> None:
    torch = FakeTorch()
    predictor = YOLOv5Predictor(torch.model, torch)

    detections = predictor(
        np.zeros((100, 200, 3), dtype=np.uint8),
        image_size=640,
        confidence=0.5,
        iou=0.3,
        max_detections=10,
    )

    assert torch.model.calls == [{"size": 640, "conf": 0.5, "iou": 0.3, "max": 10}]
    assert torch.inference_mode_entered == 1
    assert detections == [
        RawDetection(10.0, 20.0, 110.0, 60.0, 0.9, "Invoice Date"),
        RawDetection(5.0, 5.0, 50.0, 30.0, 0.6, "7"),
    ]


def test_loaded_model_version_without_fingerprint() -> None:
    loaded = LoadedModel(
        predictor=YOLOv5Predictor(None, None), device="cpu", hub_repo="r", source="s"
    )
    assert loaded.version == "r s"
