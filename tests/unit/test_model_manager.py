"""Model lifecycle: lazy loading, locking, reloads and threshold handling."""

from __future__ import annotations

import threading
import time

from typing import Any

import numpy as np
import pytest

from invoice_digitizer.config.settings import ModelSettings
from invoice_digitizer.core.model_manager import ModelManager
from invoice_digitizer.core.yolov5_backend import LoadedModel
from tests.conftest import FakeLoader, FakePredictor

IMAGE = np.zeros((32, 32, 3), dtype=np.uint8)


def test_loads_lazily_and_only_once(model_manager: ModelManager, fake_loader: FakeLoader) -> None:
    assert model_manager.is_loaded is False
    assert fake_loader.calls == 0

    model_manager.predict(IMAGE)
    model_manager.predict(IMAGE)

    assert model_manager.is_loaded is True
    assert fake_loader.calls == 1


def test_concurrent_first_use_loads_once() -> None:
    """Many threads hitting a cold manager must trigger exactly one load."""
    calls = 0
    lock = threading.Lock()

    def slow_loader(_: ModelSettings) -> LoadedModel:
        nonlocal calls
        with lock:
            calls += 1
        time.sleep(0.05)
        return LoadedModel(FakePredictor(), "cpu", "fake/repo:tag", "slow fake")

    manager = ModelManager(ModelSettings(), loader=slow_loader)
    threads = [threading.Thread(target=manager.load) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert calls == 1


def test_predictions_are_serialised() -> None:
    """The YOLOv5 wrapper mutates shared thresholds, so calls must never overlap."""
    active = 0
    peak = 0
    lock = threading.Lock()

    class SlowPredictor:
        def __call__(self, image: Any, **_: Any) -> list[Any]:
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.01)
            with lock:
                active -= 1
            return []

    manager = ModelManager(
        ModelSettings(),
        loader=lambda _: LoadedModel(SlowPredictor(), "cpu", "fake/repo:tag", "slow"),
    )
    threads = [threading.Thread(target=manager.predict, args=(IMAGE,)) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert peak == 1


def test_per_call_overrides_do_not_leak(
    model_manager: ModelManager, fake_predictor: FakePredictor
) -> None:
    """The old code wrote per-request thresholds onto the shared model and never reset them."""
    model_manager.predict(IMAGE, confidence=0.9, iou=0.2)
    model_manager.predict(IMAGE)

    first, second = fake_predictor.calls
    assert (first["confidence"], first["iou"]) == (0.9, 0.2)
    assert (second["confidence"], second["iou"]) == (0.4, 0.45)


def test_update_thresholds_changes_later_defaults(
    model_manager: ModelManager, fake_predictor: FakePredictor
) -> None:
    current = model_manager.update_thresholds(confidence=0.7, max_detections=5)
    model_manager.predict(IMAGE)

    assert current == {"confidence_threshold": 0.7, "iou_threshold": 0.45, "max_detections": 5}
    assert fake_predictor.calls[-1]["confidence"] == 0.7
    assert fake_predictor.calls[-1]["max_detections"] == 5


def test_image_size_comes_from_settings(fake_loader: FakeLoader) -> None:
    manager = ModelManager(ModelSettings(image_size=1280), loader=fake_loader)
    manager.predict(IMAGE)
    assert fake_loader.predictor.calls[-1]["image_size"] == 1280


def test_reload_swaps_in_a_new_model(model_manager: ModelManager, fake_loader: FakeLoader) -> None:
    _, first = model_manager.predict(IMAGE)
    model_manager.reload()
    _, second = model_manager.predict(IMAGE)

    assert fake_loader.calls == 2
    assert first is not second
    assert second.source == "fake weights #2"


def test_failed_reload_keeps_the_old_model() -> None:
    attempts = 0

    def flaky_loader(_: ModelSettings) -> LoadedModel:
        nonlocal attempts
        attempts += 1
        if attempts > 1:
            raise RuntimeError("new weights are corrupt")
        return LoadedModel(FakePredictor(), "cpu", "fake/repo:tag", "good weights")

    manager = ModelManager(ModelSettings(), loader=flaky_loader)
    original = manager.load()

    with pytest.raises(RuntimeError, match="corrupt"):
        manager.reload()

    assert manager.load() is original


def test_info_does_not_force_a_load(model_manager: ModelManager, fake_loader: FakeLoader) -> None:
    before = model_manager.info()
    assert before["loaded"] is False
    assert before["version"] is None
    assert fake_loader.calls == 0

    model_manager.load()
    after = model_manager.info()

    assert after["loaded"] is True
    assert after["device"] == "cpu"
    assert after["version"].startswith("fake/yolov5:test weights sha256:")
    assert after["confidence_threshold"] == 0.4
    assert len(after["classes"]) == 6
