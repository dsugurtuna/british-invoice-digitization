"""Prometheus metrics, registered once per process."""

from __future__ import annotations

from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "http_requests_total",
    "HTTP requests by method, route template and status code.",
    ["method", "route", "status"],
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency by method and route template.",
    ["method", "route"],
)
INFERENCE_COUNT = Counter(
    "inference_images_total",
    "Images sent to the model, by outcome.",
    ["outcome"],
)
INFERENCE_LATENCY = Histogram(
    "inference_duration_seconds",
    "Time to process one image, including decoding and post-processing.",
)
