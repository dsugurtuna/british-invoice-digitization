# Architecture

This page describes how the code is put together and how it behaves under concurrency and
failure. For the reasons behind each choice, see [WHY.md](WHY.md).

## Components

| Module | Responsibility |
| --- | --- |
| `api/main.py` | App factory. Builds settings, one `ModelManager` and one `InvoiceDigitizer`, attaches them to `app.state`, adds middleware. The lifespan handler preloads the model. |
| `api/routes.py` | Endpoints. Validates uploads (type, size, decodability) before any model work. Admin endpoints depend on `require_admin`. |
| `api/middleware.py` | Request ID, timing, logging and Prometheus metrics; in-memory rate limit. |
| `config/settings.py` | Settings from init arguments, then `INVOICE_DIGITIZER_*` environment variables, then an optional YAML file. |
| `core/images.py` | Decodes files, uploads and arrays to RGB `uint8`, applies EXIF orientation, enforces size limits. |
| `core/yolov5_backend.py` | The only module that imports PyTorch. Loads YOLOv5 through `torch.hub` from a pinned tag and adapts its output to `RawDetection`. |
| `core/model_manager.py` | Owns the loaded model for the process: lazy load, locking, reload, thresholds. |
| `core/detector.py` | Maps raw boxes to `InvoiceField`s (labels, clipping, ignored count) and draws annotated images. |
| `core/digitizer.py` | Sync, async and batch entry points; a thread pool for async calls. |
| `schemas/` | Pydantic models for results, requests and responses. |

## Request flow

```mermaid
sequenceDiagram
    participant Client
    participant Route as routes.py
    participant Dig as InvoiceDigitizer
    participant Det as InvoiceFieldDetector
    participant MM as ModelManager
    participant Model as YOLOv5 (torch.hub)
    Client->>Route: POST /api/v1/inference (image)
    Route->>Route: check type, size; decode to RGB
    Route->>Dig: process_async(image, overrides)
    Dig->>Det: detect() in a worker thread
    Det->>MM: predict(image, confidence, iou)
    MM->>MM: load once (lock), then take predict lock
    MM->>Model: set thresholds, run
    Model-->>MM: raw boxes
    MM-->>Det: raw boxes + model identity
    Det->>Det: map labels, clip, count ignored, sort
    Det-->>Route: DetectionResult
    Route-->>Client: 200 JSON (or 4xx/5xx with a reason)
```

## Concurrency

- **Loading.** `ModelManager.load()` takes a lock, so many threads arriving at a cold process
  cause one load.
- **Inference.** `ModelManager.predict()` holds a second lock for the whole call. The YOLOv5
  wrapper reads its thresholds from attributes on the model, so overlapping calls would race.
  Throughput is therefore one image at a time per process. The thread pool in
  `InvoiceDigitizer` keeps the event loop free and lets decoding overlap with inference; it
  does not run the model in parallel.
- **Reload.** The new model loads while the old one keeps serving. The reference is swapped only
  on success. A failed reload leaves the old model in place and returns `500`.
- **Scaling.** Run more processes or containers. Each loads its own copy of the model and has
  its own thresholds and rate-limit counters.

## Failure behaviour

| Situation | Behaviour |
| --- | --- |
| No weights, preload on (default) | Start-up fails with a message naming the path and the notebook. |
| No weights, preload off | Server starts; `/health` is `degraded`, `/health/ready` is `503`, inference is `503` with the same message. |
| Wrong file type / too large / not decodable | `415` / `413` / `400`, before any model work. |
| Model raises during inference | `500` with the request ID; the exception text stays in the server log. |
| One bad file in a batch | Listed in `errors`; other files still processed; `job_status` is `partial`. |
| Model returns classes that are not invoice fields | Dropped and counted in `ignored_detections`. |
| Admin endpoint without a configured key | `403`: admin endpoints are off. Wrong or missing key: `401`. |

## Observability

- Each response carries `X-Request-ID` (matching `request_id` in the body) and `X-Process-Time`.
- Log lines record method, route template, status and duration. They do not record file
  names or client addresses.
- `/metrics` exposes `http_requests_total`, `http_request_duration_seconds`,
  `inference_images_total{outcome}` and `inference_duration_seconds`, labelled by route
  template so label cardinality stays bounded.
- With `environment=production`, logs are JSON lines.

## Security notes

- Default bind address is `127.0.0.1`; the container sets `0.0.0.0` explicitly.
- CORS is off unless origins are configured.
- Admin endpoints use a constant-time key comparison and are disabled without a key.
- Inference endpoints are unauthenticated; put a gateway in front for anything shared.
- `torch.hub.load(..., trust_repo=True)` downloads and executes YOLOv5 code from GitHub on first
  load, and YOLOv5 checkpoints are Python pickles. Pin the tag, and load only weights you trust.
