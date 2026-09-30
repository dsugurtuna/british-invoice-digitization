"""HTTP endpoints.

Reading is open; changing the running service is not. Endpoints that alter
thresholds or reload the model need the admin key, and are switched off when no
key is configured.
"""

from __future__ import annotations

import asyncio
import secrets
import time

from typing import Annotated, Any
from uuid import UUID

import numpy as np
import structlog

from fastapi import APIRouter, Depends, File, Header, HTTPException, Query, Request, UploadFile
from fastapi import status as http_status
from fastapi.responses import JSONResponse

from invoice_digitizer._version import __version__
from invoice_digitizer.api.metrics import INFERENCE_COUNT, INFERENCE_LATENCY
from invoice_digitizer.config.settings import Settings
from invoice_digitizer.core.digitizer import InvoiceDigitizer
from invoice_digitizer.core.images import ImageValidationError, decode_image_bytes
from invoice_digitizer.core.model_manager import ModelManager
from invoice_digitizer.schemas.detection import DetectionResult
from invoice_digitizer.schemas.request import ModelConfigRequest
from invoice_digitizer.schemas.response import (
    BatchJobStatus,
    BatchProcessResponse,
    ComponentHealth,
    ErrorDetail,
    HealthResponse,
    HealthStatus,
    InferenceResponse,
    ModelConfigResponse,
    ResponseStatus,
)

logger = structlog.get_logger(__name__)

router = APIRouter()

ALLOWED_CONTENT_TYPES = frozenset(
    {"image/jpeg", "image/png", "image/tiff", "image/bmp", "image/webp"}
)

ConfidenceQuery = Annotated[
    float | None, Query(ge=0.0, le=1.0, description="Override the confidence threshold")
]
IouQuery = Annotated[
    float | None, Query(ge=0.0, le=1.0, description="Override the NMS IoU threshold")
]


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


def get_app_settings(request: Request) -> Settings:
    """Settings attached to the app when it was created."""
    settings: Settings = request.app.state.settings
    return settings


def get_digitizer(request: Request) -> InvoiceDigitizer:
    """The app's shared digitizer."""
    digitizer: InvoiceDigitizer = request.app.state.digitizer
    return digitizer


def get_model_manager(request: Request) -> ModelManager:
    """The app's shared model manager."""
    manager: ModelManager = request.app.state.model_manager
    return manager


def require_admin(
    settings: Annotated[Settings, Depends(get_app_settings)],
    x_admin_key: Annotated[str | None, Header()] = None,
) -> None:
    """Allow the request only with the configured admin key."""
    configured = settings.api.admin_api_key
    if configured is None:
        raise HTTPException(
            status_code=http_status.HTTP_403_FORBIDDEN,
            detail="Admin endpoints are disabled. Set INVOICE_DIGITIZER_API__ADMIN_API_KEY.",
        )
    supplied = (x_admin_key or "").encode()
    if not secrets.compare_digest(supplied, configured.get_secret_value().encode()):
        raise HTTPException(
            status_code=http_status.HTTP_401_UNAUTHORIZED, detail="Missing or wrong admin key."
        )


def _request_id(request: Request) -> UUID:
    return UUID(request.state.request_id)


async def _read_upload(upload: UploadFile, settings: Settings) -> np.ndarray:
    """Check type and size, then decode. Raises HTTPException with a client error."""
    if upload.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=http_status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {upload.content_type}",
        )
    limit = settings.preprocessing.max_file_size_mb * 1024 * 1024
    contents = await upload.read(limit + 1)  # read at most one byte past the limit
    if len(contents) > limit:
        raise HTTPException(
            status_code=413,  # Content Too Large (constant name differs across Starlette versions)
            detail=f"File is larger than {settings.preprocessing.max_file_size_mb} MB.",
        )
    try:
        return decode_image_bytes(contents, settings.preprocessing)
    except ImageValidationError as exc:
        raise HTTPException(status_code=http_status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check(
    request: Request, manager: Annotated[ModelManager, Depends(get_model_manager)]
) -> HealthResponse:
    """Service health. Degraded until the model is in memory."""
    info = manager.info()
    if info["loaded"]:
        model = ComponentHealth(
            name="ml_model", status=HealthStatus.HEALTHY, message=f"Loaded: {info['version']}"
        )
    else:
        model = ComponentHealth(
            name="ml_model", status=HealthStatus.DEGRADED, message="Model not loaded yet"
        )
    return HealthResponse(
        status=model.status,
        version=__version__,
        uptime_seconds=max(0.0, time.time() - request.app.state.started_at),
        components=[model],
    )


@router.get("/health/ready", tags=["health"])
async def readiness_check(
    manager: Annotated[ModelManager, Depends(get_model_manager)],
) -> JSONResponse:
    """Ready means the model is in memory. Does not trigger a load."""
    if manager.is_loaded:
        return JSONResponse({"status": "ready"})
    return JSONResponse(
        {"status": "not_ready"}, status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE
    )


@router.get("/health/live", tags=["health"])
async def liveness_check() -> dict[str, str]:
    """The process is up and serving HTTP."""
    return {"status": "alive"}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


@router.post("/api/v1/inference", response_model=InferenceResponse, tags=["inference"])
async def process_invoice(
    request: Request,
    file: Annotated[UploadFile, File(description="Invoice page image")],
    digitizer: Annotated[InvoiceDigitizer, Depends(get_digitizer)],
    settings: Annotated[Settings, Depends(get_app_settings)],
    confidence_threshold: ConfidenceQuery = None,
    iou_threshold: IouQuery = None,
) -> InferenceResponse:
    """Locate invoice fields on one page image (JPEG, PNG, TIFF, BMP or WebP).

    Returns boxes and confidence scores. Text is not read (no OCR).
    """
    image = await _read_upload(file, settings)
    start = time.perf_counter()
    try:
        result = await digitizer.process_async(
            image,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            source_name="upload",
        )
    except Exception as exc:
        INFERENCE_COUNT.labels("error").inc()
        logger.exception("inference_failed", request_id=request.state.request_id)
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed. Quote request ID {request.state.request_id}.",
        ) from exc
    INFERENCE_COUNT.labels("success").inc()
    INFERENCE_LATENCY.observe(time.perf_counter() - start)
    return InferenceResponse(request_id=_request_id(request), result=result)


@router.post("/api/v1/inference/batch", response_model=BatchProcessResponse, tags=["inference"])
async def batch_process_invoices(
    request: Request,
    files: Annotated[list[UploadFile], File(description="Invoice page images")],
    digitizer: Annotated[InvoiceDigitizer, Depends(get_digitizer)],
    settings: Annotated[Settings, Depends(get_app_settings)],
    confidence_threshold: ConfidenceQuery = None,
) -> BatchProcessResponse:
    """Process several images in one request and wait for all of them.

    Files that fail validation or inference are listed in ``errors``; the rest are
    returned in ``results``.
    """
    if len(files) > settings.api.max_batch_files:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail=f"At most {settings.api.max_batch_files} files per batch.",
        )

    errors: list[ErrorDetail] = []
    images: list[np.ndarray] = []
    names: list[str | None] = []
    for upload in files:
        try:
            images.append(await _read_upload(upload, settings))
            names.append(upload.filename)
        except HTTPException as exc:
            errors.append(
                ErrorDetail(code="INVALID_FILE", message=str(exc.detail), field=upload.filename)
            )

    start = time.perf_counter()
    outcomes = await digitizer.process_batch(list(images), confidence_threshold)
    results: list[DetectionResult] = []
    for name, outcome in zip(names, outcomes, strict=True):
        if outcome.succeeded:
            INFERENCE_COUNT.labels("success").inc()
            results.append(outcome)
        else:
            INFERENCE_COUNT.labels("error").inc()
            errors.append(
                ErrorDetail(code="PROCESSING_ERROR", message="Inference failed.", field=name)
            )
    if images:
        INFERENCE_LATENCY.observe((time.perf_counter() - start) / len(images))

    if not errors:
        job_status, status = BatchJobStatus.COMPLETED, ResponseStatus.SUCCESS
    elif results:
        job_status, status = BatchJobStatus.PARTIAL, ResponseStatus.PARTIAL
    else:
        job_status, status = BatchJobStatus.FAILED, ResponseStatus.ERROR

    return BatchProcessResponse(
        status=status,
        request_id=_request_id(request),
        job_status=job_status,
        total_images=len(files),
        processed_images=len(results),
        failed_images=len(errors),
        results=results,
        errors=errors,
    )


# ---------------------------------------------------------------------------
# Model information and administration
# ---------------------------------------------------------------------------


@router.get("/api/v1/model/info", tags=["model"])
async def get_model_info(
    manager: Annotated[ModelManager, Depends(get_model_manager)],
) -> dict[str, Any]:
    """What is loaded, from where, on which device, with which thresholds."""
    return manager.info()


@router.get("/api/v1/classes", tags=["model"])
async def list_classes(
    settings: Annotated[Settings, Depends(get_app_settings)],
) -> dict[str, list[str]]:
    """The field types the model is expected to detect."""
    return {"classes": list(settings.model.classes)}


@router.put(
    "/api/v1/model/config",
    response_model=ModelConfigResponse,
    tags=["admin"],
    dependencies=[Depends(require_admin)],
)
async def update_model_config(
    config: ModelConfigRequest,
    manager: Annotated[ModelManager, Depends(get_model_manager)],
) -> ModelConfigResponse:
    """Change default thresholds for this process. Needs the admin key."""
    updates = config.get_updates()
    if not updates:
        return ModelConfigResponse(message="No updates provided", thresholds=manager.thresholds())
    thresholds = manager.update_thresholds(
        confidence=updates.get("confidence_threshold"),
        iou=updates.get("iou_threshold"),
        max_detections=updates.get("max_detections"),
    )
    return ModelConfigResponse(message="Thresholds updated", thresholds=thresholds)


@router.post("/api/v1/model/reload", tags=["admin"], dependencies=[Depends(require_admin)])
async def reload_model(
    manager: Annotated[ModelManager, Depends(get_model_manager)],
) -> dict[str, Any]:
    """Load the weights again and swap them in. Needs the admin key.

    The old model keeps serving until the new one is ready; if loading fails, the old
    model stays. This affects only the process that handles the request.
    """
    try:
        await asyncio.to_thread(manager.reload)
    except Exception as exc:
        logger.exception("model_reload_failed")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Reload failed; the previous model is still serving.",
        ) from exc
    return {"message": "Model reloaded", "model": manager.info()}
