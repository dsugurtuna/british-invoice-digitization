"""API routes for invoice digitization."""

from __future__ import annotations

import io
import time
from typing import TYPE_CHECKING, Annotated, Any
from uuid import uuid4

import structlog
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import JSONResponse, StreamingResponse
import numpy as np
from PIL import Image

from src.core.digitizer import InvoiceDigitizer
from src.schemas.detection import DetectionResult
from src.schemas.request import BatchProcessRequest, InferenceRequest, ModelConfigRequest
from src.schemas.response import (
    APIResponse,
    BatchJobStatus,
    BatchProcessResponse,
    ComponentHealth,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    HealthStatus,
    InferenceResponse,
    ResponseStatus,
)

if TYPE_CHECKING:
    from src.config.settings import Settings

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["Invoice Digitization"])


# Dependency injection
def get_settings(request: Request) -> "Settings":
    """Get settings from app state."""
    return request.app.state.settings


def get_digitizer(request: Request) -> InvoiceDigitizer:
    """Get or create digitizer instance."""
    if not hasattr(request.app.state, "digitizer"):
        request.app.state.digitizer = InvoiceDigitizer(request.app.state.settings)
    return request.app.state.digitizer


# ============================================================================
# Health Check Endpoints
# ============================================================================


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check the health status of the API and its components.",
)
async def health_check(request: Request) -> HealthResponse:
    """Perform health check on all components."""
    components = []
    overall_status = HealthStatus.HEALTHY

    # Check model
    try:
        model_manager = request.app.state.model_manager
        model_info = model_manager.get_model_info()
        components.append(
            ComponentHealth(
                name="ml_model",
                status=HealthStatus.HEALTHY,
                message=f"Model loaded: {model_info['architecture']}",
            )
        )
    except Exception as e:
        components.append(
            ComponentHealth(
                name="ml_model",
                status=HealthStatus.UNHEALTHY,
                message=str(e),
            )
        )
        overall_status = HealthStatus.UNHEALTHY

    # Calculate uptime
    uptime = time.time() - request.app.state.start_time

    return HealthResponse(
        status=overall_status,
        version="2.0.0",
        uptime_seconds=uptime,
        components=components,
    )


@router.get(
    "/health/ready",
    summary="Readiness Check",
    description="Check if the API is ready to accept requests.",
)
async def readiness_check(request: Request) -> JSONResponse:
    """Kubernetes readiness probe endpoint."""
    try:
        # Verify model is loaded and responsive
        model_manager = request.app.state.model_manager
        _ = model_manager.model
        return JSONResponse(content={"status": "ready"}, status_code=200)
    except Exception:
        return JSONResponse(content={"status": "not_ready"}, status_code=503)


@router.get(
    "/health/live",
    summary="Liveness Check",
    description="Check if the API process is alive.",
)
async def liveness_check() -> JSONResponse:
    """Kubernetes liveness probe endpoint."""
    return JSONResponse(content={"status": "alive"}, status_code=200)


# ============================================================================
# Inference Endpoints
# ============================================================================


@router.post(
    "/api/v1/inference",
    response_model=InferenceResponse,
    summary="Process Invoice",
    description="""
Process a single invoice image and extract structured field data.

## Supported Formats
- JPEG, PNG, TIFF, BMP, WebP

## Detected Fields
- Invoice Date
- Invoice Number
- Vendor Name
- Total Amount
- VAT Amount
- Line Items

## Response
Returns detected fields with bounding boxes, confidence scores,
and optionally extracted text via OCR.
    """,
    responses={
        200: {"description": "Successful processing"},
        400: {"description": "Invalid input"},
        413: {"description": "File too large"},
        415: {"description": "Unsupported media type"},
        500: {"description": "Internal server error"},
    },
)
async def process_invoice(
    file: Annotated[UploadFile, File(description="Invoice image file")],
    confidence_threshold: Annotated[
        float | None,
        Query(ge=0.0, le=1.0, description="Override confidence threshold"),
    ] = None,
    extract_text: Annotated[
        bool,
        Query(description="Run OCR on detected regions"),
    ] = True,
    include_visualization: Annotated[
        bool,
        Query(description="Include annotated image URL"),
    ] = False,
    digitizer: InvoiceDigitizer = Depends(get_digitizer),
    settings: "Settings" = Depends(get_settings),
) -> InferenceResponse:
    """Process a single invoice image."""
    request_id = uuid4()

    logger.info(
        "Processing invoice",
        request_id=str(request_id),
        filename=file.filename,
        content_type=file.content_type,
    )

    # Validate file
    if file.content_type not in [
        "image/jpeg",
        "image/png",
        "image/tiff",
        "image/bmp",
        "image/webp",
    ]:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.content_type}",
        )

    # Check file size
    contents = await file.read()
    max_size = settings.preprocessing.max_file_size_mb * 1024 * 1024
    if len(contents) > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum of {settings.preprocessing.max_file_size_mb}MB",
        )

    try:
        # Convert to numpy array
        image = Image.open(io.BytesIO(contents))
        image_array = np.array(image)

        # Process
        result = await digitizer.process_async(
            image_array,
            confidence_threshold=confidence_threshold,
            extract_text=extract_text,
        )

        logger.info(
            "Invoice processed",
            request_id=str(request_id),
            detections=result.detection_count,
            processing_time_ms=result.metadata.processing_time_ms,
        )

        return InferenceResponse(
            request_id=request_id,
            result=result,
            visualization_url=None,  # Would be S3 URL in production
        )

    except Exception as e:
        logger.exception("Inference failed", request_id=str(request_id), error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e


@router.post(
    "/api/v1/inference/batch",
    response_model=BatchProcessResponse,
    summary="Batch Process Invoices",
    description="""
Process multiple invoice images in a batch.

Files are processed concurrently for optimal throughput.
Results are returned when all files have been processed.
    """,
)
async def batch_process_invoices(
    files: Annotated[list[UploadFile], File(description="Invoice image files")],
    confidence_threshold: Annotated[
        float | None,
        Query(ge=0.0, le=1.0, description="Override confidence threshold"),
    ] = None,
    extract_text: Annotated[
        bool,
        Query(description="Run OCR on detected regions"),
    ] = True,
    digitizer: InvoiceDigitizer = Depends(get_digitizer),
    settings: "Settings" = Depends(get_settings),
) -> BatchProcessResponse:
    """Process multiple invoice images."""
    job_id = uuid4()
    request_id = uuid4()

    logger.info(
        "Starting batch processing",
        job_id=str(job_id),
        file_count=len(files),
    )

    if len(files) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 100 files per batch",
        )

    results: list[DetectionResult] = []
    errors: list[ErrorDetail] = []

    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            image_array = np.array(image)

            result = await digitizer.process_async(
                image_array,
                confidence_threshold=confidence_threshold,
                extract_text=extract_text,
            )
            results.append(result)

        except Exception as e:
            errors.append(
                ErrorDetail(
                    code="PROCESSING_ERROR",
                    message=str(e),
                    field=file.filename,
                )
            )

    job_status = BatchJobStatus.COMPLETED
    if errors and not results:
        job_status = BatchJobStatus.FAILED
    elif errors:
        job_status = BatchJobStatus.PARTIAL

    logger.info(
        "Batch processing complete",
        job_id=str(job_id),
        processed=len(results),
        failed=len(errors),
    )

    return BatchProcessResponse(
        request_id=request_id,
        job_id=job_id,
        job_status=job_status,
        total_images=len(files),
        processed_images=len(results),
        failed_images=len(errors),
        results=results if results else None,
        errors=errors if errors else None,
    )


# ============================================================================
# Model Management Endpoints
# ============================================================================


@router.get(
    "/api/v1/model/info",
    summary="Get Model Information",
    description="Get information about the loaded ML model.",
)
async def get_model_info(request: Request) -> dict[str, Any]:
    """Get model information and statistics."""
    model_manager = request.app.state.model_manager
    return model_manager.get_model_info()


@router.put(
    "/api/v1/model/config",
    summary="Update Model Configuration",
    description="Update model inference configuration (thresholds, etc.).",
)
async def update_model_config(
    config: ModelConfigRequest,
    request: Request,
) -> dict[str, str]:
    """Update model configuration dynamically."""
    model_manager = request.app.state.model_manager

    updates = config.get_updates()
    if not updates:
        return {"message": "No updates provided"}

    if "confidence_threshold" in updates or "iou_threshold" in updates:
        model_manager.update_thresholds(
            confidence=updates.get("confidence_threshold"),
            iou=updates.get("iou_threshold"),
        )

    logger.info("Model config updated", updates=updates)
    return {"message": "Configuration updated", "updates": updates}


@router.post(
    "/api/v1/model/reload",
    summary="Reload Model",
    description="Force reload the ML model from disk.",
)
async def reload_model(request: Request) -> dict[str, str]:
    """Reload the model weights."""
    model_manager = request.app.state.model_manager
    model_manager.reload_model()

    # Also recreate the digitizer
    if hasattr(request.app.state, "digitizer"):
        del request.app.state.digitizer

    return {"message": "Model reloaded successfully"}


# ============================================================================
# Utility Endpoints
# ============================================================================


@router.get(
    "/api/v1/classes",
    summary="List Detection Classes",
    description="Get the list of invoice fields the model can detect.",
)
async def list_classes(settings: "Settings" = Depends(get_settings)) -> dict[str, list[str]]:
    """Get list of detection classes."""
    return {"classes": settings.model.classes}
