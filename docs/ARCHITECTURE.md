# RoyalAudit Digitizer - System Architecture

## 1. Overview

The RoyalAudit Digitizer is a high-performance, production-grade system designed to extract structured data from handwritten British invoices. It leverages state-of-the-art Computer Vision (YOLOv5) and modern API standards (FastAPI) to provide a scalable solution for financial auditing.

## 2. System Components

### 2.1 Core Application (`src/core`)
- **Model Manager**: Thread-safe singleton that handles model loading, caching, and inference. Supports lazy loading to optimize startup time.
- **Detector**: Wrapper around the YOLOv5 model that handles image preprocessing (resizing, normalization) and post-processing (NMS, coordinate scaling).
- **Digitizer**: High-level orchestrator that manages batch processing using `ThreadPoolExecutor` for concurrent inference.

### 2.2 API Layer (`src/api`)
- **FastAPI**: Provides a high-performance, async REST API.
- **Endpoints**:
  - `POST /detect`: Single image inference.
  - `POST /batch`: Batch image inference.
  - `GET /health`: System health checks (liveness/readiness).
  - `GET /model/info`: Model metadata and status.
- **Middleware**:
  - Request logging with correlation IDs.
  - Rate limiting to prevent abuse.
  - CORS configuration for frontend integration.

### 2.3 Data Models (`src/schemas`)
- **Pydantic v2**: Used for strict data validation and serialization.
- **Schemas**:
  - `DetectionResult`: Standardized bounding box and class info.
  - `InvoiceField`: Domain-specific field representation.
  - `ProcessingRequest`: API request validation.
  - `APIResponse`: Uniform response structure.

### 2.4 Infrastructure
- **Docker**: Multi-stage builds for optimized production images.
  - `builder`: Compiles dependencies.
  - `runtime`: Minimal footprint image (distroless/slim).
- **Docker Compose**: Orchestrates the API, Redis (optional cache), and Prometheus (monitoring).
- **GitHub Actions**: CI/CD pipeline for testing, linting, and security scanning.

## 3. Data Flow

1.  **Request**: Client sends an image (or batch) to the API.
2.  **Validation**: Pydantic schemas validate the input payload.
3.  **Preprocessing**: Image is resized to 640x640 and normalized.
4.  **Inference**: YOLOv5 model predicts bounding boxes and classes.
5.  **Post-processing**: Non-Maximum Suppression (NMS) filters overlapping boxes.
6.  **Formatting**: Results are mapped to `InvoiceField` objects.
7.  **Response**: JSON response is returned to the client.

## 4. Technology Stack

- **Language**: Python 3.11
- **Framework**: FastAPI
- **ML Engine**: PyTorch, YOLOv5 (v7.0)
- **Validation**: Pydantic v2
- **Containerization**: Docker
- **CI/CD**: GitHub Actions
- **Testing**: Pytest, Coverage

## 5. Security & Performance

- **Non-root User**: Docker container runs as a non-privileged user.
- **Health Checks**: Integrated health endpoints for orchestrators (K8s/Swarm).
- **Concurrency**: Async API with thread pool for CPU-bound ML tasks.
- **Type Safety**: 100% type-annotated codebase checked with MyPy.
*   **Pattern:** Singleton-like instantiation of the YOLOv5 model to minimize VRAM overhead.
*   **Optimization:** Uses `torch.hub` for model loading with FP16 (half-precision) inference capabilities enabled when CUDA is detected.
*   **Scalability:** Designed to be stateless. Multiple instances can be spun up behind a load balancer (e.g., NGINX) to handle high throughput.

### 2.2 Data Pipeline
*   **Augmentation:** To handle the variability of historical British documents (faded ink, coffee stains, folds), we employ a rigorous augmentation pipeline:
    *   Mosaic Augmentation (combining 4 images)
    *   HSV Color Space manipulation
    *   Random affine transformations (rotation, scaling)

## 3. Security & Compliance
*   **Data Privacy:** No PII is logged in the application logs.
*   **Audit Trails:** All inference requests are timestamped and tagged with a unique request ID.

## 4. Future Roadmap
*   **Phase 2:** Integration of LayoutLMv3 for multimodal (text + image) understanding.
*   **Phase 3:** Real-time edge deployment on mobile devices for field auditors.
