# Container image for the API.
#
# Status: not built in CI. It has not been built since these fixes (the author's
# environment had no Docker daemon), so treat it as a starting point.
#
# The image does not contain trained weights. Mount them at /app/models, for example:
#   docker run -p 8000:8000 -v "$PWD/models:/app/models:ro" british-invoice-digitization
# On first start, torch.hub downloads the pinned YOLOv5 v7.0 code from GitHub.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libglib2.0-0: needed by opencv-python-headless, which the YOLOv5 code imports.
# curl: health check. tini: forwards signals so the server shuts down cleanly.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 curl tini \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1000 app && useradd --uid 1000 --gid app --create-home app
WORKDIR /app

# CPU-only PyTorch: a fraction of the size of the default CUDA build.
# Override TORCH_INDEX_URL to build a GPU image.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install torch torchvision --index-url "${TORCH_INDEX_URL}"

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[yolov5]"

COPY config ./config
RUN mkdir -p models && chown -R app:app /app
USER app

ENV INVOICE_DIGITIZER_API__HOST=0.0.0.0 \
    INVOICE_DIGITIZER_ENVIRONMENT=production \
    INVOICE_DIGITIZER_CONFIG_FILE=/app/config/default.yaml

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl --fail --silent http://127.0.0.1:8000/health/live || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
# One process per container: each process loads its own copy of the model, and
# runtime threshold changes and reloads apply per process. Scale with replicas.
CMD ["invoice-digitizer"]
