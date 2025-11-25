# ============================================================================
# RoyalAudit Digitizer - Production Dockerfile
# Multi-stage build for optimized image size and security
# ============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder
# Install dependencies and build wheels
# -----------------------------------------------------------------------------
FROM python:3.11-slim as builder

# Build arguments
ARG TARGETPLATFORM
ARG BUILDPLATFORM

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml requirements.txt ./

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Runtime
# Minimal image for production deployment
# -----------------------------------------------------------------------------
FROM python:3.11-slim as runtime

# Labels for container metadata
LABEL maintainer="dsugurtuna" \
      version="2.0.0" \
      description="RoyalAudit Digitizer - Enterprise Invoice Extraction System" \
      org.opencontainers.image.source="https://github.com/dsugurtuna/british-invoice-digitization" \
      org.opencontainers.image.licenses="MIT"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    # App settings
    ROYALAUDIT_ENVIRONMENT=production \
    ROYALAUDIT_API__HOST=0.0.0.0 \
    ROYALAUDIT_API__PORT=8000 \
    # Paths
    APP_HOME=/app \
    PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR ${APP_HOME}

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # OpenCV dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    # Utilities
    curl \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY --chown=appuser:appgroup src/ ./src/
COPY --chown=appuser:appgroup config/ ./config/
COPY --chown=appuser:appgroup pyproject.toml ./

# Create necessary directories
RUN mkdir -p models output logs && \
    chown -R appuser:appgroup ${APP_HOME}

# Switch to non-root user
USER appuser

# Expose ports
# 8000 - FastAPI
# 8501 - Streamlit
# 9090 - Prometheus metrics
EXPOSE 8000 8501 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:8000/health/live || exit 1

# Use tini as init system
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default command - run FastAPI server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# -----------------------------------------------------------------------------
# Stage 3: Development (optional)
# Full development environment with dev dependencies
# -----------------------------------------------------------------------------
FROM runtime as development

USER root

# Install development dependencies
RUN pip install -e ".[dev]"

# Install additional dev tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    git \
    && rm -rf /var/lib/apt/lists/*

USER appuser

# Override command for development
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
