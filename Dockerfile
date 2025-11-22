# Base image with Python 3.9 and minimal dependencies
FROM python:3.9-slim

# Set metadata
LABEL maintainer="dsugurtuna"
LABEL description="RoyalAudit Digitizer - Invoice Extraction Inference API"

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and general utility
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY models/ ./models/

# Expose port for Streamlit
EXPOSE 8501

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the application
ENTRYPOINT ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
