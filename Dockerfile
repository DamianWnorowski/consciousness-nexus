# 🐳 CONSCIOUSNESS COMPUTING SUITE - DOCKER IMAGE
# ===============================================
#
# Multi-stage Docker build for universal deployment of Consciousness Suite.
# Makes enterprise AI safety available in any environment via containerization.

# Build stage
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install gunicorn uvicorn[standard] fastapi

# Copy source code
COPY . /app
WORKDIR /app

# Install the package itself
RUN pip install -e .

# Production stage
FROM python:3.11-slim as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash consciousness

# Copy virtual environment
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --from=builder /app /app
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/configs && \
    chown -R consciousness:consciousness /app

# Switch to non-root user
USER consciousness

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default environment variables
ENV CONSCIOUSNESS_API_HOST=0.0.0.0 \
    CONSCIOUSNESS_API_PORT=8000 \
    CONSCIOUSNESS_API_WORKERS=4 \
    CONSCIOUSNESS_API_DEBUG=false \
    CONSCIOUSNESS_API_AUTH=true

# Default command
CMD ["python", "consciousness_api_server.py"]

# Labels for metadata
LABEL maintainer="Consciousness AI <consciousness@ai.example.com>" \
      description="Enterprise-grade Consciousness Computing Suite API Server" \
      version="2.0.0" \
      org.opencontainers.image.source="https://github.com/DAMIANWNOROWSKI/consciousness-suite" \
      org.opencontainers.image.description="Universal AI safety and evolution platform" \
      org.opencontainers.image.licenses="MIT"
