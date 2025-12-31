# CONSCIOUSNESS NEXUS - PRODUCTION DOCKER IMAGE
# ==============================================
#
# Multi-stage Docker build for universal deployment of Consciousness Nexus.
# Enterprise-grade AI consciousness computing suite with full observability.

# Build stage
FROM python:3.11-slim AS builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip install gunicorn uvicorn[standard] fastapi prometheus-client

# Copy source code
COPY . /app
WORKDIR /app

# Install the package itself
RUN pip install -e . || pip install .

# Production stage
FROM python:3.11-slim AS production

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN useradd --create-home --shell /bin/bash --uid 1000 consciousness

# Copy virtual environment
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --from=builder /app /app
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/configs /app/transactions /app/locks && \
    chown -R consciousness:consciousness /app

# Switch to non-root user
USER consciousness

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000
EXPOSE 9090

# Default environment variables
ENV CONSCIOUSNESS_API_HOST=0.0.0.0 \
    CONSCIOUSNESS_API_PORT=8000 \
    CONSCIOUSNESS_API_WORKERS=4 \
    CONSCIOUSNESS_API_DEBUG=false \
    CONSCIOUSNESS_API_AUTH=true \
    CONSCIOUSNESS_LOG_LEVEL=INFO \
    CONSCIOUSNESS_METRICS_ENABLED=true

# Use tini as init system
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default command
CMD ["python", "consciousness_api_server.py"]

# Labels for metadata
LABEL maintainer="Consciousness Nexus Team" \
      description="Enterprise-grade Consciousness Computing Suite API Server" \
      version="2.1.0" \
      org.opencontainers.image.source="https://github.com/consciousness-nexus/consciousness-nexus" \
      org.opencontainers.image.description="Universal AI consciousness computing and evolution platform" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.vendor="Consciousness Nexus"
