"""
Observability Middleware for FastAPI/Starlette

Combines:
- OpenTelemetry trace propagation
- Prometheus metrics collection
- Request/response logging
- Error tracking
"""

from typing import Callable, Optional, Dict, Any, List
import time
import logging
from contextlib import asynccontextmanager

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from .propagation import extract_trace_context, inject_trace_context
from .tracing import ConsciousnessTracer
from .metrics import ConsciousnessMetrics, get_consciousness_metrics

logger = logging.getLogger(__name__)


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """
    Unified observability middleware combining:
    - Distributed tracing with W3C context propagation
    - Prometheus metrics collection
    - Structured request logging
    - Error tracking and recording

    Usage:
        app = FastAPI()
        app.add_middleware(ObservabilityMiddleware)

        # Or with configuration
        app.add_middleware(
            ObservabilityMiddleware,
            service_name="my-service",
            exclude_paths=["/health", "/metrics"],
        )
    """

    def __init__(
        self,
        app: ASGIApp,
        service_name: str = "consciousness-nexus",
        exclude_paths: Optional[List[str]] = None,
        log_request_body: bool = False,
        log_response_body: bool = False,
        record_exception_details: bool = True,
    ):
        super().__init__(app)
        self.service_name = service_name
        self.exclude_paths = exclude_paths or ["/metrics", "/health", "/ready", "/live"]
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
        self.record_exception_details = record_exception_details

        # Initialize components
        self._tracer = ConsciousnessTracer(name=service_name)
        self._metrics: Optional[ConsciousnessMetrics] = None

    @property
    def metrics(self) -> ConsciousnessMetrics:
        """Lazy load metrics to avoid initialization order issues."""
        if self._metrics is None:
            self._metrics = get_consciousness_metrics()
        return self._metrics

    def _should_trace(self, path: str) -> bool:
        """Check if this path should be traced."""
        return path not in self.exclude_paths

    def _normalize_path(self, request: Request) -> str:
        """
        Normalize path for metrics labeling.

        Replaces dynamic segments to avoid high cardinality.
        """
        path = request.url.path

        # Try to get route pattern if available
        if hasattr(request, "scope") and "route" in request.scope:
            route = request.scope.get("route")
            if route and hasattr(route, "path"):
                return route.path

        # Simple normalization for IDs and UUIDs
        parts = path.split("/")
        normalized = []
        for part in parts:
            if part.isdigit():
                normalized.append("{id}")
            elif len(part) == 36 and part.count("-") == 4:
                normalized.append("{uuid}")
            else:
                normalized.append(part)

        return "/".join(normalized)

    def _extract_request_attributes(self, request: Request) -> Dict[str, Any]:
        """Extract span attributes from request."""
        attrs = {
            "http.method": request.method,
            "http.url": str(request.url),
            "http.host": request.url.hostname or "",
            "http.scheme": request.url.scheme,
            "http.target": request.url.path,
            "http.user_agent": request.headers.get("user-agent", ""),
            "http.request_content_length": request.headers.get("content-length", 0),
            "net.peer.ip": request.client.host if request.client else "",
        }

        # Add custom headers if present
        if "x-request-id" in request.headers:
            attrs["http.request_id"] = request.headers["x-request-id"]

        if "x-correlation-id" in request.headers:
            attrs["correlation.id"] = request.headers["x-correlation-id"]

        return attrs

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with full observability."""
        path = request.url.path

        # Skip excluded paths
        if not self._should_trace(path):
            return await call_next(request)

        # Extract incoming trace context
        ctx = extract_trace_context(dict(request.headers))

        # Get normalized path for metrics
        normalized_path = self._normalize_path(request)
        method = request.method

        # Extract request attributes
        span_attrs = self._extract_request_attributes(request)

        # Track active requests
        self.metrics._updown_counters.get("active_requests", None)

        start_time = time.perf_counter()

        # Create span with extracted context as parent
        tracer = trace.get_tracer(self.service_name)

        with tracer.start_as_current_span(
            f"{method} {normalized_path}",
            context=ctx,
            kind=SpanKind.SERVER,
            attributes=span_attrs,
        ) as span:
            try:
                # Log request
                logger.info(
                    f"Request started: {method} {path}",
                    extra={
                        "method": method,
                        "path": path,
                        "client_ip": request.client.host if request.client else None,
                    }
                )

                # Process request
                response = await call_next(request)

                # Calculate duration
                duration = time.perf_counter() - start_time

                # Set response attributes
                span.set_attribute("http.status_code", response.status_code)

                response_size = response.headers.get("content-length")
                if response_size:
                    span.set_attribute("http.response_content_length", int(response_size))

                # Set span status based on HTTP status
                if response.status_code >= 500:
                    span.set_status(Status(StatusCode.ERROR, f"HTTP {response.status_code}"))
                elif response.status_code >= 400:
                    span.set_status(Status(StatusCode.ERROR, f"HTTP {response.status_code}"))
                else:
                    span.set_status(Status(StatusCode.OK))

                # Record metrics
                self.metrics.record_request(
                    endpoint=normalized_path,
                    method=method,
                    status_code=response.status_code,
                    duration_seconds=duration,
                    request_size=int(request.headers.get("content-length", 0)),
                    response_size=int(response_size) if response_size else 0,
                )

                # Log completion
                logger.info(
                    f"Request completed: {method} {path} -> {response.status_code} ({duration:.3f}s)",
                    extra={
                        "method": method,
                        "path": path,
                        "status_code": response.status_code,
                        "duration_seconds": duration,
                    }
                )

                # Inject trace context into response headers for debugging
                trace_headers = inject_trace_context()
                for key, value in trace_headers.items():
                    response.headers[key] = value

                return response

            except Exception as e:
                duration = time.perf_counter() - start_time

                # Record exception in span
                span.set_status(Status(StatusCode.ERROR, str(e)))

                if self.record_exception_details:
                    span.record_exception(e)

                span.set_attribute("http.status_code", 500)
                span.set_attribute("error.type", type(e).__name__)
                span.set_attribute("error.message", str(e))

                # Record error metrics
                self.metrics.record_request(
                    endpoint=normalized_path,
                    method=method,
                    status_code=500,
                    duration_seconds=duration,
                )

                # Log error
                logger.error(
                    f"Request failed: {method} {path} -> {type(e).__name__}: {e}",
                    extra={
                        "method": method,
                        "path": path,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "duration_seconds": duration,
                    },
                    exc_info=True,
                )

                raise


class HealthCheckMiddleware:
    """
    Middleware for health check endpoints.

    Provides:
    - /health - Overall health status
    - /ready - Readiness probe
    - /live - Liveness probe
    """

    def __init__(
        self,
        app: ASGIApp,
        health_path: str = "/health",
        ready_path: str = "/ready",
        live_path: str = "/live",
    ):
        self.app = app
        self.health_path = health_path
        self.ready_path = ready_path
        self.live_path = live_path

        # Health check functions
        self._health_checks: List[Callable[[], bool]] = []
        self._ready_checks: List[Callable[[], bool]] = []

    def add_health_check(self, check: Callable[[], bool]):
        """Add a health check function."""
        self._health_checks.append(check)

    def add_readiness_check(self, check: Callable[[], bool]):
        """Add a readiness check function."""
        self._ready_checks.append(check)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope["path"]

        if path == self.health_path:
            await self._health_response(scope, receive, send)
        elif path == self.ready_path:
            await self._ready_response(scope, receive, send)
        elif path == self.live_path:
            await self._live_response(scope, receive, send)
        else:
            await self.app(scope, receive, send)

    async def _health_response(self, scope, receive, send):
        """Handle /health endpoint."""
        import json

        checks_passed = all(check() for check in self._health_checks)
        status = 200 if checks_passed else 503

        body = json.dumps({
            "status": "healthy" if checks_passed else "unhealthy",
            "checks": len(self._health_checks),
        }).encode()

        await self._send_response(send, status, body)

    async def _ready_response(self, scope, receive, send):
        """Handle /ready endpoint."""
        import json

        checks_passed = all(check() for check in self._ready_checks)
        status = 200 if checks_passed else 503

        body = json.dumps({
            "status": "ready" if checks_passed else "not_ready",
        }).encode()

        await self._send_response(send, status, body)

    async def _live_response(self, scope, receive, send):
        """Handle /live endpoint."""
        import json

        body = json.dumps({"status": "alive"}).encode()
        await self._send_response(send, 200, body)

    async def _send_response(self, send, status: int, body: bytes):
        """Send HTTP response."""
        await send({
            "type": "http.response.start",
            "status": status,
            "headers": [
                [b"content-type", b"application/json"],
                [b"content-length", str(len(body)).encode()],
            ],
        })
        await send({
            "type": "http.response.body",
            "body": body,
        })


def setup_fastapi_observability(app, config=None):
    """
    Setup complete observability for a FastAPI application.

    Args:
        app: FastAPI application instance
        config: ObservabilityConfig (optional)

    Usage:
        from fastapi import FastAPI
        from consciousness_suite.observability import setup_fastapi_observability

        app = FastAPI()
        setup_fastapi_observability(app)
    """
    from fastapi import FastAPI
    from .prometheus.middleware import PrometheusMiddleware, get_metrics_handler

    if config is None:
        from . import ObservabilityConfig
        config = ObservabilityConfig.from_env()

    # Initialize observability
    from . import setup_observability
    setup_observability(config)

    # Add middleware (order matters - first added = outermost)
    app.add_middleware(ObservabilityMiddleware, service_name=config.service_name)

    # Add health check endpoints
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "service": config.service_name}

    @app.get("/ready")
    async def readiness_check():
        return {"status": "ready"}

    @app.get("/live")
    async def liveness_check():
        return {"status": "alive"}

    # Add metrics endpoint
    @app.get("/metrics")
    async def metrics_endpoint():
        return get_metrics_handler()()

    logger.info(f"Observability configured for {config.service_name}")
