"""
Prometheus Middleware for FastAPI/Starlette

Provides:
- Automatic request metrics collection
- /metrics endpoint handler
- Custom metric registration
- Histogram buckets for latency
"""

from typing import Callable, Optional, List, Dict, Any
import time
import logging
from functools import wraps

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, PlainTextResponse
from starlette.routing import Route
from starlette.types import ASGIApp

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
    REGISTRY,
    multiprocess,
)

logger = logging.getLogger(__name__)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """
    ASGI middleware for collecting HTTP request metrics.

    Collects:
    - Request counts by path, method, status
    - Request latency histograms
    - Request/response sizes
    - Active request count

    Usage:
        app = FastAPI()
        app.add_middleware(PrometheusMiddleware)
    """

    # Default histogram buckets for HTTP latency
    DEFAULT_BUCKETS = [
        0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75,
        1.0, 2.5, 5.0, 7.5, 10.0
    ]

    def __init__(
        self,
        app: ASGIApp,
        app_name: str = "consciousness_nexus",
        prefix: str = "http",
        buckets: Optional[List[float]] = None,
        exclude_paths: Optional[List[str]] = None,
        group_paths: bool = True,
        registry: Optional[CollectorRegistry] = None,
    ):
        super().__init__(app)
        self.app_name = app_name
        self.prefix = prefix
        self.buckets = buckets or self.DEFAULT_BUCKETS
        self.exclude_paths = exclude_paths or ["/metrics", "/health", "/ready"]
        self.group_paths = group_paths
        self.registry = registry or REGISTRY

        # Create metrics
        self._setup_metrics()

    def _setup_metrics(self):
        """Initialize Prometheus metrics."""
        prefix = f"{self.app_name}_{self.prefix}"

        # Request counter
        self.requests_total = Counter(
            f"{prefix}_requests_total",
            "Total HTTP requests",
            ["method", "path", "status"],
            registry=self.registry,
        )

        # Request latency
        self.request_duration = Histogram(
            f"{prefix}_request_duration_seconds",
            "HTTP request latency",
            ["method", "path"],
            buckets=self.buckets,
            registry=self.registry,
        )

        # Request size
        self.request_size = Histogram(
            f"{prefix}_request_size_bytes",
            "HTTP request body size",
            ["method", "path"],
            buckets=[100, 500, 1000, 5000, 10000, 50000, 100000, 500000],
            registry=self.registry,
        )

        # Response size
        self.response_size = Histogram(
            f"{prefix}_response_size_bytes",
            "HTTP response body size",
            ["method", "path"],
            buckets=[100, 500, 1000, 5000, 10000, 50000, 100000, 500000],
            registry=self.registry,
        )

        # Active requests
        self.requests_in_progress = Gauge(
            f"{prefix}_requests_in_progress",
            "HTTP requests currently being processed",
            ["method", "path"],
            registry=self.registry,
        )

        # Exceptions
        self.exceptions_total = Counter(
            f"{prefix}_exceptions_total",
            "Total HTTP exceptions",
            ["method", "path", "exception_type"],
            registry=self.registry,
        )

    def _get_path_label(self, request: Request) -> str:
        """
        Get path label for metrics, optionally grouping paths.

        Groups paths like /users/123 into /users/{id} to avoid
        high cardinality issues.
        """
        path = request.url.path

        if not self.group_paths:
            return path

        # Try to get the route pattern from the request
        if hasattr(request, "scope") and "route" in request.scope:
            route = request.scope["route"]
            if hasattr(route, "path"):
                return route.path

        # Fallback: simple path normalization
        # Replace numeric path segments with {id}
        parts = path.split("/")
        normalized = []
        for part in parts:
            if part.isdigit():
                normalized.append("{id}")
            elif part and len(part) == 36 and "-" in part:
                # UUID pattern
                normalized.append("{uuid}")
            else:
                normalized.append(part)
        return "/".join(normalized)

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request and collect metrics."""
        # Skip excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        method = request.method
        path = self._get_path_label(request)

        # Track in-progress requests
        self.requests_in_progress.labels(method=method, path=path).inc()

        start_time = time.perf_counter()

        try:
            # Get request size
            content_length = request.headers.get("content-length")
            if content_length:
                self.request_size.labels(method=method, path=path).observe(
                    int(content_length)
                )

            # Process request
            response = await call_next(request)

            # Record metrics
            duration = time.perf_counter() - start_time
            status = str(response.status_code)

            self.requests_total.labels(
                method=method, path=path, status=status
            ).inc()

            self.request_duration.labels(
                method=method, path=path
            ).observe(duration)

            # Response size
            response_size = response.headers.get("content-length")
            if response_size:
                self.response_size.labels(method=method, path=path).observe(
                    int(response_size)
                )

            return response

        except Exception as e:
            # Record exception
            self.exceptions_total.labels(
                method=method,
                path=path,
                exception_type=type(e).__name__,
            ).inc()

            # Still record duration and count
            duration = time.perf_counter() - start_time
            self.requests_total.labels(
                method=method, path=path, status="500"
            ).inc()
            self.request_duration.labels(method=method, path=path).observe(duration)

            raise

        finally:
            self.requests_in_progress.labels(method=method, path=path).dec()


def get_metrics_handler(registry: Optional[CollectorRegistry] = None) -> Callable:
    """
    Create a /metrics endpoint handler for Prometheus scraping.

    Args:
        registry: Custom registry (uses default if not provided)

    Returns:
        Async handler function for /metrics endpoint

    Usage:
        from fastapi import FastAPI
        app = FastAPI()

        @app.get("/metrics")
        async def metrics():
            return get_metrics_handler()()
    """
    def handler() -> Response:
        if registry is None:
            # Handle multiprocess mode
            try:
                from prometheus_client import multiprocess, CollectorRegistry
                import os
                if "prometheus_multiproc_dir" in os.environ:
                    reg = CollectorRegistry()
                    multiprocess.MultiProcessCollector(reg)
                    metrics_output = generate_latest(reg)
                else:
                    metrics_output = generate_latest(REGISTRY)
            except ImportError:
                metrics_output = generate_latest(REGISTRY)
        else:
            metrics_output = generate_latest(registry)

        return PlainTextResponse(
            content=metrics_output,
            media_type=CONTENT_TYPE_LATEST,
        )

    return handler


def create_metrics_route(
    path: str = "/metrics",
    registry: Optional[CollectorRegistry] = None,
) -> Route:
    """
    Create a Starlette Route for /metrics endpoint.

    Args:
        path: URL path for metrics endpoint
        registry: Custom registry

    Returns:
        Starlette Route object
    """
    async def metrics_endpoint(request: Request) -> Response:
        return get_metrics_handler(registry)()

    return Route(path, endpoint=metrics_endpoint, methods=["GET"])


def setup_prometheus(config) -> CollectorRegistry:
    """
    Initialize Prometheus metrics with configuration.

    Args:
        config: ObservabilityConfig instance

    Returns:
        Configured CollectorRegistry
    """
    from .collectors import (
        ConsciousnessCollector,
        MeshCollector,
        LLMCollector,
        SystemCollector,
    )

    registry = REGISTRY

    # Create collectors
    consciousness_collector = ConsciousnessCollector(registry)
    mesh_collector = MeshCollector(registry)
    llm_collector = LLMCollector(registry)
    system_collector = SystemCollector(registry)

    # Store collectors for access
    _collectors["consciousness"] = consciousness_collector
    _collectors["mesh"] = mesh_collector
    _collectors["llm"] = llm_collector
    _collectors["system"] = system_collector

    logger.info("Prometheus collectors initialized")

    return registry


# Global collectors storage
_collectors: Dict[str, Any] = {}


def get_collector(name: str):
    """Get a specific collector by name."""
    return _collectors.get(name)


# Decorator for timing functions
def timed(
    metric_name: str,
    labels: Optional[Dict[str, str]] = None,
):
    """
    Decorator to time function execution.

    Args:
        metric_name: Name of histogram metric
        labels: Static labels to apply

    Usage:
        @timed("consciousness_processing_duration_seconds", {"operation": "analyze"})
        async def analyze(data):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                duration = time.perf_counter() - start
                # Record to appropriate collector
                collector = get_collector("consciousness")
                if collector:
                    collector.processing_duration.labels(
                        **(labels or {})
                    ).observe(duration)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.perf_counter() - start
                collector = get_collector("consciousness")
                if collector:
                    collector.processing_duration.labels(
                        **(labels or {})
                    ).observe(duration)

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
