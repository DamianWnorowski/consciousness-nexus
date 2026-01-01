"""
Prometheus Integration Module

Provides:
- Custom collectors for consciousness-specific metrics
- FastAPI/Starlette middleware for request metrics
- Prometheus client configuration
- /metrics endpoint handler
"""

from .collectors import (
    ConsciousnessCollector,
    MeshCollector,
    LLMCollector,
    SystemCollector,
)
from .middleware import (
    PrometheusMiddleware,
    get_metrics_handler,
    setup_prometheus,
)
from .exporters import (
    MetricsExporter,
    PushGatewayExporter,
)

__all__ = [
    # Collectors
    "ConsciousnessCollector",
    "MeshCollector",
    "LLMCollector",
    "SystemCollector",
    # Middleware
    "PrometheusMiddleware",
    "get_metrics_handler",
    "setup_prometheus",
    # Exporters
    "MetricsExporter",
    "PushGatewayExporter",
]
