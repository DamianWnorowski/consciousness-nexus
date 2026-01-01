"""
Tracing Storage Module - Jaeger/Tempo Integration

Provides distributed trace storage and querying capabilities:
- JaegerClient: Query traces from Jaeger backend
- TempoClient: Query traces from Grafana Tempo
- TraceStorage: Unified trace persistence abstraction
- TraceAnalytics: Latency histograms, error rates, service dependencies

Thread-safe implementations with Prometheus metrics instrumentation.
"""

from .core import (
    ConsciousnessTracer,
    traced,
    get_current_trace_context,
    set_trace_context_from_headers,
    get_consciousness_tracer,
)
from .jaeger_client import (
    JaegerClient,
    JaegerConfig,
    JaegerTrace,
    JaegerSpan,
    JaegerProcess,
    JaegerReference,
    JaegerLog,
    JaegerTag,
)
from .tempo_client import (
    TempoClient,
    TempoConfig,
    TempoTrace,
    TempoSpan,
    TempoSearchResult,
    TempoMetrics,
)
from .trace_storage import (
    TraceStorage,
    TraceStorageConfig,
    StoredTrace,
    StoredSpan,
    TraceQuery,
    TraceIndex,
    StorageBackend,
)
from .trace_analytics import (
    TraceAnalytics,
    AnalyticsConfig,
    LatencyHistogram,
    ErrorRateMetrics,
    ServiceDependency,
    DependencyGraph,
    LatencyPercentiles,
    ServiceStats,
    TraceAnomalyDetector,
)

__all__ = [
    # Core
    "ConsciousnessTracer",
    "traced",
    "get_current_trace_context",
    "set_trace_context_from_headers",
    "get_consciousness_tracer",
    # Jaeger
    "JaegerClient",
    "JaegerConfig",
    "JaegerTrace",
    "JaegerSpan",
    "JaegerProcess",
    "JaegerReference",
    "JaegerLog",
    "JaegerTag",
    # Tempo
    "TempoClient",
    "TempoConfig",
    "TempoTrace",
    "TempoSpan",
    "TempoSearchResult",
    "TempoMetrics",
    # Storage
    "TraceStorage",
    "TraceStorageConfig",
    "StoredTrace",
    "StoredSpan",
    "TraceQuery",
    "TraceIndex",
    "StorageBackend",
    # Analytics
    "TraceAnalytics",
    "AnalyticsConfig",
    "LatencyHistogram",
    "ErrorRateMetrics",
    "ServiceDependency",
    "DependencyGraph",
    "LatencyPercentiles",
    "ServiceStats",
    "TraceAnomalyDetector",
]

__version__ = "1.0.0"
