"""
Prometheus Metrics Export Module

Provides:
- ConsciousnessMetrics class with all domain metrics
- Request, processing, mesh, and LLM metrics
- Histogram, counter, and gauge instruments
- Automatic labeling and cardinality control
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import time
import logging

from opentelemetry import metrics
from opentelemetry.metrics import (
    Counter,
    Histogram,
    UpDownCounter,
    ObservableGauge,
    Meter,
)

logger = logging.getLogger(__name__)


class MetricUnit(str, Enum):
    """Standard metric units."""
    SECONDS = "s"
    MILLISECONDS = "ms"
    BYTES = "By"
    KILOBYTES = "KBy"
    MEGABYTES = "MBy"
    COUNT = "1"
    PERCENT = "%"
    DOLLARS = "$"
    TOKENS = "tokens"


@dataclass
class MetricDefinition:
    """Definition for a metric instrument."""
    name: str
    description: str
    unit: MetricUnit
    instrument_type: str  # counter, histogram, gauge, updown_counter


class ConsciousnessMetrics:
    """
    Central metrics registry for Consciousness Nexus.

    Provides all observability metrics across domains:
    - Request metrics (HTTP, gRPC, WebSocket)
    - Processing metrics (consciousness operations)
    - Mesh metrics (service mesh, routing, circuit breaker)
    - LLM metrics (calls, tokens, cost, latency)
    - Business metrics (SLO, revenue, user experience)
    - System metrics (CPU, memory, connections)
    """

    # Histogram buckets for different metric types
    LATENCY_BUCKETS = [
        0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75,
        1.0, 2.5, 5.0, 7.5, 10.0, 30.0, 60.0
    ]
    TOKEN_BUCKETS = [
        10, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000
    ]
    SIZE_BUCKETS = [
        100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000
    ]

    def __init__(
        self,
        service_name: str = "consciousness-nexus",
        version: str = "1.0.0",
    ):
        self.service_name = service_name
        self.version = version
        self._meter: Optional[Meter] = None
        self._initialized = False

        # Metric instruments
        self._counters: Dict[str, Counter] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._gauges: Dict[str, Any] = {}
        self._updown_counters: Dict[str, UpDownCounter] = {}

    @property
    def meter(self) -> Meter:
        """Lazy initialization of meter."""
        if self._meter is None:
            self._meter = metrics.get_meter(self.service_name, self.version)
        return self._meter

    def initialize(self):
        """Initialize all metric instruments."""
        if self._initialized:
            return

        self._create_request_metrics()
        self._create_processing_metrics()
        self._create_mesh_metrics()
        self._create_llm_metrics()
        self._create_business_metrics()
        self._create_system_metrics()

        self._initialized = True
        logger.info("ConsciousnessMetrics initialized with all instruments")

    def _create_request_metrics(self):
        """Create HTTP/API request metrics."""
        # Request counter
        self._counters["requests_total"] = self.meter.create_counter(
            name="consciousness_requests_total",
            description="Total number of requests received",
            unit=MetricUnit.COUNT.value,
        )

        # Request duration histogram
        self._histograms["request_duration"] = self.meter.create_histogram(
            name="consciousness_request_duration_seconds",
            description="Request duration in seconds",
            unit=MetricUnit.SECONDS.value,
        )

        # Request size histogram
        self._histograms["request_size"] = self.meter.create_histogram(
            name="consciousness_request_size_bytes",
            description="Request body size in bytes",
            unit=MetricUnit.BYTES.value,
        )

        # Response size histogram
        self._histograms["response_size"] = self.meter.create_histogram(
            name="consciousness_response_size_bytes",
            description="Response body size in bytes",
            unit=MetricUnit.BYTES.value,
        )

        # Active requests gauge
        self._updown_counters["active_requests"] = self.meter.create_up_down_counter(
            name="consciousness_active_requests",
            description="Number of requests currently being processed",
            unit=MetricUnit.COUNT.value,
        )

        # Errors counter
        self._counters["request_errors"] = self.meter.create_counter(
            name="consciousness_request_errors_total",
            description="Total number of request errors",
            unit=MetricUnit.COUNT.value,
        )

    def _create_processing_metrics(self):
        """Create consciousness processing metrics."""
        # Processing duration
        self._histograms["processing_duration"] = self.meter.create_histogram(
            name="consciousness_processing_duration_seconds",
            description="Duration of consciousness processing operations",
            unit=MetricUnit.SECONDS.value,
        )

        # Processing success counter
        self._counters["processing_success"] = self.meter.create_counter(
            name="consciousness_processing_success_total",
            description="Total successful processing operations",
            unit=MetricUnit.COUNT.value,
        )

        # Processing errors counter
        self._counters["processing_errors"] = self.meter.create_counter(
            name="consciousness_processing_errors_total",
            description="Total processing errors",
            unit=MetricUnit.COUNT.value,
        )

        # Thought queue size
        self._updown_counters["thought_queue_size"] = self.meter.create_up_down_counter(
            name="consciousness_thought_queue_size",
            description="Number of thoughts in processing queue",
            unit=MetricUnit.COUNT.value,
        )

        # Vector operations
        self._histograms["vector_operation_duration"] = self.meter.create_histogram(
            name="consciousness_vector_operation_duration_seconds",
            description="Duration of vector matrix operations",
            unit=MetricUnit.SECONDS.value,
        )

    def _create_mesh_metrics(self):
        """Create service mesh metrics."""
        # Node count
        self._updown_counters["mesh_nodes"] = self.meter.create_up_down_counter(
            name="consciousness_mesh_nodes_total",
            description="Total number of mesh nodes",
            unit=MetricUnit.COUNT.value,
        )

        # Connection count
        self._updown_counters["mesh_connections"] = self.meter.create_up_down_counter(
            name="consciousness_mesh_connections_total",
            description="Total number of mesh connections",
            unit=MetricUnit.COUNT.value,
        )

        # Routing latency
        self._histograms["mesh_routing_latency"] = self.meter.create_histogram(
            name="consciousness_mesh_routing_latency_seconds",
            description="Mesh routing latency",
            unit=MetricUnit.SECONDS.value,
        )

        # Circuit breaker state changes
        self._counters["circuit_breaker_transitions"] = self.meter.create_counter(
            name="consciousness_circuit_breaker_transitions_total",
            description="Circuit breaker state transitions",
            unit=MetricUnit.COUNT.value,
        )

        # Mesh quality score
        self._histograms["mesh_quality"] = self.meter.create_histogram(
            name="consciousness_mesh_quality_score",
            description="Mesh connection quality score (0-1)",
            unit=MetricUnit.COUNT.value,
        )

    def _create_llm_metrics(self):
        """Create LLM observability metrics."""
        # LLM calls counter
        self._counters["llm_calls"] = self.meter.create_counter(
            name="consciousness_llm_calls_total",
            description="Total LLM API calls",
            unit=MetricUnit.COUNT.value,
        )

        # Token usage counter
        self._counters["llm_tokens_input"] = self.meter.create_counter(
            name="consciousness_llm_tokens_input_total",
            description="Total input tokens sent to LLM",
            unit=MetricUnit.TOKENS.value,
        )

        self._counters["llm_tokens_output"] = self.meter.create_counter(
            name="consciousness_llm_tokens_output_total",
            description="Total output tokens received from LLM",
            unit=MetricUnit.TOKENS.value,
        )

        # LLM latency
        self._histograms["llm_latency"] = self.meter.create_histogram(
            name="consciousness_llm_latency_seconds",
            description="LLM API call latency",
            unit=MetricUnit.SECONDS.value,
        )

        # LLM cost
        self._counters["llm_cost"] = self.meter.create_counter(
            name="consciousness_llm_cost_dollars",
            description="Total LLM API cost in dollars",
            unit=MetricUnit.DOLLARS.value,
        )

        # LLM errors
        self._counters["llm_errors"] = self.meter.create_counter(
            name="consciousness_llm_errors_total",
            description="Total LLM API errors",
            unit=MetricUnit.COUNT.value,
        )

        # Token usage histogram (for distribution analysis)
        self._histograms["llm_tokens_per_call"] = self.meter.create_histogram(
            name="consciousness_llm_tokens_per_call",
            description="Token usage per LLM call",
            unit=MetricUnit.TOKENS.value,
        )

    def _create_business_metrics(self):
        """Create business/SLO metrics."""
        # SLO compliance
        self._counters["slo_requests_in_budget"] = self.meter.create_counter(
            name="consciousness_slo_requests_in_budget_total",
            description="Requests meeting SLO targets",
            unit=MetricUnit.COUNT.value,
        )

        self._counters["slo_requests_out_of_budget"] = self.meter.create_counter(
            name="consciousness_slo_requests_out_of_budget_total",
            description="Requests violating SLO targets",
            unit=MetricUnit.COUNT.value,
        )

        # User experience scores
        self._histograms["user_experience_score"] = self.meter.create_histogram(
            name="consciousness_user_experience_score",
            description="User experience satisfaction score",
            unit=MetricUnit.COUNT.value,
        )

        # Feature usage
        self._counters["feature_usage"] = self.meter.create_counter(
            name="consciousness_feature_usage_total",
            description="Feature usage counts",
            unit=MetricUnit.COUNT.value,
        )

    def _create_system_metrics(self):
        """Create system/infrastructure metrics."""
        # Connection pool
        self._updown_counters["connection_pool_size"] = self.meter.create_up_down_counter(
            name="consciousness_connection_pool_size",
            description="Number of connections in pool",
            unit=MetricUnit.COUNT.value,
        )

        # Cache metrics
        self._counters["cache_hits"] = self.meter.create_counter(
            name="consciousness_cache_hits_total",
            description="Cache hit count",
            unit=MetricUnit.COUNT.value,
        )

        self._counters["cache_misses"] = self.meter.create_counter(
            name="consciousness_cache_misses_total",
            description="Cache miss count",
            unit=MetricUnit.COUNT.value,
        )

        # Queue depth
        self._updown_counters["queue_depth"] = self.meter.create_up_down_counter(
            name="consciousness_queue_depth",
            description="Message queue depth",
            unit=MetricUnit.COUNT.value,
        )

    # Recording methods
    def record_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        duration_seconds: float,
        request_size: int = 0,
        response_size: int = 0,
    ):
        """Record an HTTP request with all relevant metrics."""
        labels = {
            "endpoint": endpoint,
            "method": method,
            "status": str(status_code),
            "status_class": f"{status_code // 100}xx",
        }

        self._counters["requests_total"].add(1, labels)
        self._histograms["request_duration"].record(duration_seconds, labels)

        if request_size > 0:
            self._histograms["request_size"].record(request_size, {"endpoint": endpoint})

        if response_size > 0:
            self._histograms["response_size"].record(response_size, {"endpoint": endpoint})

        if status_code >= 400:
            error_labels = {
                "endpoint": endpoint,
                "error_type": "client_error" if status_code < 500 else "server_error",
            }
            self._counters["request_errors"].add(1, error_labels)

    def record_processing(
        self,
        operation: str,
        duration_seconds: float,
        success: bool = True,
        error_type: Optional[str] = None,
    ):
        """Record a processing operation."""
        labels = {"operation": operation}

        self._histograms["processing_duration"].record(duration_seconds, labels)

        if success:
            self._counters["processing_success"].add(1, labels)
        else:
            error_labels = {**labels, "error_type": error_type or "unknown"}
            self._counters["processing_errors"].add(1, error_labels)

    def record_llm_call(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        duration_seconds: float,
        cost_dollars: float = 0.0,
        success: bool = True,
        error_type: Optional[str] = None,
    ):
        """Record an LLM API call with full metrics."""
        labels = {"provider": provider, "model": model}

        self._counters["llm_calls"].add(1, labels)
        self._counters["llm_tokens_input"].add(input_tokens, labels)
        self._counters["llm_tokens_output"].add(output_tokens, labels)
        self._histograms["llm_latency"].record(duration_seconds, labels)
        self._histograms["llm_tokens_per_call"].record(input_tokens + output_tokens, labels)

        if cost_dollars > 0:
            self._counters["llm_cost"].add(cost_dollars, labels)

        if not success:
            error_labels = {**labels, "error_type": error_type or "unknown"}
            self._counters["llm_errors"].add(1, error_labels)

    def record_mesh_routing(
        self,
        source: str,
        target: str,
        latency_seconds: float,
        quality: float,
    ):
        """Record mesh routing metrics."""
        labels = {"source": source, "target": target}

        self._histograms["mesh_routing_latency"].record(latency_seconds, labels)
        self._histograms["mesh_quality"].record(quality, labels)

    def record_circuit_breaker_transition(
        self,
        source: str,
        target: str,
        from_state: str,
        to_state: str,
    ):
        """Record circuit breaker state transition."""
        labels = {
            "source": source,
            "target": target,
            "from_state": from_state,
            "to_state": to_state,
        }
        self._counters["circuit_breaker_transitions"].add(1, labels)

    def record_slo(
        self,
        slo_name: str,
        in_budget: bool,
    ):
        """Record SLO compliance."""
        labels = {"slo_name": slo_name}
        if in_budget:
            self._counters["slo_requests_in_budget"].add(1, labels)
        else:
            self._counters["slo_requests_out_of_budget"].add(1, labels)

    # Context managers for timing
    class TimingContext:
        """Context manager for timing operations."""
        def __init__(self, histogram: Histogram, labels: Dict[str, str]):
            self.histogram = histogram
            self.labels = labels
            self.start_time: float = 0

        def __enter__(self):
            self.start_time = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            duration = time.perf_counter() - self.start_time
            self.histogram.record(duration, self.labels)
            return False

    def time_request(self, endpoint: str, method: str) -> TimingContext:
        """Context manager for timing requests."""
        return self.TimingContext(
            self._histograms["request_duration"],
            {"endpoint": endpoint, "method": method},
        )

    def time_processing(self, operation: str) -> TimingContext:
        """Context manager for timing processing operations."""
        return self.TimingContext(
            self._histograms["processing_duration"],
            {"operation": operation},
        )

    def time_llm_call(self, provider: str, model: str) -> TimingContext:
        """Context manager for timing LLM calls."""
        return self.TimingContext(
            self._histograms["llm_latency"],
            {"provider": provider, "model": model},
        )


# Global metrics instance
_default_metrics: Optional[ConsciousnessMetrics] = None


def get_consciousness_metrics() -> ConsciousnessMetrics:
    """Get the default ConsciousnessMetrics instance."""
    global _default_metrics
    if _default_metrics is None:
        _default_metrics = ConsciousnessMetrics()
        _default_metrics.initialize()
    return _default_metrics
