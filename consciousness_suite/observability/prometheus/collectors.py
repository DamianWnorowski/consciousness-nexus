"""
Custom Prometheus Collectors

Provides collectors for:
- Consciousness processing metrics
- Mesh network metrics
- LLM usage metrics
- System resource metrics
"""

from typing import Any, Callable, Dict, Generator, List, Optional
import time
import logging
import psutil
from dataclasses import dataclass

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    Summary,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from prometheus_client.core import GaugeMetricFamily, CounterMetricFamily

logger = logging.getLogger(__name__)


@dataclass
class MetricValue:
    """Container for metric value with labels."""
    value: float
    labels: Dict[str, str]


class ConsciousnessCollector:
    """
    Collector for consciousness processing metrics.

    Collects:
    - Thought processing counts and latencies
    - Vector matrix operations
    - Analysis pipeline metrics
    - Queue depths and backlogs
    """

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()

        # Processing counters
        self.thoughts_processed = Counter(
            "consciousness_thoughts_processed_total",
            "Total thoughts processed",
            ["processor_type", "status"],
            registry=self.registry,
        )

        self.thoughts_queued = Gauge(
            "consciousness_thoughts_queued",
            "Current thoughts in queue",
            ["queue_name"],
            registry=self.registry,
        )

        # Processing histograms
        self.processing_duration = Histogram(
            "consciousness_processing_duration_seconds",
            "Time spent processing thoughts",
            ["operation", "processor_type"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
            registry=self.registry,
        )

        self.vector_operation_duration = Histogram(
            "consciousness_vector_operation_duration_seconds",
            "Time spent on vector operations",
            ["operation"],
            buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
            registry=self.registry,
        )

        # Analysis metrics
        self.analysis_depth = Gauge(
            "consciousness_analysis_depth",
            "Current recursion depth in analysis",
            ["analysis_type"],
            registry=self.registry,
        )

        self.confidence_score = Histogram(
            "consciousness_confidence_score",
            "Confidence scores from analysis",
            ["analysis_type"],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
            registry=self.registry,
        )

        # Info metric
        self.info = Info(
            "consciousness",
            "Consciousness Nexus information",
            registry=self.registry,
        )
        self.info.info({
            "version": "1.0.0",
            "environment": "production",
        })

    def record_thought_processed(
        self,
        processor_type: str,
        status: str = "success",
    ):
        """Record a processed thought."""
        self.thoughts_processed.labels(
            processor_type=processor_type,
            status=status,
        ).inc()

    def set_queue_depth(self, queue_name: str, depth: int):
        """Set current queue depth."""
        self.thoughts_queued.labels(queue_name=queue_name).set(depth)

    def observe_processing_duration(
        self,
        duration: float,
        operation: str,
        processor_type: str,
    ):
        """Record processing duration."""
        self.processing_duration.labels(
            operation=operation,
            processor_type=processor_type,
        ).observe(duration)

    def observe_vector_operation(self, duration: float, operation: str):
        """Record vector operation duration."""
        self.vector_operation_duration.labels(operation=operation).observe(duration)

    def set_analysis_depth(self, analysis_type: str, depth: int):
        """Set current analysis recursion depth."""
        self.analysis_depth.labels(analysis_type=analysis_type).set(depth)

    def observe_confidence(self, analysis_type: str, score: float):
        """Record confidence score."""
        self.confidence_score.labels(analysis_type=analysis_type).observe(score)


class MeshCollector:
    """
    Collector for service mesh metrics.

    Collects:
    - Node counts and states
    - Connection metrics
    - Routing latencies
    - Circuit breaker states
    - Quality scores
    """

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()

        # Node metrics
        self.mesh_nodes = Gauge(
            "consciousness_mesh_nodes",
            "Number of mesh nodes",
            ["state", "zone"],
            registry=self.registry,
        )

        self.mesh_connections = Gauge(
            "consciousness_mesh_connections",
            "Number of mesh connections",
            ["source_zone", "target_zone", "state"],
            registry=self.registry,
        )

        # Routing metrics
        self.routing_latency = Histogram(
            "consciousness_mesh_routing_latency_seconds",
            "Mesh routing latency",
            ["source", "target", "route_type"],
            buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1],
            registry=self.registry,
        )

        self.routing_errors = Counter(
            "consciousness_mesh_routing_errors_total",
            "Mesh routing errors",
            ["source", "target", "error_type"],
            registry=self.registry,
        )

        # Circuit breaker
        self.circuit_breaker_state = Gauge(
            "consciousness_circuit_breaker_state",
            "Circuit breaker state (0=closed, 1=half-open, 2=open)",
            ["source", "target"],
            registry=self.registry,
        )

        self.circuit_breaker_transitions = Counter(
            "consciousness_circuit_breaker_transitions_total",
            "Circuit breaker state transitions",
            ["source", "target", "from_state", "to_state"],
            registry=self.registry,
        )

        # Quality metrics
        self.connection_quality = Histogram(
            "consciousness_mesh_connection_quality",
            "Connection quality scores (0-1)",
            ["source", "target"],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
            registry=self.registry,
        )

        self.mesh_health_score = Gauge(
            "consciousness_mesh_health_score",
            "Overall mesh health score",
            ["zone"],
            registry=self.registry,
        )

    def set_node_count(self, state: str, zone: str, count: int):
        """Set node count for state/zone."""
        self.mesh_nodes.labels(state=state, zone=zone).set(count)

    def set_connection_count(
        self,
        source_zone: str,
        target_zone: str,
        state: str,
        count: int,
    ):
        """Set connection count."""
        self.mesh_connections.labels(
            source_zone=source_zone,
            target_zone=target_zone,
            state=state,
        ).set(count)

    def observe_routing_latency(
        self,
        source: str,
        target: str,
        route_type: str,
        latency: float,
    ):
        """Record routing latency."""
        self.routing_latency.labels(
            source=source,
            target=target,
            route_type=route_type,
        ).observe(latency)

    def record_routing_error(
        self,
        source: str,
        target: str,
        error_type: str,
    ):
        """Record routing error."""
        self.routing_errors.labels(
            source=source,
            target=target,
            error_type=error_type,
        ).inc()

    def set_circuit_breaker_state(
        self,
        source: str,
        target: str,
        state: int,
    ):
        """Set circuit breaker state (0=closed, 1=half-open, 2=open)."""
        self.circuit_breaker_state.labels(
            source=source,
            target=target,
        ).set(state)

    def record_circuit_breaker_transition(
        self,
        source: str,
        target: str,
        from_state: str,
        to_state: str,
    ):
        """Record circuit breaker transition."""
        self.circuit_breaker_transitions.labels(
            source=source,
            target=target,
            from_state=from_state,
            to_state=to_state,
        ).inc()

    def observe_connection_quality(
        self,
        source: str,
        target: str,
        quality: float,
    ):
        """Record connection quality score."""
        self.connection_quality.labels(
            source=source,
            target=target,
        ).observe(quality)

    def set_mesh_health(self, zone: str, score: float):
        """Set overall mesh health score."""
        self.mesh_health_score.labels(zone=zone).set(score)


class LLMCollector:
    """
    Collector for LLM observability metrics.

    Collects:
    - API call counts and latencies
    - Token usage (input/output)
    - Cost tracking
    - Error rates by model/provider
    """

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()

        # Call metrics
        self.llm_calls = Counter(
            "consciousness_llm_calls_total",
            "Total LLM API calls",
            ["provider", "model", "operation", "status"],
            registry=self.registry,
        )

        self.llm_latency = Histogram(
            "consciousness_llm_latency_seconds",
            "LLM API call latency",
            ["provider", "model", "operation"],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
            registry=self.registry,
        )

        # Token metrics
        self.llm_tokens_input = Counter(
            "consciousness_llm_tokens_input_total",
            "Total input tokens",
            ["provider", "model"],
            registry=self.registry,
        )

        self.llm_tokens_output = Counter(
            "consciousness_llm_tokens_output_total",
            "Total output tokens",
            ["provider", "model"],
            registry=self.registry,
        )

        self.llm_tokens_per_call = Histogram(
            "consciousness_llm_tokens_per_call",
            "Tokens per LLM call",
            ["provider", "model", "direction"],
            buckets=[10, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 50000],
            registry=self.registry,
        )

        # Cost metrics
        self.llm_cost = Counter(
            "consciousness_llm_cost_dollars_total",
            "Total LLM API cost in dollars",
            ["provider", "model"],
            registry=self.registry,
        )

        # Rate limiting
        self.llm_rate_limit_hits = Counter(
            "consciousness_llm_rate_limit_hits_total",
            "Rate limit hits",
            ["provider", "model"],
            registry=self.registry,
        )

        # Active requests
        self.llm_active_requests = Gauge(
            "consciousness_llm_active_requests",
            "Currently active LLM requests",
            ["provider", "model"],
            registry=self.registry,
        )

    def record_call(
        self,
        provider: str,
        model: str,
        operation: str,
        status: str,
        latency: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost: float = 0.0,
    ):
        """Record an LLM API call."""
        labels = {"provider": provider, "model": model}

        self.llm_calls.labels(**labels, operation=operation, status=status).inc()
        self.llm_latency.labels(**labels, operation=operation).observe(latency)

        if input_tokens > 0:
            self.llm_tokens_input.labels(**labels).inc(input_tokens)
            self.llm_tokens_per_call.labels(
                **labels, direction="input"
            ).observe(input_tokens)

        if output_tokens > 0:
            self.llm_tokens_output.labels(**labels).inc(output_tokens)
            self.llm_tokens_per_call.labels(
                **labels, direction="output"
            ).observe(output_tokens)

        if cost > 0:
            self.llm_cost.labels(**labels).inc(cost)

    def record_rate_limit(self, provider: str, model: str):
        """Record rate limit hit."""
        self.llm_rate_limit_hits.labels(provider=provider, model=model).inc()

    def set_active_requests(self, provider: str, model: str, count: int):
        """Set active request count."""
        self.llm_active_requests.labels(provider=provider, model=model).set(count)


class SystemCollector:
    """
    Collector for system resource metrics.

    Collects:
    - CPU usage
    - Memory usage
    - Disk I/O
    - Network I/O
    - Process metrics
    """

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()

        # CPU metrics
        self.cpu_usage = Gauge(
            "consciousness_cpu_usage_percent",
            "CPU usage percentage",
            ["cpu"],
            registry=self.registry,
        )

        # Memory metrics
        self.memory_usage = Gauge(
            "consciousness_memory_usage_bytes",
            "Memory usage in bytes",
            ["type"],
            registry=self.registry,
        )

        self.memory_percent = Gauge(
            "consciousness_memory_usage_percent",
            "Memory usage percentage",
            registry=self.registry,
        )

        # Process metrics
        self.process_cpu = Gauge(
            "consciousness_process_cpu_percent",
            "Process CPU usage",
            registry=self.registry,
        )

        self.process_memory = Gauge(
            "consciousness_process_memory_bytes",
            "Process memory usage",
            ["type"],
            registry=self.registry,
        )

        self.process_threads = Gauge(
            "consciousness_process_threads",
            "Number of process threads",
            registry=self.registry,
        )

        self.process_open_files = Gauge(
            "consciousness_process_open_files",
            "Number of open file descriptors",
            registry=self.registry,
        )

        # Connection metrics
        self.process_connections = Gauge(
            "consciousness_process_connections",
            "Number of network connections",
            ["status"],
            registry=self.registry,
        )

    def collect(self):
        """Collect all system metrics."""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(percpu=True)
            for i, percent in enumerate(cpu_percent):
                self.cpu_usage.labels(cpu=str(i)).set(percent)

            # Memory
            mem = psutil.virtual_memory()
            self.memory_usage.labels(type="used").set(mem.used)
            self.memory_usage.labels(type="available").set(mem.available)
            self.memory_usage.labels(type="total").set(mem.total)
            self.memory_percent.set(mem.percent)

            # Process
            process = psutil.Process()
            self.process_cpu.set(process.cpu_percent())

            mem_info = process.memory_info()
            self.process_memory.labels(type="rss").set(mem_info.rss)
            self.process_memory.labels(type="vms").set(mem_info.vms)

            self.process_threads.set(process.num_threads())

            try:
                self.process_open_files.set(len(process.open_files()))
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                pass

            # Connections
            try:
                connections = process.connections()
                status_counts: Dict[str, int] = {}
                for conn in connections:
                    status = conn.status
                    status_counts[status] = status_counts.get(status, 0) + 1
                for status, count in status_counts.items():
                    self.process_connections.labels(status=status).set(count)
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                pass

        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")
