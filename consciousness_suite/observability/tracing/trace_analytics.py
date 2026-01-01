"""
Trace Analytics - Distributed Tracing Analysis

Provides comprehensive trace analysis capabilities:
- Latency histograms and percentile calculations
- Error rate tracking and trends
- Service dependency graph construction
- Span-level aggregations
- Anomaly detection for latency spikes

Thread-safe with Prometheus metrics instrumentation.
"""

from __future__ import annotations

import logging
import math
import statistics
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from prometheus_client import Counter, Gauge, Histogram

from .trace_storage import StoredTrace, StoredSpan, TraceStorage, TraceQuery

logger = logging.getLogger(__name__)


@dataclass
class LatencyPercentiles:
    """Latency percentile values."""
    p50: float = 0.0
    p75: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    min: float = 0.0
    max: float = 0.0
    mean: float = 0.0
    stddev: float = 0.0
    count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "p50": self.p50,
            "p75": self.p75,
            "p90": self.p90,
            "p95": self.p95,
            "p99": self.p99,
            "min": self.min,
            "max": self.max,
            "mean": self.mean,
            "stddev": self.stddev,
            "count": self.count,
        }

    @classmethod
    def from_values(cls, values: List[float]) -> LatencyPercentiles:
        if not values:
            return cls()

        sorted_values = sorted(values)
        n = len(sorted_values)

        def percentile(p: float) -> float:
            if n == 0:
                return 0.0
            idx = (p / 100) * (n - 1)
            lower = int(idx)
            upper = min(lower + 1, n - 1)
            weight = idx - lower
            return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight

        mean = statistics.mean(sorted_values)
        stddev = statistics.stdev(sorted_values) if n > 1 else 0.0

        return cls(
            p50=percentile(50),
            p75=percentile(75),
            p90=percentile(90),
            p95=percentile(95),
            p99=percentile(99),
            min=sorted_values[0],
            max=sorted_values[-1],
            mean=mean,
            stddev=stddev,
            count=n,
        )


@dataclass
class LatencyBucket:
    """A single histogram bucket."""
    le: float  # Less than or equal to (upper bound)
    count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"le": self.le, "count": self.count}


@dataclass
class LatencyHistogram:
    """Latency distribution histogram."""
    service: str
    operation: str
    buckets: List[LatencyBucket] = field(default_factory=list)
    total_count: int = 0
    sum_ms: float = 0.0
    percentiles: Optional[LatencyPercentiles] = None
    time_window_start: Optional[datetime] = None
    time_window_end: Optional[datetime] = None

    # Default bucket boundaries in milliseconds
    DEFAULT_BUCKETS = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, float("inf")]

    def __post_init__(self):
        if not self.buckets:
            self.buckets = [LatencyBucket(le=b) for b in self.DEFAULT_BUCKETS]

    def add(self, latency_ms: float):
        """Add a latency value to the histogram."""
        self.total_count += 1
        self.sum_ms += latency_ms

        for bucket in self.buckets:
            if latency_ms <= bucket.le:
                bucket.count += 1

    @property
    def average_ms(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.sum_ms / self.total_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "service": self.service,
            "operation": self.operation,
            "buckets": [b.to_dict() for b in self.buckets],
            "total_count": self.total_count,
            "sum_ms": self.sum_ms,
            "average_ms": self.average_ms,
            "percentiles": self.percentiles.to_dict() if self.percentiles else None,
            "time_window_start": self.time_window_start.isoformat() if self.time_window_start else None,
            "time_window_end": self.time_window_end.isoformat() if self.time_window_end else None,
        }


@dataclass
class ErrorRateMetrics:
    """Error rate metrics for a service/operation."""
    service: str
    operation: str
    total_count: int = 0
    error_count: int = 0
    error_by_type: Dict[str, int] = field(default_factory=dict)
    time_window_start: Optional[datetime] = None
    time_window_end: Optional[datetime] = None

    @property
    def error_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.error_count / self.total_count

    @property
    def success_rate(self) -> float:
        return 1.0 - self.error_rate

    def to_dict(self) -> Dict[str, Any]:
        return {
            "service": self.service,
            "operation": self.operation,
            "total_count": self.total_count,
            "error_count": self.error_count,
            "error_rate": self.error_rate,
            "success_rate": self.success_rate,
            "error_by_type": self.error_by_type,
            "time_window_start": self.time_window_start.isoformat() if self.time_window_start else None,
            "time_window_end": self.time_window_end.isoformat() if self.time_window_end else None,
        }


@dataclass
class ServiceDependency:
    """A dependency link between two services."""
    source: str
    target: str
    call_count: int = 0
    error_count: int = 0
    total_latency_ms: float = 0.0

    @property
    def average_latency_ms(self) -> float:
        if self.call_count == 0:
            return 0.0
        return self.total_latency_ms / self.call_count

    @property
    def error_rate(self) -> float:
        if self.call_count == 0:
            return 0.0
        return self.error_count / self.call_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "call_count": self.call_count,
            "error_count": self.error_count,
            "average_latency_ms": self.average_latency_ms,
            "error_rate": self.error_rate,
        }


@dataclass
class ServiceStats:
    """Aggregated statistics for a service."""
    service: str
    span_count: int = 0
    trace_count: int = 0
    error_count: int = 0
    total_latency_ms: float = 0.0
    operations: Set[str] = field(default_factory=set)
    incoming_dependencies: List[str] = field(default_factory=list)
    outgoing_dependencies: List[str] = field(default_factory=list)

    @property
    def average_latency_ms(self) -> float:
        if self.span_count == 0:
            return 0.0
        return self.total_latency_ms / self.span_count

    @property
    def error_rate(self) -> float:
        if self.span_count == 0:
            return 0.0
        return self.error_count / self.span_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "service": self.service,
            "span_count": self.span_count,
            "trace_count": self.trace_count,
            "error_count": self.error_count,
            "average_latency_ms": self.average_latency_ms,
            "error_rate": self.error_rate,
            "operation_count": len(self.operations),
            "operations": list(self.operations),
            "incoming_dependencies": self.incoming_dependencies,
            "outgoing_dependencies": self.outgoing_dependencies,
        }


@dataclass
class DependencyGraph:
    """Service dependency graph."""
    nodes: List[ServiceStats] = field(default_factory=list)
    edges: List[ServiceDependency] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "generated_at": self.generated_at.isoformat(),
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
        }

    def get_service(self, name: str) -> Optional[ServiceStats]:
        for node in self.nodes:
            if node.service == name:
                return node
        return None

    def get_dependency(self, source: str, target: str) -> Optional[ServiceDependency]:
        for edge in self.edges:
            if edge.source == source and edge.target == target:
                return edge
        return None


@dataclass
class LatencyAnomaly:
    """A detected latency anomaly."""
    trace_id: str
    span_id: str
    service: str
    operation: str
    latency_ms: float
    expected_latency_ms: float
    deviation_factor: float  # How many stddevs from mean
    detected_at: datetime = field(default_factory=datetime.now)

    @property
    def severity(self) -> str:
        if self.deviation_factor >= 5:
            return "critical"
        elif self.deviation_factor >= 3:
            return "high"
        elif self.deviation_factor >= 2:
            return "medium"
        return "low"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "service": self.service,
            "operation": self.operation,
            "latency_ms": self.latency_ms,
            "expected_latency_ms": self.expected_latency_ms,
            "deviation_factor": self.deviation_factor,
            "severity": self.severity,
            "detected_at": self.detected_at.isoformat(),
        }


class TraceAnomalyDetector:
    """Detects anomalies in trace latencies.

    Uses statistical analysis to identify spans with
    unusually high latency compared to historical data.
    """

    def __init__(
        self,
        min_samples: int = 30,
        deviation_threshold: float = 3.0,
        learning_window_hours: int = 1,
    ):
        self.min_samples = min_samples
        self.deviation_threshold = deviation_threshold
        self.learning_window_hours = learning_window_hours

        self._baselines: Dict[Tuple[str, str], LatencyPercentiles] = {}
        self._lock = threading.Lock()

    def update_baseline(
        self,
        service: str,
        operation: str,
        latency_values: List[float],
    ):
        """Update baseline statistics for an operation.

        Args:
            service: Service name
            operation: Operation name
            latency_values: Historical latency values
        """
        if len(latency_values) < self.min_samples:
            return

        key = (service, operation)
        with self._lock:
            self._baselines[key] = LatencyPercentiles.from_values(latency_values)

    def detect(
        self,
        service: str,
        operation: str,
        latency_ms: float,
        trace_id: str = "",
        span_id: str = "",
    ) -> Optional[LatencyAnomaly]:
        """Detect if a latency value is anomalous.

        Args:
            service: Service name
            operation: Operation name
            latency_ms: Observed latency
            trace_id: Trace ID
            span_id: Span ID

        Returns:
            LatencyAnomaly if detected, None otherwise
        """
        key = (service, operation)
        with self._lock:
            baseline = self._baselines.get(key)

        if baseline is None or baseline.count < self.min_samples:
            return None

        if baseline.stddev == 0:
            return None

        # Calculate z-score (number of standard deviations from mean)
        deviation = (latency_ms - baseline.mean) / baseline.stddev

        if deviation >= self.deviation_threshold:
            return LatencyAnomaly(
                trace_id=trace_id,
                span_id=span_id,
                service=service,
                operation=operation,
                latency_ms=latency_ms,
                expected_latency_ms=baseline.mean,
                deviation_factor=deviation,
            )

        return None

    def get_baseline(
        self,
        service: str,
        operation: str,
    ) -> Optional[LatencyPercentiles]:
        """Get baseline statistics for an operation."""
        key = (service, operation)
        with self._lock:
            return self._baselines.get(key)


@dataclass
class AnalyticsConfig:
    """Trace analytics configuration."""
    namespace: str = "consciousness"
    histogram_buckets: List[float] = field(
        default_factory=lambda: [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
    )
    anomaly_detection_enabled: bool = True
    anomaly_deviation_threshold: float = 3.0
    anomaly_min_samples: int = 30


class TraceAnalytics:
    """Comprehensive trace analytics engine.

    Provides latency analysis, error rate tracking,
    service dependency mapping, and anomaly detection.

    Usage:
        storage = TraceStorage(config)
        analytics = TraceAnalytics(storage)

        # Compute latency histogram
        histogram = analytics.compute_latency_histogram(
            service="api-gateway",
            operation="POST /api/v1/process",
            from_time=datetime.now() - timedelta(hours=1),
        )

        # Get error rates
        error_rates = analytics.compute_error_rates(
            service="api-gateway",
        )

        # Build dependency graph
        graph = analytics.build_dependency_graph()

        # Detect anomalies
        anomalies = analytics.detect_anomalies(traces)
    """

    def __init__(
        self,
        storage: TraceStorage,
        config: Optional[AnalyticsConfig] = None,
    ):
        self.storage = storage
        self.config = config or AnalyticsConfig()
        self._lock = threading.Lock()

        # Anomaly detector
        if self.config.anomaly_detection_enabled:
            self._anomaly_detector = TraceAnomalyDetector(
                min_samples=self.config.anomaly_min_samples,
                deviation_threshold=self.config.anomaly_deviation_threshold,
            )
        else:
            self._anomaly_detector = None

        # Metrics
        ns = self.config.namespace
        self.analysis_duration = Histogram(
            f"{ns}_trace_analytics_duration_seconds",
            "Analytics operation duration",
            ["operation"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
        )
        self.anomalies_detected = Counter(
            f"{ns}_trace_analytics_anomalies_total",
            "Anomalies detected",
            ["severity"],
        )
        self.traces_analyzed = Counter(
            f"{ns}_trace_analytics_traces_analyzed_total",
            "Traces analyzed",
        )

    def compute_latency_histogram(
        self,
        service: Optional[str] = None,
        operation: Optional[str] = None,
        from_time: Optional[datetime] = None,
        to_time: Optional[datetime] = None,
        compute_percentiles: bool = True,
    ) -> List[LatencyHistogram]:
        """Compute latency histograms.

        Args:
            service: Filter by service
            operation: Filter by operation
            from_time: Start time
            to_time: End time
            compute_percentiles: Calculate percentiles

        Returns:
            List of latency histograms
        """
        start = time.time()

        query = TraceQuery(
            services=[service] if service else None,
            operations=[operation] if operation else None,
            from_time=from_time,
            to_time=to_time,
            limit=10000,
        )

        traces = self.storage.query(query)
        self.traces_analyzed.inc(len(traces))

        # Group by service/operation
        histograms: Dict[Tuple[str, str], LatencyHistogram] = {}
        latency_values: Dict[Tuple[str, str], List[float]] = defaultdict(list)

        for trace in traces:
            root = trace.root_span
            if not root:
                continue

            key = (root.service_name, root.operation_name)

            if key not in histograms:
                histograms[key] = LatencyHistogram(
                    service=root.service_name,
                    operation=root.operation_name,
                    time_window_start=from_time,
                    time_window_end=to_time,
                )

            histograms[key].add(trace.duration_ms)
            latency_values[key].append(trace.duration_ms)

        # Calculate percentiles
        if compute_percentiles:
            for key, histogram in histograms.items():
                values = latency_values[key]
                if values:
                    histogram.percentiles = LatencyPercentiles.from_values(values)

                    # Update anomaly detector baseline
                    if self._anomaly_detector:
                        self._anomaly_detector.update_baseline(
                            service=key[0],
                            operation=key[1],
                            latency_values=values,
                        )

        self.analysis_duration.labels(operation="latency_histogram").observe(
            time.time() - start
        )

        return list(histograms.values())

    def compute_error_rates(
        self,
        service: Optional[str] = None,
        operation: Optional[str] = None,
        from_time: Optional[datetime] = None,
        to_time: Optional[datetime] = None,
    ) -> List[ErrorRateMetrics]:
        """Compute error rates.

        Args:
            service: Filter by service
            operation: Filter by operation
            from_time: Start time
            to_time: End time

        Returns:
            List of error rate metrics
        """
        start = time.time()

        query = TraceQuery(
            services=[service] if service else None,
            operations=[operation] if operation else None,
            from_time=from_time,
            to_time=to_time,
            limit=10000,
        )

        traces = self.storage.query(query)

        # Group by service/operation
        metrics: Dict[Tuple[str, str], ErrorRateMetrics] = {}

        for trace in traces:
            for span in trace.spans:
                key = (span.service_name, span.operation_name)

                if key not in metrics:
                    metrics[key] = ErrorRateMetrics(
                        service=span.service_name,
                        operation=span.operation_name,
                        time_window_start=from_time,
                        time_window_end=to_time,
                    )

                m = metrics[key]
                m.total_count += 1

                if span.is_error:
                    m.error_count += 1
                    error_type = span.tags.get("error.type", span.status_message or "unknown")
                    m.error_by_type[error_type] = m.error_by_type.get(error_type, 0) + 1

        self.analysis_duration.labels(operation="error_rates").observe(
            time.time() - start
        )

        return list(metrics.values())

    def build_dependency_graph(
        self,
        from_time: Optional[datetime] = None,
        to_time: Optional[datetime] = None,
    ) -> DependencyGraph:
        """Build service dependency graph.

        Args:
            from_time: Start time
            to_time: End time

        Returns:
            DependencyGraph with nodes and edges
        """
        start = time.time()

        query = TraceQuery(
            from_time=from_time,
            to_time=to_time,
            limit=10000,
        )

        traces = self.storage.query(query)

        # Collect service stats and dependencies
        service_stats: Dict[str, ServiceStats] = {}
        dependencies: Dict[Tuple[str, str], ServiceDependency] = {}
        trace_services: Dict[str, Set[str]] = defaultdict(set)

        for trace in traces:
            span_map: Dict[str, StoredSpan] = {s.span_id: s for s in trace.spans}

            for span in trace.spans:
                service = span.service_name
                trace_services[trace.trace_id].add(service)

                # Update service stats
                if service not in service_stats:
                    service_stats[service] = ServiceStats(service=service)

                stats = service_stats[service]
                stats.span_count += 1
                stats.total_latency_ms += span.duration_ms
                stats.operations.add(span.operation_name)

                if span.is_error:
                    stats.error_count += 1

                # Track dependencies via parent relationships
                if span.parent_span_id:
                    parent = span_map.get(span.parent_span_id)
                    if parent and parent.service_name != span.service_name:
                        source = parent.service_name
                        target = span.service_name
                        dep_key = (source, target)

                        if dep_key not in dependencies:
                            dependencies[dep_key] = ServiceDependency(
                                source=source,
                                target=target,
                            )

                        dep = dependencies[dep_key]
                        dep.call_count += 1
                        dep.total_latency_ms += span.duration_ms

                        if span.is_error:
                            dep.error_count += 1

        # Count traces per service and build dependency lists
        for trace_id, services in trace_services.items():
            for service in services:
                if service in service_stats:
                    service_stats[service].trace_count += 1

        for (source, target), dep in dependencies.items():
            if source in service_stats:
                if target not in service_stats[source].outgoing_dependencies:
                    service_stats[source].outgoing_dependencies.append(target)
            if target in service_stats:
                if source not in service_stats[target].incoming_dependencies:
                    service_stats[target].incoming_dependencies.append(source)

        graph = DependencyGraph(
            nodes=list(service_stats.values()),
            edges=list(dependencies.values()),
        )

        self.analysis_duration.labels(operation="dependency_graph").observe(
            time.time() - start
        )

        return graph

    def compute_span_statistics(
        self,
        service: Optional[str] = None,
        operation: Optional[str] = None,
        from_time: Optional[datetime] = None,
        to_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Compute per-span statistics.

        Args:
            service: Filter by service
            operation: Filter by operation
            from_time: Start time
            to_time: End time

        Returns:
            List of span statistics dictionaries
        """
        query = TraceQuery(
            services=[service] if service else None,
            operations=[operation] if operation else None,
            from_time=from_time,
            to_time=to_time,
            limit=10000,
        )

        traces = self.storage.query(query)

        # Group spans by service/operation
        span_groups: Dict[Tuple[str, str], List[StoredSpan]] = defaultdict(list)

        for trace in traces:
            for span in trace.spans:
                key = (span.service_name, span.operation_name)
                span_groups[key].append(span)

        results = []
        for (service_name, operation_name), spans in span_groups.items():
            latencies = [s.duration_ms for s in spans]
            percentiles = LatencyPercentiles.from_values(latencies)

            error_count = sum(1 for s in spans if s.is_error)

            results.append({
                "service": service_name,
                "operation": operation_name,
                "call_count": len(spans),
                "error_count": error_count,
                "error_rate": error_count / len(spans) if spans else 0,
                "latency": percentiles.to_dict(),
            })

        return results

    def detect_anomalies(
        self,
        traces: Optional[List[StoredTrace]] = None,
        from_time: Optional[datetime] = None,
        to_time: Optional[datetime] = None,
    ) -> List[LatencyAnomaly]:
        """Detect latency anomalies in traces.

        Args:
            traces: Traces to analyze (or query from storage)
            from_time: Start time for query
            to_time: End time for query

        Returns:
            List of detected anomalies
        """
        if self._anomaly_detector is None:
            logger.warning("Anomaly detection disabled")
            return []

        start = time.time()

        if traces is None:
            query = TraceQuery(
                from_time=from_time,
                to_time=to_time,
                limit=10000,
            )
            traces = self.storage.query(query)

        anomalies = []

        for trace in traces:
            for span in trace.spans:
                anomaly = self._anomaly_detector.detect(
                    service=span.service_name,
                    operation=span.operation_name,
                    latency_ms=span.duration_ms,
                    trace_id=span.trace_id,
                    span_id=span.span_id,
                )

                if anomaly:
                    anomalies.append(anomaly)
                    self.anomalies_detected.labels(severity=anomaly.severity).inc()

        self.analysis_duration.labels(operation="anomaly_detection").observe(
            time.time() - start
        )

        return anomalies

    def get_service_overview(
        self,
        from_time: Optional[datetime] = None,
        to_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Get high-level service overview.

        Args:
            from_time: Start time
            to_time: End time

        Returns:
            Overview dictionary with key metrics
        """
        histograms = self.compute_latency_histogram(
            from_time=from_time,
            to_time=to_time,
        )

        error_rates = self.compute_error_rates(
            from_time=from_time,
            to_time=to_time,
        )

        graph = self.build_dependency_graph(
            from_time=from_time,
            to_time=to_time,
        )

        # Aggregate
        total_requests = sum(h.total_count for h in histograms)
        total_errors = sum(e.error_count for e in error_rates)

        latencies = []
        for h in histograms:
            if h.percentiles:
                latencies.append(h.percentiles.p95)

        return {
            "time_range": {
                "from": from_time.isoformat() if from_time else None,
                "to": to_time.isoformat() if to_time else None,
            },
            "summary": {
                "total_requests": total_requests,
                "total_errors": total_errors,
                "error_rate": total_errors / total_requests if total_requests else 0,
                "avg_p95_latency_ms": statistics.mean(latencies) if latencies else 0,
            },
            "services": {
                "count": len(graph.nodes),
                "names": [n.service for n in graph.nodes],
            },
            "dependencies": {
                "count": len(graph.edges),
            },
            "top_error_services": sorted(
                [{"service": e.service, "operation": e.operation, "error_rate": e.error_rate}
                 for e in error_rates],
                key=lambda x: x["error_rate"],
                reverse=True,
            )[:5],
            "slowest_operations": sorted(
                [{"service": h.service, "operation": h.operation, "p95_ms": h.percentiles.p95 if h.percentiles else 0}
                 for h in histograms if h.percentiles],
                key=lambda x: x["p95_ms"],
                reverse=True,
            )[:5],
        }
