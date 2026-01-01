"""Gateway Metrics

API gateway-level statistics and traffic analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import threading
import time
import logging

from prometheus_client import Gauge, Counter, Histogram, Summary

logger = logging.getLogger(__name__)


class RequestStatus(str, Enum):
    """Request status categories."""
    SUCCESS = "success"  # 2xx
    CLIENT_ERROR = "client_error"  # 4xx
    SERVER_ERROR = "server_error"  # 5xx
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"


class TrafficPattern(str, Enum):
    """Detected traffic patterns."""
    NORMAL = "normal"
    SPIKE = "spike"
    SUSTAINED_HIGH = "sustained_high"
    DECLINING = "declining"
    PERIODIC = "periodic"


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    request_id: str
    method: str
    path: str
    status_code: int
    duration_ms: float
    timestamp: datetime = field(default_factory=datetime.now)
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None
    api_key: Optional[str] = None
    request_size_bytes: int = 0
    response_size_bytes: int = 0
    upstream_duration_ms: float = 0
    cache_status: Optional[str] = None  # HIT, MISS, BYPASS

    @property
    def status_category(self) -> RequestStatus:
        """Categorize status code."""
        if 200 <= self.status_code < 300:
            return RequestStatus.SUCCESS
        elif self.status_code == 429:
            return RequestStatus.RATE_LIMITED
        elif self.status_code == 504:
            return RequestStatus.TIMEOUT
        elif 400 <= self.status_code < 500:
            return RequestStatus.CLIENT_ERROR
        else:
            return RequestStatus.SERVER_ERROR


@dataclass
class EndpointStats:
    """Statistics for an endpoint."""
    endpoint: str
    method: str
    request_count: int = 0
    success_count: int = 0
    error_count: int = 0
    avg_duration_ms: float = 0
    p50_duration_ms: float = 0
    p95_duration_ms: float = 0
    p99_duration_ms: float = 0
    max_duration_ms: float = 0
    min_duration_ms: float = float("inf")
    total_request_bytes: int = 0
    total_response_bytes: int = 0
    cache_hit_ratio: float = 0
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)

    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        if self.request_count == 0:
            return 0
        return self.error_count / self.request_count

    @property
    def avg_request_size(self) -> float:
        """Average request size."""
        if self.request_count == 0:
            return 0
        return self.total_request_bytes / self.request_count

    @property
    def avg_response_size(self) -> float:
        """Average response size."""
        if self.request_count == 0:
            return 0
        return self.total_response_bytes / self.request_count


@dataclass
class TrafficAnalysis:
    """Traffic pattern analysis."""
    timestamp: datetime = field(default_factory=datetime.now)
    current_rps: float = 0
    avg_rps_1m: float = 0
    avg_rps_5m: float = 0
    avg_rps_15m: float = 0
    peak_rps: float = 0
    pattern: TrafficPattern = TrafficPattern.NORMAL
    anomaly_score: float = 0  # 0-1, higher = more anomalous
    top_endpoints: List[Tuple[str, float]] = field(default_factory=list)
    top_clients: List[Tuple[str, int]] = field(default_factory=list)


@dataclass
class GatewayHealth:
    """Gateway health status."""
    healthy: bool = True
    uptime_seconds: float = 0
    total_requests: int = 0
    current_connections: int = 0
    max_connections: int = 0
    avg_latency_ms: float = 0
    error_rate: float = 0
    upstream_health: Dict[str, bool] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class GatewayMetrics:
    """Collects and exposes API gateway metrics.

    Usage:
        gateway = GatewayMetrics()

        # Record request
        gateway.record_request(RequestMetrics(
            request_id="abc123",
            method="GET",
            path="/api/users",
            status_code=200,
            duration_ms=45,
        ))

        # Get endpoint stats
        stats = gateway.get_endpoint_stats("/api/users", "GET")

        # Analyze traffic
        analysis = gateway.analyze_traffic()

        # Get health
        health = gateway.get_health()
    """

    def __init__(
        self,
        namespace: str = "consciousness",
        max_requests: int = 100000,
    ):
        self.namespace = namespace
        self.max_requests = max_requests
        self.start_time = datetime.now()

        self._requests: List[RequestMetrics] = []
        self._endpoint_stats: Dict[str, EndpointStats] = {}
        self._durations: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

        # Traffic tracking
        self._request_times: List[float] = []
        self._peak_rps = 0.0

        # Prometheus metrics
        self.request_total = Counter(
            f"{namespace}_gateway_requests_total",
            "Total requests",
            ["method", "endpoint", "status"],
        )

        self.request_duration = Histogram(
            f"{namespace}_gateway_request_duration_seconds",
            "Request duration",
            ["method", "endpoint"],
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        )

        self.request_size = Histogram(
            f"{namespace}_gateway_request_size_bytes",
            "Request body size",
            ["method", "endpoint"],
            buckets=[100, 1000, 10000, 100000, 1000000],
        )

        self.response_size = Histogram(
            f"{namespace}_gateway_response_size_bytes",
            "Response body size",
            ["method", "endpoint"],
            buckets=[100, 1000, 10000, 100000, 1000000],
        )

        self.upstream_duration = Histogram(
            f"{namespace}_gateway_upstream_duration_seconds",
            "Upstream request duration",
            ["endpoint", "upstream"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
        )

        self.current_requests = Gauge(
            f"{namespace}_gateway_current_requests",
            "Current in-flight requests",
            ["method"],
        )

        self.requests_per_second = Gauge(
            f"{namespace}_gateway_requests_per_second",
            "Current requests per second",
        )

        self.error_rate = Gauge(
            f"{namespace}_gateway_error_rate",
            "Error rate (5xx responses)",
            ["endpoint"],
        )

        self.cache_hits = Counter(
            f"{namespace}_gateway_cache_hits_total",
            "Cache hits",
            ["endpoint"],
        )

        self.cache_misses = Counter(
            f"{namespace}_gateway_cache_misses_total",
            "Cache misses",
            ["endpoint"],
        )

    def record_request(self, metrics: RequestMetrics):
        """Record a request.

        Args:
            metrics: Request metrics
        """
        now = time.time()

        with self._lock:
            # Store request
            self._requests.append(metrics)
            if len(self._requests) > self.max_requests:
                self._requests = self._requests[-self.max_requests // 2:]

            # Track request times for RPS calculation
            self._request_times.append(now)
            # Keep only last minute
            cutoff = now - 60
            self._request_times = [t for t in self._request_times if t > cutoff]

            # Update endpoint stats
            key = f"{metrics.method}:{metrics.path}"
            if key not in self._endpoint_stats:
                self._endpoint_stats[key] = EndpointStats(
                    endpoint=metrics.path,
                    method=metrics.method,
                )

            stats = self._endpoint_stats[key]
            stats.request_count += 1
            stats.last_seen = datetime.now()

            if metrics.status_category == RequestStatus.SUCCESS:
                stats.success_count += 1
            else:
                stats.error_count += 1

            # Update duration stats
            if key not in self._durations:
                self._durations[key] = []
            self._durations[key].append(metrics.duration_ms)
            if len(self._durations[key]) > 1000:
                self._durations[key] = self._durations[key][-500:]

            durations = self._durations[key]
            stats.avg_duration_ms = sum(durations) / len(durations)
            stats.max_duration_ms = max(stats.max_duration_ms, metrics.duration_ms)
            stats.min_duration_ms = min(stats.min_duration_ms, metrics.duration_ms)

            sorted_durations = sorted(durations)
            stats.p50_duration_ms = sorted_durations[len(sorted_durations) // 2]
            stats.p95_duration_ms = sorted_durations[int(len(sorted_durations) * 0.95)]
            stats.p99_duration_ms = sorted_durations[int(len(sorted_durations) * 0.99)]

            # Size tracking
            stats.total_request_bytes += metrics.request_size_bytes
            stats.total_response_bytes += metrics.response_size_bytes

            # Cache tracking
            if metrics.cache_status == "HIT":
                cache_hits = sum(1 for r in self._requests[-100:]
                               if r.path == metrics.path and r.cache_status == "HIT")
                cache_total = sum(1 for r in self._requests[-100:]
                                if r.path == metrics.path and r.cache_status)
                stats.cache_hit_ratio = cache_hits / cache_total if cache_total > 0 else 0

        # Update Prometheus metrics
        endpoint = self._normalize_endpoint(metrics.path)

        self.request_total.labels(
            method=metrics.method,
            endpoint=endpoint,
            status=str(metrics.status_code),
        ).inc()

        self.request_duration.labels(
            method=metrics.method,
            endpoint=endpoint,
        ).observe(metrics.duration_ms / 1000)

        self.request_size.labels(
            method=metrics.method,
            endpoint=endpoint,
        ).observe(metrics.request_size_bytes)

        self.response_size.labels(
            method=metrics.method,
            endpoint=endpoint,
        ).observe(metrics.response_size_bytes)

        if metrics.cache_status == "HIT":
            self.cache_hits.labels(endpoint=endpoint).inc()
        elif metrics.cache_status == "MISS":
            self.cache_misses.labels(endpoint=endpoint).inc()

        # Update RPS
        rps = len(self._request_times)
        self.requests_per_second.set(rps)
        self._peak_rps = max(self._peak_rps, rps)

        # Update error rate
        recent_requests = self._requests[-100:]
        endpoint_requests = [r for r in recent_requests if r.path == metrics.path]
        if endpoint_requests:
            errors = sum(1 for r in endpoint_requests
                        if r.status_category in [RequestStatus.SERVER_ERROR, RequestStatus.TIMEOUT])
            self.error_rate.labels(endpoint=endpoint).set(errors / len(endpoint_requests))

    def _normalize_endpoint(self, path: str) -> str:
        """Normalize endpoint path to reduce cardinality."""
        import re
        # Replace UUIDs
        normalized = re.sub(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            "{id}",
            path,
            flags=re.IGNORECASE,
        )
        # Replace numeric IDs
        normalized = re.sub(r"/\d+", "/{id}", normalized)
        return normalized

    def get_endpoint_stats(
        self,
        endpoint: str,
        method: str,
    ) -> Optional[EndpointStats]:
        """Get statistics for an endpoint.

        Args:
            endpoint: Endpoint path
            method: HTTP method

        Returns:
            EndpointStats or None
        """
        with self._lock:
            key = f"{method}:{endpoint}"
            return self._endpoint_stats.get(key)

    def get_all_endpoints(self) -> List[EndpointStats]:
        """Get all endpoint statistics.

        Returns:
            List of EndpointStats
        """
        with self._lock:
            return list(self._endpoint_stats.values())

    def analyze_traffic(self) -> TrafficAnalysis:
        """Analyze current traffic patterns.

        Returns:
            TrafficAnalysis
        """
        now = time.time()

        with self._lock:
            request_times = list(self._request_times)
            recent_requests = list(self._requests[-1000:])

        # Calculate RPS over different windows
        rps_1m = len([t for t in request_times if t > now - 60])
        rps_5m = len([t for t in request_times if t > now - 300]) / 5
        rps_15m = len([t for t in request_times if t > now - 900]) / 15

        # Detect pattern
        pattern = self._detect_pattern(rps_1m, rps_5m, rps_15m)

        # Calculate anomaly score
        anomaly_score = self._calculate_anomaly_score(rps_1m, rps_5m, rps_15m)

        # Top endpoints
        endpoint_counts: Dict[str, int] = {}
        for r in recent_requests:
            endpoint_counts[r.path] = endpoint_counts.get(r.path, 0) + 1
        top_endpoints = sorted(endpoint_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        top_endpoints_rps = [(e, c / 60) for e, c in top_endpoints]

        # Top clients
        client_counts: Dict[str, int] = {}
        for r in recent_requests:
            if r.client_ip:
                client_counts[r.client_ip] = client_counts.get(r.client_ip, 0) + 1
        top_clients = sorted(client_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        return TrafficAnalysis(
            current_rps=rps_1m / 60,
            avg_rps_1m=rps_1m / 60,
            avg_rps_5m=rps_5m / 60,
            avg_rps_15m=rps_15m / 60,
            peak_rps=self._peak_rps / 60,
            pattern=pattern,
            anomaly_score=anomaly_score,
            top_endpoints=top_endpoints_rps,
            top_clients=top_clients,
        )

    def _detect_pattern(
        self,
        rps_1m: float,
        rps_5m: float,
        rps_15m: float,
    ) -> TrafficPattern:
        """Detect traffic pattern."""
        if rps_1m > rps_5m * 2:
            return TrafficPattern.SPIKE
        elif rps_1m > rps_15m * 1.5:
            return TrafficPattern.SUSTAINED_HIGH
        elif rps_1m < rps_5m * 0.5:
            return TrafficPattern.DECLINING
        return TrafficPattern.NORMAL

    def _calculate_anomaly_score(
        self,
        rps_1m: float,
        rps_5m: float,
        rps_15m: float,
    ) -> float:
        """Calculate anomaly score (0-1)."""
        if rps_15m == 0:
            return 0

        # Deviation from 15-minute average
        deviation = abs(rps_1m / 60 - rps_15m / 60) / (rps_15m / 60)
        return min(1.0, deviation)

    def get_health(self) -> GatewayHealth:
        """Get gateway health status.

        Returns:
            GatewayHealth
        """
        uptime = (datetime.now() - self.start_time).total_seconds()

        with self._lock:
            total_requests = sum(s.request_count for s in self._endpoint_stats.values())
            total_errors = sum(s.error_count for s in self._endpoint_stats.values())
            all_durations = []
            for durations in self._durations.values():
                all_durations.extend(durations)

        error_rate = total_errors / total_requests if total_requests > 0 else 0
        avg_latency = sum(all_durations) / len(all_durations) if all_durations else 0

        issues = []
        healthy = True

        if error_rate > 0.05:
            issues.append(f"High error rate: {error_rate:.1%}")
            healthy = False

        if avg_latency > 1000:
            issues.append(f"High average latency: {avg_latency:.0f}ms")
            healthy = error_rate <= 0.1  # Still healthy if just slow

        return GatewayHealth(
            healthy=healthy,
            uptime_seconds=uptime,
            total_requests=total_requests,
            avg_latency_ms=avg_latency,
            error_rate=error_rate,
            issues=issues,
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get gateway summary.

        Returns:
            Summary dictionary
        """
        health = self.get_health()
        traffic = self.analyze_traffic()

        with self._lock:
            endpoints = list(self._endpoint_stats.values())

        # Find slowest endpoints
        slowest = sorted(endpoints, key=lambda e: e.avg_duration_ms, reverse=True)[:5]

        # Find highest error rate endpoints
        error_prone = sorted(
            [e for e in endpoints if e.request_count > 10],
            key=lambda e: e.error_rate,
            reverse=True
        )[:5]

        return {
            "healthy": health.healthy,
            "uptime_seconds": health.uptime_seconds,
            "total_requests": health.total_requests,
            "current_rps": traffic.current_rps,
            "peak_rps": traffic.peak_rps,
            "avg_latency_ms": health.avg_latency_ms,
            "error_rate": health.error_rate,
            "traffic_pattern": traffic.pattern.value,
            "endpoint_count": len(endpoints),
            "slowest_endpoints": [
                {"endpoint": e.endpoint, "avg_ms": e.avg_duration_ms}
                for e in slowest
            ],
            "error_prone_endpoints": [
                {"endpoint": e.endpoint, "error_rate": e.error_rate}
                for e in error_prone
            ],
            "issues": health.issues,
        }
