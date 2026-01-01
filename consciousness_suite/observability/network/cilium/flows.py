"""Network Flow Analysis

Analysis and classification of network flows from Hubble.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict
import threading
import logging
import statistics

from prometheus_client import Counter, Gauge, Histogram

from .hubble_client import HubbleClient, HubbleFlow, FlowFilter

logger = logging.getLogger(__name__)


class FlowDirection(str, Enum):
    """Network flow direction."""
    INGRESS = "INGRESS"
    EGRESS = "EGRESS"
    UNKNOWN = "UNKNOWN"


class FlowVerdict(str, Enum):
    """Flow verdict from network policies."""
    FORWARDED = "FORWARDED"
    DROPPED = "DROPPED"
    REDIRECTED = "REDIRECTED"
    ERROR = "ERROR"
    AUDIT = "AUDIT"
    UNKNOWN = "UNKNOWN"


class FlowType(str, Enum):
    """Classification of flow type."""
    INTER_SERVICE = "inter_service"
    EXTERNAL_INGRESS = "external_ingress"
    EXTERNAL_EGRESS = "external_egress"
    INTRA_NAMESPACE = "intra_namespace"
    CROSS_NAMESPACE = "cross_namespace"
    DNS = "dns"
    HEALTH_CHECK = "health_check"
    METRICS = "metrics"
    UNKNOWN = "unknown"


class L7Protocol(str, Enum):
    """Layer 7 protocol types."""
    HTTP = "HTTP"
    HTTPS = "HTTPS"
    GRPC = "gRPC"
    DNS = "DNS"
    KAFKA = "Kafka"
    MYSQL = "MySQL"
    REDIS = "Redis"
    POSTGRES = "PostgreSQL"
    MONGODB = "MongoDB"
    UNKNOWN = "UNKNOWN"


@dataclass
class FlowRecord:
    """Enriched flow record with analysis data.

    Attributes:
        flow: Original HubbleFlow
        flow_type: Classified flow type
        service_pair: Source and destination service names
        latency_ms: Observed latency if available
        request_size_bytes: Request size if available
        response_size_bytes: Response size if available
        is_error: Whether this represents an error
        error_category: Category of error if applicable
        slo_relevant: Whether this flow is SLO-relevant
        tags: Additional classification tags
    """
    flow: HubbleFlow
    flow_type: FlowType = FlowType.UNKNOWN
    service_pair: Tuple[str, str] = ("unknown", "unknown")
    latency_ms: Optional[float] = None
    request_size_bytes: Optional[int] = None
    response_size_bytes: Optional[int] = None
    is_error: bool = False
    error_category: Optional[str] = None
    slo_relevant: bool = True
    tags: List[str] = field(default_factory=list)

    @property
    def source_service(self) -> str:
        return self.service_pair[0]

    @property
    def destination_service(self) -> str:
        return self.service_pair[1]


@dataclass
class FlowAggregation:
    """Aggregated flow statistics.

    Attributes:
        source_service: Source service name
        destination_service: Destination service name
        protocol: Network protocol
        l7_protocol: L7 protocol if applicable
        flow_count: Total number of flows
        bytes_sent: Total bytes sent
        bytes_received: Total bytes received
        error_count: Number of errors
        dropped_count: Number of dropped flows
        avg_latency_ms: Average latency
        p50_latency_ms: P50 latency
        p95_latency_ms: P95 latency
        p99_latency_ms: P99 latency
        first_seen: First flow timestamp
        last_seen: Last flow timestamp
    """
    source_service: str
    destination_service: str
    protocol: str = "TCP"
    l7_protocol: Optional[str] = None
    flow_count: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    error_count: int = 0
    dropped_count: int = 0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None


@dataclass
class FlowStats:
    """Overall flow statistics.

    Attributes:
        total_flows: Total flows analyzed
        forwarded_flows: Flows forwarded
        dropped_flows: Flows dropped
        error_flows: Flows with errors
        flows_per_second: Current flow rate
        unique_services: Number of unique services
        unique_connections: Number of unique service pairs
        top_talkers: Top bandwidth consumers
        top_destinations: Most accessed destinations
        protocol_breakdown: Breakdown by protocol
        namespace_breakdown: Breakdown by namespace
    """
    total_flows: int = 0
    forwarded_flows: int = 0
    dropped_flows: int = 0
    error_flows: int = 0
    flows_per_second: float = 0.0
    unique_services: int = 0
    unique_connections: int = 0
    top_talkers: List[Tuple[str, int]] = field(default_factory=list)
    top_destinations: List[Tuple[str, int]] = field(default_factory=list)
    protocol_breakdown: Dict[str, int] = field(default_factory=dict)
    namespace_breakdown: Dict[str, int] = field(default_factory=dict)


class FlowAnalyzer:
    """Analyzes and classifies network flows.

    Provides flow classification, aggregation, and analysis capabilities.

    Usage:
        client = HubbleClient(config)
        analyzer = FlowAnalyzer(client)

        # Analyze flows
        flows = analyzer.get_flows(limit=1000)
        for record in flows:
            if record.is_error:
                handle_error(record)

        # Get aggregations
        aggs = analyzer.get_aggregations(window_minutes=5)
        for agg in aggs:
            print(f"{agg.source_service} -> {agg.destination_service}: {agg.flow_count}")

        # Get statistics
        stats = analyzer.get_stats()
        print(f"Flow rate: {stats.flows_per_second}/s")
    """

    def __init__(
        self,
        client: Optional[HubbleClient] = None,
        namespace: str = "consciousness",
    ):
        self.client = client
        self.namespace = namespace
        self._lock = threading.Lock()

        # Flow storage
        self._flows: List[FlowRecord] = []
        self._aggregations: Dict[Tuple[str, str], FlowAggregation] = {}
        self._latencies: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        self._max_flows = 100000

        # Classification rules
        self._health_check_ports = {8080, 8081, 9090, 10254}
        self._metrics_paths = {"/metrics", "/healthz", "/ready", "/live"}
        self._dns_port = 53

        # Prometheus metrics
        self.flows_analyzed = Counter(
            f"{namespace}_flow_analyzer_flows_total",
            "Total flows analyzed",
            ["flow_type", "verdict"],
        )

        self.flow_latency = Histogram(
            f"{namespace}_flow_analyzer_latency_seconds",
            "Flow latency distribution",
            ["source_service", "destination_service"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
        )

        self.flow_errors = Counter(
            f"{namespace}_flow_analyzer_errors_total",
            "Flow errors by category",
            ["error_category", "source_service", "destination_service"],
        )

        self.active_connections = Gauge(
            f"{namespace}_flow_analyzer_active_connections",
            "Active service connections",
        )

    def analyze_flow(self, flow: HubbleFlow) -> FlowRecord:
        """Analyze and classify a single flow.

        Args:
            flow: HubbleFlow to analyze

        Returns:
            FlowRecord with classification
        """
        start_time = time.perf_counter()

        # Determine flow type
        flow_type = self._classify_flow(flow)

        # Extract service names
        source_service = self._extract_service_name(
            flow.source_pod, flow.source_namespace, flow.source_labels
        )
        destination_service = self._extract_service_name(
            flow.destination_pod, flow.destination_namespace, flow.destination_labels
        )

        # Check for errors
        is_error = False
        error_category = None

        if flow.verdict == "DROPPED":
            is_error = True
            error_category = flow.drop_reason or "policy_drop"
        elif flow.http_info:
            status_code = flow.http_info.get("code", 0)
            if status_code >= 400:
                is_error = True
                error_category = f"http_{status_code // 100}xx"

        # Extract latency if available
        latency_ms = None
        if flow.http_info and "latency_ms" in flow.raw_data:
            latency_ms = flow.raw_data["latency_ms"]

        # Determine SLO relevance
        slo_relevant = flow_type not in {
            FlowType.HEALTH_CHECK,
            FlowType.METRICS,
            FlowType.DNS,
        }

        # Build tags
        tags = self._build_tags(flow, flow_type)

        record = FlowRecord(
            flow=flow,
            flow_type=flow_type,
            service_pair=(source_service, destination_service),
            latency_ms=latency_ms,
            is_error=is_error,
            error_category=error_category,
            slo_relevant=slo_relevant,
            tags=tags,
        )

        # Update metrics
        self.flows_analyzed.labels(
            flow_type=flow_type.value,
            verdict=flow.verdict,
        ).inc()

        if latency_ms is not None:
            self.flow_latency.labels(
                source_service=source_service[:50],
                destination_service=destination_service[:50],
            ).observe(latency_ms / 1000)

        if is_error and error_category:
            self.flow_errors.labels(
                error_category=error_category,
                source_service=source_service[:50],
                destination_service=destination_service[:50],
            ).inc()

        return record

    def _classify_flow(self, flow: HubbleFlow) -> FlowType:
        """Classify flow into a type."""
        # DNS flow
        if flow.destination_port == self._dns_port or flow.dns_info:
            return FlowType.DNS

        # Health check
        if flow.destination_port in self._health_check_ports:
            if flow.http_info:
                path = flow.http_info.get("url", "")
                if any(p in path for p in self._metrics_paths):
                    return FlowType.HEALTH_CHECK
            return FlowType.HEALTH_CHECK

        # Metrics endpoint
        if flow.http_info:
            path = flow.http_info.get("url", "")
            if "/metrics" in path:
                return FlowType.METRICS

        # External traffic
        if not flow.source_pod:
            return FlowType.EXTERNAL_INGRESS
        if not flow.destination_pod:
            return FlowType.EXTERNAL_EGRESS

        # Namespace-based classification
        if flow.source_namespace and flow.destination_namespace:
            if flow.source_namespace == flow.destination_namespace:
                return FlowType.INTRA_NAMESPACE
            else:
                return FlowType.CROSS_NAMESPACE

        return FlowType.INTER_SERVICE

    def _extract_service_name(
        self,
        pod_name: Optional[str],
        namespace: Optional[str],
        labels: Dict[str, str],
    ) -> str:
        """Extract service name from pod information."""
        # Try app label first
        if "app" in labels:
            service = labels["app"]
        elif "app.kubernetes.io/name" in labels:
            service = labels["app.kubernetes.io/name"]
        elif pod_name:
            # Strip deployment/replicaset suffix
            parts = pod_name.rsplit("-", 2)
            service = parts[0] if parts else pod_name
        else:
            service = "unknown"

        if namespace:
            return f"{namespace}/{service}"
        return service

    def _build_tags(self, flow: HubbleFlow, flow_type: FlowType) -> List[str]:
        """Build classification tags for flow."""
        tags = [flow_type.value]

        if flow.protocol:
            tags.append(f"protocol:{flow.protocol}")

        if flow.l7_protocol:
            tags.append(f"l7:{flow.l7_protocol}")

        if flow.http_info:
            method = flow.http_info.get("method", "")
            if method:
                tags.append(f"http_method:{method}")
            status = flow.http_info.get("code", 0)
            if status:
                tags.append(f"http_status:{status}")

        if flow.is_reply:
            tags.append("reply")

        if flow.traffic_direction == "EGRESS":
            tags.append("egress")
        else:
            tags.append("ingress")

        return tags

    def get_flows(
        self,
        filter: Optional[FlowFilter] = None,
        limit: int = 1000,
        since: Optional[timedelta] = None,
    ) -> List[FlowRecord]:
        """Get analyzed flows.

        Args:
            filter: Optional filter for flows
            limit: Maximum flows to return
            since: Get flows since this time ago

        Returns:
            List of analyzed FlowRecords
        """
        if not self.client:
            return []

        raw_flows = self.client.get_flows(filter=filter, limit=limit, since=since)
        records = []

        for flow in raw_flows:
            record = self.analyze_flow(flow)
            records.append(record)

            # Update aggregations
            self._update_aggregation(record)

        # Store flows
        with self._lock:
            self._flows.extend(records)
            if len(self._flows) > self._max_flows:
                self._flows = self._flows[-self._max_flows // 2:]

        return records

    def _update_aggregation(self, record: FlowRecord):
        """Update flow aggregation with new record."""
        key = record.service_pair

        with self._lock:
            if key not in self._aggregations:
                self._aggregations[key] = FlowAggregation(
                    source_service=record.source_service,
                    destination_service=record.destination_service,
                    protocol=record.flow.protocol,
                    l7_protocol=record.flow.l7_protocol,
                    first_seen=record.flow.time,
                )

            agg = self._aggregations[key]
            agg.flow_count += 1
            agg.last_seen = record.flow.time

            if record.is_error:
                agg.error_count += 1
            if record.flow.verdict == "DROPPED":
                agg.dropped_count += 1

            # Track latencies
            if record.latency_ms is not None:
                self._latencies[key].append(record.latency_ms)
                # Keep only last 1000 latencies
                if len(self._latencies[key]) > 1000:
                    self._latencies[key] = self._latencies[key][-500:]

                latencies = self._latencies[key]
                agg.avg_latency_ms = statistics.mean(latencies)
                if len(latencies) >= 2:
                    sorted_latencies = sorted(latencies)
                    n = len(sorted_latencies)
                    agg.p50_latency_ms = sorted_latencies[n // 2]
                    agg.p95_latency_ms = sorted_latencies[int(n * 0.95)]
                    agg.p99_latency_ms = sorted_latencies[int(n * 0.99)]

    def get_aggregations(
        self,
        window_minutes: int = 5,
        min_flow_count: int = 1,
    ) -> List[FlowAggregation]:
        """Get flow aggregations.

        Args:
            window_minutes: Time window for aggregation
            min_flow_count: Minimum flows to include

        Returns:
            List of FlowAggregations
        """
        cutoff = datetime.now() - timedelta(minutes=window_minutes)

        with self._lock:
            aggs = [
                agg for agg in self._aggregations.values()
                if agg.flow_count >= min_flow_count
                and agg.last_seen and agg.last_seen >= cutoff
            ]

        return sorted(aggs, key=lambda a: a.flow_count, reverse=True)

    def get_stats(self) -> FlowStats:
        """Get overall flow statistics.

        Returns:
            FlowStats with current statistics
        """
        with self._lock:
            flows = list(self._flows)
            aggregations = dict(self._aggregations)

        if not flows:
            return FlowStats()

        # Calculate time window
        now = datetime.now()
        one_minute_ago = now - timedelta(minutes=1)
        recent_flows = [f for f in flows if f.flow.time >= one_minute_ago]

        # Protocol breakdown
        protocol_counts: Dict[str, int] = defaultdict(int)
        namespace_counts: Dict[str, int] = defaultdict(int)
        talker_counts: Dict[str, int] = defaultdict(int)
        destination_counts: Dict[str, int] = defaultdict(int)

        total_flows = len(flows)
        forwarded = sum(1 for f in flows if f.flow.verdict == "FORWARDED")
        dropped = sum(1 for f in flows if f.flow.verdict == "DROPPED")
        errors = sum(1 for f in flows if f.is_error)

        for record in flows:
            protocol_counts[record.flow.protocol] += 1
            if record.flow.source_namespace:
                namespace_counts[record.flow.source_namespace] += 1
            talker_counts[record.source_service] += 1
            destination_counts[record.destination_service] += 1

        # Get unique counts
        services = set()
        for agg in aggregations.values():
            services.add(agg.source_service)
            services.add(agg.destination_service)

        return FlowStats(
            total_flows=total_flows,
            forwarded_flows=forwarded,
            dropped_flows=dropped,
            error_flows=errors,
            flows_per_second=len(recent_flows) / 60.0 if recent_flows else 0.0,
            unique_services=len(services),
            unique_connections=len(aggregations),
            top_talkers=sorted(talker_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            top_destinations=sorted(destination_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            protocol_breakdown=dict(protocol_counts),
            namespace_breakdown=dict(namespace_counts),
        )

    def get_service_dependencies(self) -> Dict[str, List[str]]:
        """Get service dependency map.

        Returns:
            Dict mapping services to their dependencies
        """
        dependencies: Dict[str, set] = defaultdict(set)

        with self._lock:
            for (src, dst), agg in self._aggregations.items():
                if agg.flow_count > 0:
                    dependencies[src].add(dst)

        return {k: sorted(v) for k, v in dependencies.items()}

    def get_error_summary(
        self,
        window_minutes: int = 60,
    ) -> Dict[str, Any]:
        """Get summary of flow errors.

        Args:
            window_minutes: Time window for analysis

        Returns:
            Error summary dict
        """
        cutoff = datetime.now() - timedelta(minutes=window_minutes)

        with self._lock:
            recent_flows = [
                f for f in self._flows
                if f.flow.time >= cutoff
            ]

        error_flows = [f for f in recent_flows if f.is_error]

        # Group by category
        by_category: Dict[str, int] = defaultdict(int)
        by_service_pair: Dict[Tuple[str, str], int] = defaultdict(int)

        for f in error_flows:
            if f.error_category:
                by_category[f.error_category] += 1
            by_service_pair[f.service_pair] += 1

        return {
            "total_errors": len(error_flows),
            "total_flows": len(recent_flows),
            "error_rate": len(error_flows) / len(recent_flows) if recent_flows else 0.0,
            "by_category": dict(by_category),
            "top_error_pairs": sorted(
                by_service_pair.items(), key=lambda x: x[1], reverse=True
            )[:10],
            "window_minutes": window_minutes,
        }

    def clear_history(self):
        """Clear stored flow history."""
        with self._lock:
            self._flows.clear()
            self._aggregations.clear()
            self._latencies.clear()
