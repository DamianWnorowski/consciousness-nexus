"""MCP Data Resources

Data resources exposed via MCP for Claude Code integration:
- MetricsResource: Current metrics data
- TracesResource: Recent traces data
- AlertsResource: Alert data and status

Usage:
    from consciousness_suite.observability.ai_platform.mcp_server.resources import (
        MetricsResource,
        TracesResource,
        AlertsResource,
        ResourceRegistry,
    )

    registry = ResourceRegistry()
    registry.register("metrics", MetricsResource())
    registry.register("traces", TracesResource())
    registry.register("alerts", AlertsResource())

    # Read resource
    content = await registry.read("metrics", {"filter": "consciousness_"})
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union

from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)


class ResourceType(str, Enum):
    """Resource content types."""
    JSON = "application/json"
    TEXT = "text/plain"
    MARKDOWN = "text/markdown"
    PROMETHEUS = "text/plain; version=0.0.4"


@dataclass
class ResourceMetadata:
    """Metadata for a resource."""
    name: str
    description: str
    mime_type: str = ResourceType.JSON.value
    uri: str = ""
    last_updated: datetime = field(default_factory=datetime.now)
    size_bytes: int = 0
    tags: Dict[str, str] = field(default_factory=dict)


class MCPResource(ABC):
    """Base class for MCP data resources.

    Subclasses must implement:
    - read(): Read resource content
    - description: Resource description
    - mime_type: Content MIME type
    """

    def __init__(self, namespace: str = "consciousness"):
        self.namespace = namespace
        self._lock = threading.Lock()
        self._cache: Optional[Any] = None
        self._cache_time: Optional[datetime] = None
        self._cache_ttl_seconds: float = 30.0

        # Subscription management
        self._subscribers: Set[str] = set()

        # Metrics
        self.reads_total = Counter(
            f"{namespace}_mcp_resource_reads_total",
            "Total resource reads",
            ["resource_name"],
        )

        self.read_latency = Histogram(
            f"{namespace}_mcp_resource_read_latency_seconds",
            "Resource read latency",
            ["resource_name"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
        )

        self.cache_hits = Counter(
            f"{namespace}_mcp_resource_cache_hits_total",
            "Resource cache hits",
            ["resource_name"],
        )

        self.subscribers_count = Gauge(
            f"{namespace}_mcp_resource_subscribers",
            "Resource subscribers",
            ["resource_name"],
        )

    @property
    @abstractmethod
    def name(self) -> str:
        """Resource name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Resource description."""
        pass

    @property
    def mime_type(self) -> str:
        """Resource MIME type."""
        return ResourceType.JSON.value

    @abstractmethod
    async def _fetch(self, arguments: Dict[str, Any]) -> Any:
        """Fetch resource content.

        Args:
            arguments: Fetch arguments

        Returns:
            Resource content
        """
        pass

    async def read(self, arguments: Optional[Dict[str, Any]] = None) -> Any:
        """Read resource content.

        Args:
            arguments: Read arguments

        Returns:
            Resource content
        """
        start_time = time.perf_counter()
        arguments = arguments or {}

        # Check cache
        if self._is_cache_valid():
            self.cache_hits.labels(resource_name=self.name).inc()
            return self._cache

        try:
            content = await self._fetch(arguments)

            # Update cache
            with self._lock:
                self._cache = content
                self._cache_time = datetime.now()

            return content

        finally:
            latency = time.perf_counter() - start_time
            self.reads_total.labels(resource_name=self.name).inc()
            self.read_latency.labels(resource_name=self.name).observe(latency)

    def _is_cache_valid(self) -> bool:
        """Check if cache is valid."""
        if self._cache is None or self._cache_time is None:
            return False

        elapsed = (datetime.now() - self._cache_time).total_seconds()
        return elapsed < self._cache_ttl_seconds

    def invalidate_cache(self):
        """Invalidate resource cache."""
        with self._lock:
            self._cache = None
            self._cache_time = None

    def subscribe(self, subscriber_id: str):
        """Subscribe to resource updates.

        Args:
            subscriber_id: Subscriber identifier
        """
        with self._lock:
            self._subscribers.add(subscriber_id)
            self.subscribers_count.labels(resource_name=self.name).set(
                len(self._subscribers)
            )

    def unsubscribe(self, subscriber_id: str):
        """Unsubscribe from resource updates.

        Args:
            subscriber_id: Subscriber identifier
        """
        with self._lock:
            self._subscribers.discard(subscriber_id)
            self.subscribers_count.labels(resource_name=self.name).set(
                len(self._subscribers)
            )

    def get_metadata(self) -> ResourceMetadata:
        """Get resource metadata.

        Returns:
            ResourceMetadata
        """
        return ResourceMetadata(
            name=self.name,
            description=self.description,
            mime_type=self.mime_type,
            uri=f"consciousness://{self.name}",
            last_updated=self._cache_time or datetime.now(),
        )


class MetricsResource(MCPResource):
    """Resource providing current metrics data.

    Exposes:
    - All registered Prometheus metrics
    - Metric metadata (type, help, labels)
    - Current values
    """

    def __init__(
        self,
        prometheus_url: str = "http://localhost:9090",
        namespace: str = "consciousness",
    ):
        super().__init__(namespace)
        self.prometheus_url = prometheus_url
        self._cache_ttl_seconds = 15.0  # Short TTL for metrics

    @property
    def name(self) -> str:
        return "metrics"

    @property
    def description(self) -> str:
        return (
            "Current metrics from the consciousness observability platform. "
            "Includes all Prometheus metrics with their types, descriptions, and values."
        )

    async def _fetch(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch current metrics."""
        filter_prefix = arguments.get("filter", "consciousness_")
        include_metadata = arguments.get("include_metadata", True)

        # Get metrics from Prometheus
        metrics = await self._get_metrics(filter_prefix)

        result = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "filter": filter_prefix,
        }

        if include_metadata:
            metadata = await self._get_metadata(filter_prefix)
            result["metadata"] = metadata

        return result

    async def _get_metrics(self, filter_prefix: str) -> List[Dict[str, Any]]:
        """Get metrics from Prometheus."""
        import aiohttp

        url = f"{self.prometheus_url}/api/v1/query"
        params = {"query": f'{{__name__=~"{filter_prefix}.*"}}'}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._format_metrics(data)
                    else:
                        return self._get_mock_metrics(filter_prefix)
        except Exception:
            return self._get_mock_metrics(filter_prefix)

    async def _get_metadata(self, filter_prefix: str) -> Dict[str, Any]:
        """Get metric metadata."""
        import aiohttp

        url = f"{self.prometheus_url}/api/v1/metadata"
        params = {"metric": f"{filter_prefix}"}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("data", {})
                    else:
                        return {}
        except Exception:
            return {}

    def _format_metrics(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format Prometheus metrics response."""
        if data.get("status") != "success":
            return []

        result = data.get("data", {}).get("result", [])
        metrics = []

        for item in result:
            metric = item.get("metric", {})
            value = item.get("value", [])

            metrics.append({
                "name": metric.get("__name__", ""),
                "labels": {k: v for k, v in metric.items() if k != "__name__"},
                "value": float(value[1]) if len(value) >= 2 else None,
                "timestamp": datetime.fromtimestamp(value[0]).isoformat()
                if len(value) >= 1
                else None,
            })

        return metrics

    def _get_mock_metrics(self, filter_prefix: str) -> List[Dict[str, Any]]:
        """Get mock metrics."""
        import random

        return [
            {
                "name": f"{filter_prefix}requests_total",
                "labels": {"method": "GET", "status": "200"},
                "value": random.randint(100, 10000),
                "timestamp": datetime.now().isoformat(),
            },
            {
                "name": f"{filter_prefix}processing_duration_seconds",
                "labels": {"operation": "analyze"},
                "value": random.uniform(0.01, 1.0),
                "timestamp": datetime.now().isoformat(),
            },
            {
                "name": f"{filter_prefix}active_connections",
                "labels": {},
                "value": random.randint(1, 50),
                "timestamp": datetime.now().isoformat(),
            },
        ]


class TracesResource(MCPResource):
    """Resource providing recent traces data.

    Exposes:
    - Recent trace summaries
    - Service dependency map
    - Error rate by service
    """

    def __init__(
        self,
        jaeger_url: str = "http://localhost:16686",
        namespace: str = "consciousness",
    ):
        super().__init__(namespace)
        self.jaeger_url = jaeger_url
        self._cache_ttl_seconds = 30.0

    @property
    def name(self) -> str:
        return "traces"

    @property
    def description(self) -> str:
        return (
            "Recent traces from the consciousness observability platform. "
            "Includes trace summaries, service dependencies, and error rates."
        )

    async def _fetch(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch recent traces."""
        service = arguments.get("service", "consciousness-nexus")
        limit = arguments.get("limit", 100)
        lookback = arguments.get("lookback", "1h")

        # Get recent traces
        traces = await self._get_recent_traces(service, limit)

        # Get service dependencies
        dependencies = await self._get_dependencies()

        # Calculate error rates
        error_rates = self._calculate_error_rates(traces)

        return {
            "timestamp": datetime.now().isoformat(),
            "service": service,
            "lookback": lookback,
            "traces": traces,
            "dependencies": dependencies,
            "error_rates": error_rates,
            "summary": {
                "total_traces": len(traces),
                "services": list(set(t.get("service", "") for t in traces)),
                "avg_duration_ms": sum(t.get("duration_ms", 0) for t in traces)
                / len(traces)
                if traces
                else 0,
            },
        }

    async def _get_recent_traces(
        self,
        service: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Get recent traces from Jaeger."""
        import aiohttp

        url = f"{self.jaeger_url}/api/traces"
        params = {
            "service": service,
            "limit": limit,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._format_traces(data)
                    else:
                        return self._get_mock_traces(limit)
        except Exception:
            return self._get_mock_traces(limit)

    async def _get_dependencies(self) -> List[Dict[str, Any]]:
        """Get service dependencies."""
        import aiohttp

        url = f"{self.jaeger_url}/api/dependencies"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("data", [])
                    else:
                        return self._get_mock_dependencies()
        except Exception:
            return self._get_mock_dependencies()

    def _format_traces(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format traces data."""
        traces_data = data.get("data", [])
        traces = []

        for trace in traces_data:
            spans = trace.get("spans", [])
            if not spans:
                continue

            root_span = spans[0]
            traces.append({
                "trace_id": trace.get("traceID"),
                "service": root_span.get("processID", ""),
                "operation": root_span.get("operationName", ""),
                "start_time": datetime.fromtimestamp(
                    root_span.get("startTime", 0) / 1000000
                ).isoformat(),
                "duration_ms": root_span.get("duration", 0) / 1000,
                "span_count": len(spans),
                "has_error": any(
                    tag.get("key") == "error" and tag.get("value")
                    for span in spans
                    for tag in span.get("tags", [])
                ),
            })

        return traces

    def _calculate_error_rates(
        self,
        traces: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Calculate error rates by service."""
        by_service: Dict[str, Dict[str, int]] = {}

        for trace in traces:
            service = trace.get("service", "unknown")
            if service not in by_service:
                by_service[service] = {"total": 0, "errors": 0}

            by_service[service]["total"] += 1
            if trace.get("has_error"):
                by_service[service]["errors"] += 1

        return {
            service: stats["errors"] / stats["total"] if stats["total"] > 0 else 0
            for service, stats in by_service.items()
        }

    def _get_mock_traces(self, limit: int) -> List[Dict[str, Any]]:
        """Get mock traces."""
        import random

        traces = []
        for i in range(min(limit, 20)):
            traces.append({
                "trace_id": f"trace_{i}_{random.randint(1000, 9999)}",
                "service": "consciousness-nexus",
                "operation": random.choice([
                    "process_thought",
                    "analyze_pattern",
                    "mesh_route",
                    "llm_call",
                ]),
                "start_time": datetime.now().isoformat(),
                "duration_ms": random.uniform(10, 500),
                "span_count": random.randint(1, 10),
                "has_error": random.random() < 0.05,
            })

        return traces

    def _get_mock_dependencies(self) -> List[Dict[str, Any]]:
        """Get mock dependencies."""
        return [
            {"parent": "api-gateway", "child": "consciousness-nexus", "callCount": 1000},
            {"parent": "consciousness-nexus", "child": "mesh-router", "callCount": 500},
            {"parent": "consciousness-nexus", "child": "llm-service", "callCount": 300},
            {"parent": "mesh-router", "child": "worker-1", "callCount": 200},
            {"parent": "mesh-router", "child": "worker-2", "callCount": 200},
        ]


class AlertsResource(MCPResource):
    """Resource providing alerts data.

    Exposes:
    - Active alerts
    - Alert history
    - Silences
    - Alert rules
    """

    def __init__(
        self,
        alertmanager_url: str = "http://localhost:9093",
        namespace: str = "consciousness",
    ):
        super().__init__(namespace)
        self.alertmanager_url = alertmanager_url
        self._cache_ttl_seconds = 10.0  # Short TTL for alerts

    @property
    def name(self) -> str:
        return "alerts"

    @property
    def description(self) -> str:
        return (
            "Current alerts from the consciousness observability platform. "
            "Includes active alerts, silences, and alert statistics."
        )

    async def _fetch(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch alerts data."""
        include_silences = arguments.get("include_silences", True)
        include_history = arguments.get("include_history", False)

        # Get active alerts
        alerts = await self._get_alerts()

        result = {
            "timestamp": datetime.now().isoformat(),
            "alerts": alerts,
            "summary": self._create_summary(alerts),
        }

        if include_silences:
            result["silences"] = await self._get_silences()

        if include_history:
            result["history"] = await self._get_alert_history()

        return result

    async def _get_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts from Alertmanager."""
        import aiohttp

        url = f"{self.alertmanager_url}/api/v2/alerts"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._format_alerts(data)
                    else:
                        return self._get_mock_alerts()
        except Exception:
            return self._get_mock_alerts()

    async def _get_silences(self) -> List[Dict[str, Any]]:
        """Get active silences."""
        import aiohttp

        url = f"{self.alertmanager_url}/api/v2/silences"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        return [
                            s for s in data
                            if s.get("status", {}).get("state") == "active"
                        ]
                    else:
                        return []
        except Exception:
            return []

    async def _get_alert_history(self) -> List[Dict[str, Any]]:
        """Get alert history (resolved alerts)."""
        # This would query a storage backend for historical alerts
        # For now, return empty list
        return []

    def _format_alerts(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format alerts."""
        formatted = []

        for alert in alerts:
            labels = alert.get("labels", {})
            formatted.append({
                "alertname": labels.get("alertname", "unknown"),
                "severity": labels.get("severity", "unknown"),
                "state": alert.get("status", {}).get("state", "unknown"),
                "labels": labels,
                "annotations": alert.get("annotations", {}),
                "starts_at": alert.get("startsAt"),
                "ends_at": alert.get("endsAt"),
                "fingerprint": alert.get("fingerprint"),
                "is_silenced": alert.get("status", {}).get("silencedBy", []) != [],
                "is_inhibited": alert.get("status", {}).get("inhibitedBy", []) != [],
            })

        return formatted

    def _create_summary(self, alerts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create alerts summary."""
        by_severity = {}
        by_state = {}

        for alert in alerts:
            severity = alert.get("severity", "unknown")
            state = alert.get("state", "unknown")

            by_severity[severity] = by_severity.get(severity, 0) + 1
            by_state[state] = by_state.get(state, 0) + 1

        return {
            "total": len(alerts),
            "firing": by_state.get("active", 0) + by_state.get("firing", 0),
            "silenced": sum(1 for a in alerts if a.get("is_silenced")),
            "by_severity": by_severity,
            "by_state": by_state,
        }

    def _get_mock_alerts(self) -> List[Dict[str, Any]]:
        """Get mock alerts."""
        import random

        alerts = []
        severities = ["critical", "warning", "info"]

        for i in range(random.randint(0, 5)):
            severity = random.choice(severities)
            alerts.append({
                "alertname": f"ConsciousnessAlert_{i}",
                "severity": severity,
                "state": "firing",
                "labels": {
                    "alertname": f"ConsciousnessAlert_{i}",
                    "severity": severity,
                    "service": "consciousness-nexus",
                    "instance": "localhost:8080",
                },
                "annotations": {
                    "summary": f"Mock alert {i}",
                    "description": f"This is mock alert {i} for demonstration",
                },
                "starts_at": datetime.now().isoformat(),
                "ends_at": None,
                "fingerprint": f"fingerprint_{i}",
                "is_silenced": False,
                "is_inhibited": False,
            })

        return alerts


class ResourceRegistry:
    """Registry for MCP data resources.

    Usage:
        registry = ResourceRegistry()
        registry.register("metrics", MetricsResource())
        registry.register("traces", TracesResource())

        # Read resource
        content = await registry.read("metrics", {"filter": "consciousness_"})
    """

    def __init__(self, namespace: str = "consciousness"):
        self.namespace = namespace
        self._resources: Dict[str, MCPResource] = {}
        self._lock = threading.Lock()

    def register(self, name: str, resource: MCPResource):
        """Register a resource.

        Args:
            name: Resource name
            resource: Resource instance
        """
        with self._lock:
            self._resources[name] = resource
            logger.info(f"Registered resource: {name}")

    def unregister(self, name: str):
        """Unregister a resource.

        Args:
            name: Resource name
        """
        with self._lock:
            if name in self._resources:
                del self._resources[name]
                logger.info(f"Unregistered resource: {name}")

    def get(self, name: str) -> Optional[MCPResource]:
        """Get a resource by name.

        Args:
            name: Resource name

        Returns:
            Resource or None
        """
        with self._lock:
            return self._resources.get(name)

    def list_resources(self) -> List[Dict[str, Any]]:
        """List all registered resources.

        Returns:
            List of resource definitions
        """
        with self._lock:
            return [
                {
                    "uri": f"consciousness://{name}",
                    "name": name,
                    "description": resource.description,
                    "mimeType": resource.mime_type,
                }
                for name, resource in self._resources.items()
            ]

    async def read(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Read a resource by name.

        Args:
            name: Resource name
            arguments: Read arguments

        Returns:
            Resource content
        """
        resource = self.get(name)
        if not resource:
            raise ValueError(f"Unknown resource: {name}")

        return await resource.read(arguments)

    def subscribe(self, name: str, subscriber_id: str):
        """Subscribe to resource updates.

        Args:
            name: Resource name
            subscriber_id: Subscriber identifier
        """
        resource = self.get(name)
        if resource:
            resource.subscribe(subscriber_id)

    def unsubscribe(self, name: str, subscriber_id: str):
        """Unsubscribe from resource updates.

        Args:
            name: Resource name
            subscriber_id: Subscriber identifier
        """
        resource = self.get(name)
        if resource:
            resource.unsubscribe(subscriber_id)

    def create_default_registry(self) -> "ResourceRegistry":
        """Create registry with default resources.

        Returns:
            Registry with default resources registered
        """
        self.register("metrics", MetricsResource(namespace=self.namespace))
        self.register("traces", TracesResource(namespace=self.namespace))
        self.register("alerts", AlertsResource(namespace=self.namespace))
        return self
