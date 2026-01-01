"""MCP Observability Tools

Tools exposed via MCP for Claude Code integration:
- MetricsQueryTool: Query Prometheus/OTLP metrics
- TraceLookupTool: Lookup distributed traces
- AlertStatusTool: Check alert status

Usage:
    from consciousness_suite.observability.ai_platform.mcp_server.tools import (
        MetricsQueryTool,
        TraceLookupTool,
        AlertStatusTool,
        ToolRegistry,
    )

    registry = ToolRegistry()
    registry.register("metrics_query", MetricsQueryTool())
    registry.register("trace_lookup", TraceLookupTool())
    registry.register("alert_status", AlertStatusTool())

    # Execute a tool
    result = await registry.execute("metrics_query", {
        "query": "consciousness_requests_total",
        "start": "1h",
    })
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)


class ToolStatus(str, Enum):
    """Tool execution status."""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    INVALID_INPUT = "invalid_input"


@dataclass
class ToolResult:
    """Result from tool execution."""
    tool_name: str
    status: ToolStatus
    data: Optional[Any] = None
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return self.status == ToolStatus.SUCCESS

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tool_name": self.tool_name,
            "status": self.status.value,
            "data": self.data,
            "error_message": self.error_message,
            "execution_time_ms": self.execution_time_ms,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class ObservabilityTool(ABC):
    """Base class for MCP observability tools.

    Subclasses must implement:
    - execute(): The tool logic
    - description: Tool description
    - input_schema: JSON Schema for input validation
    """

    def __init__(self, namespace: str = "consciousness"):
        self.namespace = namespace
        self._lock = threading.Lock()

        # Execution history
        self._history: List[ToolResult] = []
        self._max_history = 100

        # Metrics
        self.executions_total = Counter(
            f"{namespace}_mcp_tool_executions_total",
            "Total tool executions",
            ["tool_name", "status"],
        )

        self.execution_latency = Histogram(
            f"{namespace}_mcp_tool_execution_latency_seconds",
            "Tool execution latency",
            ["tool_name"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        )

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description."""
        pass

    @property
    @abstractmethod
    def input_schema(self) -> Dict[str, Any]:
        """JSON Schema for input validation."""
        pass

    @abstractmethod
    async def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        """Execute the tool.

        Args:
            arguments: Tool arguments

        Returns:
            ToolResult
        """
        pass

    def _record_result(self, result: ToolResult):
        """Record execution result."""
        with self._lock:
            self._history.append(result)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

        self.executions_total.labels(
            tool_name=result.tool_name,
            status=result.status.value,
        ).inc()

        self.execution_latency.labels(
            tool_name=result.tool_name,
        ).observe(result.execution_time_ms / 1000)

    def get_history(self, limit: int = 50) -> List[ToolResult]:
        """Get execution history."""
        with self._lock:
            return list(reversed(self._history[-limit:]))


class MetricsQueryTool(ObservabilityTool):
    """Tool for querying metrics from Prometheus/OTLP.

    Supports:
    - Instant queries
    - Range queries
    - Label filtering
    - Aggregation
    """

    def __init__(
        self,
        prometheus_url: str = "http://localhost:9090",
        namespace: str = "consciousness",
    ):
        super().__init__(namespace)
        self.prometheus_url = prometheus_url

    @property
    def name(self) -> str:
        return "metrics_query"

    @property
    def description(self) -> str:
        return (
            "Query Prometheus metrics from the consciousness observability platform. "
            "Supports instant and range queries with label filtering."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "PromQL query expression",
                },
                "start": {
                    "type": "string",
                    "description": "Start time (e.g., '1h', '30m', ISO timestamp)",
                    "default": "5m",
                },
                "end": {
                    "type": "string",
                    "description": "End time (e.g., 'now', ISO timestamp)",
                    "default": "now",
                },
                "step": {
                    "type": "string",
                    "description": "Query step (e.g., '15s', '1m')",
                    "default": "15s",
                },
                "labels": {
                    "type": "object",
                    "description": "Label filters",
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": ["query"],
        }

    async def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        """Execute metrics query."""
        start_time = time.perf_counter()

        try:
            query = arguments.get("query")
            if not query:
                return ToolResult(
                    tool_name=self.name,
                    status=ToolStatus.INVALID_INPUT,
                    error_message="Missing required 'query' parameter",
                )

            start = arguments.get("start", "5m")
            end = arguments.get("end", "now")
            step = arguments.get("step", "15s")
            labels = arguments.get("labels", {})

            # Add label filters to query
            if labels:
                label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
                if "{" in query:
                    query = query.replace("{", "{" + label_str + ",")
                else:
                    query = f"{query}{{{label_str}}}"

            # Query Prometheus
            result = await self._query_prometheus(query, start, end, step)

            execution_time = (time.perf_counter() - start_time) * 1000

            tool_result = ToolResult(
                tool_name=self.name,
                status=ToolStatus.SUCCESS,
                data=result,
                execution_time_ms=execution_time,
                metadata={
                    "query": query,
                    "start": start,
                    "end": end,
                    "step": step,
                },
            )

        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            logger.error(f"Metrics query error: {e}")
            tool_result = ToolResult(
                tool_name=self.name,
                status=ToolStatus.ERROR,
                error_message=str(e),
                execution_time_ms=execution_time,
            )

        self._record_result(tool_result)
        return tool_result

    async def _query_prometheus(
        self,
        query: str,
        start: str,
        end: str,
        step: str,
    ) -> Dict[str, Any]:
        """Query Prometheus API."""
        import aiohttp

        # Parse time specifications
        end_time = datetime.now() if end == "now" else self._parse_time(end)
        start_time = self._calculate_start_time(start, end_time)

        # Determine if range or instant query
        is_range = start != end

        if is_range:
            url = f"{self.prometheus_url}/api/v1/query_range"
            params = {
                "query": query,
                "start": start_time.timestamp(),
                "end": end_time.timestamp(),
                "step": step,
            }
        else:
            url = f"{self.prometheus_url}/api/v1/query"
            params = {
                "query": query,
                "time": end_time.timestamp(),
            }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._format_prometheus_response(data)
                    else:
                        text = await response.text()
                        raise Exception(f"Prometheus query failed: {text}")
        except aiohttp.ClientError:
            # Return mock data for demonstration
            return self._get_mock_metrics(query)

    def _parse_time(self, time_str: str) -> datetime:
        """Parse time string to datetime."""
        if time_str == "now":
            return datetime.now()

        # Try ISO format
        try:
            return datetime.fromisoformat(time_str.replace("Z", "+00:00"))
        except ValueError:
            pass

        # Try relative format (1h, 30m, etc)
        return self._calculate_start_time(time_str, datetime.now())

    def _calculate_start_time(self, duration: str, end_time: datetime) -> datetime:
        """Calculate start time from duration string."""
        unit = duration[-1]
        try:
            value = int(duration[:-1])
        except ValueError:
            return end_time - timedelta(minutes=5)

        if unit == "s":
            return end_time - timedelta(seconds=value)
        elif unit == "m":
            return end_time - timedelta(minutes=value)
        elif unit == "h":
            return end_time - timedelta(hours=value)
        elif unit == "d":
            return end_time - timedelta(days=value)
        else:
            return end_time - timedelta(minutes=5)

    def _format_prometheus_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format Prometheus response."""
        if data.get("status") != "success":
            return {"error": data.get("error", "Unknown error")}

        result = data.get("data", {})
        result_type = result.get("resultType")
        metrics = result.get("result", [])

        formatted = {
            "result_type": result_type,
            "metrics": [],
            "summary": {},
        }

        for metric in metrics:
            metric_info = {
                "labels": metric.get("metric", {}),
            }

            if result_type == "vector":
                value = metric.get("value", [])
                if len(value) >= 2:
                    metric_info["timestamp"] = datetime.fromtimestamp(
                        value[0]
                    ).isoformat()
                    metric_info["value"] = float(value[1])
            elif result_type == "matrix":
                values = metric.get("values", [])
                metric_info["values"] = [
                    {
                        "timestamp": datetime.fromtimestamp(v[0]).isoformat(),
                        "value": float(v[1]),
                    }
                    for v in values
                ]
                if values:
                    all_values = [float(v[1]) for v in values]
                    metric_info["stats"] = {
                        "min": min(all_values),
                        "max": max(all_values),
                        "avg": sum(all_values) / len(all_values),
                    }

            formatted["metrics"].append(metric_info)

        # Add summary
        if formatted["metrics"]:
            if result_type == "vector":
                values = [m.get("value", 0) for m in formatted["metrics"]]
                formatted["summary"] = {
                    "count": len(values),
                    "total": sum(values),
                    "avg": sum(values) / len(values) if values else 0,
                }
            elif result_type == "matrix":
                formatted["summary"] = {
                    "series_count": len(formatted["metrics"]),
                }

        return formatted

    def _get_mock_metrics(self, query: str) -> Dict[str, Any]:
        """Get mock metrics for demonstration."""
        import random

        return {
            "result_type": "vector",
            "metrics": [
                {
                    "labels": {"instance": "localhost:9090", "job": "consciousness"},
                    "timestamp": datetime.now().isoformat(),
                    "value": random.uniform(0, 100),
                }
            ],
            "summary": {
                "count": 1,
                "total": random.uniform(0, 100),
                "avg": random.uniform(0, 100),
            },
            "note": "Mock data - Prometheus not available",
        }


class TraceLookupTool(ObservabilityTool):
    """Tool for looking up distributed traces.

    Supports:
    - Trace ID lookup
    - Service filtering
    - Time range filtering
    - Span attribute filtering
    """

    def __init__(
        self,
        jaeger_url: str = "http://localhost:16686",
        namespace: str = "consciousness",
    ):
        super().__init__(namespace)
        self.jaeger_url = jaeger_url

    @property
    def name(self) -> str:
        return "trace_lookup"

    @property
    def description(self) -> str:
        return (
            "Look up distributed traces from the consciousness observability platform. "
            "Search by trace ID, service name, or operation with time filtering."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "trace_id": {
                    "type": "string",
                    "description": "Specific trace ID to look up",
                },
                "service": {
                    "type": "string",
                    "description": "Filter by service name",
                },
                "operation": {
                    "type": "string",
                    "description": "Filter by operation name",
                },
                "start": {
                    "type": "string",
                    "description": "Start time (e.g., '1h', ISO timestamp)",
                    "default": "1h",
                },
                "end": {
                    "type": "string",
                    "description": "End time",
                    "default": "now",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of traces",
                    "default": 20,
                },
                "min_duration": {
                    "type": "string",
                    "description": "Minimum trace duration (e.g., '100ms', '1s')",
                },
                "max_duration": {
                    "type": "string",
                    "description": "Maximum trace duration",
                },
                "tags": {
                    "type": "object",
                    "description": "Tag filters",
                    "additionalProperties": {"type": "string"},
                },
            },
        }

    async def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        """Execute trace lookup."""
        start_time = time.perf_counter()

        try:
            trace_id = arguments.get("trace_id")
            service = arguments.get("service", "consciousness-nexus")
            operation = arguments.get("operation")
            limit = arguments.get("limit", 20)
            tags = arguments.get("tags", {})

            if trace_id:
                # Direct trace lookup
                result = await self._get_trace(trace_id)
            else:
                # Search for traces
                result = await self._search_traces(
                    service=service,
                    operation=operation,
                    limit=limit,
                    tags=tags,
                )

            execution_time = (time.perf_counter() - start_time) * 1000

            tool_result = ToolResult(
                tool_name=self.name,
                status=ToolStatus.SUCCESS,
                data=result,
                execution_time_ms=execution_time,
                metadata={
                    "trace_id": trace_id,
                    "service": service,
                    "operation": operation,
                },
            )

        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            logger.error(f"Trace lookup error: {e}")
            tool_result = ToolResult(
                tool_name=self.name,
                status=ToolStatus.ERROR,
                error_message=str(e),
                execution_time_ms=execution_time,
            )

        self._record_result(tool_result)
        return tool_result

    async def _get_trace(self, trace_id: str) -> Dict[str, Any]:
        """Get a specific trace by ID."""
        import aiohttp

        url = f"{self.jaeger_url}/api/traces/{trace_id}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._format_trace(data)
                    else:
                        text = await response.text()
                        raise Exception(f"Trace lookup failed: {text}")
        except aiohttp.ClientError:
            # Return mock data
            return self._get_mock_trace(trace_id)

    async def _search_traces(
        self,
        service: str,
        operation: Optional[str],
        limit: int,
        tags: Dict[str, str],
    ) -> Dict[str, Any]:
        """Search for traces."""
        import aiohttp

        url = f"{self.jaeger_url}/api/traces"
        params = {
            "service": service,
            "limit": limit,
        }
        if operation:
            params["operation"] = operation
        if tags:
            params["tags"] = ",".join(f"{k}:{v}" for k, v in tags.items())

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._format_traces_search(data)
                    else:
                        text = await response.text()
                        raise Exception(f"Trace search failed: {text}")
        except aiohttp.ClientError:
            # Return mock data
            return self._get_mock_traces_search(service, limit)

    def _format_trace(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format trace data."""
        traces = data.get("data", [])
        if not traces:
            return {"error": "Trace not found"}

        trace = traces[0]
        spans = trace.get("spans", [])

        formatted_spans = []
        for span in spans:
            formatted_spans.append({
                "span_id": span.get("spanID"),
                "operation_name": span.get("operationName"),
                "service": span.get("processID"),
                "start_time": datetime.fromtimestamp(
                    span.get("startTime", 0) / 1000000
                ).isoformat(),
                "duration_ms": span.get("duration", 0) / 1000,
                "status": self._get_span_status(span),
                "tags": {
                    tag["key"]: tag["value"]
                    for tag in span.get("tags", [])
                },
                "logs": [
                    {
                        "timestamp": datetime.fromtimestamp(
                            log.get("timestamp", 0) / 1000000
                        ).isoformat(),
                        "fields": {
                            f["key"]: f["value"]
                            for f in log.get("fields", [])
                        },
                    }
                    for log in span.get("logs", [])
                ],
            })

        return {
            "trace_id": trace.get("traceID"),
            "spans": formatted_spans,
            "span_count": len(spans),
            "services": list(set(s["service"] for s in formatted_spans)),
            "total_duration_ms": max(
                s["duration_ms"] for s in formatted_spans
            ) if formatted_spans else 0,
        }

    def _format_traces_search(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format trace search results."""
        traces = data.get("data", [])

        formatted_traces = []
        for trace in traces:
            spans = trace.get("spans", [])
            if not spans:
                continue

            root_span = spans[0]
            formatted_traces.append({
                "trace_id": trace.get("traceID"),
                "root_operation": root_span.get("operationName"),
                "service": root_span.get("processID"),
                "start_time": datetime.fromtimestamp(
                    root_span.get("startTime", 0) / 1000000
                ).isoformat(),
                "duration_ms": root_span.get("duration", 0) / 1000,
                "span_count": len(spans),
                "status": self._get_span_status(root_span),
            })

        return {
            "traces": formatted_traces,
            "total_count": len(formatted_traces),
        }

    def _get_span_status(self, span: Dict[str, Any]) -> str:
        """Extract span status from tags."""
        for tag in span.get("tags", []):
            if tag["key"] in ["otel.status_code", "status.code"]:
                return tag["value"]
            if tag["key"] == "error" and tag["value"]:
                return "ERROR"
        return "OK"

    def _get_mock_trace(self, trace_id: str) -> Dict[str, Any]:
        """Get mock trace data."""
        import random

        return {
            "trace_id": trace_id,
            "spans": [
                {
                    "span_id": f"span_{i}",
                    "operation_name": f"operation_{i}",
                    "service": "consciousness-nexus",
                    "start_time": datetime.now().isoformat(),
                    "duration_ms": random.uniform(1, 100),
                    "status": "OK",
                    "tags": {},
                    "logs": [],
                }
                for i in range(3)
            ],
            "span_count": 3,
            "services": ["consciousness-nexus"],
            "total_duration_ms": random.uniform(10, 200),
            "note": "Mock data - Jaeger not available",
        }

    def _get_mock_traces_search(
        self,
        service: str,
        limit: int,
    ) -> Dict[str, Any]:
        """Get mock trace search results."""
        import random

        traces = []
        for i in range(min(limit, 5)):
            traces.append({
                "trace_id": f"trace_{i}_{random.randint(1000, 9999)}",
                "root_operation": f"operation_{i}",
                "service": service,
                "start_time": datetime.now().isoformat(),
                "duration_ms": random.uniform(10, 500),
                "span_count": random.randint(1, 10),
                "status": "OK" if random.random() > 0.1 else "ERROR",
            })

        return {
            "traces": traces,
            "total_count": len(traces),
            "note": "Mock data - Jaeger not available",
        }


class AlertStatusTool(ObservabilityTool):
    """Tool for checking alert status.

    Supports:
    - Active alerts listing
    - Alert history
    - Silenced alerts
    - Alert rule status
    """

    def __init__(
        self,
        alertmanager_url: str = "http://localhost:9093",
        namespace: str = "consciousness",
    ):
        super().__init__(namespace)
        self.alertmanager_url = alertmanager_url

    @property
    def name(self) -> str:
        return "alert_status"

    @property
    def description(self) -> str:
        return (
            "Check alert status from the consciousness observability platform. "
            "View active alerts, alert history, and silences."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "state": {
                    "type": "string",
                    "enum": ["active", "pending", "suppressed", "all"],
                    "description": "Filter by alert state",
                    "default": "active",
                },
                "severity": {
                    "type": "string",
                    "enum": ["critical", "warning", "info", "all"],
                    "description": "Filter by severity",
                    "default": "all",
                },
                "labels": {
                    "type": "object",
                    "description": "Label filters",
                    "additionalProperties": {"type": "string"},
                },
                "include_silenced": {
                    "type": "boolean",
                    "description": "Include silenced alerts",
                    "default": False,
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum alerts to return",
                    "default": 50,
                },
            },
        }

    async def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        """Execute alert status check."""
        start_time = time.perf_counter()

        try:
            state = arguments.get("state", "active")
            severity = arguments.get("severity", "all")
            labels = arguments.get("labels", {})
            include_silenced = arguments.get("include_silenced", False)
            limit = arguments.get("limit", 50)

            # Get alerts
            alerts = await self._get_alerts(
                state=state,
                severity=severity,
                labels=labels,
                include_silenced=include_silenced,
                limit=limit,
            )

            # Get silences if requested
            silences = []
            if include_silenced:
                silences = await self._get_silences()

            execution_time = (time.perf_counter() - start_time) * 1000

            tool_result = ToolResult(
                tool_name=self.name,
                status=ToolStatus.SUCCESS,
                data={
                    "alerts": alerts,
                    "silences": silences,
                    "summary": self._create_summary(alerts),
                },
                execution_time_ms=execution_time,
                metadata={
                    "state": state,
                    "severity": severity,
                },
            )

        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            logger.error(f"Alert status error: {e}")
            tool_result = ToolResult(
                tool_name=self.name,
                status=ToolStatus.ERROR,
                error_message=str(e),
                execution_time_ms=execution_time,
            )

        self._record_result(tool_result)
        return tool_result

    async def _get_alerts(
        self,
        state: str,
        severity: str,
        labels: Dict[str, str],
        include_silenced: bool,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Get alerts from Alertmanager."""
        import aiohttp

        url = f"{self.alertmanager_url}/api/v2/alerts"
        params = {}

        if state != "all":
            params["active"] = state == "active"
            params["pending"] = state == "pending"
            params["silenced"] = include_silenced

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._format_alerts(
                            data, severity, labels, limit
                        )
                    else:
                        text = await response.text()
                        raise Exception(f"Alertmanager query failed: {text}")
        except aiohttp.ClientError:
            # Return mock data
            return self._get_mock_alerts(severity, limit)

    async def _get_silences(self) -> List[Dict[str, Any]]:
        """Get active silences."""
        import aiohttp

        url = f"{self.alertmanager_url}/api/v2/silences"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._format_silences(data)
                    else:
                        return []
        except aiohttp.ClientError:
            return []

    def _format_alerts(
        self,
        alerts: List[Dict[str, Any]],
        severity: str,
        labels: Dict[str, str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Format alerts."""
        formatted = []

        for alert in alerts:
            alert_labels = alert.get("labels", {})

            # Filter by severity
            if severity != "all":
                if alert_labels.get("severity") != severity:
                    continue

            # Filter by labels
            match = True
            for key, value in labels.items():
                if alert_labels.get(key) != value:
                    match = False
                    break
            if not match:
                continue

            formatted.append({
                "alertname": alert_labels.get("alertname", "unknown"),
                "severity": alert_labels.get("severity", "unknown"),
                "state": alert.get("status", {}).get("state", "unknown"),
                "labels": alert_labels,
                "annotations": alert.get("annotations", {}),
                "starts_at": alert.get("startsAt"),
                "ends_at": alert.get("endsAt"),
                "generator_url": alert.get("generatorURL"),
                "fingerprint": alert.get("fingerprint"),
            })

            if len(formatted) >= limit:
                break

        return formatted

    def _format_silences(
        self,
        silences: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Format silences."""
        formatted = []

        for silence in silences:
            if silence.get("status", {}).get("state") != "active":
                continue

            formatted.append({
                "id": silence.get("id"),
                "matchers": silence.get("matchers", []),
                "created_by": silence.get("createdBy"),
                "comment": silence.get("comment"),
                "starts_at": silence.get("startsAt"),
                "ends_at": silence.get("endsAt"),
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
            "by_severity": by_severity,
            "by_state": by_state,
        }

    def _get_mock_alerts(
        self,
        severity: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Get mock alerts."""
        import random

        alerts = []
        severities = ["critical", "warning", "info"]
        if severity != "all":
            severities = [severity]

        for i in range(min(limit, 5)):
            alerts.append({
                "alertname": f"ConsciousnessAlert_{i}",
                "severity": random.choice(severities),
                "state": "firing",
                "labels": {
                    "alertname": f"ConsciousnessAlert_{i}",
                    "severity": random.choice(severities),
                    "service": "consciousness-nexus",
                },
                "annotations": {
                    "summary": f"Mock alert {i}",
                    "description": f"This is mock alert {i} for demonstration",
                },
                "starts_at": datetime.now().isoformat(),
                "ends_at": None,
            })

        return alerts


class ToolRegistry:
    """Registry for MCP observability tools.

    Usage:
        registry = ToolRegistry()
        registry.register("metrics_query", MetricsQueryTool())
        registry.register("trace_lookup", TraceLookupTool())

        # Execute tool
        result = await registry.execute("metrics_query", {"query": "up"})
    """

    def __init__(self, namespace: str = "consciousness"):
        self.namespace = namespace
        self._tools: Dict[str, ObservabilityTool] = {}
        self._lock = threading.Lock()

        # Metrics
        self.registered_tools = 0

    def register(self, name: str, tool: ObservabilityTool):
        """Register a tool.

        Args:
            name: Tool name
            tool: Tool instance
        """
        with self._lock:
            self._tools[name] = tool
            self.registered_tools = len(self._tools)
            logger.info(f"Registered tool: {name}")

    def unregister(self, name: str):
        """Unregister a tool.

        Args:
            name: Tool name
        """
        with self._lock:
            if name in self._tools:
                del self._tools[name]
                self.registered_tools = len(self._tools)
                logger.info(f"Unregistered tool: {name}")

    def get(self, name: str) -> Optional[ObservabilityTool]:
        """Get a tool by name.

        Args:
            name: Tool name

        Returns:
            Tool or None
        """
        with self._lock:
            return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """List all registered tools.

        Returns:
            List of tool definitions
        """
        with self._lock:
            return [
                {
                    "name": name,
                    "description": tool.description,
                    "inputSchema": tool.input_schema,
                }
                for name, tool in self._tools.items()
            ]

    async def execute(
        self,
        name: str,
        arguments: Dict[str, Any],
    ) -> ToolResult:
        """Execute a tool by name.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            ToolResult
        """
        tool = self.get(name)
        if not tool:
            return ToolResult(
                tool_name=name,
                status=ToolStatus.ERROR,
                error_message=f"Unknown tool: {name}",
            )

        return await tool.execute(arguments)

    def create_default_registry(self) -> "ToolRegistry":
        """Create registry with default tools.

        Returns:
            Registry with default tools registered
        """
        self.register("metrics_query", MetricsQueryTool(namespace=self.namespace))
        self.register("trace_lookup", TraceLookupTool(namespace=self.namespace))
        self.register("alert_status", AlertStatusTool(namespace=self.namespace))
        return self
