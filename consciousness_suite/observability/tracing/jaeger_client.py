"""
Jaeger Client - Distributed Trace Querying

Provides integration with Jaeger tracing backend for:
- Trace retrieval by ID, service, operation, tags
- Service discovery and operation enumeration
- Span details with logs and process information

Thread-safe with connection pooling and Prometheus metrics.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import requests
from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)


class JaegerReferenceType(str, Enum):
    """Span reference types."""
    CHILD_OF = "CHILD_OF"
    FOLLOWS_FROM = "FOLLOWS_FROM"


@dataclass
class JaegerTag:
    """Jaeger span tag."""
    key: str
    type: str  # string, bool, int64, float64, binary
    value: Any

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "type": self.type, "value": self.value}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> JaegerTag:
        return cls(
            key=data.get("key", ""),
            type=data.get("type", "string"),
            value=data.get("value"),
        )


@dataclass
class JaegerLog:
    """Jaeger span log entry."""
    timestamp: int  # microseconds since epoch
    fields: List[JaegerTag] = field(default_factory=list)

    @property
    def datetime(self) -> datetime:
        """Convert timestamp to datetime."""
        return datetime.fromtimestamp(self.timestamp / 1_000_000)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "fields": [f.to_dict() for f in self.fields],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> JaegerLog:
        return cls(
            timestamp=data.get("timestamp", 0),
            fields=[JaegerTag.from_dict(f) for f in data.get("fields", [])],
        )


@dataclass
class JaegerReference:
    """Reference to another span."""
    ref_type: JaegerReferenceType
    trace_id: str
    span_id: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "refType": self.ref_type.value,
            "traceID": self.trace_id,
            "spanID": self.span_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> JaegerReference:
        return cls(
            ref_type=JaegerReferenceType(data.get("refType", "CHILD_OF")),
            trace_id=data.get("traceID", ""),
            span_id=data.get("spanID", ""),
        )


@dataclass
class JaegerProcess:
    """Process information for a span."""
    service_name: str
    tags: List[JaegerTag] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "serviceName": self.service_name,
            "tags": [t.to_dict() for t in self.tags],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> JaegerProcess:
        return cls(
            service_name=data.get("serviceName", "unknown"),
            tags=[JaegerTag.from_dict(t) for t in data.get("tags", [])],
        )


@dataclass
class JaegerSpan:
    """A single Jaeger span."""
    trace_id: str
    span_id: str
    operation_name: str
    process: JaegerProcess
    start_time: int  # microseconds since epoch
    duration: int  # microseconds
    references: List[JaegerReference] = field(default_factory=list)
    tags: List[JaegerTag] = field(default_factory=list)
    logs: List[JaegerLog] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    flags: int = 0

    @property
    def start_datetime(self) -> datetime:
        """Convert start time to datetime."""
        return datetime.fromtimestamp(self.start_time / 1_000_000)

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        return self.duration / 1000.0

    @property
    def duration_seconds(self) -> float:
        """Duration in seconds."""
        return self.duration / 1_000_000.0

    @property
    def parent_span_id(self) -> Optional[str]:
        """Get parent span ID if exists."""
        for ref in self.references:
            if ref.ref_type == JaegerReferenceType.CHILD_OF:
                return ref.span_id
        return None

    @property
    def is_error(self) -> bool:
        """Check if span has error tag."""
        for tag in self.tags:
            if tag.key == "error" and tag.value is True:
                return True
        return False

    def get_tag(self, key: str) -> Optional[Any]:
        """Get tag value by key."""
        for tag in self.tags:
            if tag.key == key:
                return tag.value
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "traceID": self.trace_id,
            "spanID": self.span_id,
            "operationName": self.operation_name,
            "process": self.process.to_dict(),
            "startTime": self.start_time,
            "duration": self.duration,
            "references": [r.to_dict() for r in self.references],
            "tags": [t.to_dict() for t in self.tags],
            "logs": [log.to_dict() for log in self.logs],
            "warnings": self.warnings,
            "flags": self.flags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], process: JaegerProcess) -> JaegerSpan:
        return cls(
            trace_id=data.get("traceID", ""),
            span_id=data.get("spanID", ""),
            operation_name=data.get("operationName", ""),
            process=process,
            start_time=data.get("startTime", 0),
            duration=data.get("duration", 0),
            references=[JaegerReference.from_dict(r) for r in data.get("references", [])],
            tags=[JaegerTag.from_dict(t) for t in data.get("tags", [])],
            logs=[JaegerLog.from_dict(log) for log in data.get("logs", [])],
            warnings=data.get("warnings", []),
            flags=data.get("flags", 0),
        )


@dataclass
class JaegerTrace:
    """A complete Jaeger trace with all spans."""
    trace_id: str
    spans: List[JaegerSpan] = field(default_factory=list)
    processes: Dict[str, JaegerProcess] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def root_span(self) -> Optional[JaegerSpan]:
        """Get root span (no parent)."""
        for span in self.spans:
            if span.parent_span_id is None:
                return span
        return self.spans[0] if self.spans else None

    @property
    def service_name(self) -> str:
        """Get primary service name from root span."""
        root = self.root_span
        if root:
            return root.process.service_name
        return "unknown"

    @property
    def operation_name(self) -> str:
        """Get operation name from root span."""
        root = self.root_span
        if root:
            return root.operation_name
        return "unknown"

    @property
    def duration_ms(self) -> float:
        """Total trace duration in milliseconds."""
        if not self.spans:
            return 0.0
        start = min(s.start_time for s in self.spans)
        end = max(s.start_time + s.duration for s in self.spans)
        return (end - start) / 1000.0

    @property
    def start_time(self) -> Optional[datetime]:
        """Trace start time."""
        if not self.spans:
            return None
        min_start = min(s.start_time for s in self.spans)
        return datetime.fromtimestamp(min_start / 1_000_000)

    @property
    def span_count(self) -> int:
        """Number of spans in trace."""
        return len(self.spans)

    @property
    def error_count(self) -> int:
        """Number of error spans."""
        return sum(1 for s in self.spans if s.is_error)

    @property
    def has_errors(self) -> bool:
        """Check if any span has errors."""
        return self.error_count > 0

    def get_spans_by_service(self, service_name: str) -> List[JaegerSpan]:
        """Get all spans for a service."""
        return [s for s in self.spans if s.process.service_name == service_name]

    def get_spans_by_operation(self, operation_name: str) -> List[JaegerSpan]:
        """Get all spans for an operation."""
        return [s for s in self.spans if s.operation_name == operation_name]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "traceID": self.trace_id,
            "spans": [s.to_dict() for s in self.spans],
            "processes": {k: v.to_dict() for k, v in self.processes.items()},
            "warnings": self.warnings,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> JaegerTrace:
        # Parse processes first
        processes = {}
        for pid, pdata in data.get("processes", {}).items():
            processes[pid] = JaegerProcess.from_dict(pdata)

        # Parse spans with process references
        spans = []
        for span_data in data.get("spans", []):
            process_id = span_data.get("processID", "p1")
            process = processes.get(process_id, JaegerProcess(service_name="unknown"))
            spans.append(JaegerSpan.from_dict(span_data, process))

        return cls(
            trace_id=data.get("traceID", ""),
            spans=spans,
            processes=processes,
            warnings=data.get("warnings", []),
        )


@dataclass
class JaegerConfig:
    """Jaeger client configuration."""
    base_url: str = "http://localhost:16686"
    api_path: str = "/api"
    timeout_seconds: float = 30.0
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    verify_ssl: bool = True
    auth_token: Optional[str] = None
    namespace: str = "consciousness"

    @property
    def api_url(self) -> str:
        return f"{self.base_url.rstrip('/')}{self.api_path}"


class JaegerClient:
    """Client for Jaeger distributed tracing backend.

    Thread-safe client with connection pooling, retry logic,
    and Prometheus metrics instrumentation.

    Usage:
        config = JaegerConfig(base_url="http://jaeger:16686")
        client = JaegerClient(config)

        # Get trace by ID
        trace = client.get_trace("abc123")

        # Search traces
        traces = client.search_traces(
            service="api-gateway",
            operation="POST /api/v1/process",
            tags={"http.status_code": "500"},
            start=datetime.now() - timedelta(hours=1),
            limit=20,
        )

        # List services
        services = client.get_services()

        # List operations for service
        operations = client.get_operations("api-gateway")
    """

    def __init__(self, config: Optional[JaegerConfig] = None):
        self.config = config or JaegerConfig()
        self._session: Optional[requests.Session] = None
        self._lock = threading.Lock()

        # Metrics
        ns = self.config.namespace
        self.requests_total = Counter(
            f"{ns}_jaeger_requests_total",
            "Total Jaeger API requests",
            ["endpoint", "status"],
        )
        self.request_latency = Histogram(
            f"{ns}_jaeger_request_latency_seconds",
            "Jaeger API request latency",
            ["endpoint"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        )
        self.traces_fetched = Counter(
            f"{ns}_jaeger_traces_fetched_total",
            "Total traces fetched from Jaeger",
        )
        self.connection_errors = Counter(
            f"{ns}_jaeger_connection_errors_total",
            "Jaeger connection errors",
        )

    @property
    def session(self) -> requests.Session:
        """Get or create HTTP session."""
        if self._session is None:
            with self._lock:
                if self._session is None:
                    self._session = requests.Session()
                    if self.config.auth_token:
                        self._session.headers["Authorization"] = f"Bearer {self.config.auth_token}"
        return self._session

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make HTTP request with retry logic.

        Args:
            method: HTTP method
            endpoint: API endpoint path
            params: Query parameters
            json_data: JSON body

        Returns:
            Response JSON data

        Raises:
            JaegerClientError: On request failure
        """
        url = f"{self.config.api_url}{endpoint}"
        last_error = None

        for attempt in range(self.config.max_retries):
            start_time = time.time()
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json_data,
                    timeout=self.config.timeout_seconds,
                    verify=self.config.verify_ssl,
                )

                duration = time.time() - start_time
                self.request_latency.labels(endpoint=endpoint).observe(duration)

                if response.ok:
                    self.requests_total.labels(endpoint=endpoint, status="success").inc()
                    return response.json()
                else:
                    self.requests_total.labels(endpoint=endpoint, status="error").inc()
                    last_error = f"HTTP {response.status_code}: {response.text}"

            except requests.exceptions.ConnectionError as e:
                self.connection_errors.inc()
                last_error = f"Connection error: {e}"
            except requests.exceptions.Timeout as e:
                last_error = f"Request timeout: {e}"
            except requests.exceptions.RequestException as e:
                last_error = f"Request error: {e}"

            # Wait before retry
            if attempt < self.config.max_retries - 1:
                time.sleep(self.config.retry_delay_seconds * (attempt + 1))

        raise JaegerClientError(f"Request failed after {self.config.max_retries} attempts: {last_error}")

    def get_services(self) -> List[str]:
        """Get list of available services.

        Returns:
            List of service names
        """
        response = self._request("GET", "/services")
        return response.get("data", [])

    def get_operations(self, service: str) -> List[str]:
        """Get operations for a service.

        Args:
            service: Service name

        Returns:
            List of operation names
        """
        response = self._request("GET", f"/services/{service}/operations")
        return response.get("data", [])

    def get_trace(self, trace_id: str) -> Optional[JaegerTrace]:
        """Get a trace by ID.

        Args:
            trace_id: Trace ID

        Returns:
            JaegerTrace or None if not found
        """
        try:
            response = self._request("GET", f"/traces/{trace_id}")
            traces_data = response.get("data", [])
            if traces_data:
                self.traces_fetched.inc()
                return JaegerTrace.from_dict(traces_data[0])
        except JaegerClientError as e:
            logger.warning(f"Failed to get trace {trace_id}: {e}")
        return None

    def search_traces(
        self,
        service: str,
        operation: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        min_duration: Optional[int] = None,
        max_duration: Optional[int] = None,
        limit: int = 20,
    ) -> List[JaegerTrace]:
        """Search for traces.

        Args:
            service: Service name (required)
            operation: Operation name filter
            tags: Tag filters (key=value)
            start: Start time (default: 1 hour ago)
            end: End time (default: now)
            min_duration: Minimum duration in microseconds
            max_duration: Maximum duration in microseconds
            limit: Maximum traces to return

        Returns:
            List of matching traces
        """
        params: Dict[str, Any] = {"service": service, "limit": limit}

        if operation:
            params["operation"] = operation

        if tags:
            # Format: key=value pairs
            params["tags"] = json.dumps(tags)

        if start:
            params["start"] = int(start.timestamp() * 1_000_000)
        else:
            params["start"] = int((datetime.now() - timedelta(hours=1)).timestamp() * 1_000_000)

        if end:
            params["end"] = int(end.timestamp() * 1_000_000)
        else:
            params["end"] = int(datetime.now().timestamp() * 1_000_000)

        if min_duration:
            params["minDuration"] = f"{min_duration}us"

        if max_duration:
            params["maxDuration"] = f"{max_duration}us"

        response = self._request("GET", "/traces", params=params)
        traces_data = response.get("data", [])

        traces = [JaegerTrace.from_dict(t) for t in traces_data]
        self.traces_fetched.inc(len(traces))

        return traces

    def get_dependencies(
        self,
        end_ts: Optional[datetime] = None,
        lookback: int = 86400000,  # 1 day in ms
    ) -> List[Dict[str, Any]]:
        """Get service dependencies.

        Args:
            end_ts: End timestamp (default: now)
            lookback: Lookback duration in milliseconds

        Returns:
            List of dependency links
        """
        params = {
            "lookback": lookback,
        }
        if end_ts:
            params["endTs"] = int(end_ts.timestamp() * 1000)
        else:
            params["endTs"] = int(datetime.now().timestamp() * 1000)

        response = self._request("GET", "/dependencies", params=params)
        return response.get("data", [])

    def health_check(self) -> bool:
        """Check if Jaeger is healthy.

        Returns:
            True if healthy
        """
        try:
            services = self.get_services()
            return True
        except JaegerClientError:
            return False

    def close(self):
        """Close the client session."""
        with self._lock:
            if self._session:
                self._session.close()
                self._session = None

    def __enter__(self) -> JaegerClient:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class JaegerClientError(Exception):
    """Jaeger client error."""
    pass


# Import json for tag serialization
import json
