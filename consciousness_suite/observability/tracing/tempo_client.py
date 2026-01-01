"""
Grafana Tempo Client - Distributed Trace Backend

Provides integration with Grafana Tempo for:
- Trace retrieval by ID
- TraceQL queries for advanced search
- Service graph and metrics
- Span metrics aggregation

Thread-safe with connection pooling and Prometheus metrics.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import requests
from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)


class SpanStatus(str, Enum):
    """Span status codes."""
    UNSET = "UNSET"
    OK = "OK"
    ERROR = "ERROR"


class SpanKind(str, Enum):
    """Span kind values."""
    UNSPECIFIED = "SPAN_KIND_UNSPECIFIED"
    INTERNAL = "SPAN_KIND_INTERNAL"
    SERVER = "SPAN_KIND_SERVER"
    CLIENT = "SPAN_KIND_CLIENT"
    PRODUCER = "SPAN_KIND_PRODUCER"
    CONSUMER = "SPAN_KIND_CONSUMER"


@dataclass
class TempoAttribute:
    """Tempo span attribute."""
    key: str
    value: Any
    value_type: str = "string"

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "value": {"stringValue": str(self.value)}}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TempoAttribute:
        key = data.get("key", "")
        value_data = data.get("value", {})

        # Extract value based on type
        if "stringValue" in value_data:
            value = value_data["stringValue"]
            value_type = "string"
        elif "intValue" in value_data:
            value = int(value_data["intValue"])
            value_type = "int"
        elif "boolValue" in value_data:
            value = value_data["boolValue"]
            value_type = "bool"
        elif "doubleValue" in value_data:
            value = float(value_data["doubleValue"])
            value_type = "double"
        else:
            value = str(value_data)
            value_type = "string"

        return cls(key=key, value=value, value_type=value_type)


@dataclass
class TempoEvent:
    """Tempo span event (log)."""
    name: str
    timestamp_ns: int
    attributes: List[TempoAttribute] = field(default_factory=list)

    @property
    def datetime(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp_ns / 1_000_000_000)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "timeUnixNano": str(self.timestamp_ns),
            "attributes": [a.to_dict() for a in self.attributes],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TempoEvent:
        return cls(
            name=data.get("name", ""),
            timestamp_ns=int(data.get("timeUnixNano", 0)),
            attributes=[TempoAttribute.from_dict(a) for a in data.get("attributes", [])],
        )


@dataclass
class TempoResource:
    """Tempo resource representing the entity producing spans."""
    attributes: List[TempoAttribute] = field(default_factory=list)

    @property
    def service_name(self) -> str:
        for attr in self.attributes:
            if attr.key == "service.name":
                return str(attr.value)
        return "unknown"

    def get_attribute(self, key: str) -> Optional[Any]:
        for attr in self.attributes:
            if attr.key == key:
                return attr.value
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {"attributes": [a.to_dict() for a in self.attributes]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TempoResource:
        return cls(
            attributes=[TempoAttribute.from_dict(a) for a in data.get("attributes", [])]
        )


@dataclass
class TempoSpan:
    """A single Tempo span."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    name: str
    kind: SpanKind
    start_time_ns: int
    end_time_ns: int
    status: SpanStatus = SpanStatus.UNSET
    status_message: str = ""
    attributes: List[TempoAttribute] = field(default_factory=list)
    events: List[TempoEvent] = field(default_factory=list)
    resource: Optional[TempoResource] = None

    @property
    def start_time(self) -> datetime:
        return datetime.fromtimestamp(self.start_time_ns / 1_000_000_000)

    @property
    def end_time(self) -> datetime:
        return datetime.fromtimestamp(self.end_time_ns / 1_000_000_000)

    @property
    def duration_ns(self) -> int:
        return self.end_time_ns - self.start_time_ns

    @property
    def duration_ms(self) -> float:
        return self.duration_ns / 1_000_000

    @property
    def duration_seconds(self) -> float:
        return self.duration_ns / 1_000_000_000

    @property
    def is_error(self) -> bool:
        return self.status == SpanStatus.ERROR

    @property
    def is_root(self) -> bool:
        return self.parent_span_id is None or self.parent_span_id == ""

    @property
    def service_name(self) -> str:
        if self.resource:
            return self.resource.service_name
        return "unknown"

    def get_attribute(self, key: str) -> Optional[Any]:
        for attr in self.attributes:
            if attr.key == key:
                return attr.value
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "traceId": self.trace_id,
            "spanId": self.span_id,
            "parentSpanId": self.parent_span_id or "",
            "name": self.name,
            "kind": self.kind.value,
            "startTimeUnixNano": str(self.start_time_ns),
            "endTimeUnixNano": str(self.end_time_ns),
            "status": {"code": self.status.value, "message": self.status_message},
            "attributes": [a.to_dict() for a in self.attributes],
            "events": [e.to_dict() for e in self.events],
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        resource: Optional[TempoResource] = None
    ) -> TempoSpan:
        status_data = data.get("status", {})
        status_code = status_data.get("code", "UNSET")
        if isinstance(status_code, int):
            status_map = {0: SpanStatus.UNSET, 1: SpanStatus.OK, 2: SpanStatus.ERROR}
            status = status_map.get(status_code, SpanStatus.UNSET)
        else:
            try:
                status = SpanStatus(status_code)
            except ValueError:
                status = SpanStatus.UNSET

        kind_value = data.get("kind", "SPAN_KIND_UNSPECIFIED")
        if isinstance(kind_value, int):
            kind_map = {
                0: SpanKind.UNSPECIFIED,
                1: SpanKind.INTERNAL,
                2: SpanKind.SERVER,
                3: SpanKind.CLIENT,
                4: SpanKind.PRODUCER,
                5: SpanKind.CONSUMER,
            }
            kind = kind_map.get(kind_value, SpanKind.UNSPECIFIED)
        else:
            try:
                kind = SpanKind(kind_value)
            except ValueError:
                kind = SpanKind.UNSPECIFIED

        return cls(
            trace_id=data.get("traceId", ""),
            span_id=data.get("spanId", ""),
            parent_span_id=data.get("parentSpanId") or None,
            name=data.get("name", ""),
            kind=kind,
            start_time_ns=int(data.get("startTimeUnixNano", 0)),
            end_time_ns=int(data.get("endTimeUnixNano", 0)),
            status=status,
            status_message=status_data.get("message", ""),
            attributes=[TempoAttribute.from_dict(a) for a in data.get("attributes", [])],
            events=[TempoEvent.from_dict(e) for e in data.get("events", [])],
            resource=resource,
        )


@dataclass
class TempoTrace:
    """A complete Tempo trace."""
    trace_id: str
    spans: List[TempoSpan] = field(default_factory=list)

    @property
    def root_span(self) -> Optional[TempoSpan]:
        for span in self.spans:
            if span.is_root:
                return span
        return self.spans[0] if self.spans else None

    @property
    def service_name(self) -> str:
        root = self.root_span
        return root.service_name if root else "unknown"

    @property
    def operation_name(self) -> str:
        root = self.root_span
        return root.name if root else "unknown"

    @property
    def duration_ms(self) -> float:
        if not self.spans:
            return 0.0
        start = min(s.start_time_ns for s in self.spans)
        end = max(s.end_time_ns for s in self.spans)
        return (end - start) / 1_000_000

    @property
    def start_time(self) -> Optional[datetime]:
        if not self.spans:
            return None
        min_start = min(s.start_time_ns for s in self.spans)
        return datetime.fromtimestamp(min_start / 1_000_000_000)

    @property
    def span_count(self) -> int:
        return len(self.spans)

    @property
    def error_count(self) -> int:
        return sum(1 for s in self.spans if s.is_error)

    @property
    def has_errors(self) -> bool:
        return self.error_count > 0

    def get_spans_by_service(self, service_name: str) -> List[TempoSpan]:
        return [s for s in self.spans if s.service_name == service_name]

    def get_spans_by_name(self, name: str) -> List[TempoSpan]:
        return [s for s in self.spans if s.name == name]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "traceId": self.trace_id,
            "spans": [s.to_dict() for s in self.spans],
        }

    @classmethod
    def from_otlp_response(cls, data: Dict[str, Any]) -> TempoTrace:
        """Parse trace from OTLP format response."""
        spans = []
        trace_id = ""

        for batch in data.get("batches", []):
            resource = None
            resource_data = batch.get("resource", {})
            if resource_data:
                resource = TempoResource.from_dict(resource_data)

            for scope_spans in batch.get("scopeSpans", []):
                for span_data in scope_spans.get("spans", []):
                    span = TempoSpan.from_dict(span_data, resource)
                    spans.append(span)
                    if not trace_id:
                        trace_id = span.trace_id

        return cls(trace_id=trace_id, spans=spans)


@dataclass
class TempoSearchResult:
    """Result from a Tempo search query."""
    traces: List[TempoTrace] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TempoMetrics:
    """Aggregated span metrics from Tempo."""
    service: str
    span_name: str
    span_kind: SpanKind
    call_count: int = 0
    error_count: int = 0
    duration_p50_ms: float = 0.0
    duration_p95_ms: float = 0.0
    duration_p99_ms: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    @property
    def error_rate(self) -> float:
        if self.call_count == 0:
            return 0.0
        return self.error_count / self.call_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "service": self.service,
            "spanName": self.span_name,
            "spanKind": self.span_kind.value,
            "callCount": self.call_count,
            "errorCount": self.error_count,
            "durationP50Ms": self.duration_p50_ms,
            "durationP95Ms": self.duration_p95_ms,
            "durationP99Ms": self.duration_p99_ms,
            "errorRate": self.error_rate,
        }


@dataclass
class TempoConfig:
    """Tempo client configuration."""
    base_url: str = "http://localhost:3200"
    timeout_seconds: float = 30.0
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    verify_ssl: bool = True
    auth_token: Optional[str] = None
    org_id: Optional[str] = None  # For multi-tenant Tempo
    namespace: str = "consciousness"


class TempoClient:
    """Client for Grafana Tempo tracing backend.

    Thread-safe with connection pooling, retry logic,
    and Prometheus metrics instrumentation.

    Usage:
        config = TempoConfig(base_url="http://tempo:3200")
        client = TempoClient(config)

        # Get trace by ID
        trace = client.get_trace("abc123def456")

        # Search with TraceQL
        results = client.search_traceql(
            query='{span.http.status_code >= 500}',
            start=datetime.now() - timedelta(hours=1),
            limit=20,
        )

        # Get span metrics
        metrics = client.get_span_metrics("api-gateway")
    """

    def __init__(self, config: Optional[TempoConfig] = None):
        self.config = config or TempoConfig()
        self._session: Optional[requests.Session] = None
        self._lock = threading.Lock()

        # Metrics
        ns = self.config.namespace
        self.requests_total = Counter(
            f"{ns}_tempo_requests_total",
            "Total Tempo API requests",
            ["endpoint", "status"],
        )
        self.request_latency = Histogram(
            f"{ns}_tempo_request_latency_seconds",
            "Tempo API request latency",
            ["endpoint"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        )
        self.traces_fetched = Counter(
            f"{ns}_tempo_traces_fetched_total",
            "Total traces fetched from Tempo",
        )
        self.connection_errors = Counter(
            f"{ns}_tempo_connection_errors_total",
            "Tempo connection errors",
        )

    @property
    def session(self) -> requests.Session:
        """Get or create HTTP session."""
        if self._session is None:
            with self._lock:
                if self._session is None:
                    self._session = requests.Session()
                    headers = {}
                    if self.config.auth_token:
                        headers["Authorization"] = f"Bearer {self.config.auth_token}"
                    if self.config.org_id:
                        headers["X-Scope-OrgID"] = self.config.org_id
                    self._session.headers.update(headers)
        return self._session

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        accept: str = "application/json",
    ) -> Union[Dict[str, Any], bytes]:
        """Make HTTP request with retry logic.

        Args:
            method: HTTP method
            endpoint: API endpoint path
            params: Query parameters
            accept: Accept header

        Returns:
            Response data

        Raises:
            TempoClientError: On request failure
        """
        url = f"{self.config.base_url.rstrip('/')}{endpoint}"
        last_error = None

        for attempt in range(self.config.max_retries):
            start_time = time.time()
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    headers={"Accept": accept},
                    timeout=self.config.timeout_seconds,
                    verify=self.config.verify_ssl,
                )

                duration = time.time() - start_time
                self.request_latency.labels(endpoint=endpoint).observe(duration)

                if response.ok:
                    self.requests_total.labels(endpoint=endpoint, status="success").inc()
                    if accept == "application/json":
                        return response.json()
                    return response.content
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

            if attempt < self.config.max_retries - 1:
                time.sleep(self.config.retry_delay_seconds * (attempt + 1))

        raise TempoClientError(f"Request failed after {self.config.max_retries} attempts: {last_error}")

    def get_trace(self, trace_id: str) -> Optional[TempoTrace]:
        """Get a trace by ID.

        Args:
            trace_id: Trace ID (hex string)

        Returns:
            TempoTrace or None if not found
        """
        try:
            response = self._request("GET", f"/api/traces/{trace_id}")
            if response:
                self.traces_fetched.inc()
                return TempoTrace.from_otlp_response(response)
        except TempoClientError as e:
            logger.warning(f"Failed to get trace {trace_id}: {e}")
        return None

    def search_traceql(
        self,
        query: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 20,
        spss: int = 3,  # Spans per span set
    ) -> TempoSearchResult:
        """Search traces using TraceQL.

        Args:
            query: TraceQL query (e.g., '{span.http.status_code >= 500}')
            start: Start time (default: 1 hour ago)
            end: End time (default: now)
            limit: Maximum traces
            spss: Spans per span set

        Returns:
            TempoSearchResult with matching traces
        """
        params: Dict[str, Any] = {
            "q": query,
            "limit": limit,
            "spss": spss,
        }

        if start:
            params["start"] = int(start.timestamp())
        else:
            params["start"] = int((datetime.now() - timedelta(hours=1)).timestamp())

        if end:
            params["end"] = int(end.timestamp())
        else:
            params["end"] = int(datetime.now().timestamp())

        response = self._request("GET", "/api/search", params=params)

        traces = []
        for trace_data in response.get("traces", []):
            trace_id = trace_data.get("traceID", "")
            # Minimal trace info from search - get full trace if needed
            trace = TempoTrace(trace_id=trace_id)
            # Parse root span info if available
            if "rootServiceName" in trace_data:
                root_span = TempoSpan(
                    trace_id=trace_id,
                    span_id="",
                    parent_span_id=None,
                    name=trace_data.get("rootTraceName", ""),
                    kind=SpanKind.SERVER,
                    start_time_ns=int(trace_data.get("startTimeUnixNano", 0)),
                    end_time_ns=int(trace_data.get("startTimeUnixNano", 0)) +
                               int(trace_data.get("durationMs", 0)) * 1_000_000,
                    resource=TempoResource(attributes=[
                        TempoAttribute(key="service.name", value=trace_data.get("rootServiceName", ""))
                    ]),
                )
                trace.spans.append(root_span)
            traces.append(trace)

        self.traces_fetched.inc(len(traces))

        return TempoSearchResult(
            traces=traces,
            metrics=response.get("metrics", {}),
        )

    def search_tags(
        self,
        scope: str = "span",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[str]:
        """Get available tag keys.

        Args:
            scope: Tag scope ('span', 'resource', or 'intrinsic')
            start: Start time
            end: End time

        Returns:
            List of tag keys
        """
        params: Dict[str, Any] = {"scope": scope}

        if start:
            params["start"] = int(start.timestamp())
        if end:
            params["end"] = int(end.timestamp())

        response = self._request("GET", "/api/v2/search/tags", params=params)
        return [tag.get("name", "") for tag in response.get("scopes", [{}])[0].get("tags", [])]

    def search_tag_values(
        self,
        tag: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[str]:
        """Get values for a tag.

        Args:
            tag: Tag key
            start: Start time
            end: End time

        Returns:
            List of tag values
        """
        params: Dict[str, Any] = {}

        if start:
            params["start"] = int(start.timestamp())
        if end:
            params["end"] = int(end.timestamp())

        response = self._request("GET", f"/api/v2/search/tag/{tag}/values", params=params)
        return [v.get("value", "") for v in response.get("tagValues", [])]

    def get_service_graph(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Get service dependency graph.

        Args:
            start: Start time (default: 1 hour ago)
            end: End time (default: now)

        Returns:
            Service graph data with nodes and edges
        """
        params: Dict[str, Any] = {}

        if start:
            params["start"] = int(start.timestamp())
        else:
            params["start"] = int((datetime.now() - timedelta(hours=1)).timestamp())

        if end:
            params["end"] = int(end.timestamp())
        else:
            params["end"] = int(datetime.now().timestamp())

        try:
            response = self._request("GET", "/api/metrics/query_range", params={
                **params,
                "query": "traces_service_graph_request_total",
            })
            return response
        except TempoClientError:
            # Fallback for older Tempo versions
            return {"nodes": [], "edges": []}

    def get_span_metrics(
        self,
        service: Optional[str] = None,
        span_name: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[TempoMetrics]:
        """Get aggregated span metrics.

        Args:
            service: Filter by service
            span_name: Filter by span name
            start: Start time
            end: End time

        Returns:
            List of aggregated metrics
        """
        # This requires Tempo with span metrics enabled
        # Returns mock data if not available
        metrics = []

        try:
            # Query span metrics using metrics-generator data
            params: Dict[str, Any] = {}
            if start:
                params["start"] = int(start.timestamp())
            if end:
                params["end"] = int(end.timestamp())

            # Try to get metrics from Tempo's span metrics
            query = "traces_spanmetrics_latency_bucket"
            if service:
                query = f'{query}{{service_name="{service}"}}'

            response = self._request("GET", "/api/metrics/query", params={
                **params,
                "query": query,
            })

            # Parse metrics from response
            for result in response.get("data", {}).get("result", []):
                metric = result.get("metric", {})
                metrics.append(TempoMetrics(
                    service=metric.get("service_name", "unknown"),
                    span_name=metric.get("span_name", ""),
                    span_kind=SpanKind.UNSPECIFIED,
                ))

        except TempoClientError:
            logger.debug("Span metrics not available from Tempo")

        return metrics

    def health_check(self) -> bool:
        """Check if Tempo is healthy.

        Returns:
            True if healthy
        """
        try:
            response = self._request("GET", "/ready")
            return True
        except TempoClientError:
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get Tempo status information.

        Returns:
            Status dictionary
        """
        try:
            return self._request("GET", "/status/config")
        except TempoClientError:
            return {}

    def close(self):
        """Close the client session."""
        with self._lock:
            if self._session:
                self._session.close()
                self._session = None

    def __enter__(self) -> TempoClient:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class TempoClientError(Exception):
    """Tempo client error."""
    pass
