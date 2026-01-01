"""Invocation Tracing

AWS Lambda invocation tracing with OpenTelemetry:
- Distributed tracing with W3C Trace Context
- X-Ray integration
- Span creation and context propagation
- Automatic instrumentation
"""

from __future__ import annotations

import os
import time
import uuid
import functools
import threading
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from contextlib import contextmanager
from enum import Enum

from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class SpanKind(int, Enum):
    """OpenTelemetry span kinds."""
    INTERNAL = 0
    SERVER = 1
    CLIENT = 2
    PRODUCER = 3
    CONSUMER = 4


class SpanStatus(str, Enum):
    """Span status codes."""
    UNSET = "UNSET"
    OK = "OK"
    ERROR = "ERROR"


@dataclass
class TracingConfig:
    """Configuration for invocation tracing."""
    # Service identification
    service_name: str = field(
        default_factory=lambda: os.environ.get("AWS_LAMBDA_FUNCTION_NAME", "lambda")
    )
    service_version: str = field(
        default_factory=lambda: os.environ.get("AWS_LAMBDA_FUNCTION_VERSION", "$LATEST")
    )
    # Sampling
    sample_rate: float = 1.0  # 1.0 = 100% sampling
    # X-Ray integration
    xray_enabled: bool = True
    xray_daemon_address: str = field(
        default_factory=lambda: os.environ.get("AWS_XRAY_DAEMON_ADDRESS", "127.0.0.1:2000")
    )
    # Context propagation
    propagation_format: str = "w3c"  # w3c, xray, b3
    # Exporter
    export_enabled: bool = True
    export_endpoint: str = "http://localhost:4317"
    # Features
    capture_request: bool = True
    capture_response: bool = False  # Be careful with sensitive data
    max_attribute_length: int = 1024


@dataclass
class InvocationContext:
    """Lambda invocation context for tracing."""
    request_id: str
    function_name: str
    function_version: str
    function_arn: str
    memory_limit_mb: int
    log_group: str
    log_stream: str
    remaining_time_ms: int
    # Trace context
    trace_id: str = ""
    span_id: str = ""
    parent_span_id: str = ""
    sampled: bool = True
    # X-Ray
    xray_trace_id: Optional[str] = None
    # Timing
    start_time: float = field(default_factory=time.time)
    deadline_ms: int = 0

    @classmethod
    def from_lambda_context(cls, context: Any) -> "InvocationContext":
        """Create from Lambda context object.

        Args:
            context: Lambda context object

        Returns:
            InvocationContext
        """
        # Parse X-Ray trace header if present
        xray_trace_id = os.environ.get("_X_AMZN_TRACE_ID")

        return cls(
            request_id=getattr(context, "aws_request_id", str(uuid.uuid4())),
            function_name=getattr(context, "function_name", os.environ.get("AWS_LAMBDA_FUNCTION_NAME", "unknown")),
            function_version=getattr(context, "function_version", os.environ.get("AWS_LAMBDA_FUNCTION_VERSION", "$LATEST")),
            function_arn=getattr(context, "invoked_function_arn", ""),
            memory_limit_mb=int(getattr(context, "memory_limit_in_mb", os.environ.get("AWS_LAMBDA_FUNCTION_MEMORY_SIZE", 128))),
            log_group=getattr(context, "log_group_name", ""),
            log_stream=getattr(context, "log_stream_name", ""),
            remaining_time_ms=getattr(context, "get_remaining_time_in_millis", lambda: 0)(),
            xray_trace_id=xray_trace_id,
        )


@dataclass
class InvocationSpan:
    """Represents a span in the invocation trace."""
    name: str
    trace_id: str
    span_id: str
    parent_span_id: str = ""
    kind: SpanKind = SpanKind.INTERNAL
    status: SpanStatus = SpanStatus.UNSET
    # Timing
    start_time_ns: int = 0
    end_time_ns: int = 0
    # Data
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    links: List[Dict[str, str]] = field(default_factory=list)
    # Error
    error: Optional[str] = None
    error_type: Optional[str] = None

    @property
    def duration_ns(self) -> int:
        return self.end_time_ns - self.start_time_ns if self.end_time_ns else 0

    @property
    def duration_ms(self) -> float:
        return self.duration_ns / 1_000_000

    def set_attribute(self, key: str, value: Any):
        """Set a span attribute."""
        self.attributes[key] = value

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add an event to the span."""
        self.events.append({
            "name": name,
            "timestamp_ns": time.time_ns(),
            "attributes": attributes or {},
        })

    def record_exception(self, exception: Exception):
        """Record an exception."""
        self.status = SpanStatus.ERROR
        self.error = str(exception)
        self.error_type = type(exception).__name__
        self.add_event("exception", {
            "exception.type": self.error_type,
            "exception.message": self.error,
        })

    def end(self, status: Optional[SpanStatus] = None):
        """End the span."""
        self.end_time_ns = time.time_ns()
        if status:
            self.status = status
        elif self.status == SpanStatus.UNSET:
            self.status = SpanStatus.OK

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "kind": self.kind.value,
            "start_time": self.start_time_ns,
            "end_time": self.end_time_ns,
            "attributes": self.attributes,
            "events": self.events,
            "links": self.links,
            "status": {
                "code": 1 if self.status == SpanStatus.OK else 2 if self.status == SpanStatus.ERROR else 0,
                "message": self.error or "",
            },
        }


class InvocationTracer:
    """Lambda invocation tracer with OpenTelemetry support.

    Provides distributed tracing for Lambda functions:
    - Automatic trace context extraction/injection
    - Span creation and management
    - X-Ray integration
    - Export to OTLP backends

    Usage:
        tracer = InvocationTracer()

        @tracer.trace
        def handler(event, context):
            # Create child spans
            with tracer.span("database_query") as span:
                span.set_attribute("db.statement", "SELECT ...")
                result = db.query(...)
            return result

        # Or manual tracing
        tracer.start_invocation(context)
        try:
            result = process(event)
        finally:
            tracer.end_invocation()
    """

    # Thread-local storage for active spans
    _local = threading.local()

    def __init__(
        self,
        config: Optional[TracingConfig] = None,
        namespace: str = "consciousness",
    ):
        self.config = config or TracingConfig()
        self.namespace = namespace

        self._spans: List[InvocationSpan] = []
        self._current_context: Optional[InvocationContext] = None
        self._lock = threading.Lock()

        # Setup metrics
        self._setup_metrics()

    def _setup_metrics(self):
        """Initialize Prometheus metrics."""
        self.invocations_total = Counter(
            f"{self.namespace}_lambda_invocations_total",
            "Total Lambda invocations",
            ["function_name", "status"],
        )

        self.invocation_duration = Histogram(
            f"{self.namespace}_lambda_invocation_duration_seconds",
            "Lambda invocation duration",
            ["function_name"],
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
        )

        self.spans_created = Counter(
            f"{self.namespace}_lambda_spans_created_total",
            "Total spans created",
            ["function_name", "span_kind"],
        )

        self.active_spans = Gauge(
            f"{self.namespace}_lambda_active_spans",
            "Currently active spans",
            ["function_name"],
        )

    @property
    def current_span(self) -> Optional[InvocationSpan]:
        """Get the current active span."""
        stack = getattr(self._local, "span_stack", [])
        return stack[-1] if stack else None

    @property
    def current_trace_id(self) -> str:
        """Get current trace ID."""
        if self._current_context:
            return self._current_context.trace_id
        return ""

    def _generate_trace_id(self) -> str:
        """Generate a new trace ID (32 hex chars)."""
        return uuid.uuid4().hex

    def _generate_span_id(self) -> str:
        """Generate a new span ID (16 hex chars)."""
        return uuid.uuid4().hex[:16]

    def _should_sample(self) -> bool:
        """Determine if this trace should be sampled."""
        import random
        return random.random() < self.config.sample_rate

    def _extract_trace_context(
        self,
        event: Dict[str, Any],
    ) -> tuple[str, str, bool]:
        """Extract trace context from incoming event.

        Args:
            event: Lambda event

        Returns:
            Tuple of (trace_id, parent_span_id, sampled)
        """
        headers = event.get("headers", {}) or {}
        # Normalize header keys to lowercase
        headers = {k.lower(): v for k, v in headers.items()}

        # Try W3C Trace Context
        if "traceparent" in headers:
            return self._parse_traceparent(headers["traceparent"])

        # Try X-Ray
        xray_header = os.environ.get("_X_AMZN_TRACE_ID", "")
        if xray_header:
            return self._parse_xray_header(xray_header)

        # Try B3
        if "x-b3-traceid" in headers:
            return (
                headers["x-b3-traceid"],
                headers.get("x-b3-spanid", ""),
                headers.get("x-b3-sampled", "1") == "1",
            )

        # No context found, generate new
        return self._generate_trace_id(), "", self._should_sample()

    def _parse_traceparent(self, header: str) -> tuple[str, str, bool]:
        """Parse W3C traceparent header.

        Format: version-trace_id-parent_id-flags
        Example: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01
        """
        try:
            parts = header.split("-")
            if len(parts) >= 4:
                trace_id = parts[1]
                parent_span_id = parts[2]
                sampled = int(parts[3], 16) & 1 == 1
                return trace_id, parent_span_id, sampled
        except Exception as e:
            logger.warning(f"Failed to parse traceparent: {e}")

        return self._generate_trace_id(), "", self._should_sample()

    def _parse_xray_header(self, header: str) -> tuple[str, str, bool]:
        """Parse X-Ray trace header.

        Format: Root=1-xxx-yyy;Parent=zzz;Sampled=1
        """
        trace_id = ""
        parent_id = ""
        sampled = True

        try:
            for part in header.split(";"):
                if part.startswith("Root="):
                    # Convert X-Ray format to OTel
                    xray_root = part[5:]
                    parts = xray_root.split("-")
                    if len(parts) >= 3:
                        trace_id = parts[1] + parts[2]
                elif part.startswith("Parent="):
                    parent_id = part[7:]
                elif part.startswith("Sampled="):
                    sampled = part[8:] == "1"
        except Exception as e:
            logger.warning(f"Failed to parse X-Ray header: {e}")

        if not trace_id:
            trace_id = self._generate_trace_id()

        return trace_id, parent_id, sampled

    def _inject_trace_context(
        self,
        headers: Dict[str, str],
    ) -> Dict[str, str]:
        """Inject trace context into outgoing headers.

        Args:
            headers: Headers to inject into

        Returns:
            Updated headers
        """
        if not self._current_context:
            return headers

        current_span = self.current_span

        # W3C Trace Context
        if self.config.propagation_format == "w3c":
            sampled_flag = "01" if self._current_context.sampled else "00"
            span_id = current_span.span_id if current_span else self._generate_span_id()
            headers["traceparent"] = (
                f"00-{self._current_context.trace_id}-{span_id}-{sampled_flag}"
            )

        # B3
        elif self.config.propagation_format == "b3":
            span_id = current_span.span_id if current_span else self._generate_span_id()
            headers["X-B3-TraceId"] = self._current_context.trace_id
            headers["X-B3-SpanId"] = span_id
            headers["X-B3-Sampled"] = "1" if self._current_context.sampled else "0"

        return headers

    def start_invocation(
        self,
        context: Any,
        event: Optional[Dict[str, Any]] = None,
    ) -> InvocationContext:
        """Start tracing a Lambda invocation.

        Args:
            context: Lambda context object
            event: Lambda event (for trace context extraction)

        Returns:
            InvocationContext
        """
        # Create invocation context
        inv_context = InvocationContext.from_lambda_context(context)

        # Extract trace context from event
        if event:
            trace_id, parent_span_id, sampled = self._extract_trace_context(event)
            inv_context.trace_id = trace_id
            inv_context.parent_span_id = parent_span_id
            inv_context.sampled = sampled
        else:
            inv_context.trace_id = self._generate_trace_id()
            inv_context.sampled = self._should_sample()

        self._current_context = inv_context

        # Create root span
        root_span = self._create_span(
            name=f"{inv_context.function_name}",
            kind=SpanKind.SERVER,
            parent_span_id=inv_context.parent_span_id,
        )

        # Add Lambda-specific attributes
        root_span.set_attribute("faas.trigger", "http")
        root_span.set_attribute("faas.invocation_id", inv_context.request_id)
        root_span.set_attribute("faas.name", inv_context.function_name)
        root_span.set_attribute("faas.version", inv_context.function_version)
        root_span.set_attribute("cloud.provider", "aws")
        root_span.set_attribute("cloud.platform", "aws_lambda")
        root_span.set_attribute("cloud.region", os.environ.get("AWS_REGION", "unknown"))

        # Capture request if configured
        if self.config.capture_request and event:
            self._capture_request(root_span, event)

        # Push to span stack
        if not hasattr(self._local, "span_stack"):
            self._local.span_stack = []
        self._local.span_stack.append(root_span)

        self.active_spans.labels(
            function_name=inv_context.function_name
        ).set(len(self._local.span_stack))

        return inv_context

    def end_invocation(
        self,
        response: Optional[Any] = None,
        error: Optional[Exception] = None,
    ):
        """End the invocation trace.

        Args:
            response: Lambda response (optional)
            error: Exception if invocation failed
        """
        if not self._current_context:
            return

        # Get root span
        stack = getattr(self._local, "span_stack", [])
        if not stack:
            return

        root_span = stack[0]

        # Record error if present
        if error:
            root_span.record_exception(error)
            status = "error"
        else:
            root_span.status = SpanStatus.OK
            status = "success"

        # Capture response if configured
        if self.config.capture_response and response:
            self._capture_response(root_span, response)

        # End all spans in stack
        while stack:
            span = stack.pop()
            span.end()
            self._spans.append(span)

        # Record metrics
        duration = root_span.duration_ms / 1000
        self.invocations_total.labels(
            function_name=self._current_context.function_name,
            status=status,
        ).inc()
        self.invocation_duration.labels(
            function_name=self._current_context.function_name,
        ).observe(duration)

        self.active_spans.labels(
            function_name=self._current_context.function_name
        ).set(0)

        # Export spans
        if self.config.export_enabled:
            self._export_spans()

        self._current_context = None

    def _create_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        parent_span_id: str = "",
    ) -> InvocationSpan:
        """Create a new span.

        Args:
            name: Span name
            kind: Span kind
            parent_span_id: Parent span ID

        Returns:
            InvocationSpan
        """
        if not self._current_context:
            raise RuntimeError("No active invocation context")

        # Use current span as parent if not specified
        if not parent_span_id:
            current = self.current_span
            if current:
                parent_span_id = current.span_id

        span = InvocationSpan(
            name=name,
            trace_id=self._current_context.trace_id,
            span_id=self._generate_span_id(),
            parent_span_id=parent_span_id,
            kind=kind,
            start_time_ns=time.time_ns(),
        )

        self.spans_created.labels(
            function_name=self._current_context.function_name,
            span_kind=kind.name.lower(),
        ).inc()

        return span

    @contextmanager
    def span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """Context manager for creating a span.

        Args:
            name: Span name
            kind: Span kind
            attributes: Initial attributes

        Yields:
            InvocationSpan
        """
        span = self._create_span(name, kind)

        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)

        # Push to stack
        stack = getattr(self._local, "span_stack", [])
        stack.append(span)

        if self._current_context:
            self.active_spans.labels(
                function_name=self._current_context.function_name
            ).set(len(stack))

        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            raise
        finally:
            # Pop from stack
            if stack and stack[-1] is span:
                stack.pop()
            span.end()

            with self._lock:
                self._spans.append(span)

            if self._current_context:
                self.active_spans.labels(
                    function_name=self._current_context.function_name
                ).set(len(stack))

    def trace(
        self,
        name: Optional[str] = None,
        kind: SpanKind = SpanKind.SERVER,
    ) -> Callable[[F], F]:
        """Decorator for tracing Lambda handlers.

        Args:
            name: Custom span name
            kind: Span kind

        Returns:
            Decorated function
        """
        def decorator(func: F) -> F:
            @functools.wraps(func)
            def wrapper(event, context, *args, **kwargs):
                span_name = name or func.__name__
                self.start_invocation(context, event)

                try:
                    result = func(event, context, *args, **kwargs)
                    self.end_invocation(response=result)
                    return result
                except Exception as e:
                    self.end_invocation(error=e)
                    raise

            @functools.wraps(func)
            async def async_wrapper(event, context, *args, **kwargs):
                span_name = name or func.__name__
                self.start_invocation(context, event)

                try:
                    result = await func(event, context, *args, **kwargs)
                    self.end_invocation(response=result)
                    return result
                except Exception as e:
                    self.end_invocation(error=e)
                    raise

            import asyncio
            if asyncio.iscoroutinefunction(func):
                return async_wrapper  # type: ignore
            return wrapper  # type: ignore

        return decorator

    def _capture_request(self, span: InvocationSpan, event: Dict[str, Any]):
        """Capture request attributes from event."""
        # HTTP-specific attributes
        if "httpMethod" in event:
            span.set_attribute("http.method", event["httpMethod"])
        if "path" in event:
            span.set_attribute("http.target", event["path"])
        if "headers" in event and event["headers"]:
            headers = event["headers"]
            if "user-agent" in headers:
                span.set_attribute("http.user_agent", headers["user-agent"][:256])
            if "host" in headers:
                span.set_attribute("http.host", headers["host"])

        # API Gateway attributes
        if "requestContext" in event:
            ctx = event["requestContext"]
            if "stage" in ctx:
                span.set_attribute("faas.trigger.stage", ctx["stage"])
            if "apiId" in ctx:
                span.set_attribute("aws.api_gateway.api_id", ctx["apiId"])

    def _capture_response(self, span: InvocationSpan, response: Any):
        """Capture response attributes."""
        if isinstance(response, dict):
            if "statusCode" in response:
                span.set_attribute("http.status_code", response["statusCode"])

    def _export_spans(self):
        """Export collected spans."""
        with self._lock:
            spans = list(self._spans)
            self._spans.clear()

        if not spans:
            return

        # Convert to export format
        span_dicts = [span.to_dict() for span in spans]

        # Import exporter
        try:
            from .lambda_layer.exporter import LambdaOTLPExporter, ExportConfig

            exporter = LambdaOTLPExporter(
                config=ExportConfig(
                    endpoint=self.config.export_endpoint,
                    service_name=self.config.service_name,
                )
            )

            for span_dict in span_dicts:
                exporter.add_span(span_dict)

            exporter.flush()
        except Exception as e:
            logger.warning(f"Failed to export spans: {e}")

    def get_propagation_headers(self) -> Dict[str, str]:
        """Get headers for outgoing request propagation.

        Returns:
            Headers dict with trace context
        """
        return self._inject_trace_context({})

    def get_spans(self) -> List[InvocationSpan]:
        """Get collected spans.

        Returns:
            List of spans
        """
        with self._lock:
            return list(self._spans)
