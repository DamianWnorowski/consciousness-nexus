"""
W3C Trace Context Propagation Module

Provides:
- HTTP header injection/extraction
- Context carrier utilities
- Integration with popular HTTP clients (httpx, aiohttp, requests)
- WebSocket trace propagation
- gRPC metadata propagation
"""

from typing import Any, Callable, Dict, Mapping, Optional, TypeVar
from functools import wraps
import logging

from opentelemetry import trace
from opentelemetry.context import Context, get_current, attach, detach
from opentelemetry.propagate import inject, extract
from opentelemetry.propagators.textmap import Getter, Setter
from opentelemetry.trace import Span, SpanKind, Status, StatusCode

logger = logging.getLogger(__name__)

T = TypeVar('T')


class DictGetter(Getter[Dict[str, str]]):
    """Getter implementation for dict-like carriers."""

    def get(self, carrier: Dict[str, str], key: str) -> Optional[str]:
        return carrier.get(key)

    def keys(self, carrier: Dict[str, str]) -> list:
        return list(carrier.keys())


class DictSetter(Setter[Dict[str, str]]):
    """Setter implementation for dict-like carriers."""

    def set(self, carrier: Dict[str, str], key: str, value: str) -> None:
        carrier[key] = value


# Default instances
_dict_getter = DictGetter()
_dict_setter = DictSetter()


def inject_trace_context(headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """
    Inject current trace context into HTTP headers.

    Adds W3C Trace Context headers:
    - traceparent: trace-id, parent-id, trace-flags
    - tracestate: vendor-specific state (optional)

    Args:
        headers: Existing headers dict (optional)

    Returns:
        Headers dict with trace context injected

    Example:
        headers = inject_trace_context()
        response = httpx.get("http://service/api", headers=headers)
    """
    if headers is None:
        headers = {}

    inject(headers, setter=_dict_setter)
    return headers


def extract_trace_context(
    headers: Mapping[str, str],
) -> Context:
    """
    Extract trace context from incoming HTTP headers.

    Parses W3C Trace Context headers and creates
    a context for continuing the distributed trace.

    Args:
        headers: Incoming request headers

    Returns:
        Context with extracted trace information

    Example:
        ctx = extract_trace_context(request.headers)
        with tracer.start_as_current_span("handler", context=ctx):
            # handle request
    """
    # Convert headers to dict if needed (e.g., from starlette Headers)
    if not isinstance(headers, dict):
        headers = dict(headers)

    return extract(headers, getter=_dict_getter)


def get_traceparent() -> Optional[str]:
    """
    Get the traceparent header value for the current span.

    Returns:
        Traceparent string or None if no active span
    """
    span = trace.get_current_span()
    if not span.is_recording():
        return None

    ctx = span.get_span_context()
    # Format: {version}-{trace-id}-{parent-id}-{trace-flags}
    return f"00-{ctx.trace_id:032x}-{ctx.span_id:016x}-{ctx.trace_flags:02x}"


def get_trace_id() -> Optional[str]:
    """Get current trace ID as hex string."""
    span = trace.get_current_span()
    if span.is_recording():
        return format(span.get_span_context().trace_id, "032x")
    return None


def get_span_id() -> Optional[str]:
    """Get current span ID as hex string."""
    span = trace.get_current_span()
    if span.is_recording():
        return format(span.get_span_context().span_id, "016x")
    return None


class PropagatedContext:
    """
    Context manager for handling incoming trace context.

    Automatically extracts context from headers, creates
    a child span, and ensures proper cleanup.

    Example:
        async def handler(request):
            with PropagatedContext(request.headers, "handle_request"):
                # Your handler code
                return response
    """

    def __init__(
        self,
        headers: Mapping[str, str],
        span_name: str,
        kind: SpanKind = SpanKind.SERVER,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        self.headers = headers
        self.span_name = span_name
        self.kind = kind
        self.attributes = attributes or {}
        self._token = None
        self._span: Optional[Span] = None

    def __enter__(self) -> Span:
        # Extract parent context
        ctx = extract_trace_context(self.headers)

        # Attach the context
        self._token = attach(ctx)

        # Start span with extracted context as parent
        tracer = trace.get_tracer(__name__)
        self._span = tracer.start_span(
            self.span_name,
            kind=self.kind,
            context=ctx,
            attributes=self.attributes,
        )

        # Make it the current span
        trace.use_span(self._span, end_on_exit=False).__enter__()

        return self._span

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._span:
            if exc_type:
                self._span.set_status(Status(StatusCode.ERROR, str(exc_val)))
                self._span.record_exception(exc_val)
            else:
                self._span.set_status(Status(StatusCode.OK))
            self._span.end()

        if self._token:
            detach(self._token)

        return False


# HTTP Client Instrumentation Helpers

def traced_httpx_client():
    """
    Create an httpx client with automatic trace propagation.

    Returns:
        httpx.AsyncClient with event hooks for trace injection

    Example:
        async with traced_httpx_client() as client:
            response = await client.get("http://service/api")
    """
    import httpx

    def inject_headers(request: httpx.Request):
        """Event hook to inject trace context."""
        headers = dict(request.headers)
        inject(headers, setter=_dict_setter)
        request.headers.update(headers)

    return httpx.AsyncClient(
        event_hooks={"request": [inject_headers]}
    )


def traced_aiohttp_session():
    """
    Create an aiohttp session with trace propagation.

    Returns context manager for traced aiohttp session.

    Example:
        async with traced_aiohttp_session() as session:
            async with session.get("http://service/api") as response:
                data = await response.json()
    """
    import aiohttp

    class TracedSession:
        def __init__(self):
            self._session = None

        async def __aenter__(self):
            trace_config = aiohttp.TraceConfig()

            async def on_request_start(session, ctx, params):
                headers = dict(params.headers)
                inject(headers, setter=_dict_setter)
                params.headers.update(headers)

            trace_config.on_request_start.append(on_request_start)
            self._session = aiohttp.ClientSession(trace_configs=[trace_config])
            return self._session

        async def __aexit__(self, *args):
            if self._session:
                await self._session.close()

    return TracedSession()


def propagate_to_websocket(websocket_headers: Dict[str, str]) -> Dict[str, str]:
    """
    Inject trace context for WebSocket connections.

    Args:
        websocket_headers: WebSocket connection headers

    Returns:
        Headers with trace context
    """
    return inject_trace_context(websocket_headers)


def propagate_to_grpc_metadata(metadata: Optional[list] = None) -> list:
    """
    Inject trace context into gRPC metadata.

    Args:
        metadata: Existing gRPC metadata (list of tuples)

    Returns:
        Metadata with trace context

    Example:
        metadata = propagate_to_grpc_metadata()
        response = stub.SomeMethod(request, metadata=metadata)
    """
    if metadata is None:
        metadata = []

    headers = {}
    inject(headers, setter=_dict_setter)

    for key, value in headers.items():
        metadata.append((key.lower(), value))

    return metadata


def extract_from_grpc_metadata(metadata) -> Context:
    """
    Extract trace context from gRPC metadata.

    Args:
        metadata: gRPC invocation metadata

    Returns:
        Extracted context
    """
    # Convert metadata to dict
    headers = {}
    for key, value in metadata:
        headers[key] = value

    return extract(headers, getter=_dict_getter)


# Decorator for automatic context propagation
def with_propagation(
    extract_headers: Callable[[Any], Mapping[str, str]],
    span_name: Optional[str] = None,
    kind: SpanKind = SpanKind.SERVER,
):
    """
    Decorator for automatic trace context propagation.

    Args:
        extract_headers: Function to extract headers from first argument
        span_name: Span name (defaults to function name)
        kind: Span kind

    Example:
        @with_propagation(lambda req: req.headers)
        async def handle_request(request):
            # Automatically in context of parent trace
            return response
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            headers = extract_headers(args[0]) if args else {}
            name = span_name or func.__name__

            with PropagatedContext(headers, name, kind):
                return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            headers = extract_headers(args[0]) if args else {}
            name = span_name or func.__name__

            with PropagatedContext(headers, name, kind):
                return func(*args, **kwargs)

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
