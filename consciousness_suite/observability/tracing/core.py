"""
Distributed Tracing Module

Provides:
- ConsciousnessTracer wrapper for creating spans
- Decorators for automatic function tracing
- Context propagation utilities
- Span enrichment with consciousness-specific attributes
"""

from typing import Any, Callable, Dict, Optional, TypeVar, Union
from functools import wraps
import asyncio
import time
import logging

from opentelemetry import trace
from opentelemetry.trace import (
    Span,
    SpanKind,
    Status,
    StatusCode,
    Tracer,
    get_current_span,
    set_span_in_context,
)
from opentelemetry.trace.propagation import set_span_in_context
from opentelemetry.context import Context, get_current
from opentelemetry.semconv.trace import SpanAttributes

logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., Any])


class ConsciousnessTracer:
    """
    Enhanced tracer wrapper for Consciousness Nexus.

    Provides:
    - Automatic span creation with consciousness metadata
    - Error handling and status propagation
    - LLM call tracing with token tracking
    - Mesh routing trace decoration
    """

    def __init__(
        self,
        name: str = "consciousness-nexus",
        version: str = "1.0.0",
    ):
        self.name = name
        self.version = version
        self._tracer: Optional[Tracer] = None

    @property
    def tracer(self) -> Tracer:
        """Lazy initialization of tracer."""
        if self._tracer is None:
            self._tracer = trace.get_tracer(self.name, self.version)
        return self._tracer

    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
        context: Optional[Context] = None,
    ) -> Span:
        """
        Start a new span with consciousness metadata.

        Args:
            name: Span name (operation being traced)
            kind: Span kind (INTERNAL, SERVER, CLIENT, PRODUCER, CONSUMER)
            attributes: Additional span attributes
            context: Parent context (optional, uses current if not provided)

        Returns:
            Active span
        """
        attrs = {
            "consciousness.tracer.name": self.name,
            "consciousness.tracer.version": self.version,
        }
        if attributes:
            attrs.update(attributes)

        return self.tracer.start_span(
            name,
            kind=kind,
            attributes=attrs,
            context=context,
        )

    def start_as_current_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """
        Context manager for starting span as current.

        Usage:
            with tracer.start_as_current_span("operation") as span:
                # do work
                span.set_attribute("key", "value")
        """
        attrs = {
            "consciousness.tracer.name": self.name,
        }
        if attributes:
            attrs.update(attributes)

        return self.tracer.start_as_current_span(
            name,
            kind=kind,
            attributes=attrs,
        )

    def trace_llm_call(
        self,
        model: str,
        provider: str = "anthropic",
        operation: str = "completion",
    ):
        """
        Decorator for tracing LLM API calls.

        Automatically captures:
        - Model name and provider
        - Token counts (input/output)
        - Latency
        - Error details

        Usage:
            @tracer.trace_llm_call(model="claude-3-opus", provider="anthropic")
            async def call_claude(prompt: str) -> str:
                ...
        """
        def decorator(func: F) -> F:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                with self.start_as_current_span(
                    f"llm.{provider}.{operation}",
                    kind=SpanKind.CLIENT,
                    attributes={
                        "llm.model": model,
                        "llm.provider": provider,
                        "llm.operation": operation,
                    },
                ) as span:
                    start_time = time.perf_counter()
                    try:
                        result = await func(*args, **kwargs)

                        # Extract token counts if available
                        if hasattr(result, 'usage'):
                            span.set_attribute("llm.tokens.input", result.usage.input_tokens)
                            span.set_attribute("llm.tokens.output", result.usage.output_tokens)
                            span.set_attribute("llm.tokens.total", result.usage.total_tokens)

                        span.set_status(Status(StatusCode.OK))
                        return result
                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        raise
                    finally:
                        duration = time.perf_counter() - start_time
                        span.set_attribute("llm.duration_seconds", duration)

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                with self.start_as_current_span(
                    f"llm.{provider}.{operation}",
                    kind=SpanKind.CLIENT,
                    attributes={
                        "llm.model": model,
                        "llm.provider": provider,
                        "llm.operation": operation,
                    },
                ) as span:
                    start_time = time.perf_counter()
                    try:
                        result = func(*args, **kwargs)
                        span.set_status(Status(StatusCode.OK))
                        return result
                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        raise
                    finally:
                        duration = time.perf_counter() - start_time
                        span.set_attribute("llm.duration_seconds", duration)

            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return sync_wrapper
        return decorator

    def trace_mesh_routing(
        self,
        source_node: str,
        target_node: str,
    ):
        """
        Decorator for tracing mesh routing operations.

        Captures:
        - Source and target nodes
        - Routing latency
        - Circuit breaker state
        - Quality metrics
        """
        def decorator(func: F) -> F:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                with self.start_as_current_span(
                    "mesh.route",
                    kind=SpanKind.INTERNAL,
                    attributes={
                        "mesh.source_node": source_node,
                        "mesh.target_node": target_node,
                    },
                ) as span:
                    start_time = time.perf_counter()
                    try:
                        result = await func(*args, **kwargs)

                        # Extract routing info if available
                        if isinstance(result, dict):
                            if "quality" in result:
                                span.set_attribute("mesh.quality", result["quality"])
                            if "circuit_breaker_state" in result:
                                span.set_attribute("mesh.circuit_breaker", result["circuit_breaker_state"])

                        span.set_status(Status(StatusCode.OK))
                        return result
                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        raise
                    finally:
                        duration = time.perf_counter() - start_time
                        span.set_attribute("mesh.routing_latency_seconds", duration)

            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return func
        return decorator

    def trace_processing(
        self,
        operation: str,
        processor_type: str = "base",
    ):
        """
        Decorator for tracing consciousness processing operations.

        Usage:
            @tracer.trace_processing("analyze", "thought")
            async def analyze_thought(thought: str) -> Analysis:
                ...
        """
        def decorator(func: F) -> F:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                with self.start_as_current_span(
                    f"consciousness.{processor_type}.{operation}",
                    kind=SpanKind.INTERNAL,
                    attributes={
                        "consciousness.operation": operation,
                        "consciousness.processor_type": processor_type,
                    },
                ) as span:
                    try:
                        result = await func(*args, **kwargs)
                        span.set_status(Status(StatusCode.OK))
                        return result
                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        raise

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                with self.start_as_current_span(
                    f"consciousness.{processor_type}.{operation}",
                    kind=SpanKind.INTERNAL,
                    attributes={
                        "consciousness.operation": operation,
                        "consciousness.processor_type": processor_type,
                    },
                ) as span:
                    try:
                        result = func(*args, **kwargs)
                        span.set_status(Status(StatusCode.OK))
                        return result
                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        raise

            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return sync_wrapper
        return decorator


def traced(
    name: Optional[str] = None,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Optional[Dict[str, Any]] = None,
):
    """
    Simple decorator for tracing any function.

    Usage:
        @traced("my_operation")
        def my_function():
            ...

        @traced(attributes={"custom.attr": "value"})
        async def my_async_function():
            ...
    """
    def decorator(func: F) -> F:
        span_name = name or f"{func.__module__}.{func.__qualname__}"
        tracer = trace.get_tracer(__name__)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with tracer.start_as_current_span(
                span_name,
                kind=kind,
                attributes=attributes or {},
            ) as span:
                try:
                    result = await func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with tracer.start_as_current_span(
                span_name,
                kind=kind,
                attributes=attributes or {},
            ) as span:
                try:
                    result = func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator


def get_current_trace_context() -> Dict[str, str]:
    """
    Extract current trace context for propagation.

    Returns dict with traceparent and tracestate headers
    suitable for HTTP propagation.
    """
    from opentelemetry.propagate import inject

    carrier = {}
    inject(carrier)
    return carrier


def set_trace_context_from_headers(headers: Dict[str, str]) -> Context:
    """
    Create context from incoming trace headers.

    Used for receiving distributed traces from upstream services.
    """
    from opentelemetry.propagate import extract

    return extract(headers)


# Global tracer instance
_default_tracer: Optional[ConsciousnessTracer] = None


def get_consciousness_tracer() -> ConsciousnessTracer:
    """Get the default ConsciousnessTracer instance."""
    global _default_tracer
    if _default_tracer is None:
        _default_tracer = ConsciousnessTracer()
    return _default_tracer
