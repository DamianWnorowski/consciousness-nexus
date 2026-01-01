"""Langfuse Integration for LLM Observability

Provides comprehensive LLM call tracing with:
- Automatic trace creation
- Token usage tracking
- Cost attribution
- Latency measurement
- Model versioning
"""

from __future__ import annotations

import os
import time
import functools
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from contextlib import contextmanager
from dataclasses import dataclass, field
import logging

try:
    from langfuse import Langfuse
    from langfuse.decorators import observe, langfuse_context
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    Langfuse = None
    observe = None
    langfuse_context = None

from ..metrics import ConsciousnessMetrics

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class LLMCallMetadata:
    """Metadata for an LLM call."""
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    ttft_ms: Optional[float] = None  # Time to first token
    cost_usd: float = 0.0
    success: bool = True
    error: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stop_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class LangfuseObservability:
    """Langfuse integration for LLM observability.

    Usage:
        obs = LangfuseObservability()

        # Using decorator
        @obs.trace_generation
        async def call_llm(prompt: str) -> str:
            response = await llm.complete(prompt)
            return response

        # Using context manager
        async with obs.generation("my-model") as gen:
            response = await llm.complete(prompt)
            gen.update(output=response, usage={"tokens": 100})
    """

    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
        enabled: bool = True,
        metrics: Optional[ConsciousnessMetrics] = None,
    ):
        self.enabled = enabled and LANGFUSE_AVAILABLE
        self.metrics = metrics
        self._client: Optional[Langfuse] = None

        if self.enabled:
            try:
                self._client = Langfuse(
                    public_key=public_key or os.getenv("LANGFUSE_PUBLIC_KEY"),
                    secret_key=secret_key or os.getenv("LANGFUSE_SECRET_KEY"),
                    host=host or os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
                )
                logger.info("Langfuse observability initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Langfuse: {e}")
                self.enabled = False

    @property
    def client(self) -> Optional[Langfuse]:
        return self._client

    def trace(
        self,
        name: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ):
        """Create a new trace.

        Args:
            name: Name of the trace
            user_id: Optional user identifier
            session_id: Optional session identifier
            metadata: Additional metadata
            tags: Tags for filtering

        Returns:
            Trace object or None if disabled
        """
        if not self.enabled or not self._client:
            return None

        return self._client.trace(
            name=name,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata or {},
            tags=tags or [],
        )

    @contextmanager
    def generation(
        self,
        name: str,
        model: str,
        model_parameters: Optional[Dict[str, Any]] = None,
        input_text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Context manager for LLM generation tracing.

        Usage:
            with obs.generation("chat", "gpt-4", input_text=prompt) as gen:
                response = await llm.complete(prompt)
                gen.end(output=response.text, usage={"input": 100, "output": 50})
        """
        start_time = time.perf_counter()
        gen_data = {
            "name": name,
            "model": model,
            "input": input_text,
            "output": None,
            "usage": None,
            "metadata": metadata or {},
            "model_parameters": model_parameters or {},
        }

        class GenerationContext:
            def __init__(self, obs: LangfuseObservability, data: dict):
                self.obs = obs
                self.data = data
                self._trace = None
                self._generation = None

            def update(
                self,
                output: Optional[str] = None,
                usage: Optional[Dict[str, int]] = None,
                **kwargs,
            ):
                if output:
                    self.data["output"] = output
                if usage:
                    self.data["usage"] = usage
                self.data["metadata"].update(kwargs)

            def end(
                self,
                output: Optional[str] = None,
                usage: Optional[Dict[str, int]] = None,
                error: Optional[str] = None,
            ):
                if output:
                    self.data["output"] = output
                if usage:
                    self.data["usage"] = usage
                if error:
                    self.data["metadata"]["error"] = error

        ctx = GenerationContext(self, gen_data)

        try:
            yield ctx
        except Exception as e:
            ctx.data["metadata"]["error"] = str(e)
            raise
        finally:
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Record to Langfuse
            if self.enabled and self._client:
                try:
                    trace = self._client.trace(name=name)
                    trace.generation(
                        name=f"{name}_generation",
                        model=model,
                        input=gen_data["input"],
                        output=gen_data["output"],
                        usage=gen_data["usage"],
                        metadata={
                            **gen_data["metadata"],
                            "latency_ms": latency_ms,
                        },
                        model_parameters=gen_data["model_parameters"],
                    )
                except Exception as e:
                    logger.warning(f"Failed to record generation to Langfuse: {e}")

            # Record metrics
            if self.metrics:
                usage = gen_data.get("usage") or {}
                self._record_metrics(
                    model=model,
                    provider=gen_data["metadata"].get("provider", "unknown"),
                    input_tokens=usage.get("input", usage.get("input_tokens", 0)),
                    output_tokens=usage.get("output", usage.get("output_tokens", 0)),
                    latency_ms=latency_ms,
                    success="error" not in gen_data["metadata"],
                )

    def trace_generation(
        self,
        model: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Callable[[F], F]:
        """Decorator for tracing LLM generation calls.

        Args:
            model: Model name (can be extracted from function)
            name: Custom name for the trace

        Usage:
            @obs.trace_generation(model="gpt-4")
            async def generate(prompt: str) -> str:
                return await llm.complete(prompt)
        """
        def decorator(func: F) -> F:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                trace_name = name or func.__name__
                trace_model = model or kwargs.get("model", "unknown")

                start_time = time.perf_counter()
                error = None
                result = None

                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    error = str(e)
                    raise
                finally:
                    latency_ms = (time.perf_counter() - start_time) * 1000

                    # Extract usage from result if available
                    usage = None
                    if hasattr(result, "usage"):
                        usage = {
                            "input": getattr(result.usage, "input_tokens", 0),
                            "output": getattr(result.usage, "output_tokens", 0),
                        }

                    # Record to Langfuse
                    if self.enabled and self._client:
                        try:
                            trace = self._client.trace(name=trace_name)
                            trace.generation(
                                name=f"{trace_name}_generation",
                                model=trace_model,
                                input=str(args[0]) if args else str(kwargs.get("prompt", "")),
                                output=str(result) if result else None,
                                usage=usage,
                                metadata={
                                    "latency_ms": latency_ms,
                                    "error": error,
                                },
                            )
                        except Exception as e:
                            logger.warning(f"Failed to record to Langfuse: {e}")

                    # Record metrics
                    if self.metrics:
                        self._record_metrics(
                            model=trace_model,
                            provider="unknown",
                            input_tokens=usage.get("input", 0) if usage else 0,
                            output_tokens=usage.get("output", 0) if usage else 0,
                            latency_ms=latency_ms,
                            success=error is None,
                        )

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                trace_name = name or func.__name__
                trace_model = model or kwargs.get("model", "unknown")

                start_time = time.perf_counter()
                error = None
                result = None

                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    error = str(e)
                    raise
                finally:
                    latency_ms = (time.perf_counter() - start_time) * 1000

                    if self.enabled and self._client:
                        try:
                            trace = self._client.trace(name=trace_name)
                            trace.generation(
                                name=f"{trace_name}_generation",
                                model=trace_model,
                                input=str(args[0]) if args else str(kwargs.get("prompt", "")),
                                output=str(result) if result else None,
                                metadata={
                                    "latency_ms": latency_ms,
                                    "error": error,
                                },
                            )
                        except Exception as e:
                            logger.warning(f"Failed to record to Langfuse: {e}")

            import asyncio
            if asyncio.iscoroutinefunction(func):
                return async_wrapper  # type: ignore
            return sync_wrapper  # type: ignore

        return decorator

    def _record_metrics(
        self,
        model: str,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        success: bool,
    ):
        """Record LLM metrics."""
        if not self.metrics:
            return

        # Record to Prometheus metrics
        labels = {"model": model, "provider": provider}

        self.metrics.llm_calls_total.labels(**labels).inc()
        self.metrics.llm_tokens_total.labels(
            model=model, provider=provider, direction="input"
        ).inc(input_tokens)
        self.metrics.llm_tokens_total.labels(
            model=model, provider=provider, direction="output"
        ).inc(output_tokens)
        self.metrics.llm_latency.labels(**labels).observe(latency_ms / 1000)

        if not success:
            self.metrics.llm_errors_total.labels(**labels).inc()

    def score(
        self,
        trace_id: str,
        name: str,
        value: float,
        comment: Optional[str] = None,
    ):
        """Add a score to a trace.

        Args:
            trace_id: ID of the trace to score
            name: Score name (e.g., "accuracy", "relevance")
            value: Score value (0-1)
            comment: Optional comment
        """
        if not self.enabled or not self._client:
            return

        try:
            self._client.score(
                trace_id=trace_id,
                name=name,
                value=value,
                comment=comment,
            )
        except Exception as e:
            logger.warning(f"Failed to add score to Langfuse: {e}")

    def flush(self):
        """Flush pending data to Langfuse."""
        if self._client:
            try:
                self._client.flush()
            except Exception as e:
                logger.warning(f"Failed to flush Langfuse: {e}")

    def shutdown(self):
        """Shutdown the client."""
        self.flush()
        if self._client:
            try:
                self._client.shutdown()
            except Exception:
                pass


def observe_llm(
    model: Optional[str] = None,
    name: Optional[str] = None,
    capture_input: bool = True,
    capture_output: bool = True,
) -> Callable[[F], F]:
    """Decorator for observing LLM calls using Langfuse.

    This is a simplified decorator that works with the global Langfuse context.

    Args:
        model: Model name
        name: Custom trace name
        capture_input: Whether to capture input
        capture_output: Whether to capture output

    Usage:
        @observe_llm(model="gpt-4")
        async def my_llm_call(prompt: str) -> str:
            return await llm.complete(prompt)
    """
    if not LANGFUSE_AVAILABLE or observe is None:
        # Return no-op decorator if Langfuse not available
        def noop_decorator(func: F) -> F:
            return func
        return noop_decorator

    # Use Langfuse's built-in observe decorator with generation type
    return observe(
        as_type="generation",
        name=name,
        capture_input=capture_input,
        capture_output=capture_output,
    )
