"""Intelligent Telemetry Sampler

Head-based and tail-based sampling strategies for FinOps optimization.
Implements probabilistic, priority-based, and outcome-aware sampling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import threading
import logging
import random
import hashlib
import time

from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)


class SamplingDecision(str, Enum):
    """Sampling decision outcomes."""
    SAMPLE = "sample"           # Keep the data
    DROP = "drop"               # Discard the data
    DEFER = "defer"             # Defer decision (tail-based)
    FORCE_SAMPLE = "force"      # Always sample (high priority)


class SamplingMethod(str, Enum):
    """Sampling methods."""
    HEAD_BASED = "head_based"           # Decide at trace start
    TAIL_BASED = "tail_based"           # Decide at trace end
    PROBABILISTIC = "probabilistic"     # Random sampling
    RATE_LIMITED = "rate_limited"       # Token bucket
    PRIORITY = "priority"               # Priority-based
    CONSISTENT = "consistent"           # Consistent hashing (trace ID)
    ADAPTIVE = "adaptive"               # Load-adaptive
    HYBRID = "hybrid"                   # Combination strategies


@dataclass
class SamplingConfig:
    """Configuration for sampling behavior.

    Usage:
        config = SamplingConfig(
            base_rate=0.1,              # 10% base sampling
            method=SamplingMethod.HYBRID,
            priority_tags=["error", "critical"],
            min_rate=0.01,
            max_rate=1.0,
        )
    """
    base_rate: float = 0.1                          # Base sampling rate (0-1)
    method: SamplingMethod = SamplingMethod.HEAD_BASED
    priority_tags: List[str] = field(default_factory=list)
    priority_attributes: Dict[str, List[str]] = field(default_factory=dict)
    min_rate: float = 0.01                          # Minimum sampling rate
    max_rate: float = 1.0                           # Maximum sampling rate
    rate_limit_per_second: int = 100                # For rate-limited sampling
    decision_cache_ttl_seconds: int = 300           # TTL for consistent sampling
    adaptive_window_seconds: int = 60               # Window for adaptive sampling
    adaptive_target_volume: int = 1000              # Target samples per window


@dataclass
class SamplingContext:
    """Context for a sampling decision.

    Usage:
        ctx = SamplingContext(
            trace_id="abc123",
            span_id="def456",
            service="api-gateway",
            operation="GET /users",
            attributes={"http.status_code": "500"},
        )
    """
    trace_id: str
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    service: str = ""
    operation: str = ""
    attributes: Dict[str, str] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    start_time: datetime = field(default_factory=datetime.now)
    duration_ms: Optional[float] = None
    has_error: bool = False
    is_root_span: bool = False
    priority: int = 0                               # Higher = more important


@dataclass
class SamplingResult:
    """Result of a sampling decision."""
    decision: SamplingDecision
    reason: str
    probability: float
    method_used: SamplingMethod
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeferredTrace:
    """A trace deferred for tail-based sampling."""
    trace_id: str
    spans: List[SamplingContext]
    start_time: datetime
    total_duration_ms: float = 0.0
    has_error: bool = False
    priority: int = 0


class AdaptiveSampler:
    """Intelligent sampler with head/tail-based strategies.

    Usage:
        sampler = AdaptiveSampler(
            namespace="consciousness",
            config=SamplingConfig(
                base_rate=0.1,
                method=SamplingMethod.HYBRID,
            ),
        )

        # Head-based sampling
        result = sampler.should_sample(SamplingContext(
            trace_id="abc123",
            service="api",
            operation="GET /users",
        ))

        # Tail-based sampling
        sampler.record_span(context)
        sampler.finalize_trace("abc123")  # Makes final decision
    """

    def __init__(
        self,
        namespace: str = "consciousness",
        config: Optional[SamplingConfig] = None,
    ):
        self.namespace = namespace
        self.config = config or SamplingConfig()
        self._lock = threading.Lock()

        # Tracking state
        self._decision_cache: Dict[str, Tuple[SamplingResult, datetime]] = {}
        self._rate_limiter_tokens: float = float(self.config.rate_limit_per_second)
        self._rate_limiter_last_update: datetime = datetime.now()
        self._adaptive_window: deque = deque(maxlen=10000)
        self._deferred_traces: Dict[str, DeferredTrace] = {}

        # Statistics
        self._total_evaluated: int = 0
        self._total_sampled: int = 0
        self._total_dropped: int = 0
        self._total_deferred: int = 0

        # Prometheus metrics
        self.sampling_decisions = Counter(
            f"{namespace}_adaptive_sampling_decisions_total",
            "Sampling decisions made",
            ["decision", "method", "service"],
        )

        self.sampling_rate = Gauge(
            f"{namespace}_adaptive_sampling_rate",
            "Current effective sampling rate",
            ["method"],
        )

        self.deferred_traces = Gauge(
            f"{namespace}_adaptive_deferred_traces",
            "Currently deferred traces awaiting decision",
        )

        self.sampling_latency = Histogram(
            f"{namespace}_adaptive_sampling_latency_seconds",
            "Time to make sampling decision",
            buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
        )

        self.priority_samples = Counter(
            f"{namespace}_adaptive_priority_samples_total",
            "Samples kept due to priority",
            ["reason"],
        )

    def should_sample(self, context: SamplingContext) -> SamplingResult:
        """Determine if data should be sampled.

        Args:
            context: Sampling context

        Returns:
            SamplingResult with decision
        """
        start = time.perf_counter()
        self._total_evaluated += 1

        try:
            # Check cache for consistent sampling
            if self.config.method == SamplingMethod.CONSISTENT:
                cached = self._get_cached_decision(context.trace_id)
                if cached:
                    return cached

            # Route to appropriate strategy
            if self.config.method == SamplingMethod.HEAD_BASED:
                result = self._head_based_sample(context)
            elif self.config.method == SamplingMethod.TAIL_BASED:
                result = self._tail_based_sample(context)
            elif self.config.method == SamplingMethod.PROBABILISTIC:
                result = self._probabilistic_sample(context)
            elif self.config.method == SamplingMethod.RATE_LIMITED:
                result = self._rate_limited_sample(context)
            elif self.config.method == SamplingMethod.PRIORITY:
                result = self._priority_sample(context)
            elif self.config.method == SamplingMethod.CONSISTENT:
                result = self._consistent_sample(context)
            elif self.config.method == SamplingMethod.ADAPTIVE:
                result = self._adaptive_sample(context)
            elif self.config.method == SamplingMethod.HYBRID:
                result = self._hybrid_sample(context)
            else:
                result = self._probabilistic_sample(context)

            # Update metrics
            self._update_metrics(result, context)

            # Cache decision for consistent sampling
            if self.config.method in (SamplingMethod.CONSISTENT, SamplingMethod.HYBRID):
                self._cache_decision(context.trace_id, result)

            return result

        finally:
            duration = time.perf_counter() - start
            self.sampling_latency.observe(duration)

    def _head_based_sample(self, context: SamplingContext) -> SamplingResult:
        """Head-based sampling: decide at trace/span start.

        Considers:
        - Base sampling rate
        - Priority tags/attributes
        - Service-specific rates
        """
        # Check for force-sample conditions
        force_reason = self._check_force_sample(context)
        if force_reason:
            return SamplingResult(
                decision=SamplingDecision.FORCE_SAMPLE,
                reason=force_reason,
                probability=1.0,
                method_used=SamplingMethod.HEAD_BASED,
            )

        # Apply base rate with priority adjustment
        effective_rate = self._calculate_effective_rate(context)

        if random.random() < effective_rate:
            return SamplingResult(
                decision=SamplingDecision.SAMPLE,
                reason="head_based_probability",
                probability=effective_rate,
                method_used=SamplingMethod.HEAD_BASED,
            )

        return SamplingResult(
            decision=SamplingDecision.DROP,
            reason="below_sampling_threshold",
            probability=effective_rate,
            method_used=SamplingMethod.HEAD_BASED,
        )

    def _tail_based_sample(self, context: SamplingContext) -> SamplingResult:
        """Tail-based sampling: defer decision until trace completes.

        Defers the decision and collects spans until trace finalizes.
        """
        # Always defer for tail-based
        self._defer_span(context)

        return SamplingResult(
            decision=SamplingDecision.DEFER,
            reason="tail_based_deferred",
            probability=self.config.base_rate,
            method_used=SamplingMethod.TAIL_BASED,
        )

    def _probabilistic_sample(self, context: SamplingContext) -> SamplingResult:
        """Simple probabilistic (random) sampling."""
        if random.random() < self.config.base_rate:
            return SamplingResult(
                decision=SamplingDecision.SAMPLE,
                reason="probabilistic",
                probability=self.config.base_rate,
                method_used=SamplingMethod.PROBABILISTIC,
            )

        return SamplingResult(
            decision=SamplingDecision.DROP,
            reason="probabilistic_drop",
            probability=self.config.base_rate,
            method_used=SamplingMethod.PROBABILISTIC,
        )

    def _rate_limited_sample(self, context: SamplingContext) -> SamplingResult:
        """Token bucket rate limiting."""
        with self._lock:
            # Refill tokens
            now = datetime.now()
            elapsed = (now - self._rate_limiter_last_update).total_seconds()
            self._rate_limiter_tokens = min(
                float(self.config.rate_limit_per_second),
                self._rate_limiter_tokens + elapsed * self.config.rate_limit_per_second,
            )
            self._rate_limiter_last_update = now

            # Try to consume token
            if self._rate_limiter_tokens >= 1.0:
                self._rate_limiter_tokens -= 1.0
                effective_rate = self._rate_limiter_tokens / self.config.rate_limit_per_second
                return SamplingResult(
                    decision=SamplingDecision.SAMPLE,
                    reason="rate_limit_allowed",
                    probability=effective_rate,
                    method_used=SamplingMethod.RATE_LIMITED,
                )

        return SamplingResult(
            decision=SamplingDecision.DROP,
            reason="rate_limit_exceeded",
            probability=0.0,
            method_used=SamplingMethod.RATE_LIMITED,
        )

    def _priority_sample(self, context: SamplingContext) -> SamplingResult:
        """Priority-based sampling."""
        # Check explicit priority
        if context.priority > 0:
            priority_rate = min(1.0, self.config.base_rate * (1 + context.priority * 0.5))
            if random.random() < priority_rate:
                return SamplingResult(
                    decision=SamplingDecision.SAMPLE,
                    reason=f"priority_{context.priority}",
                    probability=priority_rate,
                    method_used=SamplingMethod.PRIORITY,
                )

        # Check priority tags
        if context.tags & set(self.config.priority_tags):
            matching_tags = context.tags & set(self.config.priority_tags)
            return SamplingResult(
                decision=SamplingDecision.FORCE_SAMPLE,
                reason=f"priority_tag:{list(matching_tags)[0]}",
                probability=1.0,
                method_used=SamplingMethod.PRIORITY,
            )

        # Check priority attributes
        for attr, values in self.config.priority_attributes.items():
            if context.attributes.get(attr) in values:
                return SamplingResult(
                    decision=SamplingDecision.FORCE_SAMPLE,
                    reason=f"priority_attr:{attr}={context.attributes[attr]}",
                    probability=1.0,
                    method_used=SamplingMethod.PRIORITY,
                )

        # Fall back to base rate
        if random.random() < self.config.base_rate:
            return SamplingResult(
                decision=SamplingDecision.SAMPLE,
                reason="priority_base_rate",
                probability=self.config.base_rate,
                method_used=SamplingMethod.PRIORITY,
            )

        return SamplingResult(
            decision=SamplingDecision.DROP,
            reason="priority_dropped",
            probability=self.config.base_rate,
            method_used=SamplingMethod.PRIORITY,
        )

    def _consistent_sample(self, context: SamplingContext) -> SamplingResult:
        """Consistent hash-based sampling (same trace ID = same decision)."""
        # Hash trace ID to get deterministic value
        hash_value = int(hashlib.md5(context.trace_id.encode()).hexdigest()[:8], 16)
        threshold = int(self.config.base_rate * 0xFFFFFFFF)

        if hash_value < threshold:
            return SamplingResult(
                decision=SamplingDecision.SAMPLE,
                reason="consistent_hash",
                probability=self.config.base_rate,
                method_used=SamplingMethod.CONSISTENT,
            )

        return SamplingResult(
            decision=SamplingDecision.DROP,
            reason="consistent_hash_drop",
            probability=self.config.base_rate,
            method_used=SamplingMethod.CONSISTENT,
        )

    def _adaptive_sample(self, context: SamplingContext) -> SamplingResult:
        """Adaptive sampling based on current load."""
        with self._lock:
            # Clean old entries from window
            cutoff = datetime.now() - timedelta(seconds=self.config.adaptive_window_seconds)
            while self._adaptive_window and self._adaptive_window[0] < cutoff:
                self._adaptive_window.popleft()

            current_volume = len(self._adaptive_window)

        # Calculate adaptive rate
        if current_volume < self.config.adaptive_target_volume:
            # Under target: increase rate
            scale = self.config.adaptive_target_volume / max(current_volume, 1)
            adaptive_rate = min(self.config.max_rate, self.config.base_rate * scale)
        else:
            # Over target: decrease rate
            scale = self.config.adaptive_target_volume / current_volume
            adaptive_rate = max(self.config.min_rate, self.config.base_rate * scale)

        self.sampling_rate.labels(method="adaptive").set(adaptive_rate)

        if random.random() < adaptive_rate:
            with self._lock:
                self._adaptive_window.append(datetime.now())
            return SamplingResult(
                decision=SamplingDecision.SAMPLE,
                reason="adaptive",
                probability=adaptive_rate,
                method_used=SamplingMethod.ADAPTIVE,
                metadata={"current_volume": current_volume},
            )

        return SamplingResult(
            decision=SamplingDecision.DROP,
            reason="adaptive_drop",
            probability=adaptive_rate,
            method_used=SamplingMethod.ADAPTIVE,
            metadata={"current_volume": current_volume},
        )

    def _hybrid_sample(self, context: SamplingContext) -> SamplingResult:
        """Hybrid sampling combining multiple strategies.

        Priority: Force-sample conditions > Priority > Consistent > Adaptive
        """
        # 1. Check force-sample conditions (errors, specific tags)
        force_reason = self._check_force_sample(context)
        if force_reason:
            self.priority_samples.labels(reason="force").inc()
            return SamplingResult(
                decision=SamplingDecision.FORCE_SAMPLE,
                reason=force_reason,
                probability=1.0,
                method_used=SamplingMethod.HYBRID,
            )

        # 2. Priority-based boost
        priority_result = self._priority_sample(context)
        if priority_result.decision == SamplingDecision.FORCE_SAMPLE:
            self.priority_samples.labels(reason="priority").inc()
            return SamplingResult(
                decision=priority_result.decision,
                reason=priority_result.reason,
                probability=priority_result.probability,
                method_used=SamplingMethod.HYBRID,
            )

        # 3. Use consistent hashing for trace coherence
        consistent = self._consistent_sample(context)

        # 4. Apply adaptive rate adjustment
        adaptive = self._adaptive_sample(context)

        # Combine: if either says sample, sample
        if consistent.decision == SamplingDecision.SAMPLE or \
           adaptive.decision == SamplingDecision.SAMPLE:
            final_probability = max(consistent.probability, adaptive.probability)
            return SamplingResult(
                decision=SamplingDecision.SAMPLE,
                reason="hybrid_combined",
                probability=final_probability,
                method_used=SamplingMethod.HYBRID,
                metadata={
                    "consistent_decision": consistent.decision.value,
                    "adaptive_decision": adaptive.decision.value,
                },
            )

        return SamplingResult(
            decision=SamplingDecision.DROP,
            reason="hybrid_dropped",
            probability=min(consistent.probability, adaptive.probability),
            method_used=SamplingMethod.HYBRID,
        )

    def _check_force_sample(self, context: SamplingContext) -> Optional[str]:
        """Check conditions that force sampling."""
        # Always sample errors
        if context.has_error:
            return "error_trace"

        # Always sample root spans with specific attributes
        if context.is_root_span and context.priority >= 2:
            return "high_priority_root"

        # Check for error status codes
        status_code = context.attributes.get("http.status_code", "")
        if status_code.startswith("5"):
            return "server_error"

        # Check for specific tags
        critical_tags = {"error", "critical", "alert", "incident"}
        if context.tags & critical_tags:
            return f"critical_tag:{list(context.tags & critical_tags)[0]}"

        return None

    def _calculate_effective_rate(self, context: SamplingContext) -> float:
        """Calculate effective sampling rate based on context."""
        rate = self.config.base_rate

        # Boost for priority
        if context.priority > 0:
            rate = min(1.0, rate * (1 + context.priority * 0.25))

        # Boost for root spans
        if context.is_root_span:
            rate = min(1.0, rate * 1.2)

        # Clamp to configured bounds
        return max(self.config.min_rate, min(self.config.max_rate, rate))

    def _defer_span(self, context: SamplingContext):
        """Defer a span for tail-based sampling."""
        with self._lock:
            if context.trace_id not in self._deferred_traces:
                self._deferred_traces[context.trace_id] = DeferredTrace(
                    trace_id=context.trace_id,
                    spans=[],
                    start_time=context.start_time,
                )
                self._total_deferred += 1

            trace = self._deferred_traces[context.trace_id]
            trace.spans.append(context)

            if context.has_error:
                trace.has_error = True
            if context.duration_ms:
                trace.total_duration_ms += context.duration_ms
            trace.priority = max(trace.priority, context.priority)

        self.deferred_traces.set(len(self._deferred_traces))

    def record_span(self, context: SamplingContext):
        """Record a span for tail-based sampling.

        Args:
            context: Span context
        """
        if self.config.method in (SamplingMethod.TAIL_BASED, SamplingMethod.HYBRID):
            self._defer_span(context)

    def finalize_trace(self, trace_id: str) -> Optional[SamplingResult]:
        """Finalize and make decision for a deferred trace.

        Args:
            trace_id: Trace ID to finalize

        Returns:
            Final sampling result or None if not deferred
        """
        with self._lock:
            if trace_id not in self._deferred_traces:
                return None

            trace = self._deferred_traces.pop(trace_id)

        # Make final decision based on complete trace
        # Force sample errors
        if trace.has_error:
            result = SamplingResult(
                decision=SamplingDecision.FORCE_SAMPLE,
                reason="tail_error_trace",
                probability=1.0,
                method_used=SamplingMethod.TAIL_BASED,
                metadata={"span_count": len(trace.spans)},
            )
        # Force sample slow traces (> 5s)
        elif trace.total_duration_ms > 5000:
            result = SamplingResult(
                decision=SamplingDecision.FORCE_SAMPLE,
                reason="tail_slow_trace",
                probability=1.0,
                method_used=SamplingMethod.TAIL_BASED,
                metadata={
                    "duration_ms": trace.total_duration_ms,
                    "span_count": len(trace.spans),
                },
            )
        # Force sample high priority
        elif trace.priority >= 2:
            result = SamplingResult(
                decision=SamplingDecision.FORCE_SAMPLE,
                reason="tail_high_priority",
                probability=1.0,
                method_used=SamplingMethod.TAIL_BASED,
            )
        # Apply base rate
        elif random.random() < self.config.base_rate:
            result = SamplingResult(
                decision=SamplingDecision.SAMPLE,
                reason="tail_sampled",
                probability=self.config.base_rate,
                method_used=SamplingMethod.TAIL_BASED,
            )
        else:
            result = SamplingResult(
                decision=SamplingDecision.DROP,
                reason="tail_dropped",
                probability=self.config.base_rate,
                method_used=SamplingMethod.TAIL_BASED,
            )

        self._update_metrics(result, trace.spans[0] if trace.spans else SamplingContext(trace_id=trace_id))
        self.deferred_traces.set(len(self._deferred_traces))

        return result

    def _get_cached_decision(self, trace_id: str) -> Optional[SamplingResult]:
        """Get cached decision for trace ID."""
        with self._lock:
            if trace_id in self._decision_cache:
                result, timestamp = self._decision_cache[trace_id]
                if (datetime.now() - timestamp).seconds < self.config.decision_cache_ttl_seconds:
                    return result
                del self._decision_cache[trace_id]
        return None

    def _cache_decision(self, trace_id: str, result: SamplingResult):
        """Cache a sampling decision."""
        with self._lock:
            self._decision_cache[trace_id] = (result, datetime.now())

            # Clean old entries periodically
            if len(self._decision_cache) > 10000:
                cutoff = datetime.now() - timedelta(seconds=self.config.decision_cache_ttl_seconds)
                self._decision_cache = {
                    k: (r, t) for k, (r, t) in self._decision_cache.items()
                    if t > cutoff
                }

    def _update_metrics(self, result: SamplingResult, context: SamplingContext):
        """Update Prometheus metrics."""
        self.sampling_decisions.labels(
            decision=result.decision.value,
            method=result.method_used.value,
            service=context.service or "unknown",
        ).inc()

        if result.decision in (SamplingDecision.SAMPLE, SamplingDecision.FORCE_SAMPLE):
            self._total_sampled += 1
        elif result.decision == SamplingDecision.DROP:
            self._total_dropped += 1

        self.sampling_rate.labels(method=result.method_used.value).set(result.probability)

    def get_statistics(self) -> Dict[str, Any]:
        """Get sampler statistics.

        Returns:
            Statistics dictionary
        """
        with self._lock:
            deferred_count = len(self._deferred_traces)
            cache_size = len(self._decision_cache)

        return {
            "total_evaluated": self._total_evaluated,
            "total_sampled": self._total_sampled,
            "total_dropped": self._total_dropped,
            "total_deferred": self._total_deferred,
            "effective_sample_rate": (
                self._total_sampled / self._total_evaluated
                if self._total_evaluated > 0 else 0.0
            ),
            "currently_deferred": deferred_count,
            "decision_cache_size": cache_size,
            "config": {
                "method": self.config.method.value,
                "base_rate": self.config.base_rate,
                "rate_limit_per_second": self.config.rate_limit_per_second,
            },
        }

    def cleanup_stale_traces(self, max_age_seconds: int = 300):
        """Clean up stale deferred traces.

        Args:
            max_age_seconds: Maximum age for deferred traces
        """
        cutoff = datetime.now() - timedelta(seconds=max_age_seconds)

        with self._lock:
            stale = [
                trace_id for trace_id, trace in self._deferred_traces.items()
                if trace.start_time < cutoff
            ]
            for trace_id in stale:
                del self._deferred_traces[trace_id]

        if stale:
            logger.info(f"Cleaned up {len(stale)} stale deferred traces")
            self.deferred_traces.set(len(self._deferred_traces))
