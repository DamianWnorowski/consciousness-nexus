"""Cold Start Tracking and Optimization

AWS Lambda cold start observability:
- Cold start detection and measurement
- Init duration breakdown
- Provisioned concurrency recommendations
- Warm instance tracking
"""

from __future__ import annotations

import os
import time
import threading
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict
from functools import wraps

from prometheus_client import Counter, Gauge, Histogram, Summary

logger = logging.getLogger(__name__)


class StartType(str, Enum):
    """Lambda start type."""
    COLD = "cold"
    WARM = "warm"
    PROVISIONED = "provisioned"


class InitPhase(str, Enum):
    """Lambda initialization phases."""
    RUNTIME_INIT = "runtime_init"
    EXTENSION_INIT = "extension_init"
    HANDLER_INIT = "handler_init"
    DEPENDENCY_LOAD = "dependency_load"
    CONNECTION_SETUP = "connection_setup"
    CACHE_WARMUP = "cache_warmup"


@dataclass
class ColdStartEvent:
    """Cold start event data."""
    request_id: str
    function_name: str
    function_version: str
    start_type: StartType
    init_duration_ms: float
    billed_duration_ms: float
    memory_size_mb: int
    memory_used_mb: int
    timestamp: float = field(default_factory=time.time)
    # Init phase breakdown
    phase_durations: Dict[InitPhase, float] = field(default_factory=dict)
    # Environment
    runtime: str = field(default_factory=lambda: os.environ.get("AWS_EXECUTION_ENV", "unknown"))
    region: str = field(default_factory=lambda: os.environ.get("AWS_REGION", "unknown"))
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_cold_start(self) -> bool:
        return self.start_type == StartType.COLD

    @property
    def total_init_ms(self) -> float:
        return sum(self.phase_durations.values())


@dataclass
class ColdStartMetrics:
    """Aggregated cold start metrics."""
    cold_start_count: int = 0
    warm_start_count: int = 0
    provisioned_start_count: int = 0
    avg_cold_start_duration_ms: float = 0.0
    p50_cold_start_duration_ms: float = 0.0
    p95_cold_start_duration_ms: float = 0.0
    p99_cold_start_duration_ms: float = 0.0
    cold_start_rate: float = 0.0
    # Phase breakdown
    avg_phase_durations: Dict[InitPhase, float] = field(default_factory=dict)
    # Memory correlation
    memory_cold_start_correlation: float = 0.0

    @property
    def total_starts(self) -> int:
        return self.cold_start_count + self.warm_start_count + self.provisioned_start_count


class ColdStartTracker:
    """Track Lambda cold starts and initialization.

    Detects and measures:
    - Cold vs warm starts
    - Init duration breakdown by phase
    - Memory/cold start correlation
    - Cold start rate over time

    Usage:
        tracker = ColdStartTracker()

        @tracker.track_init
        def handler(event, context):
            # Your handler code
            return response

        # Or manual tracking
        with tracker.track_phase(InitPhase.DEPENDENCY_LOAD):
            import heavy_dependency

        tracker.record_invocation(context)
    """

    # Global flag to detect first invocation
    _first_invocation = True
    _init_start_time: Optional[float] = None
    _instance_id: str = ""

    def __init__(
        self,
        function_name: Optional[str] = None,
        function_version: Optional[str] = None,
        namespace: str = "consciousness",
    ):
        self.function_name = function_name or os.environ.get("AWS_LAMBDA_FUNCTION_NAME", "unknown")
        self.function_version = function_version or os.environ.get("AWS_LAMBDA_FUNCTION_VERSION", "$LATEST")
        self.namespace = namespace

        self._events: List[ColdStartEvent] = []
        self._phase_timers: Dict[InitPhase, float] = {}
        self._lock = threading.Lock()

        # Track instance
        if not ColdStartTracker._instance_id:
            ColdStartTracker._instance_id = f"{time.time()}-{id(self)}"
            ColdStartTracker._init_start_time = time.time()

        # Setup metrics
        self._setup_metrics()

    def _setup_metrics(self):
        """Initialize Prometheus metrics."""
        self.cold_start_total = Counter(
            f"{self.namespace}_lambda_cold_starts_total",
            "Total cold starts",
            ["function_name", "runtime"],
        )

        self.warm_start_total = Counter(
            f"{self.namespace}_lambda_warm_starts_total",
            "Total warm starts",
            ["function_name"],
        )

        self.init_duration = Histogram(
            f"{self.namespace}_lambda_init_duration_seconds",
            "Lambda init duration",
            ["function_name", "phase"],
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        )

        self.cold_start_rate = Gauge(
            f"{self.namespace}_lambda_cold_start_rate",
            "Cold start rate (0-1)",
            ["function_name"],
        )

        self.memory_used = Gauge(
            f"{self.namespace}_lambda_memory_used_mb",
            "Memory used in MB",
            ["function_name", "start_type"],
        )

        self.phase_duration = Summary(
            f"{self.namespace}_lambda_phase_duration_seconds",
            "Duration by init phase",
            ["function_name", "phase"],
        )

    def is_cold_start(self) -> bool:
        """Check if current invocation is a cold start.

        Returns:
            True if cold start, False otherwise
        """
        return ColdStartTracker._first_invocation

    def get_start_type(self) -> StartType:
        """Get the start type for current invocation.

        Returns:
            StartType enum
        """
        if os.environ.get("AWS_LAMBDA_INITIALIZATION_TYPE") == "provisioned-concurrency":
            return StartType.PROVISIONED
        elif ColdStartTracker._first_invocation:
            return StartType.COLD
        else:
            return StartType.WARM

    def track_phase(self, phase: InitPhase):
        """Context manager for tracking init phase duration.

        Args:
            phase: Init phase to track

        Usage:
            with tracker.track_phase(InitPhase.DEPENDENCY_LOAD):
                import pandas
                import numpy
        """
        class PhaseTimer:
            def __init__(self, tracker: ColdStartTracker, phase: InitPhase):
                self.tracker = tracker
                self.phase = phase
                self.start_time = 0.0

            def __enter__(self):
                self.start_time = time.perf_counter()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                duration = time.perf_counter() - self.start_time
                with self.tracker._lock:
                    self.tracker._phase_timers[self.phase] = duration

                self.tracker.phase_duration.labels(
                    function_name=self.tracker.function_name,
                    phase=self.phase.value,
                ).observe(duration)

                return False

        return PhaseTimer(self, phase)

    def track_init(self, func: Callable) -> Callable:
        """Decorator to track handler initialization.

        Args:
            func: Lambda handler function

        Returns:
            Wrapped function

        Usage:
            @tracker.track_init
            def handler(event, context):
                return process(event)
        """
        @wraps(func)
        def wrapper(event, context):
            start_type = self.get_start_type()

            # Record invocation
            self.record_invocation(context, start_type)

            # Execute handler
            return func(event, context)

        return wrapper

    def record_invocation(
        self,
        context: Any,
        start_type: Optional[StartType] = None,
    ) -> ColdStartEvent:
        """Record an invocation.

        Args:
            context: Lambda context object
            start_type: Override start type detection

        Returns:
            ColdStartEvent
        """
        if start_type is None:
            start_type = self.get_start_type()

        # Get init duration from environment or calculate
        init_duration = 0.0
        if start_type == StartType.COLD and ColdStartTracker._init_start_time:
            init_duration = (time.time() - ColdStartTracker._init_start_time) * 1000

        # Get memory info
        memory_size = int(os.environ.get("AWS_LAMBDA_FUNCTION_MEMORY_SIZE", 128))
        memory_used = self._get_memory_used()

        # Create event
        event = ColdStartEvent(
            request_id=getattr(context, "aws_request_id", "unknown"),
            function_name=self.function_name,
            function_version=self.function_version,
            start_type=start_type,
            init_duration_ms=init_duration,
            billed_duration_ms=0,  # Set after invocation completes
            memory_size_mb=memory_size,
            memory_used_mb=memory_used,
            phase_durations=dict(self._phase_timers),
        )

        # Store event
        with self._lock:
            self._events.append(event)
            # Keep last 1000 events
            if len(self._events) > 1000:
                self._events = self._events[-500:]

        # Update metrics
        self._update_metrics(event)

        # Mark as warm for subsequent invocations
        ColdStartTracker._first_invocation = False

        return event

    def _get_memory_used(self) -> int:
        """Get current memory usage in MB."""
        try:
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF)
            return int(usage.ru_maxrss / 1024)  # Convert to MB
        except Exception:
            return 0

    def _update_metrics(self, event: ColdStartEvent):
        """Update Prometheus metrics from event."""
        if event.start_type == StartType.COLD:
            self.cold_start_total.labels(
                function_name=self.function_name,
                runtime=event.runtime,
            ).inc()
        else:
            self.warm_start_total.labels(
                function_name=self.function_name,
            ).inc()

        # Init duration
        if event.init_duration_ms > 0:
            self.init_duration.labels(
                function_name=self.function_name,
                phase="total",
            ).observe(event.init_duration_ms / 1000)

        # Memory
        self.memory_used.labels(
            function_name=self.function_name,
            start_type=event.start_type.value,
        ).set(event.memory_used_mb)

        # Calculate cold start rate
        with self._lock:
            if self._events:
                cold_count = sum(1 for e in self._events if e.start_type == StartType.COLD)
                rate = cold_count / len(self._events)
                self.cold_start_rate.labels(function_name=self.function_name).set(rate)

    def get_metrics(self) -> ColdStartMetrics:
        """Get aggregated cold start metrics.

        Returns:
            ColdStartMetrics
        """
        with self._lock:
            events = list(self._events)

        if not events:
            return ColdStartMetrics()

        cold_events = [e for e in events if e.start_type == StartType.COLD]
        warm_events = [e for e in events if e.start_type == StartType.WARM]
        provisioned_events = [e for e in events if e.start_type == StartType.PROVISIONED]

        # Calculate percentiles for cold starts
        cold_durations = sorted([e.init_duration_ms for e in cold_events])
        p50 = p95 = p99 = avg = 0.0

        if cold_durations:
            avg = sum(cold_durations) / len(cold_durations)
            p50 = cold_durations[len(cold_durations) // 2]
            p95 = cold_durations[int(len(cold_durations) * 0.95)]
            p99 = cold_durations[int(len(cold_durations) * 0.99)]

        # Average phase durations
        avg_phases: Dict[InitPhase, float] = defaultdict(float)
        phase_counts: Dict[InitPhase, int] = defaultdict(int)

        for event in cold_events:
            for phase, duration in event.phase_durations.items():
                avg_phases[phase] += duration
                phase_counts[phase] += 1

        for phase in avg_phases:
            if phase_counts[phase] > 0:
                avg_phases[phase] /= phase_counts[phase]

        return ColdStartMetrics(
            cold_start_count=len(cold_events),
            warm_start_count=len(warm_events),
            provisioned_start_count=len(provisioned_events),
            avg_cold_start_duration_ms=avg,
            p50_cold_start_duration_ms=p50,
            p95_cold_start_duration_ms=p95,
            p99_cold_start_duration_ms=p99,
            cold_start_rate=len(cold_events) / len(events) if events else 0,
            avg_phase_durations=dict(avg_phases),
        )

    def get_recent_events(
        self,
        count: int = 100,
        start_type: Optional[StartType] = None,
    ) -> List[ColdStartEvent]:
        """Get recent cold start events.

        Args:
            count: Number of events to return
            start_type: Filter by start type

        Returns:
            List of events
        """
        with self._lock:
            events = list(reversed(self._events))

        if start_type:
            events = [e for e in events if e.start_type == start_type]

        return events[:count]


class ColdStartOptimizer:
    """Cold start optimization recommendations.

    Analyzes cold start patterns and provides recommendations:
    - Provisioned concurrency recommendations
    - Memory optimization
    - Dependency optimization
    - Connection pooling

    Usage:
        optimizer = ColdStartOptimizer(tracker)
        recommendations = optimizer.analyze()

        for rec in recommendations:
            print(f"{rec['type']}: {rec['description']}")
    """

    def __init__(
        self,
        tracker: ColdStartTracker,
        cost_per_gb_second: float = 0.0000166667,
        provisioned_cost_per_gb_hour: float = 0.000004646,
    ):
        self.tracker = tracker
        self.cost_per_gb_second = cost_per_gb_second
        self.provisioned_cost_per_gb_hour = provisioned_cost_per_gb_hour

    def analyze(self) -> List[Dict[str, Any]]:
        """Analyze cold start patterns and generate recommendations.

        Returns:
            List of recommendations
        """
        metrics = self.tracker.get_metrics()
        recommendations = []

        # Check cold start rate
        if metrics.cold_start_rate > 0.1:
            recommendations.append(self._recommend_provisioned_concurrency(metrics))

        # Check init duration
        if metrics.avg_cold_start_duration_ms > 1000:
            recommendations.extend(self._recommend_init_optimization(metrics))

        # Check phase breakdown
        if metrics.avg_phase_durations:
            recommendations.extend(self._recommend_phase_optimization(metrics))

        # Memory optimization
        recommendations.extend(self._recommend_memory_optimization(metrics))

        return recommendations

    def _recommend_provisioned_concurrency(
        self,
        metrics: ColdStartMetrics,
    ) -> Dict[str, Any]:
        """Generate provisioned concurrency recommendation."""
        # Estimate optimal provisioned concurrency
        # Based on cold start rate and cost analysis
        cold_rate = metrics.cold_start_rate
        avg_duration_ms = metrics.avg_cold_start_duration_ms

        # Time saved per cold start avoided
        time_saved_per_start = avg_duration_ms / 1000

        # Rough estimate: 1 provisioned instance per 10% cold start rate
        estimated_instances = max(1, int(cold_rate * 10))

        return {
            "type": "provisioned_concurrency",
            "priority": "high" if cold_rate > 0.2 else "medium",
            "description": (
                f"Cold start rate is {cold_rate:.1%}. Consider provisioned concurrency."
            ),
            "details": {
                "cold_start_rate": cold_rate,
                "avg_cold_start_duration_ms": avg_duration_ms,
                "recommended_instances": estimated_instances,
                "estimated_time_saved_per_cold_start_s": time_saved_per_start,
            },
            "action": f"Configure {estimated_instances} provisioned concurrent instances",
        }

    def _recommend_init_optimization(
        self,
        metrics: ColdStartMetrics,
    ) -> List[Dict[str, Any]]:
        """Generate init optimization recommendations."""
        recommendations = []
        avg_init = metrics.avg_cold_start_duration_ms

        if avg_init > 5000:
            recommendations.append({
                "type": "init_optimization",
                "priority": "critical",
                "description": (
                    f"Init duration is {avg_init:.0f}ms (>5s). "
                    "This significantly impacts user experience."
                ),
                "details": {
                    "current_init_ms": avg_init,
                    "target_init_ms": 1000,
                    "potential_improvement_ms": avg_init - 1000,
                },
                "action": (
                    "Review initialization code: lazy load dependencies, "
                    "reduce package size, use Lambda layers"
                ),
            })
        elif avg_init > 2000:
            recommendations.append({
                "type": "init_optimization",
                "priority": "high",
                "description": (
                    f"Init duration is {avg_init:.0f}ms (>2s). "
                    "Consider optimization."
                ),
                "action": "Analyze phase breakdown and optimize slowest phases",
            })

        return recommendations

    def _recommend_phase_optimization(
        self,
        metrics: ColdStartMetrics,
    ) -> List[Dict[str, Any]]:
        """Generate phase-specific optimization recommendations."""
        recommendations = []
        phases = metrics.avg_phase_durations

        # Check dependency load
        dep_load = phases.get(InitPhase.DEPENDENCY_LOAD, 0)
        if dep_load > 1.0:
            recommendations.append({
                "type": "dependency_optimization",
                "priority": "high",
                "description": (
                    f"Dependency loading takes {dep_load:.1f}s. "
                    "Consider lazy loading or reducing dependencies."
                ),
                "action": (
                    "Use lazy imports, move heavy dependencies to Lambda layers, "
                    "or bundle with webpack/esbuild"
                ),
            })

        # Check connection setup
        conn_setup = phases.get(InitPhase.CONNECTION_SETUP, 0)
        if conn_setup > 0.5:
            recommendations.append({
                "type": "connection_optimization",
                "priority": "medium",
                "description": (
                    f"Connection setup takes {conn_setup:.1f}s. "
                    "Consider connection pooling."
                ),
                "action": (
                    "Use connection pooling, reuse connections across invocations, "
                    "or use RDS Proxy"
                ),
            })

        return recommendations

    def _recommend_memory_optimization(
        self,
        metrics: ColdStartMetrics,
    ) -> List[Dict[str, Any]]:
        """Generate memory optimization recommendations."""
        recommendations = []
        events = self.tracker.get_recent_events(100)

        if not events:
            return recommendations

        # Check memory utilization
        memory_sizes = [e.memory_size_mb for e in events]
        memory_used = [e.memory_used_mb for e in events if e.memory_used_mb > 0]

        if memory_used:
            avg_used = sum(memory_used) / len(memory_used)
            avg_size = sum(memory_sizes) / len(memory_sizes)
            utilization = avg_used / avg_size

            if utilization < 0.3:
                recommendations.append({
                    "type": "memory_optimization",
                    "priority": "medium",
                    "description": (
                        f"Memory utilization is {utilization:.1%}. "
                        "Consider reducing memory allocation."
                    ),
                    "details": {
                        "current_memory_mb": int(avg_size),
                        "avg_used_mb": int(avg_used),
                        "recommended_memory_mb": max(128, int(avg_used * 2)),
                    },
                    "action": (
                        f"Reduce memory from {int(avg_size)}MB to {max(128, int(avg_used * 2))}MB"
                    ),
                })
            elif utilization > 0.8:
                recommendations.append({
                    "type": "memory_optimization",
                    "priority": "high",
                    "description": (
                        f"Memory utilization is {utilization:.1%}. "
                        "Consider increasing memory to avoid OOM."
                    ),
                    "details": {
                        "current_memory_mb": int(avg_size),
                        "avg_used_mb": int(avg_used),
                        "recommended_memory_mb": int(avg_size * 1.5),
                    },
                    "action": (
                        f"Increase memory from {int(avg_size)}MB to {int(avg_size * 1.5)}MB"
                    ),
                })

        return recommendations

    def estimate_cost_impact(
        self,
        current_memory_mb: int,
        invocations_per_hour: int,
    ) -> Dict[str, Any]:
        """Estimate cost impact of cold start optimization.

        Args:
            current_memory_mb: Current Lambda memory
            invocations_per_hour: Average invocations per hour

        Returns:
            Cost analysis dict
        """
        metrics = self.tracker.get_metrics()
        cold_rate = metrics.cold_start_rate
        avg_init_s = metrics.avg_cold_start_duration_ms / 1000

        # Current cold start cost overhead
        cold_starts_per_hour = invocations_per_hour * cold_rate
        cold_start_duration_gb_s = (
            cold_starts_per_hour * avg_init_s * (current_memory_mb / 1024)
        )
        cold_start_cost_per_hour = cold_start_duration_gb_s * self.cost_per_gb_second

        # Provisioned concurrency cost
        # Estimate 2 instances to reduce cold starts
        provisioned_instances = 2
        provisioned_cost_per_hour = (
            provisioned_instances *
            (current_memory_mb / 1024) *
            self.provisioned_cost_per_gb_hour
        )

        # Net savings
        net_savings_per_hour = cold_start_cost_per_hour - provisioned_cost_per_hour

        return {
            "cold_start_rate": cold_rate,
            "cold_starts_per_hour": cold_starts_per_hour,
            "avg_cold_start_duration_s": avg_init_s,
            "cold_start_cost_per_hour": cold_start_cost_per_hour,
            "provisioned_cost_per_hour": provisioned_cost_per_hour,
            "net_savings_per_hour": net_savings_per_hour,
            "recommended_if_net_savings_positive": net_savings_per_hour > 0,
        }
