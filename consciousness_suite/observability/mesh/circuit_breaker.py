"""Circuit Breaker Observability

Metrics and monitoring for circuit breaker patterns in the service mesh.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from enum import Enum
from datetime import datetime, timedelta
import threading
import logging

from prometheus_client import Counter, Gauge, Histogram, Info

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitEvent:
    """Event from a circuit breaker."""
    circuit_name: str
    source_service: str
    target_service: str
    state: CircuitState
    previous_state: Optional[CircuitState]
    timestamp: datetime
    failure_count: int
    success_count: int
    last_failure_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitStats:
    """Statistics for a circuit breaker."""
    name: str
    state: CircuitState
    failure_count: int
    success_count: int
    total_requests: int
    rejection_count: int
    last_state_change: datetime
    time_in_current_state: timedelta
    failure_rate: float
    avg_response_time_ms: float


class CircuitBreakerMetrics:
    """Circuit breaker metrics collection.

    Usage:
        metrics = CircuitBreakerMetrics()

        # Record state change
        metrics.record_state_change("auth-service", CircuitState.OPEN, CircuitState.CLOSED)

        # Record request outcome
        metrics.record_request("auth-service", success=True, latency_ms=50)

        # Get current status
        status = metrics.get_circuit_status("auth-service")
    """

    def __init__(self, namespace: str = "consciousness"):
        self.namespace = namespace
        self._circuits: Dict[str, CircuitStats] = {}
        self._events: List[CircuitEvent] = []
        self._lock = threading.Lock()

        # Prometheus metrics
        self.circuit_state = Gauge(
            f"{namespace}_circuit_breaker_state",
            "Current circuit breaker state (0=closed, 1=half_open, 2=open)",
            ["circuit", "source", "target"],
        )

        self.circuit_state_changes = Counter(
            f"{namespace}_circuit_breaker_state_changes_total",
            "Total circuit breaker state changes",
            ["circuit", "source", "target", "from_state", "to_state"],
        )

        self.circuit_requests = Counter(
            f"{namespace}_circuit_breaker_requests_total",
            "Total requests through circuit breaker",
            ["circuit", "source", "target", "outcome"],
        )

        self.circuit_rejections = Counter(
            f"{namespace}_circuit_breaker_rejections_total",
            "Requests rejected by open circuit",
            ["circuit", "source", "target"],
        )

        self.circuit_latency = Histogram(
            f"{namespace}_circuit_breaker_latency_seconds",
            "Request latency through circuit breaker",
            ["circuit", "source", "target"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
        )

        self.circuit_failure_rate = Gauge(
            f"{namespace}_circuit_breaker_failure_rate",
            "Current failure rate",
            ["circuit", "source", "target"],
        )

    def record_state_change(
        self,
        circuit_name: str,
        new_state: CircuitState,
        previous_state: Optional[CircuitState] = None,
        source_service: str = "unknown",
        target_service: str = "unknown",
        failure_reason: Optional[str] = None,
    ):
        """Record a circuit breaker state change.

        Args:
            circuit_name: Name of the circuit breaker
            new_state: New state
            previous_state: Previous state
            source_service: Source service name
            target_service: Target service name
            failure_reason: Reason for failure if transitioning to OPEN
        """
        now = datetime.now()

        with self._lock:
            # Get or create circuit stats
            if circuit_name not in self._circuits:
                self._circuits[circuit_name] = CircuitStats(
                    name=circuit_name,
                    state=new_state,
                    failure_count=0,
                    success_count=0,
                    total_requests=0,
                    rejection_count=0,
                    last_state_change=now,
                    time_in_current_state=timedelta(),
                    failure_rate=0.0,
                    avg_response_time_ms=0.0,
                )
            else:
                stats = self._circuits[circuit_name]
                stats.time_in_current_state = now - stats.last_state_change
                stats.last_state_change = now
                stats.state = new_state

            # Record event
            event = CircuitEvent(
                circuit_name=circuit_name,
                source_service=source_service,
                target_service=target_service,
                state=new_state,
                previous_state=previous_state,
                timestamp=now,
                failure_count=self._circuits[circuit_name].failure_count,
                success_count=self._circuits[circuit_name].success_count,
                last_failure_reason=failure_reason,
            )
            self._events.append(event)

            # Keep only last 1000 events
            if len(self._events) > 1000:
                self._events = self._events[-500:]

        # Update Prometheus metrics
        state_value = {"closed": 0, "half_open": 1, "open": 2}.get(new_state.value, 0)
        self.circuit_state.labels(
            circuit=circuit_name,
            source=source_service,
            target=target_service,
        ).set(state_value)

        if previous_state:
            self.circuit_state_changes.labels(
                circuit=circuit_name,
                source=source_service,
                target=target_service,
                from_state=previous_state.value,
                to_state=new_state.value,
            ).inc()

        logger.info(
            f"Circuit {circuit_name} state change: {previous_state} -> {new_state}"
        )

    def record_request(
        self,
        circuit_name: str,
        success: bool,
        latency_ms: float,
        source_service: str = "unknown",
        target_service: str = "unknown",
        rejected: bool = False,
    ):
        """Record a request through the circuit breaker.

        Args:
            circuit_name: Name of the circuit breaker
            success: Whether the request succeeded
            latency_ms: Request latency in milliseconds
            source_service: Source service name
            target_service: Target service name
            rejected: Whether the request was rejected by open circuit
        """
        with self._lock:
            if circuit_name not in self._circuits:
                self._circuits[circuit_name] = CircuitStats(
                    name=circuit_name,
                    state=CircuitState.CLOSED,
                    failure_count=0,
                    success_count=0,
                    total_requests=0,
                    rejection_count=0,
                    last_state_change=datetime.now(),
                    time_in_current_state=timedelta(),
                    failure_rate=0.0,
                    avg_response_time_ms=0.0,
                )

            stats = self._circuits[circuit_name]
            stats.total_requests += 1

            if rejected:
                stats.rejection_count += 1
            elif success:
                stats.success_count += 1
            else:
                stats.failure_count += 1

            # Update failure rate (sliding window would be better)
            total = stats.success_count + stats.failure_count
            if total > 0:
                stats.failure_rate = stats.failure_count / total

            # Update average response time (simplified)
            stats.avg_response_time_ms = (
                stats.avg_response_time_ms * 0.9 + latency_ms * 0.1
            )

        # Update Prometheus metrics
        labels = {
            "circuit": circuit_name,
            "source": source_service,
            "target": target_service,
        }

        if rejected:
            self.circuit_rejections.labels(**labels).inc()
            self.circuit_requests.labels(**labels, outcome="rejected").inc()
        elif success:
            self.circuit_requests.labels(**labels, outcome="success").inc()
        else:
            self.circuit_requests.labels(**labels, outcome="failure").inc()

        self.circuit_latency.labels(**labels).observe(latency_ms / 1000)
        self.circuit_failure_rate.labels(**labels).set(
            self._circuits[circuit_name].failure_rate
        )

    def get_circuit_status(self, circuit_name: str) -> Optional[CircuitStats]:
        """Get current status of a circuit breaker."""
        with self._lock:
            if circuit_name in self._circuits:
                stats = self._circuits[circuit_name]
                stats.time_in_current_state = datetime.now() - stats.last_state_change
                return stats
            return None

    def get_all_circuits(self) -> Dict[str, CircuitStats]:
        """Get status of all circuit breakers."""
        with self._lock:
            now = datetime.now()
            for stats in self._circuits.values():
                stats.time_in_current_state = now - stats.last_state_change
            return self._circuits.copy()

    def get_open_circuits(self) -> List[CircuitStats]:
        """Get all circuits currently in OPEN state."""
        with self._lock:
            return [
                stats for stats in self._circuits.values()
                if stats.state == CircuitState.OPEN
            ]

    def get_recent_events(self, count: int = 50) -> List[CircuitEvent]:
        """Get recent circuit breaker events."""
        with self._lock:
            return list(reversed(self._events[-count:]))

    def get_events_for_circuit(
        self, circuit_name: str, count: int = 20
    ) -> List[CircuitEvent]:
        """Get recent events for a specific circuit."""
        with self._lock:
            events = [e for e in self._events if e.circuit_name == circuit_name]
            return list(reversed(events[-count:]))

    def health_summary(self) -> Dict[str, Any]:
        """Get overall circuit breaker health summary."""
        with self._lock:
            total = len(self._circuits)
            open_count = sum(
                1 for s in self._circuits.values() if s.state == CircuitState.OPEN
            )
            half_open_count = sum(
                1 for s in self._circuits.values() if s.state == CircuitState.HALF_OPEN
            )
            closed_count = total - open_count - half_open_count

            avg_failure_rate = (
                sum(s.failure_rate for s in self._circuits.values()) / total
                if total > 0
                else 0.0
            )

            return {
                "total_circuits": total,
                "closed": closed_count,
                "half_open": half_open_count,
                "open": open_count,
                "avg_failure_rate": avg_failure_rate,
                "health_score": closed_count / total if total > 0 else 1.0,
                "open_circuits": [
                    s.name for s in self._circuits.values()
                    if s.state == CircuitState.OPEN
                ],
            }


class CircuitBreakerObserver:
    """Observer that can be attached to circuit breakers.

    Usage:
        metrics = CircuitBreakerMetrics()
        observer = CircuitBreakerObserver(metrics)

        # Attach to your circuit breaker library
        circuit_breaker.add_listener(observer.on_state_change)
    """

    def __init__(self, metrics: CircuitBreakerMetrics):
        self.metrics = metrics

    def on_state_change(
        self,
        circuit_name: str,
        new_state: str,
        previous_state: Optional[str] = None,
        **kwargs,
    ):
        """Handle circuit state change event."""
        state_map = {
            "closed": CircuitState.CLOSED,
            "open": CircuitState.OPEN,
            "half_open": CircuitState.HALF_OPEN,
            "half-open": CircuitState.HALF_OPEN,
        }

        new = state_map.get(new_state.lower(), CircuitState.CLOSED)
        prev = state_map.get(previous_state.lower(), None) if previous_state else None

        self.metrics.record_state_change(
            circuit_name=circuit_name,
            new_state=new,
            previous_state=prev,
            source_service=kwargs.get("source", "unknown"),
            target_service=kwargs.get("target", "unknown"),
            failure_reason=kwargs.get("reason"),
        )

    def on_request(
        self,
        circuit_name: str,
        success: bool,
        latency_ms: float,
        rejected: bool = False,
        **kwargs,
    ):
        """Handle request event."""
        self.metrics.record_request(
            circuit_name=circuit_name,
            success=success,
            latency_ms=latency_ms,
            source_service=kwargs.get("source", "unknown"),
            target_service=kwargs.get("target", "unknown"),
            rejected=rejected,
        )
