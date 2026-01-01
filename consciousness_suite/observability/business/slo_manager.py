"""SLO (Service Level Objective) Manager

Defines and tracks Service Level Objectives with alerting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime, timedelta
from enum import Enum
import threading
import logging

from prometheus_client import Counter, Gauge

logger = logging.getLogger(__name__)


class SLOType(str, Enum):
    """Types of SLOs."""
    AVAILABILITY = "availability"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    FRESHNESS = "freshness"
    CORRECTNESS = "correctness"
    CUSTOM = "custom"


class SLOPeriod(str, Enum):
    """SLO measurement periods."""
    ROLLING_7D = "rolling_7d"
    ROLLING_28D = "rolling_28d"
    ROLLING_30D = "rolling_30d"
    CALENDAR_MONTH = "calendar_month"
    CALENDAR_QUARTER = "calendar_quarter"


@dataclass
class SLODefinition:
    """Definition of a Service Level Objective.

    Usage:
        slo = SLODefinition(
            id="api-availability",
            name="API Availability",
            service="api-gateway",
            slo_type=SLOType.AVAILABILITY,
            target_percentage=99.9,
            period=SLOPeriod.ROLLING_30D,
            description="API should be available 99.9% of the time",
        )
    """
    id: str
    name: str
    service: str
    slo_type: SLOType
    target_percentage: float
    period: SLOPeriod = SLOPeriod.ROLLING_30D
    description: str = ""
    owner_team: str = "platform"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # For latency SLOs
    latency_threshold_ms: Optional[float] = None
    latency_percentile: float = 99.0

    # Alert thresholds
    warning_threshold: float = 0.1  # Alert at 10% error budget consumed
    critical_threshold: float = 0.5  # Alert at 50% error budget consumed


@dataclass
class SLOStatus:
    """Current status of an SLO."""
    slo_id: str
    current_value: float
    target_value: float
    period_start: datetime
    period_end: datetime
    is_meeting_target: bool
    error_budget_remaining: float
    error_budget_consumed: float
    trend: str  # improving, degrading, stable
    last_violation: Optional[datetime] = None
    measurements: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SLOViolation:
    """Record of an SLO violation."""
    slo_id: str
    service: str
    violation_start: datetime
    violation_end: Optional[datetime]
    expected_value: float
    actual_value: float
    duration_seconds: float
    impact_percentage: float
    resolved: bool = False
    root_cause: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SLOManager:
    """Manages SLO definitions, tracking, and alerting.

    Usage:
        manager = SLOManager()

        # Define SLO
        manager.define_slo(SLODefinition(
            id="api-availability",
            name="API Availability",
            service="api-gateway",
            slo_type=SLOType.AVAILABILITY,
            target_percentage=99.9,
        ))

        # Record measurements
        manager.record_measurement("api-availability", success=True)

        # Get status
        status = manager.get_slo_status("api-availability")

        # Register alert callback
        manager.on_violation(notify_team)
    """

    def __init__(self, namespace: str = "consciousness"):
        self.namespace = namespace
        self._slos: Dict[str, SLODefinition] = {}
        self._measurements: Dict[str, List[Dict[str, Any]]] = {}
        self._violations: Dict[str, List[SLOViolation]] = {}
        self._active_violations: Dict[str, SLOViolation] = {}
        self._callbacks: List[Callable[[SLOViolation], None]] = []
        self._lock = threading.Lock()
        self._max_measurements = 100000

        # Prometheus metrics
        self.slo_target = Gauge(
            f"{namespace}_slo_target_ratio",
            "SLO target as ratio (0-1)",
            ["slo_id", "service", "slo_type"],
        )

        self.slo_current = Gauge(
            f"{namespace}_slo_current_ratio",
            "Current SLO value as ratio (0-1)",
            ["slo_id", "service", "slo_type"],
        )

        self.slo_error_budget = Gauge(
            f"{namespace}_slo_error_budget_remaining_ratio",
            "Remaining error budget as ratio (0-1)",
            ["slo_id", "service"],
        )

        self.slo_violations = Counter(
            f"{namespace}_slo_violations_total",
            "Total SLO violations",
            ["slo_id", "service"],
        )

        self.slo_measurements = Counter(
            f"{namespace}_slo_measurements_total",
            "Total SLO measurements",
            ["slo_id", "service", "result"],
        )

    def define_slo(self, slo: SLODefinition):
        """Define a new SLO.

        Args:
            slo: SLO definition
        """
        with self._lock:
            self._slos[slo.id] = slo
            self._measurements[slo.id] = []
            self._violations[slo.id] = []

        # Set target metric
        self.slo_target.labels(
            slo_id=slo.id,
            service=slo.service,
            slo_type=slo.slo_type.value,
        ).set(slo.target_percentage / 100)

        logger.info(f"Defined SLO: {slo.id} ({slo.target_percentage}% target)")

    def remove_slo(self, slo_id: str):
        """Remove an SLO definition.

        Args:
            slo_id: SLO ID to remove
        """
        with self._lock:
            if slo_id in self._slos:
                del self._slos[slo_id]
                del self._measurements[slo_id]

    def record_measurement(
        self,
        slo_id: str,
        success: bool,
        latency_ms: Optional[float] = None,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Record an SLO measurement.

        Args:
            slo_id: SLO ID
            success: Whether the measurement was successful
            latency_ms: Latency in milliseconds (for latency SLOs)
            timestamp: Measurement timestamp
            metadata: Additional context
        """
        if slo_id not in self._slos:
            logger.warning(f"Unknown SLO: {slo_id}")
            return

        slo = self._slos[slo_id]
        ts = timestamp or datetime.now()

        measurement = {
            "success": success,
            "latency_ms": latency_ms,
            "timestamp": ts,
            "metadata": metadata or {},
        }

        with self._lock:
            self._measurements[slo_id].append(measurement)

            # Trim old measurements
            if len(self._measurements[slo_id]) > self._max_measurements:
                self._measurements[slo_id] = self._measurements[slo_id][-self._max_measurements // 2:]

        # Update metrics
        self.slo_measurements.labels(
            slo_id=slo_id,
            service=slo.service,
            result="success" if success else "failure",
        ).inc()

        # Check for violations
        self._check_violation(slo_id)

    def _check_violation(self, slo_id: str):
        """Check if SLO is violated and trigger alerts."""
        status = self.get_slo_status(slo_id)
        if not status:
            return

        slo = self._slos[slo_id]

        if not status.is_meeting_target:
            if slo_id not in self._active_violations:
                # New violation
                violation = SLOViolation(
                    slo_id=slo_id,
                    service=slo.service,
                    violation_start=datetime.now(),
                    violation_end=None,
                    expected_value=slo.target_percentage,
                    actual_value=status.current_value * 100,
                    duration_seconds=0,
                    impact_percentage=slo.target_percentage - (status.current_value * 100),
                )

                with self._lock:
                    self._active_violations[slo_id] = violation
                    self._violations[slo_id].append(violation)

                self.slo_violations.labels(
                    slo_id=slo_id,
                    service=slo.service,
                ).inc()

                # Trigger callbacks
                for callback in self._callbacks:
                    try:
                        callback(violation)
                    except Exception as e:
                        logger.error(f"SLO callback error: {e}")

                logger.warning(
                    f"SLO violation: {slo_id} at {status.current_value * 100:.2f}% "
                    f"(target: {slo.target_percentage}%)"
                )

        else:
            # Check if violation resolved
            if slo_id in self._active_violations:
                with self._lock:
                    violation = self._active_violations.pop(slo_id)
                    violation.violation_end = datetime.now()
                    violation.resolved = True
                    violation.duration_seconds = (
                        violation.violation_end - violation.violation_start
                    ).total_seconds()

                logger.info(f"SLO violation resolved: {slo_id}")

    def get_slo_status(self, slo_id: str) -> Optional[SLOStatus]:
        """Get current status of an SLO.

        Args:
            slo_id: SLO ID

        Returns:
            SLOStatus or None
        """
        if slo_id not in self._slos:
            return None

        slo = self._slos[slo_id]

        with self._lock:
            measurements = self._measurements.get(slo_id, [])

        # Calculate time window
        now = datetime.now()
        if slo.period == SLOPeriod.ROLLING_7D:
            window_start = now - timedelta(days=7)
        elif slo.period == SLOPeriod.ROLLING_28D:
            window_start = now - timedelta(days=28)
        elif slo.period == SLOPeriod.ROLLING_30D:
            window_start = now - timedelta(days=30)
        else:
            window_start = now - timedelta(days=30)

        # Filter measurements in window
        window_measurements = [
            m for m in measurements
            if m["timestamp"] >= window_start
        ]

        if not window_measurements:
            return SLOStatus(
                slo_id=slo_id,
                current_value=1.0,
                target_value=slo.target_percentage / 100,
                period_start=window_start,
                period_end=now,
                is_meeting_target=True,
                error_budget_remaining=1.0,
                error_budget_consumed=0.0,
                trend="stable",
                measurements=0,
            )

        # Calculate current value based on SLO type
        if slo.slo_type == SLOType.AVAILABILITY:
            success_count = sum(1 for m in window_measurements if m["success"])
            current_value = success_count / len(window_measurements)

        elif slo.slo_type == SLOType.LATENCY:
            if slo.latency_threshold_ms:
                good_count = sum(
                    1 for m in window_measurements
                    if m.get("latency_ms", 0) <= slo.latency_threshold_ms
                )
                current_value = good_count / len(window_measurements)
            else:
                current_value = 1.0

        elif slo.slo_type == SLOType.ERROR_RATE:
            error_count = sum(1 for m in window_measurements if not m["success"])
            current_value = 1 - (error_count / len(window_measurements))

        else:
            success_count = sum(1 for m in window_measurements if m["success"])
            current_value = success_count / len(window_measurements)

        target_value = slo.target_percentage / 100
        is_meeting = current_value >= target_value

        # Calculate error budget
        error_budget_total = 1 - target_value
        error_budget_used = max(0, target_value - current_value)
        error_budget_consumed = (
            error_budget_used / error_budget_total
            if error_budget_total > 0 else 0
        )
        error_budget_remaining = max(0, 1 - error_budget_consumed)

        # Determine trend
        half_window = (now - window_start) / 2
        mid_point = window_start + half_window

        first_half = [m for m in window_measurements if m["timestamp"] < mid_point]
        second_half = [m for m in window_measurements if m["timestamp"] >= mid_point]

        if first_half and second_half:
            first_success = sum(1 for m in first_half if m["success"]) / len(first_half)
            second_success = sum(1 for m in second_half if m["success"]) / len(second_half)

            if second_success > first_success + 0.01:
                trend = "improving"
            elif second_success < first_success - 0.01:
                trend = "degrading"
            else:
                trend = "stable"
        else:
            trend = "stable"

        # Get last violation
        last_violation = None
        with self._lock:
            violations = self._violations.get(slo_id, [])
            if violations:
                last_violation = violations[-1].violation_start

        # Update Prometheus metrics
        self.slo_current.labels(
            slo_id=slo_id,
            service=slo.service,
            slo_type=slo.slo_type.value,
        ).set(current_value)

        self.slo_error_budget.labels(
            slo_id=slo_id,
            service=slo.service,
        ).set(error_budget_remaining)

        return SLOStatus(
            slo_id=slo_id,
            current_value=current_value,
            target_value=target_value,
            period_start=window_start,
            period_end=now,
            is_meeting_target=is_meeting,
            error_budget_remaining=error_budget_remaining,
            error_budget_consumed=error_budget_consumed,
            trend=trend,
            last_violation=last_violation,
            measurements=len(window_measurements),
        )

    def get_all_slo_status(self) -> Dict[str, SLOStatus]:
        """Get status of all SLOs.

        Returns:
            Dictionary mapping SLO ID to status
        """
        return {
            slo_id: self.get_slo_status(slo_id)
            for slo_id in self._slos
        }

    def get_violations(
        self,
        slo_id: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[SLOViolation]:
        """Get SLO violations.

        Args:
            slo_id: Filter by SLO (None for all)
            since: Only violations after this time

        Returns:
            List of violations
        """
        with self._lock:
            if slo_id:
                violations = self._violations.get(slo_id, [])
            else:
                violations = []
                for v_list in self._violations.values():
                    violations.extend(v_list)

        if since:
            violations = [v for v in violations if v.violation_start >= since]

        return violations

    def get_active_violations(self) -> List[SLOViolation]:
        """Get currently active violations.

        Returns:
            List of active violations
        """
        with self._lock:
            return list(self._active_violations.values())

    def on_violation(self, callback: Callable[[SLOViolation], None]):
        """Register a callback for violations.

        Args:
            callback: Function to call on violation
        """
        self._callbacks.append(callback)

    def get_summary(self) -> Dict[str, Any]:
        """Get SLO summary.

        Returns:
            Summary dictionary
        """
        all_status = self.get_all_slo_status()

        meeting_target = sum(1 for s in all_status.values() if s and s.is_meeting_target)
        degrading = sum(1 for s in all_status.values() if s and s.trend == "degrading")

        return {
            "total_slos": len(self._slos),
            "meeting_target": meeting_target,
            "not_meeting_target": len(all_status) - meeting_target,
            "degrading": degrading,
            "active_violations": len(self._active_violations),
            "avg_error_budget_remaining": (
                sum(s.error_budget_remaining for s in all_status.values() if s) /
                len(all_status) if all_status else 1.0
            ),
        }
