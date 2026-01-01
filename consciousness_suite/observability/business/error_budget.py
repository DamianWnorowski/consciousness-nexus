"""Error Budget Management

Track and alert on error budget consumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime, timedelta
from enum import Enum
import threading
import logging

from prometheus_client import Gauge, Counter

logger = logging.getLogger(__name__)


class BudgetStatus(str, Enum):
    """Status of error budget."""
    HEALTHY = "healthy"         # > 50% remaining
    WARNING = "warning"         # 20-50% remaining
    CRITICAL = "critical"       # < 20% remaining
    EXHAUSTED = "exhausted"     # 0% remaining


@dataclass
class BurnRate:
    """Error budget burn rate calculation."""
    rate: float  # Multiplier of normal burn rate
    window_hours: int
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_fast_burn(self) -> bool:
        """Check if this is a fast burn rate (>10x)."""
        return self.rate > 10

    @property
    def is_slow_burn(self) -> bool:
        """Check if this is a slow burn (1-10x)."""
        return 1 < self.rate <= 10


@dataclass
class BudgetConsumption:
    """Error budget consumption record."""
    slo_id: str
    period_start: datetime
    period_end: datetime
    total_budget_minutes: float
    consumed_minutes: float
    remaining_minutes: float
    consumption_rate: float  # minutes/hour
    burn_rate: BurnRate
    status: BudgetStatus
    estimated_exhaustion: Optional[datetime] = None

    @property
    def remaining_percentage(self) -> float:
        """Get remaining budget as percentage."""
        return (self.remaining_minutes / self.total_budget_minutes * 100
                if self.total_budget_minutes > 0 else 0)

    @property
    def consumed_percentage(self) -> float:
        """Get consumed budget as percentage."""
        return 100 - self.remaining_percentage


@dataclass
class ErrorBudget:
    """Error budget definition and state.

    Usage:
        budget = ErrorBudget(
            slo_id="api-availability",
            slo_target=99.9,
            period_days=30,
        )
        # Budget = 30 days * 24 hours * 60 min * (1 - 0.999) = 43.2 minutes
    """
    slo_id: str
    slo_target: float  # Target as percentage (e.g., 99.9)
    period_days: int = 30
    owner_team: str = "platform"

    # Calculated fields
    total_budget_minutes: float = field(init=False)
    consumed_minutes: float = field(default=0.0)
    period_start: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        # Calculate total error budget
        total_minutes = self.period_days * 24 * 60
        error_rate = (100 - self.slo_target) / 100
        self.total_budget_minutes = total_minutes * error_rate

    @property
    def remaining_minutes(self) -> float:
        return max(0, self.total_budget_minutes - self.consumed_minutes)

    @property
    def period_end(self) -> datetime:
        return self.period_start + timedelta(days=self.period_days)


class ErrorBudgetManager:
    """Manages error budgets for SLOs.

    Usage:
        manager = ErrorBudgetManager()

        # Create budget
        manager.create_budget("api-availability", slo_target=99.9, period_days=30)

        # Record downtime
        manager.consume_budget("api-availability", duration_minutes=5)

        # Check status
        consumption = manager.get_consumption("api-availability")
        print(f"Budget remaining: {consumption.remaining_percentage}%")
    """

    def __init__(self, namespace: str = "consciousness"):
        self.namespace = namespace
        self._budgets: Dict[str, ErrorBudget] = {}
        self._consumption_history: Dict[str, List[Dict[str, Any]]] = {}
        self._alerts: List[Callable[[str, BudgetStatus], None]] = []
        self._lock = threading.Lock()

        # Prometheus metrics
        self.budget_remaining = Gauge(
            f"{namespace}_error_budget_remaining_ratio",
            "Remaining error budget as ratio (0-1)",
            ["slo_id"],
        )

        self.budget_consumed = Gauge(
            f"{namespace}_error_budget_consumed_minutes",
            "Consumed error budget in minutes",
            ["slo_id"],
        )

        self.burn_rate = Gauge(
            f"{namespace}_error_budget_burn_rate",
            "Current burn rate multiplier",
            ["slo_id", "window"],
        )

        self.budget_events = Counter(
            f"{namespace}_error_budget_consumption_events_total",
            "Error budget consumption events",
            ["slo_id", "event_type"],
        )

    def create_budget(
        self,
        slo_id: str,
        slo_target: float,
        period_days: int = 30,
        owner_team: str = "platform",
    ) -> ErrorBudget:
        """Create an error budget for an SLO.

        Args:
            slo_id: SLO identifier
            slo_target: Target percentage (e.g., 99.9)
            period_days: Budget period in days
            owner_team: Team owning this SLO

        Returns:
            Created ErrorBudget
        """
        budget = ErrorBudget(
            slo_id=slo_id,
            slo_target=slo_target,
            period_days=period_days,
            owner_team=owner_team,
        )

        with self._lock:
            self._budgets[slo_id] = budget
            self._consumption_history[slo_id] = []

        logger.info(
            f"Created error budget for {slo_id}: "
            f"{budget.total_budget_minutes:.1f} minutes for {period_days} days"
        )

        return budget

    def consume_budget(
        self,
        slo_id: str,
        duration_minutes: float,
        reason: Optional[str] = None,
        incident_id: Optional[str] = None,
    ):
        """Record error budget consumption.

        Args:
            slo_id: SLO identifier
            duration_minutes: Duration of downtime/errors in minutes
            reason: Reason for consumption
            incident_id: Related incident ID
        """
        if slo_id not in self._budgets:
            logger.warning(f"Unknown budget: {slo_id}")
            return

        with self._lock:
            budget = self._budgets[slo_id]
            old_status = self._get_status(budget)

            budget.consumed_minutes += duration_minutes

            # Record in history
            self._consumption_history[slo_id].append({
                "timestamp": datetime.now(),
                "duration_minutes": duration_minutes,
                "reason": reason,
                "incident_id": incident_id,
                "remaining_after": budget.remaining_minutes,
            })

            new_status = self._get_status(budget)

        # Update metrics
        self.budget_consumed.labels(slo_id=slo_id).set(budget.consumed_minutes)
        self.budget_remaining.labels(slo_id=slo_id).set(
            budget.remaining_minutes / budget.total_budget_minutes
            if budget.total_budget_minutes > 0 else 0
        )

        self.budget_events.labels(slo_id=slo_id, event_type="consumption").inc()

        # Trigger alerts on status change
        if new_status != old_status:
            for callback in self._alerts:
                try:
                    callback(slo_id, new_status)
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")

            logger.warning(f"Error budget status changed: {slo_id} -> {new_status.value}")

    def _get_status(self, budget: ErrorBudget) -> BudgetStatus:
        """Get status from budget."""
        remaining_pct = budget.remaining_minutes / budget.total_budget_minutes * 100

        if remaining_pct <= 0:
            return BudgetStatus.EXHAUSTED
        elif remaining_pct < 20:
            return BudgetStatus.CRITICAL
        elif remaining_pct < 50:
            return BudgetStatus.WARNING
        else:
            return BudgetStatus.HEALTHY

    def calculate_burn_rate(
        self,
        slo_id: str,
        window_hours: int = 1,
    ) -> Optional[BurnRate]:
        """Calculate current burn rate.

        Args:
            slo_id: SLO identifier
            window_hours: Window for calculation

        Returns:
            BurnRate or None
        """
        if slo_id not in self._budgets:
            return None

        budget = self._budgets[slo_id]

        with self._lock:
            history = self._consumption_history.get(slo_id, [])

        if not history:
            return BurnRate(rate=0.0, window_hours=window_hours)

        # Get consumption in window
        window_start = datetime.now() - timedelta(hours=window_hours)
        window_consumption = sum(
            h["duration_minutes"]
            for h in history
            if h["timestamp"] >= window_start
        )

        # Calculate expected consumption rate
        expected_rate = budget.total_budget_minutes / (budget.period_days * 24)

        # Actual rate
        actual_rate = window_consumption / window_hours if window_hours > 0 else 0

        # Burn rate multiplier
        rate = actual_rate / expected_rate if expected_rate > 0 else 0

        burn_rate = BurnRate(rate=rate, window_hours=window_hours)

        # Update metric
        self.burn_rate.labels(slo_id=slo_id, window=f"{window_hours}h").set(rate)

        return burn_rate

    def get_consumption(self, slo_id: str) -> Optional[BudgetConsumption]:
        """Get current budget consumption status.

        Args:
            slo_id: SLO identifier

        Returns:
            BudgetConsumption or None
        """
        if slo_id not in self._budgets:
            return None

        budget = self._budgets[slo_id]

        # Calculate burn rate
        burn_rate_1h = self.calculate_burn_rate(slo_id, 1)
        burn_rate_6h = self.calculate_burn_rate(slo_id, 6)

        # Use 1h burn rate for status
        burn_rate = burn_rate_1h or BurnRate(rate=0, window_hours=1)

        # Get status
        status = self._get_status(budget)

        # Estimate exhaustion
        estimated_exhaustion = None
        if burn_rate.rate > 0 and budget.remaining_minutes > 0:
            expected_rate = budget.total_budget_minutes / (budget.period_days * 24)
            hours_to_exhaustion = budget.remaining_minutes / (burn_rate.rate * expected_rate)
            estimated_exhaustion = datetime.now() + timedelta(hours=hours_to_exhaustion)

        # Calculate consumption rate
        with self._lock:
            history = self._consumption_history.get(slo_id, [])

        last_hour = datetime.now() - timedelta(hours=1)
        recent_consumption = sum(
            h["duration_minutes"]
            for h in history
            if h["timestamp"] >= last_hour
        )

        return BudgetConsumption(
            slo_id=slo_id,
            period_start=budget.period_start,
            period_end=budget.period_end,
            total_budget_minutes=budget.total_budget_minutes,
            consumed_minutes=budget.consumed_minutes,
            remaining_minutes=budget.remaining_minutes,
            consumption_rate=recent_consumption,
            burn_rate=burn_rate,
            status=status,
            estimated_exhaustion=estimated_exhaustion,
        )

    def get_all_consumption(self) -> Dict[str, BudgetConsumption]:
        """Get consumption for all budgets.

        Returns:
            Dictionary mapping SLO ID to consumption
        """
        return {
            slo_id: self.get_consumption(slo_id)
            for slo_id in self._budgets
        }

    def reset_budget(self, slo_id: str):
        """Reset a budget for a new period.

        Args:
            slo_id: SLO identifier
        """
        if slo_id not in self._budgets:
            return

        with self._lock:
            budget = self._budgets[slo_id]
            budget.consumed_minutes = 0
            budget.period_start = datetime.now()
            self._consumption_history[slo_id] = []

        logger.info(f"Reset error budget for {slo_id}")

    def on_status_change(
        self,
        callback: Callable[[str, BudgetStatus], None],
    ):
        """Register callback for status changes.

        Args:
            callback: Function(slo_id, new_status)
        """
        self._alerts.append(callback)

    def get_history(
        self,
        slo_id: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get consumption history.

        Args:
            slo_id: SLO identifier
            limit: Maximum records to return

        Returns:
            List of consumption records
        """
        with self._lock:
            history = self._consumption_history.get(slo_id, [])

        return list(reversed(history[-limit:]))

    def get_summary(self) -> Dict[str, Any]:
        """Get error budget summary.

        Returns:
            Summary dictionary
        """
        all_consumption = self.get_all_consumption()

        healthy = sum(
            1 for c in all_consumption.values()
            if c and c.status == BudgetStatus.HEALTHY
        )
        warning = sum(
            1 for c in all_consumption.values()
            if c and c.status == BudgetStatus.WARNING
        )
        critical = sum(
            1 for c in all_consumption.values()
            if c and c.status == BudgetStatus.CRITICAL
        )
        exhausted = sum(
            1 for c in all_consumption.values()
            if c and c.status == BudgetStatus.EXHAUSTED
        )

        return {
            "total_budgets": len(self._budgets),
            "healthy": healthy,
            "warning": warning,
            "critical": critical,
            "exhausted": exhausted,
            "avg_remaining_pct": (
                sum(c.remaining_percentage for c in all_consumption.values() if c) /
                len(all_consumption) if all_consumption else 100
            ),
        }
