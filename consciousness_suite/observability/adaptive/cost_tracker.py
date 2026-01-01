"""Telemetry Cost Tracking

Tracks and attributes telemetry costs per service, team, and metric type.
Implements cost estimation, budgeting, and chargeback reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from datetime import datetime, timedelta, date
from enum import Enum
from collections import defaultdict
import threading
import logging
import json

from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)


class TelemetryType(str, Enum):
    """Types of telemetry data."""
    METRICS = "metrics"
    LOGS = "logs"
    TRACES = "traces"
    EVENTS = "events"
    PROFILES = "profiles"


class CostModel(str, Enum):
    """Cost model types."""
    PER_DATAPOINT = "per_datapoint"         # Cost per data point
    PER_SERIES = "per_series"               # Cost per active series
    PER_GB = "per_gb"                       # Cost per GB ingested
    PER_QUERY = "per_query"                 # Cost per query
    TIERED = "tiered"                       # Tiered pricing


@dataclass
class PricingTier:
    """Pricing tier for tiered cost model."""
    name: str
    up_to: int                              # Volume up to this amount
    unit_cost: float                        # Cost per unit in this tier
    base_cost: float = 0.0                  # Base cost for this tier


@dataclass
class CostConfig:
    """Cost configuration for a telemetry type.

    Usage:
        config = CostConfig(
            telemetry_type=TelemetryType.METRICS,
            cost_model=CostModel.PER_SERIES,
            unit_cost=0.003,  # $0.003 per active series per hour
            provider="datadog",
        )
    """
    telemetry_type: TelemetryType
    cost_model: CostModel
    unit_cost: float                        # Base unit cost
    provider: str = "default"
    currency: str = "USD"
    tiers: List[PricingTier] = field(default_factory=list)
    minimum_cost: float = 0.0               # Minimum cost regardless of usage
    bytes_per_unit: int = 1                 # For per_gb model: bytes per data point


@dataclass
class ServiceCost:
    """Cost tracking for a service."""
    service: str
    team: str
    telemetry_type: TelemetryType
    period_start: datetime
    period_end: datetime
    volume: int                             # Data points, bytes, etc.
    estimated_cost: float
    currency: str = "USD"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CostBudget:
    """Budget configuration for cost control.

    Usage:
        budget = CostBudget(
            name="api-team-monthly",
            owner="api-team",
            monthly_budget=5000.0,
            alert_thresholds=[0.5, 0.8, 0.95],
            enforcement_action="throttle",
        )
    """
    name: str
    owner: str                              # Team or service
    monthly_budget: float
    currency: str = "USD"
    alert_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.8, 0.95])
    enforcement_action: str = "alert"       # alert, throttle, drop
    enabled: bool = True
    rollover_unused: bool = False           # Rollover unused budget


@dataclass
class CostAlert:
    """Cost alert notification."""
    budget_name: str
    owner: str
    threshold_pct: float
    current_spend: float
    budget_amount: float
    projected_overage: float
    timestamp: datetime = field(default_factory=datetime.now)


class TelemetryCostTracker:
    """Tracks and attributes telemetry costs.

    Usage:
        tracker = TelemetryCostTracker(namespace="consciousness")

        # Configure costs
        tracker.set_cost_config(CostConfig(
            telemetry_type=TelemetryType.METRICS,
            cost_model=CostModel.PER_SERIES,
            unit_cost=0.003,
        ))

        # Set budget
        tracker.set_budget(CostBudget(
            name="api-team",
            owner="api-team",
            monthly_budget=5000.0,
        ))

        # Track usage
        tracker.record_usage(
            service="api-gateway",
            team="api-team",
            telemetry_type=TelemetryType.METRICS,
            volume=1000,
        )

        # Get cost report
        report = tracker.get_cost_report(start_date, end_date)
    """

    def __init__(
        self,
        namespace: str = "consciousness",
        default_unit_cost: float = 0.001,
    ):
        self.namespace = namespace
        self.default_unit_cost = default_unit_cost
        self._lock = threading.Lock()

        # Configuration
        self._cost_configs: Dict[TelemetryType, CostConfig] = {}
        self._budgets: Dict[str, CostBudget] = {}

        # Usage tracking
        self._usage: Dict[str, Dict[TelemetryType, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._usage_history: List[ServiceCost] = []
        self._active_series: Dict[str, Set[str]] = defaultdict(set)

        # Budget tracking
        self._budget_spend: Dict[str, float] = defaultdict(float)
        self._alerts_sent: Dict[str, Set[float]] = defaultdict(set)

        # Callbacks
        self._alert_callbacks: List[Callable[[CostAlert], None]] = []

        # Statistics
        self._total_volume: Dict[TelemetryType, int] = defaultdict(int)
        self._total_cost: float = 0.0

        # Prometheus metrics
        self.telemetry_cost = Counter(
            f"{namespace}_telemetry_cost_dollars_total",
            "Total telemetry cost",
            ["service", "team", "type"],
        )

        self.telemetry_volume = Counter(
            f"{namespace}_telemetry_volume_total",
            "Total telemetry volume",
            ["service", "team", "type"],
        )

        self.budget_usage = Gauge(
            f"{namespace}_budget_usage_ratio",
            "Budget usage ratio (0-1)",
            ["budget", "owner"],
        )

        self.cost_rate = Gauge(
            f"{namespace}_cost_rate_per_hour",
            "Current cost rate per hour",
            ["service", "type"],
        )

        self.active_series = Gauge(
            f"{namespace}_active_series",
            "Active metric series",
            ["service"],
        )

        self.budget_alerts = Counter(
            f"{namespace}_budget_alerts_total",
            "Budget alerts triggered",
            ["budget", "threshold"],
        )

    def set_cost_config(self, config: CostConfig):
        """Set cost configuration for a telemetry type.

        Args:
            config: Cost configuration
        """
        with self._lock:
            self._cost_configs[config.telemetry_type] = config

        logger.info(
            f"Set cost config: {config.telemetry_type.value} "
            f"({config.cost_model.value}, ${config.unit_cost}/unit)"
        )

    def set_budget(self, budget: CostBudget):
        """Set a cost budget.

        Args:
            budget: Budget configuration
        """
        with self._lock:
            self._budgets[budget.name] = budget

        logger.info(
            f"Set budget: {budget.name} "
            f"(${budget.monthly_budget}/month for {budget.owner})"
        )

    def record_usage(
        self,
        service: str,
        team: str,
        telemetry_type: TelemetryType,
        volume: int,
        series_id: Optional[str] = None,
        bytes_size: Optional[int] = None,
        timestamp: Optional[datetime] = None,
    ):
        """Record telemetry usage.

        Args:
            service: Service name
            team: Team owner
            telemetry_type: Type of telemetry
            volume: Volume (data points, series, etc.)
            series_id: For metrics, the series identifier
            bytes_size: Data size in bytes
            timestamp: Usage timestamp
        """
        timestamp = timestamp or datetime.now()
        key = f"{service}:{team}"

        with self._lock:
            # Track volume
            self._usage[key][telemetry_type] += volume
            self._total_volume[telemetry_type] += volume

            # Track active series for metrics
            if telemetry_type == TelemetryType.METRICS and series_id:
                self._active_series[service].add(series_id)
                self.active_series.labels(service=service).set(
                    len(self._active_series[service])
                )

        # Calculate cost
        cost = self._calculate_cost(telemetry_type, volume, bytes_size)

        # Record cost
        with self._lock:
            self._total_cost += cost

            # Update budget spend
            for budget_name, budget in self._budgets.items():
                if budget.enabled and budget.owner in (service, team):
                    self._budget_spend[budget_name] += cost

        # Update metrics
        self.telemetry_cost.labels(
            service=service,
            team=team,
            type=telemetry_type.value,
        ).inc(cost)

        self.telemetry_volume.labels(
            service=service,
            team=team,
            type=telemetry_type.value,
        ).inc(volume)

        # Store usage history
        service_cost = ServiceCost(
            service=service,
            team=team,
            telemetry_type=telemetry_type,
            period_start=timestamp,
            period_end=timestamp,
            volume=volume,
            estimated_cost=cost,
            metadata={"bytes_size": bytes_size, "series_id": series_id},
        )

        with self._lock:
            self._usage_history.append(service_cost)

            # Trim history
            if len(self._usage_history) > 100000:
                self._usage_history = self._usage_history[-50000:]

        # Check budgets
        self._check_budgets()

    def _calculate_cost(
        self,
        telemetry_type: TelemetryType,
        volume: int,
        bytes_size: Optional[int] = None,
    ) -> float:
        """Calculate cost for usage."""
        config = self._cost_configs.get(telemetry_type)

        if not config:
            return volume * self.default_unit_cost

        if config.cost_model == CostModel.PER_DATAPOINT:
            cost = volume * config.unit_cost

        elif config.cost_model == CostModel.PER_SERIES:
            # Cost per active series per hour
            # Assume this call represents 1 minute of data
            hourly_fraction = 1 / 60
            cost = volume * config.unit_cost * hourly_fraction

        elif config.cost_model == CostModel.PER_GB:
            if bytes_size:
                gb = bytes_size / (1024 ** 3)
            else:
                gb = (volume * config.bytes_per_unit) / (1024 ** 3)
            cost = gb * config.unit_cost

        elif config.cost_model == CostModel.TIERED:
            cost = self._calculate_tiered_cost(volume, config)

        else:
            cost = volume * config.unit_cost

        return max(cost, config.minimum_cost)

    def _calculate_tiered_cost(self, volume: int, config: CostConfig) -> float:
        """Calculate tiered cost."""
        if not config.tiers:
            return volume * config.unit_cost

        total_cost = 0.0
        remaining_volume = volume

        for tier in sorted(config.tiers, key=lambda t: t.up_to):
            tier_volume = min(remaining_volume, tier.up_to)
            total_cost += tier.base_cost + (tier_volume * tier.unit_cost)
            remaining_volume -= tier_volume

            if remaining_volume <= 0:
                break

        return total_cost

    def _check_budgets(self):
        """Check budget thresholds and trigger alerts."""
        for budget_name, budget in self._budgets.items():
            if not budget.enabled:
                continue

            current_spend = self._budget_spend.get(budget_name, 0.0)
            usage_ratio = current_spend / budget.monthly_budget

            # Update metric
            self.budget_usage.labels(
                budget=budget_name,
                owner=budget.owner,
            ).set(usage_ratio)

            # Check thresholds
            for threshold in budget.alert_thresholds:
                if usage_ratio >= threshold and threshold not in self._alerts_sent[budget_name]:
                    self._alerts_sent[budget_name].add(threshold)
                    self._trigger_budget_alert(budget, threshold, current_spend)

    def _trigger_budget_alert(
        self,
        budget: CostBudget,
        threshold: float,
        current_spend: float,
    ):
        """Trigger a budget alert."""
        # Project overage based on current rate
        now = datetime.now()
        days_elapsed = now.day
        days_in_month = 30  # Approximation
        projected_spend = current_spend * (days_in_month / days_elapsed)
        projected_overage = max(0, projected_spend - budget.monthly_budget)

        alert = CostAlert(
            budget_name=budget.name,
            owner=budget.owner,
            threshold_pct=threshold * 100,
            current_spend=current_spend,
            budget_amount=budget.monthly_budget,
            projected_overage=projected_overage,
        )

        self.budget_alerts.labels(
            budget=budget.name,
            threshold=f"{int(threshold * 100)}%",
        ).inc()

        logger.warning(
            f"Budget alert: {budget.name} at {threshold * 100:.0f}% "
            f"(${current_spend:.2f}/${budget.monthly_budget:.2f})"
        )

        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

    def on_alert(self, callback: Callable[[CostAlert], None]):
        """Register callback for budget alerts.

        Args:
            callback: Function to call on alert
        """
        self._alert_callbacks.append(callback)

    def get_cost_report(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        group_by: str = "service",
    ) -> Dict[str, Any]:
        """Get cost report for a period.

        Args:
            start_date: Start of period
            end_date: End of period
            group_by: Grouping (service, team, type)

        Returns:
            Cost report dictionary
        """
        start_date = start_date or (datetime.now() - timedelta(days=30))
        end_date = end_date or datetime.now()

        with self._lock:
            filtered = [
                cost for cost in self._usage_history
                if start_date <= cost.period_start <= end_date
            ]

        # Group costs
        grouped: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"volume": 0, "cost": 0.0, "breakdown": defaultdict(float)}
        )

        for cost in filtered:
            if group_by == "service":
                key = cost.service
            elif group_by == "team":
                key = cost.team
            elif group_by == "type":
                key = cost.telemetry_type.value
            else:
                key = f"{cost.service}:{cost.team}"

            grouped[key]["volume"] += cost.volume
            grouped[key]["cost"] += cost.estimated_cost
            grouped[key]["breakdown"][cost.telemetry_type.value] += cost.estimated_cost

        # Sort by cost descending
        sorted_groups = sorted(
            grouped.items(),
            key=lambda x: x[1]["cost"],
            reverse=True,
        )

        total_cost = sum(g["cost"] for _, g in sorted_groups)
        total_volume = sum(g["volume"] for _, g in sorted_groups)

        return {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "total_cost": total_cost,
            "total_volume": total_volume,
            "currency": "USD",
            "grouped_by": group_by,
            "groups": {
                key: {
                    "cost": data["cost"],
                    "volume": data["volume"],
                    "percentage": (data["cost"] / total_cost * 100) if total_cost > 0 else 0,
                    "breakdown": dict(data["breakdown"]),
                }
                for key, data in sorted_groups
            },
            "top_spenders": [
                {"name": key, "cost": data["cost"]}
                for key, data in sorted_groups[:10]
            ],
        }

    def get_budget_status(self, budget_name: Optional[str] = None) -> Dict[str, Any]:
        """Get current budget status.

        Args:
            budget_name: Specific budget (None for all)

        Returns:
            Budget status dictionary
        """
        with self._lock:
            if budget_name:
                budgets = {budget_name: self._budgets.get(budget_name)}
            else:
                budgets = dict(self._budgets)

        result = {}
        for name, budget in budgets.items():
            if not budget:
                continue

            current_spend = self._budget_spend.get(name, 0.0)

            # Calculate projected spend
            now = datetime.now()
            days_elapsed = max(now.day, 1)
            days_in_month = 30
            projected_spend = current_spend * (days_in_month / days_elapsed)

            result[name] = {
                "owner": budget.owner,
                "monthly_budget": budget.monthly_budget,
                "current_spend": current_spend,
                "usage_ratio": current_spend / budget.monthly_budget,
                "remaining": budget.monthly_budget - current_spend,
                "projected_spend": projected_spend,
                "projected_overage": max(0, projected_spend - budget.monthly_budget),
                "status": self._get_budget_status(current_spend, budget.monthly_budget),
                "currency": budget.currency,
            }

        return result

    def _get_budget_status(self, spend: float, budget: float) -> str:
        """Get budget status string."""
        ratio = spend / budget if budget > 0 else 0

        if ratio >= 1.0:
            return "exceeded"
        elif ratio >= 0.95:
            return "critical"
        elif ratio >= 0.8:
            return "warning"
        elif ratio >= 0.5:
            return "on_track"
        else:
            return "under_budget"

    def get_service_cost_breakdown(self, service: str) -> Dict[str, Any]:
        """Get detailed cost breakdown for a service.

        Args:
            service: Service name

        Returns:
            Service cost breakdown
        """
        with self._lock:
            service_costs = [
                cost for cost in self._usage_history
                if cost.service == service
            ]

        if not service_costs:
            return {"service": service, "total_cost": 0, "breakdown": {}}

        total_cost = sum(c.estimated_cost for c in service_costs)
        by_type: Dict[str, float] = defaultdict(float)

        for cost in service_costs:
            by_type[cost.telemetry_type.value] += cost.estimated_cost

        # Calculate daily trend
        daily_costs: Dict[str, float] = defaultdict(float)
        for cost in service_costs:
            day = cost.period_start.strftime("%Y-%m-%d")
            daily_costs[day] += cost.estimated_cost

        return {
            "service": service,
            "total_cost": total_cost,
            "active_series": len(self._active_series.get(service, set())),
            "breakdown_by_type": dict(by_type),
            "daily_trend": dict(sorted(daily_costs.items())),
            "cost_per_series": (
                total_cost / len(self._active_series.get(service, set()))
                if self._active_series.get(service) else 0
            ),
        }

    def get_chargeback_report(
        self,
        period_start: datetime,
        period_end: datetime,
    ) -> Dict[str, Any]:
        """Generate chargeback report for billing.

        Args:
            period_start: Start of billing period
            period_end: End of billing period

        Returns:
            Chargeback report
        """
        with self._lock:
            filtered = [
                cost for cost in self._usage_history
                if period_start <= cost.period_start <= period_end
            ]

        # Group by team
        team_costs: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "services": defaultdict(float),
                "by_type": defaultdict(float),
                "total": 0.0,
            }
        )

        for cost in filtered:
            team_costs[cost.team]["services"][cost.service] += cost.estimated_cost
            team_costs[cost.team]["by_type"][cost.telemetry_type.value] += cost.estimated_cost
            team_costs[cost.team]["total"] += cost.estimated_cost

        # Format report
        total = sum(tc["total"] for tc in team_costs.values())

        return {
            "report_type": "chargeback",
            "period": {
                "start": period_start.isoformat(),
                "end": period_end.isoformat(),
            },
            "total_cost": total,
            "currency": "USD",
            "teams": {
                team: {
                    "total_cost": data["total"],
                    "percentage": (data["total"] / total * 100) if total > 0 else 0,
                    "services": dict(data["services"]),
                    "breakdown": dict(data["by_type"]),
                }
                for team, data in sorted(
                    team_costs.items(),
                    key=lambda x: x[1]["total"],
                    reverse=True,
                )
            },
            "generated_at": datetime.now().isoformat(),
        }

    def reset_monthly_budgets(self):
        """Reset budget tracking for new month."""
        with self._lock:
            for budget_name in self._budgets:
                self._budget_spend[budget_name] = 0.0
                self._alerts_sent[budget_name] = set()

        logger.info("Reset monthly budget tracking")

    def get_statistics(self) -> Dict[str, Any]:
        """Get cost tracking statistics.

        Returns:
            Statistics dictionary
        """
        with self._lock:
            total_series = sum(
                len(series) for series in self._active_series.values()
            )
            history_size = len(self._usage_history)

        return {
            "total_cost": self._total_cost,
            "total_volume": dict(self._total_volume),
            "total_active_series": total_series,
            "history_size": history_size,
            "configured_cost_models": len(self._cost_configs),
            "active_budgets": len(self._budgets),
        }
