"""Revenue Impact Analysis

Correlates incidents and performance with revenue impact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from enum import Enum
import threading
import logging

from prometheus_client import Gauge, Counter

logger = logging.getLogger(__name__)


class ImpactSeverity(str, Enum):
    """Severity of revenue impact."""
    NEGLIGIBLE = "negligible"  # < $100
    LOW = "low"               # $100 - $1K
    MEDIUM = "medium"         # $1K - $10K
    HIGH = "high"             # $10K - $100K
    CRITICAL = "critical"     # > $100K


@dataclass
class IncidentCost:
    """Cost calculation for an incident.

    Usage:
        cost = IncidentCost(
            incident_id="inc-123",
            service="payment-service",
            duration_minutes=30,
            affected_users=10000,
            lost_transactions=500,
            revenue_per_transaction=50.0,
        )
        print(f"Total cost: ${cost.total_cost:,.2f}")
    """
    incident_id: str
    service: str
    duration_minutes: float
    affected_users: int
    lost_transactions: int = 0
    revenue_per_transaction: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    # Cost components
    direct_revenue_loss: float = 0.0
    sla_penalties: float = 0.0
    engineering_hours: float = 0.0
    engineering_cost_per_hour: float = 150.0
    customer_support_hours: float = 0.0
    support_cost_per_hour: float = 75.0
    reputation_cost_estimate: float = 0.0

    def __post_init__(self):
        if self.direct_revenue_loss == 0 and self.lost_transactions > 0:
            self.direct_revenue_loss = self.lost_transactions * self.revenue_per_transaction

    @property
    def total_cost(self) -> float:
        """Calculate total incident cost."""
        return (
            self.direct_revenue_loss +
            self.sla_penalties +
            (self.engineering_hours * self.engineering_cost_per_hour) +
            (self.customer_support_hours * self.support_cost_per_hour) +
            self.reputation_cost_estimate
        )

    @property
    def severity(self) -> ImpactSeverity:
        """Determine severity based on total cost."""
        cost = self.total_cost
        if cost < 100:
            return ImpactSeverity.NEGLIGIBLE
        elif cost < 1000:
            return ImpactSeverity.LOW
        elif cost < 10000:
            return ImpactSeverity.MEDIUM
        elif cost < 100000:
            return ImpactSeverity.HIGH
        else:
            return ImpactSeverity.CRITICAL


@dataclass
class RevenueCorrelation:
    """Correlation between metrics and revenue."""
    metric_name: str
    correlation_coefficient: float  # -1 to 1
    confidence: float  # 0 to 1
    revenue_impact_per_unit: float  # $ per unit change in metric
    window_hours: int
    timestamp: datetime = field(default_factory=datetime.now)


class RevenueImpactAnalyzer:
    """Analyzes revenue impact of incidents and performance.

    Usage:
        analyzer = RevenueImpactAnalyzer()

        # Configure revenue model
        analyzer.set_revenue_model(
            avg_revenue_per_request=0.05,
            avg_revenue_per_user_hour=2.50,
        )

        # Calculate incident cost
        cost = analyzer.calculate_incident_cost(
            incident_id="inc-123",
            service="checkout",
            duration_minutes=15,
            affected_users=5000,
            error_rate_increase=0.15,
        )

        # Get revenue-metric correlations
        correlations = analyzer.get_correlations()
    """

    def __init__(self, namespace: str = "consciousness"):
        self.namespace = namespace

        # Revenue model parameters
        self._avg_revenue_per_request = 0.0
        self._avg_revenue_per_user_hour = 0.0
        self._avg_transaction_value = 0.0
        self._hourly_baseline_revenue = 0.0

        self._incidents: List[IncidentCost] = []
        self._revenue_data: List[Dict[str, Any]] = []
        self._metric_data: List[Dict[str, Any]] = []
        self._correlations: Dict[str, RevenueCorrelation] = {}
        self._lock = threading.Lock()

        # Prometheus metrics
        self.incident_cost = Gauge(
            f"{namespace}_incident_cost_dollars",
            "Cost of incident in dollars",
            ["incident_id", "service", "severity"],
        )

        self.total_incident_cost = Counter(
            f"{namespace}_total_incident_cost_dollars",
            "Total incident cost in dollars",
            ["service"],
        )

        self.revenue_correlation = Gauge(
            f"{namespace}_revenue_metric_correlation",
            "Correlation between metric and revenue",
            ["metric"],
        )

        self.estimated_revenue_loss = Gauge(
            f"{namespace}_estimated_revenue_loss_dollars_per_hour",
            "Estimated revenue loss per hour",
            ["service"],
        )

    def set_revenue_model(
        self,
        avg_revenue_per_request: float = 0.0,
        avg_revenue_per_user_hour: float = 0.0,
        avg_transaction_value: float = 0.0,
        hourly_baseline_revenue: float = 0.0,
    ):
        """Configure revenue model parameters.

        Args:
            avg_revenue_per_request: Average revenue per API request
            avg_revenue_per_user_hour: Average revenue per user per hour
            avg_transaction_value: Average transaction value
            hourly_baseline_revenue: Expected hourly revenue
        """
        self._avg_revenue_per_request = avg_revenue_per_request
        self._avg_revenue_per_user_hour = avg_revenue_per_user_hour
        self._avg_transaction_value = avg_transaction_value
        self._hourly_baseline_revenue = hourly_baseline_revenue

    def calculate_incident_cost(
        self,
        incident_id: str,
        service: str,
        duration_minutes: float,
        affected_users: int = 0,
        error_rate_increase: float = 0.0,
        latency_increase_ms: float = 0.0,
        requests_per_minute: float = 0.0,
        sla_violated: bool = False,
        sla_penalty: float = 0.0,
        engineering_hours: float = 0.0,
        support_hours: float = 0.0,
    ) -> IncidentCost:
        """Calculate cost of an incident.

        Args:
            incident_id: Incident identifier
            service: Affected service
            duration_minutes: Duration in minutes
            affected_users: Number of affected users
            error_rate_increase: Increase in error rate (0-1)
            latency_increase_ms: Increase in latency (ms)
            requests_per_minute: Request rate
            sla_violated: Whether SLA was violated
            sla_penalty: Explicit SLA penalty amount
            engineering_hours: Hours spent by engineering
            support_hours: Hours spent by support

        Returns:
            IncidentCost calculation
        """
        # Calculate lost transactions
        if error_rate_increase > 0 and requests_per_minute > 0:
            lost_requests = requests_per_minute * duration_minutes * error_rate_increase
            lost_transactions = int(lost_requests * 0.1)  # Assume 10% are transactions
        else:
            lost_transactions = 0

        # Calculate direct revenue loss
        direct_loss = 0.0

        # From lost transactions
        if lost_transactions > 0 and self._avg_transaction_value > 0:
            direct_loss += lost_transactions * self._avg_transaction_value

        # From lost requests
        if requests_per_minute > 0 and error_rate_increase > 0:
            lost_requests = requests_per_minute * duration_minutes * error_rate_increase
            direct_loss += lost_requests * self._avg_revenue_per_request

        # From affected users
        if affected_users > 0 and self._avg_revenue_per_user_hour > 0:
            hours = duration_minutes / 60
            direct_loss += affected_users * hours * self._avg_revenue_per_user_hour * 0.5

        # Latency impact (conversion rate reduction)
        if latency_increase_ms > 100:
            # ~1% conversion drop per 100ms latency
            conversion_drop = (latency_increase_ms / 100) * 0.01
            if self._hourly_baseline_revenue > 0:
                direct_loss += self._hourly_baseline_revenue * (duration_minutes / 60) * conversion_drop

        # Reputation cost estimate (based on severity)
        reputation_cost = 0.0
        if affected_users > 10000:
            reputation_cost = affected_users * 0.10  # $0.10 per affected user
        elif affected_users > 1000:
            reputation_cost = affected_users * 0.05

        cost = IncidentCost(
            incident_id=incident_id,
            service=service,
            duration_minutes=duration_minutes,
            affected_users=affected_users,
            lost_transactions=lost_transactions,
            revenue_per_transaction=self._avg_transaction_value,
            direct_revenue_loss=direct_loss,
            sla_penalties=sla_penalty if sla_violated else 0,
            engineering_hours=engineering_hours,
            customer_support_hours=support_hours,
            reputation_cost_estimate=reputation_cost,
        )

        with self._lock:
            self._incidents.append(cost)

        # Update metrics
        self.incident_cost.labels(
            incident_id=incident_id,
            service=service,
            severity=cost.severity.value,
        ).set(cost.total_cost)

        self.total_incident_cost.labels(service=service).inc(cost.total_cost)

        logger.info(
            f"Calculated incident cost: {incident_id} = ${cost.total_cost:,.2f} "
            f"({cost.severity.value})"
        )

        return cost

    def record_revenue_data(
        self,
        revenue: float,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Record revenue data point for correlation analysis.

        Args:
            revenue: Revenue amount
            timestamp: Data timestamp
            metadata: Additional context
        """
        with self._lock:
            self._revenue_data.append({
                "revenue": revenue,
                "timestamp": timestamp or datetime.now(),
                "metadata": metadata or {},
            })

            # Keep last 10000 points
            if len(self._revenue_data) > 10000:
                self._revenue_data = self._revenue_data[-5000:]

    def record_metric_data(
        self,
        metric_name: str,
        value: float,
        timestamp: Optional[datetime] = None,
    ):
        """Record metric data point for correlation analysis.

        Args:
            metric_name: Metric name
            value: Metric value
            timestamp: Data timestamp
        """
        with self._lock:
            self._metric_data.append({
                "metric": metric_name,
                "value": value,
                "timestamp": timestamp or datetime.now(),
            })

            # Keep last 10000 points
            if len(self._metric_data) > 10000:
                self._metric_data = self._metric_data[-5000:]

    def calculate_correlation(
        self,
        metric_name: str,
        window_hours: int = 24,
    ) -> Optional[RevenueCorrelation]:
        """Calculate correlation between metric and revenue.

        Args:
            metric_name: Metric to correlate
            window_hours: Time window for correlation

        Returns:
            RevenueCorrelation or None
        """
        cutoff = datetime.now() - timedelta(hours=window_hours)

        with self._lock:
            revenue = [
                r for r in self._revenue_data
                if r["timestamp"] >= cutoff
            ]
            metrics = [
                m for m in self._metric_data
                if m["metric"] == metric_name and m["timestamp"] >= cutoff
            ]

        if len(revenue) < 10 or len(metrics) < 10:
            return None

        # Simple correlation calculation
        # In production, use proper time series correlation
        import statistics

        revenue_values = [r["revenue"] for r in revenue]
        metric_values = [m["value"] for m in metrics]

        # Align data (simplified - just use latest N points)
        n = min(len(revenue_values), len(metric_values))
        revenue_values = revenue_values[-n:]
        metric_values = metric_values[-n:]

        if n < 5:
            return None

        # Calculate Pearson correlation
        mean_r = statistics.mean(revenue_values)
        mean_m = statistics.mean(metric_values)

        numerator = sum(
            (r - mean_r) * (m - mean_m)
            for r, m in zip(revenue_values, metric_values)
        )

        std_r = statistics.stdev(revenue_values) if len(revenue_values) > 1 else 1
        std_m = statistics.stdev(metric_values) if len(metric_values) > 1 else 1

        denominator = (n - 1) * std_r * std_m

        correlation = numerator / denominator if denominator > 0 else 0

        # Calculate impact per unit
        if std_m > 0:
            impact_per_unit = correlation * (std_r / std_m)
        else:
            impact_per_unit = 0

        result = RevenueCorrelation(
            metric_name=metric_name,
            correlation_coefficient=correlation,
            confidence=min(n / 100, 1.0),  # More data = more confidence
            revenue_impact_per_unit=impact_per_unit,
            window_hours=window_hours,
        )

        with self._lock:
            self._correlations[metric_name] = result

        # Update metric
        self.revenue_correlation.labels(metric=metric_name).set(correlation)

        return result

    def get_correlations(self) -> Dict[str, RevenueCorrelation]:
        """Get all calculated correlations.

        Returns:
            Dictionary of correlations
        """
        with self._lock:
            return self._correlations.copy()

    def get_incidents(
        self,
        service: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[IncidentCost]:
        """Get incident cost records.

        Args:
            service: Filter by service
            since: Only incidents after this time

        Returns:
            List of incidents
        """
        with self._lock:
            incidents = self._incidents.copy()

        if service:
            incidents = [i for i in incidents if i.service == service]
        if since:
            incidents = [i for i in incidents if i.timestamp >= since]

        return incidents

    def get_summary(
        self,
        window_hours: int = 24,
    ) -> Dict[str, Any]:
        """Get revenue impact summary.

        Args:
            window_hours: Time window

        Returns:
            Summary dictionary
        """
        cutoff = datetime.now() - timedelta(hours=window_hours)

        with self._lock:
            recent_incidents = [
                i for i in self._incidents
                if i.timestamp >= cutoff
            ]

        total_cost = sum(i.total_cost for i in recent_incidents)
        by_service: Dict[str, float] = {}

        for incident in recent_incidents:
            by_service[incident.service] = by_service.get(
                incident.service, 0
            ) + incident.total_cost

        return {
            "window_hours": window_hours,
            "incident_count": len(recent_incidents),
            "total_cost": total_cost,
            "avg_cost_per_incident": total_cost / len(recent_incidents) if recent_incidents else 0,
            "cost_by_service": by_service,
            "highest_cost_incident": max(
                recent_incidents, key=lambda i: i.total_cost
            ) if recent_incidents else None,
        }
