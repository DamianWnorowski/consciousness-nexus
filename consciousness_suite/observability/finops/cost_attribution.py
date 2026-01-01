"""Cost Attribution

Attribute cloud and infrastructure costs to services and teams.
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


class CostCategory(str, Enum):
    """Categories of costs."""
    COMPUTE = "compute"
    STORAGE = "storage"
    NETWORK = "network"
    DATABASE = "database"
    AI_ML = "ai_ml"
    OBSERVABILITY = "observability"
    SECURITY = "security"
    OTHER = "other"


class AllocationMethod(str, Enum):
    """Methods for cost allocation."""
    DIRECT = "direct"           # Direct cost assignment
    USAGE_BASED = "usage_based" # Based on resource usage
    HEADCOUNT = "headcount"     # Based on team size
    EQUAL_SPLIT = "equal_split" # Split equally
    CUSTOM = "custom"           # Custom allocation rules


@dataclass
class ServiceCost:
    """Cost record for a service."""
    service: str
    cost_usd: float
    category: CostCategory
    period_start: datetime
    period_end: datetime
    resource_usage: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def cost_per_day(self) -> float:
        """Get daily cost rate."""
        days = (self.period_end - self.period_start).days or 1
        return self.cost_usd / days


@dataclass
class TeamCost:
    """Aggregated cost for a team."""
    team: str
    total_cost_usd: float
    by_service: Dict[str, float]
    by_category: Dict[CostCategory, float]
    period_start: datetime
    period_end: datetime
    headcount: int = 0

    @property
    def cost_per_head(self) -> float:
        """Get cost per team member."""
        return self.total_cost_usd / self.headcount if self.headcount > 0 else 0


@dataclass
class CostAllocation:
    """Cost allocation record."""
    source_cost: float
    allocated_cost: float
    allocation_method: AllocationMethod
    target_service: str
    target_team: str
    allocation_percentage: float
    justification: str = ""


class CostAttributor:
    """Attributes costs to services and teams.

    Usage:
        attributor = CostAttributor()

        # Configure service-team mapping
        attributor.map_service_to_team("api-gateway", "platform")
        attributor.map_service_to_team("payment-service", "payments")

        # Record costs
        attributor.record_cost(ServiceCost(
            service="api-gateway",
            cost_usd=1000,
            category=CostCategory.COMPUTE,
            period_start=datetime(2024, 1, 1),
            period_end=datetime(2024, 1, 31),
        ))

        # Get team costs
        team_cost = attributor.get_team_cost("platform")
    """

    def __init__(self, namespace: str = "consciousness"):
        self.namespace = namespace
        self._service_team_map: Dict[str, str] = {}
        self._team_headcount: Dict[str, int] = {}
        self._costs: List[ServiceCost] = []
        self._allocations: List[CostAllocation] = []
        self._shared_cost_rules: Dict[str, Dict[str, float]] = {}
        self._lock = threading.Lock()

        # Prometheus metrics
        self.service_cost = Gauge(
            f"{namespace}_finops_service_cost_dollars",
            "Service cost in dollars",
            ["service", "category"],
        )

        self.team_cost = Gauge(
            f"{namespace}_finops_team_cost_dollars",
            "Team cost in dollars",
            ["team"],
        )

        self.cost_per_request = Gauge(
            f"{namespace}_finops_cost_per_request_cents",
            "Cost per request in cents",
            ["service"],
        )

        self.budget_utilization = Gauge(
            f"{namespace}_finops_budget_utilization_ratio",
            "Budget utilization ratio",
            ["team"],
        )

    def map_service_to_team(self, service: str, team: str):
        """Map a service to a team.

        Args:
            service: Service name
            team: Team name
        """
        with self._lock:
            self._service_team_map[service] = team

    def set_team_headcount(self, team: str, headcount: int):
        """Set team headcount for cost-per-head calculations.

        Args:
            team: Team name
            headcount: Number of team members
        """
        with self._lock:
            self._team_headcount[team] = headcount

    def set_shared_cost_rule(
        self,
        service: str,
        allocation: Dict[str, float],
    ):
        """Set allocation rule for shared services.

        Args:
            service: Shared service name
            allocation: Dict mapping teams to percentage (must sum to 100)
        """
        total = sum(allocation.values())
        if abs(total - 100) > 0.01:
            logger.warning(f"Allocation percentages sum to {total}, not 100")

        with self._lock:
            self._shared_cost_rules[service] = allocation

    def record_cost(self, cost: ServiceCost):
        """Record a service cost.

        Args:
            cost: Service cost record
        """
        with self._lock:
            self._costs.append(cost)

        # Update metrics
        self.service_cost.labels(
            service=cost.service,
            category=cost.category.value,
        ).set(cost.cost_usd)

    def record_usage(
        self,
        service: str,
        requests: int,
        compute_hours: float = 0,
        storage_gb: float = 0,
        network_gb: float = 0,
    ):
        """Record resource usage for cost attribution.

        Args:
            service: Service name
            requests: Number of requests
            compute_hours: Compute hours used
            storage_gb: Storage in GB
            network_gb: Network transfer in GB
        """
        # This would be used for usage-based allocation
        # Implementation would track usage over time
        pass

    def allocate_shared_cost(
        self,
        source_service: str,
        total_cost: float,
        period_start: datetime,
        period_end: datetime,
    ) -> List[CostAllocation]:
        """Allocate shared service cost to teams.

        Args:
            source_service: Shared service
            total_cost: Total cost to allocate
            period_start: Period start
            period_end: Period end

        Returns:
            List of allocations
        """
        allocations = []

        with self._lock:
            rules = self._shared_cost_rules.get(source_service, {})

        if not rules:
            # Default: split among all teams
            with self._lock:
                teams = set(self._service_team_map.values())

            if teams:
                split = 100 / len(teams)
                rules = {team: split for team in teams}

        for team, percentage in rules.items():
            allocated = total_cost * (percentage / 100)

            allocation = CostAllocation(
                source_cost=total_cost,
                allocated_cost=allocated,
                allocation_method=AllocationMethod.CUSTOM,
                target_service=source_service,
                target_team=team,
                allocation_percentage=percentage,
                justification=f"Shared cost allocation: {percentage}%",
            )

            allocations.append(allocation)

            with self._lock:
                self._allocations.append(allocation)

        return allocations

    def get_service_cost(
        self,
        service: str,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
    ) -> float:
        """Get total cost for a service.

        Args:
            service: Service name
            period_start: Start of period
            period_end: End of period

        Returns:
            Total cost in USD
        """
        with self._lock:
            costs = [c for c in self._costs if c.service == service]

        if period_start:
            costs = [c for c in costs if c.period_end >= period_start]
        if period_end:
            costs = [c for c in costs if c.period_start <= period_end]

        return sum(c.cost_usd for c in costs)

    def get_team_cost(
        self,
        team: str,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
    ) -> TeamCost:
        """Get aggregated cost for a team.

        Args:
            team: Team name
            period_start: Start of period
            period_end: End of period

        Returns:
            TeamCost
        """
        now = datetime.now()
        period_start = period_start or now - timedelta(days=30)
        period_end = period_end or now

        with self._lock:
            # Get services for team
            team_services = [
                s for s, t in self._service_team_map.items()
                if t == team
            ]

            # Get costs for team services
            costs = [
                c for c in self._costs
                if c.service in team_services
                and c.period_end >= period_start
                and c.period_start <= period_end
            ]

            # Get allocated costs
            allocations = [
                a for a in self._allocations
                if a.target_team == team
            ]

            headcount = self._team_headcount.get(team, 0)

        # Aggregate by service
        by_service: Dict[str, float] = {}
        for cost in costs:
            by_service[cost.service] = by_service.get(cost.service, 0) + cost.cost_usd

        # Add allocations
        for allocation in allocations:
            by_service[allocation.target_service] = (
                by_service.get(allocation.target_service, 0) +
                allocation.allocated_cost
            )

        # Aggregate by category
        by_category: Dict[CostCategory, float] = {}
        for cost in costs:
            by_category[cost.category] = by_category.get(cost.category, 0) + cost.cost_usd

        total = sum(by_service.values())

        team_cost = TeamCost(
            team=team,
            total_cost_usd=total,
            by_service=by_service,
            by_category=by_category,
            period_start=period_start,
            period_end=period_end,
            headcount=headcount,
        )

        # Update metric
        self.team_cost.labels(team=team).set(total)

        return team_cost

    def get_cost_by_category(
        self,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
    ) -> Dict[CostCategory, float]:
        """Get costs by category.

        Args:
            period_start: Start of period
            period_end: End of period

        Returns:
            Dictionary mapping category to cost
        """
        now = datetime.now()
        period_start = period_start or now - timedelta(days=30)
        period_end = period_end or now

        with self._lock:
            costs = [
                c for c in self._costs
                if c.period_end >= period_start
                and c.period_start <= period_end
            ]

        by_category: Dict[CostCategory, float] = {}
        for cost in costs:
            by_category[cost.category] = by_category.get(cost.category, 0) + cost.cost_usd

        return by_category

    def calculate_unit_economics(
        self,
        service: str,
        requests: int,
        period_start: datetime,
        period_end: datetime,
    ) -> Dict[str, float]:
        """Calculate unit economics for a service.

        Args:
            service: Service name
            requests: Total requests in period
            period_start: Period start
            period_end: Period end

        Returns:
            Unit economics metrics
        """
        total_cost = self.get_service_cost(service, period_start, period_end)

        cost_per_request = total_cost / requests if requests > 0 else 0
        cost_per_1k_requests = cost_per_request * 1000

        days = (period_end - period_start).days or 1
        daily_cost = total_cost / days
        daily_requests = requests / days

        # Update metric
        self.cost_per_request.labels(service=service).set(cost_per_request * 100)

        return {
            "total_cost_usd": total_cost,
            "total_requests": requests,
            "cost_per_request_usd": cost_per_request,
            "cost_per_1k_requests_usd": cost_per_1k_requests,
            "daily_cost_usd": daily_cost,
            "daily_requests": daily_requests,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get cost attribution summary.

        Returns:
            Summary dictionary
        """
        now = datetime.now()
        month_start = datetime(now.year, now.month, 1)

        with self._lock:
            total_costs = sum(
                c.cost_usd for c in self._costs
                if c.period_start >= month_start
            )
            services = set(c.service for c in self._costs)
            teams = set(self._service_team_map.values())

        by_category = self.get_cost_by_category(month_start, now)

        return {
            "total_cost_mtd": total_costs,
            "service_count": len(services),
            "team_count": len(teams),
            "by_category": {k.value: v for k, v in by_category.items()},
            "top_category": max(by_category, key=by_category.get) if by_category else None,
        }
