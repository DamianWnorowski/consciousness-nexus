"""Waste Detection

Detect unused resources and optimization opportunities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from enum import Enum
import threading
import logging

from prometheus_client import Gauge

logger = logging.getLogger(__name__)


class ResourceType(str, Enum):
    """Types of cloud resources."""
    COMPUTE = "compute"
    STORAGE = "storage"
    DATABASE = "database"
    NETWORK = "network"
    KUBERNETES = "kubernetes"
    SERVERLESS = "serverless"
    AI_ML = "ai_ml"


class WasteType(str, Enum):
    """Types of resource waste."""
    UNUSED = "unused"           # Resource not being used
    IDLE = "idle"               # Low utilization
    OVERSIZED = "oversized"     # Larger than needed
    ORPHANED = "orphaned"       # No longer attached
    EXPIRED = "expired"         # Past useful life
    DUPLICATE = "duplicate"     # Redundant resource


@dataclass
class UnusedResource:
    """An unused or underutilized resource."""
    resource_id: str
    resource_name: str
    resource_type: ResourceType
    waste_type: WasteType
    monthly_cost_usd: float
    utilization_pct: float
    last_used: Optional[datetime]
    created_at: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    recommendation: str = ""

    @property
    def days_unused(self) -> int:
        """Days since last use."""
        if self.last_used:
            return (datetime.now() - self.last_used).days
        return (datetime.now() - self.created_at).days


@dataclass
class OptimizationOpportunity:
    """An opportunity to optimize resources."""
    resource_id: str
    resource_type: ResourceType
    current_config: str
    recommended_config: str
    current_monthly_cost: float
    optimized_monthly_cost: float
    savings_pct: float
    confidence: float
    rationale: str

    @property
    def monthly_savings(self) -> float:
        return self.current_monthly_cost - self.optimized_monthly_cost


@dataclass
class WasteReport:
    """Report of detected waste."""
    generated_at: datetime
    period_days: int
    total_waste_usd: float
    unused_resources: List[UnusedResource]
    optimization_opportunities: List[OptimizationOpportunity]
    by_type: Dict[WasteType, float]
    by_resource: Dict[ResourceType, float]


class WasteDetector:
    """Detects resource waste and optimization opportunities.

    Usage:
        detector = WasteDetector()

        # Track resources
        detector.track_resource(
            resource_id="vol-12345",
            name="unused-volume",
            resource_type=ResourceType.STORAGE,
            monthly_cost=50.0,
            utilization=0.0,
        )

        # Generate report
        report = detector.generate_report()
        print(f"Total waste: ${report.total_waste_usd}")
    """

    def __init__(
        self,
        namespace: str = "consciousness",
        idle_threshold_pct: float = 10.0,
        unused_days_threshold: int = 14,
    ):
        self.namespace = namespace
        self.idle_threshold = idle_threshold_pct
        self.unused_days_threshold = unused_days_threshold

        self._resources: Dict[str, Dict[str, Any]] = {}
        self._utilization_history: Dict[str, List[Tuple[datetime, float]]] = {}
        self._lock = threading.Lock()

        # Prometheus metrics
        self.total_waste = Gauge(
            f"{namespace}_finops_total_waste_dollars",
            "Total identified waste in dollars",
        )

        self.unused_resources = Gauge(
            f"{namespace}_finops_unused_resources_count",
            "Count of unused resources",
            ["resource_type"],
        )

        self.idle_resources = Gauge(
            f"{namespace}_finops_idle_resources_count",
            "Count of idle resources",
            ["resource_type"],
        )

        self.potential_savings = Gauge(
            f"{namespace}_finops_potential_savings_dollars",
            "Potential monthly savings",
            ["optimization_type"],
        )

    def track_resource(
        self,
        resource_id: str,
        name: str,
        resource_type: ResourceType,
        monthly_cost: float,
        utilization: float,
        created_at: Optional[datetime] = None,
        last_used: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Track a resource for waste detection.

        Args:
            resource_id: Unique resource identifier
            name: Resource name
            resource_type: Type of resource
            monthly_cost: Monthly cost in USD
            utilization: Current utilization (0-100)
            created_at: Creation timestamp
            last_used: Last usage timestamp
            tags: Resource tags
            config: Resource configuration
        """
        with self._lock:
            self._resources[resource_id] = {
                "name": name,
                "type": resource_type,
                "monthly_cost": monthly_cost,
                "utilization": utilization,
                "created_at": created_at or datetime.now(),
                "last_used": last_used,
                "tags": tags or {},
                "config": config or {},
                "tracked_at": datetime.now(),
            }

            # Track utilization history
            if resource_id not in self._utilization_history:
                self._utilization_history[resource_id] = []
            self._utilization_history[resource_id].append(
                (datetime.now(), utilization)
            )

            # Trim history (keep last 30 days)
            cutoff = datetime.now() - timedelta(days=30)
            self._utilization_history[resource_id] = [
                (ts, u) for ts, u in self._utilization_history[resource_id]
                if ts > cutoff
            ]

    def update_utilization(self, resource_id: str, utilization: float):
        """Update resource utilization.

        Args:
            resource_id: Resource identifier
            utilization: Current utilization (0-100)
        """
        with self._lock:
            if resource_id in self._resources:
                self._resources[resource_id]["utilization"] = utilization
                self._resources[resource_id]["last_used"] = (
                    datetime.now() if utilization > 0 else
                    self._resources[resource_id].get("last_used")
                )

                self._utilization_history[resource_id].append(
                    (datetime.now(), utilization)
                )

    def detect_unused(self) -> List[UnusedResource]:
        """Detect unused resources.

        Returns:
            List of unused resources
        """
        unused = []

        with self._lock:
            for resource_id, data in self._resources.items():
                # Check utilization
                if data["utilization"] == 0:
                    # Check if never used or unused for threshold days
                    if data["last_used"] is None:
                        days_unused = (datetime.now() - data["created_at"]).days
                    else:
                        days_unused = (datetime.now() - data["last_used"]).days

                    if days_unused >= self.unused_days_threshold:
                        unused.append(UnusedResource(
                            resource_id=resource_id,
                            resource_name=data["name"],
                            resource_type=data["type"],
                            waste_type=WasteType.UNUSED,
                            monthly_cost_usd=data["monthly_cost"],
                            utilization_pct=0,
                            last_used=data["last_used"],
                            created_at=data["created_at"],
                            tags=data["tags"],
                            recommendation=f"Consider deleting - unused for {days_unused} days",
                        ))

        # Update metrics
        by_type: Dict[ResourceType, int] = {}
        for r in unused:
            by_type[r.resource_type] = by_type.get(r.resource_type, 0) + 1

        for rt in ResourceType:
            self.unused_resources.labels(resource_type=rt.value).set(by_type.get(rt, 0))

        return unused

    def detect_idle(self) -> List[UnusedResource]:
        """Detect idle (low utilization) resources.

        Returns:
            List of idle resources
        """
        idle = []

        with self._lock:
            for resource_id, data in self._resources.items():
                # Check average utilization over time
                history = self._utilization_history.get(resource_id, [])

                if history:
                    avg_util = sum(u for _, u in history) / len(history)
                else:
                    avg_util = data["utilization"]

                if 0 < avg_util < self.idle_threshold:
                    idle.append(UnusedResource(
                        resource_id=resource_id,
                        resource_name=data["name"],
                        resource_type=data["type"],
                        waste_type=WasteType.IDLE,
                        monthly_cost_usd=data["monthly_cost"],
                        utilization_pct=avg_util,
                        last_used=data["last_used"],
                        created_at=data["created_at"],
                        tags=data["tags"],
                        recommendation=f"Consider rightsizing - only {avg_util:.1f}% utilized",
                    ))

        # Update metrics
        by_type: Dict[ResourceType, int] = {}
        for r in idle:
            by_type[r.resource_type] = by_type.get(r.resource_type, 0) + 1

        for rt in ResourceType:
            self.idle_resources.labels(resource_type=rt.value).set(by_type.get(rt, 0))

        return idle

    def find_rightsizing_opportunities(self) -> List[OptimizationOpportunity]:
        """Find rightsizing opportunities.

        Returns:
            List of optimization opportunities
        """
        opportunities = []

        with self._lock:
            for resource_id, data in self._resources.items():
                history = self._utilization_history.get(resource_id, [])

                if not history:
                    continue

                avg_util = sum(u for _, u in history) / len(history)
                max_util = max(u for _, u in history)

                # If max utilization is well below 100%, suggest rightsizing
                if max_util < 50 and data["monthly_cost"] > 0:
                    # Calculate potential savings
                    # Rough estimate: half size = half cost for most resources
                    potential_cost = data["monthly_cost"] * (max_util / 100 + 0.2)

                    opportunities.append(OptimizationOpportunity(
                        resource_id=resource_id,
                        resource_type=data["type"],
                        current_config=str(data.get("config", {})),
                        recommended_config="Rightsize to smaller instance",
                        current_monthly_cost=data["monthly_cost"],
                        optimized_monthly_cost=potential_cost,
                        savings_pct=(data["monthly_cost"] - potential_cost) / data["monthly_cost"] * 100,
                        confidence=0.7 if len(history) > 10 else 0.5,
                        rationale=f"Peak utilization is only {max_util:.1f}%",
                    ))

        # Update metrics
        total_savings = sum(o.monthly_savings for o in opportunities)
        self.potential_savings.labels(optimization_type="rightsizing").set(total_savings)

        return opportunities

    def generate_report(self, period_days: int = 30) -> WasteReport:
        """Generate waste report.

        Args:
            period_days: Period to analyze

        Returns:
            WasteReport
        """
        unused = self.detect_unused()
        idle = self.detect_idle()
        opportunities = self.find_rightsizing_opportunities()

        all_waste = unused + idle

        # Calculate totals
        total_waste = sum(r.monthly_cost_usd for r in all_waste)

        by_type: Dict[WasteType, float] = {}
        for r in all_waste:
            by_type[r.waste_type] = by_type.get(r.waste_type, 0) + r.monthly_cost_usd

        by_resource: Dict[ResourceType, float] = {}
        for r in all_waste:
            by_resource[r.resource_type] = by_resource.get(
                r.resource_type, 0
            ) + r.monthly_cost_usd

        # Update total waste metric
        self.total_waste.set(total_waste)

        return WasteReport(
            generated_at=datetime.now(),
            period_days=period_days,
            total_waste_usd=total_waste,
            unused_resources=all_waste,
            optimization_opportunities=opportunities,
            by_type=by_type,
            by_resource=by_resource,
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get waste detection summary.

        Returns:
            Summary dictionary
        """
        report = self.generate_report()

        return {
            "total_tracked_resources": len(self._resources),
            "unused_count": len([r for r in report.unused_resources if r.waste_type == WasteType.UNUSED]),
            "idle_count": len([r for r in report.unused_resources if r.waste_type == WasteType.IDLE]),
            "total_waste_usd": report.total_waste_usd,
            "optimization_opportunities": len(report.optimization_opportunities),
            "potential_savings_usd": sum(o.monthly_savings for o in report.optimization_opportunities),
        }
