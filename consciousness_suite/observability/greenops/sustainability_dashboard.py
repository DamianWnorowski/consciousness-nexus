"""Sustainability Dashboard

Comprehensive sustainability metrics and goal tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from enum import Enum
import threading
import logging

from prometheus_client import Gauge

from .carbon_calculator import CarbonCalculator, CarbonFootprint
from .energy_tracker import EnergyTracker, PowerUsage

logger = logging.getLogger(__name__)


class GoalStatus(str, Enum):
    """Status of sustainability goal."""
    ON_TRACK = "on_track"
    AT_RISK = "at_risk"
    OFF_TRACK = "off_track"
    ACHIEVED = "achieved"
    NOT_STARTED = "not_started"


class MetricType(str, Enum):
    """Types of sustainability metrics."""
    CARBON = "carbon"
    ENERGY = "energy"
    RENEWABLE = "renewable"
    EFFICIENCY = "efficiency"
    WASTE = "waste"


@dataclass
class SustainabilityGoal:
    """A sustainability goal.

    Usage:
        goal = SustainabilityGoal(
            id="carbon-reduction-2025",
            name="50% Carbon Reduction by 2025",
            metric_type=MetricType.CARBON,
            target_value=50,
            target_type="percentage_reduction",
            baseline_value=1000000,
            deadline=datetime(2025, 12, 31),
        )
    """
    id: str
    name: str
    metric_type: MetricType
    target_value: float
    target_type: str  # absolute, percentage_reduction, percentage
    baseline_value: float
    deadline: datetime
    current_value: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    description: str = ""

    @property
    def progress_percentage(self) -> float:
        """Get progress toward goal."""
        if self.target_type == "percentage_reduction":
            reduction = self.baseline_value - self.current_value
            target_reduction = self.baseline_value * (self.target_value / 100)
            return (reduction / target_reduction * 100) if target_reduction > 0 else 0

        elif self.target_type == "absolute":
            if self.target_value < self.baseline_value:
                # Reduction goal
                reduction = self.baseline_value - self.current_value
                target_reduction = self.baseline_value - self.target_value
                return (reduction / target_reduction * 100) if target_reduction > 0 else 0
            else:
                # Increase goal
                increase = self.current_value - self.baseline_value
                target_increase = self.target_value - self.baseline_value
                return (increase / target_increase * 100) if target_increase > 0 else 0

        elif self.target_type == "percentage":
            return (self.current_value / self.target_value * 100) if self.target_value > 0 else 0

        return 0

    @property
    def status(self) -> GoalStatus:
        """Get goal status."""
        if self.progress_percentage >= 100:
            return GoalStatus.ACHIEVED

        now = datetime.now()
        if now >= self.deadline:
            return GoalStatus.OFF_TRACK

        # Calculate expected progress
        total_duration = (self.deadline - self.created_at).days
        elapsed = (now - self.created_at).days
        expected_progress = (elapsed / total_duration * 100) if total_duration > 0 else 0

        if self.progress_percentage >= expected_progress * 0.9:
            return GoalStatus.ON_TRACK
        elif self.progress_percentage >= expected_progress * 0.7:
            return GoalStatus.AT_RISK
        else:
            return GoalStatus.OFF_TRACK


@dataclass
class GreenMetrics:
    """Aggregated green/sustainability metrics."""
    carbon_footprint_gco2: float
    energy_consumption_kwh: float
    renewable_percentage: float
    pue: float
    green_regions_percentage: float
    waste_reduction_percentage: float
    sustainability_score: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SustainabilityScore:
    """Overall sustainability score."""
    overall_score: float  # 0-100
    carbon_score: float
    energy_score: float
    renewable_score: float
    efficiency_score: float
    goal_progress_score: float
    grade: str
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class SustainabilityDashboard:
    """Comprehensive sustainability dashboard.

    Usage:
        dashboard = SustainabilityDashboard(
            carbon_calculator=carbon_calc,
            energy_tracker=energy_tracker,
        )

        # Add goals
        dashboard.add_goal(SustainabilityGoal(...))

        # Get overall metrics
        metrics = dashboard.get_metrics()

        # Get sustainability score
        score = dashboard.calculate_score()
    """

    def __init__(
        self,
        carbon_calculator: Optional[CarbonCalculator] = None,
        energy_tracker: Optional[EnergyTracker] = None,
        namespace: str = "consciousness",
    ):
        self.namespace = namespace
        self.carbon_calculator = carbon_calculator or CarbonCalculator(namespace)
        self.energy_tracker = energy_tracker or EnergyTracker(namespace)

        self._goals: Dict[str, SustainabilityGoal] = {}
        self._metrics_history: List[GreenMetrics] = []
        self._lock = threading.Lock()

        # Prometheus metrics
        self.sustainability_score_metric = Gauge(
            f"{namespace}_greenops_sustainability_score",
            "Overall sustainability score",
        )

        self.goal_progress = Gauge(
            f"{namespace}_greenops_goal_progress_percentage",
            "Progress toward sustainability goal",
            ["goal_id"],
        )

        self.goal_status = Gauge(
            f"{namespace}_greenops_goal_status",
            "Goal status (0=not_started, 1=off_track, 2=at_risk, 3=on_track, 4=achieved)",
            ["goal_id"],
        )

    def add_goal(self, goal: SustainabilityGoal):
        """Add a sustainability goal.

        Args:
            goal: Sustainability goal
        """
        with self._lock:
            self._goals[goal.id] = goal

        logger.info(f"Added sustainability goal: {goal.id}")

    def update_goal_progress(self, goal_id: str, current_value: float):
        """Update progress on a goal.

        Args:
            goal_id: Goal identifier
            current_value: Current metric value
        """
        with self._lock:
            if goal_id in self._goals:
                self._goals[goal_id].current_value = current_value

        # Update metrics
        goal = self._goals.get(goal_id)
        if goal:
            self.goal_progress.labels(goal_id=goal_id).set(goal.progress_percentage)

            status_map = {
                GoalStatus.NOT_STARTED: 0,
                GoalStatus.OFF_TRACK: 1,
                GoalStatus.AT_RISK: 2,
                GoalStatus.ON_TRACK: 3,
                GoalStatus.ACHIEVED: 4,
            }
            self.goal_status.labels(goal_id=goal_id).set(status_map[goal.status])

    def get_goals(self) -> List[SustainabilityGoal]:
        """Get all goals.

        Returns:
            List of goals
        """
        with self._lock:
            return list(self._goals.values())

    def get_metrics(self) -> GreenMetrics:
        """Get current green metrics.

        Returns:
            GreenMetrics
        """
        # Get carbon footprint
        footprint = self.carbon_calculator.calculate_footprint()

        # Get energy usage
        usage = self.energy_tracker.get_usage()

        # Get efficiency
        efficiency = self.energy_tracker.calculate_efficiency()

        # Calculate renewable percentage
        renewable_pct = self.energy_tracker.get_renewable_percentage()

        # Calculate green regions percentage
        green_regions = self.carbon_calculator.get_green_regions()
        all_regions = len(self.carbon_calculator._regions)
        green_pct = (len(green_regions) / all_regions * 100) if all_regions > 0 else 0

        # Calculate sustainability score
        score = self.calculate_score()

        metrics = GreenMetrics(
            carbon_footprint_gco2=footprint.total_gco2eq,
            energy_consumption_kwh=usage.total_energy_kwh,
            renewable_percentage=renewable_pct,
            pue=efficiency.pue,
            green_regions_percentage=green_pct,
            waste_reduction_percentage=0,  # Would come from waste tracker
            sustainability_score=score.overall_score,
        )

        with self._lock:
            self._metrics_history.append(metrics)

        return metrics

    def calculate_score(self) -> SustainabilityScore:
        """Calculate overall sustainability score.

        Returns:
            SustainabilityScore
        """
        # Carbon score (lower is better)
        footprint = self.carbon_calculator.calculate_footprint()
        annual_estimate = self.carbon_calculator.estimate_annual_footprint()

        # Score based on carbon intensity - max 100 for < 1 ton/year
        carbon_score = max(0, 100 - (annual_estimate.total_gco2eq / 1000000 * 10))

        # Energy score based on efficiency
        efficiency = self.energy_tracker.calculate_efficiency()
        # PUE of 1.0 = 100, PUE of 2.0 = 50
        energy_score = max(0, 100 - (efficiency.pue - 1.0) * 50)

        # Renewable score
        renewable_pct = self.energy_tracker.get_renewable_percentage()
        renewable_score = renewable_pct  # 100% renewable = 100 score

        # Efficiency score (DCiE)
        efficiency_score = efficiency.dcie

        # Goal progress score
        goals = self.get_goals()
        if goals:
            avg_progress = sum(g.progress_percentage for g in goals) / len(goals)
            goal_score = min(100, avg_progress)
        else:
            goal_score = 50  # Neutral if no goals

        # Overall weighted score
        overall = (
            carbon_score * 0.25 +
            energy_score * 0.20 +
            renewable_score * 0.25 +
            efficiency_score * 0.15 +
            goal_score * 0.15
        )

        # Determine grade
        if overall >= 90:
            grade = "A"
        elif overall >= 80:
            grade = "B"
        elif overall >= 70:
            grade = "C"
        elif overall >= 60:
            grade = "D"
        else:
            grade = "F"

        # Generate recommendations
        recommendations = []

        if carbon_score < 70:
            recommendations.append("Consider migrating to low-carbon regions")

        if renewable_score < 50:
            recommendations.append("Increase use of renewable energy sources")

        if efficiency.pue > 1.5:
            recommendations.append("Improve data center cooling efficiency")

        if goal_score < 50:
            recommendations.append("Review and accelerate sustainability initiatives")

        score = SustainabilityScore(
            overall_score=overall,
            carbon_score=carbon_score,
            energy_score=energy_score,
            renewable_score=renewable_score,
            efficiency_score=efficiency_score,
            goal_progress_score=goal_score,
            grade=grade,
            recommendations=recommendations,
        )

        # Update metric
        self.sustainability_score_metric.set(overall)

        return score

    def get_trend(
        self,
        days: int = 30,
    ) -> List[GreenMetrics]:
        """Get metrics trend.

        Args:
            days: Number of days

        Returns:
            List of metrics over time
        """
        cutoff = datetime.now() - timedelta(days=days)

        with self._lock:
            return [
                m for m in self._metrics_history
                if m.timestamp >= cutoff
            ]

    def get_report(self) -> Dict[str, Any]:
        """Generate sustainability report.

        Returns:
            Report dictionary
        """
        metrics = self.get_metrics()
        score = self.calculate_score()
        goals = self.get_goals()

        return {
            "report_date": datetime.now().isoformat(),
            "overall_score": score.overall_score,
            "grade": score.grade,
            "metrics": {
                "carbon_footprint_gco2": metrics.carbon_footprint_gco2,
                "energy_consumption_kwh": metrics.energy_consumption_kwh,
                "renewable_percentage": metrics.renewable_percentage,
                "pue": metrics.pue,
            },
            "scores": {
                "carbon": score.carbon_score,
                "energy": score.energy_score,
                "renewable": score.renewable_score,
                "efficiency": score.efficiency_score,
                "goal_progress": score.goal_progress_score,
            },
            "goals": [
                {
                    "id": g.id,
                    "name": g.name,
                    "progress": g.progress_percentage,
                    "status": g.status.value,
                    "deadline": g.deadline.isoformat(),
                }
                for g in goals
            ],
            "recommendations": score.recommendations,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get dashboard summary.

        Returns:
            Summary dictionary
        """
        score = self.calculate_score()
        goals = self.get_goals()

        on_track = sum(1 for g in goals if g.status == GoalStatus.ON_TRACK)
        at_risk = sum(1 for g in goals if g.status == GoalStatus.AT_RISK)
        achieved = sum(1 for g in goals if g.status == GoalStatus.ACHIEVED)

        return {
            "sustainability_score": score.overall_score,
            "grade": score.grade,
            "total_goals": len(goals),
            "goals_on_track": on_track,
            "goals_at_risk": at_risk,
            "goals_achieved": achieved,
            "top_recommendation": score.recommendations[0] if score.recommendations else None,
        }
