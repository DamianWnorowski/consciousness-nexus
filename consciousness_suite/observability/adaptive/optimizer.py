"""80/20 Optimization Engine for Telemetry Budget

Applies Pareto principle to optimize telemetry spend:
- Identify the 20% of telemetry providing 80% of value
- Recommend cuts to low-value telemetry
- Automate optimization actions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import threading
import logging
import math

from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)


class OptimizationPriority(str, Enum):
    """Priority levels for optimization actions."""
    CRITICAL = "critical"       # Immediate action required
    HIGH = "high"               # Action within hours
    MEDIUM = "medium"           # Action within days
    LOW = "low"                 # Nice to have
    INFORMATIONAL = "info"      # For awareness only


class OptimizationCategory(str, Enum):
    """Categories of optimization."""
    SAMPLING = "sampling"           # Increase sampling rate
    AGGREGATION = "aggregation"     # Pre-aggregate data
    FILTERING = "filtering"         # Drop low-value data
    CARDINALITY = "cardinality"     # Reduce label cardinality
    RETENTION = "retention"         # Reduce retention period
    ARCHITECTURE = "architecture"   # Architectural changes
    REDUNDANCY = "redundancy"       # Remove duplicate data


@dataclass
class TelemetryValueScore:
    """Value score for a telemetry source.

    Higher score = higher value. Range 0-100.
    """
    source: str                             # Metric/log/trace name
    service: str
    telemetry_type: str
    value_score: float                      # 0-100
    cost_per_unit: float
    monthly_cost: float
    usage_frequency: int                    # How often queried/used
    alert_dependency: int                   # Alerts depending on this
    dashboard_usage: int                    # Dashboards using this
    slo_dependency: bool                    # Used in SLOs
    last_accessed: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationRecommendation:
    """Recommendation for telemetry optimization.

    Usage:
        rec = OptimizationRecommendation(
            id="rec_001",
            title="Reduce API latency histogram cardinality",
            description="The 'path' label has 5000 unique values...",
            category=OptimizationCategory.CARDINALITY,
            priority=OptimizationPriority.HIGH,
            estimated_savings=500.0,
            implementation_effort="low",
        )
    """
    id: str
    title: str
    description: str
    category: OptimizationCategory
    priority: OptimizationPriority
    target: str                             # Metric/service affected
    estimated_savings: float                # Monthly savings in $
    current_cost: float
    new_cost: float
    savings_percentage: float
    implementation_effort: str              # low, medium, high
    implementation_steps: List[str] = field(default_factory=list)
    risk_level: str = "low"                 # low, medium, high
    auto_applicable: bool = False           # Can be auto-applied
    requires_approval: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationPlan:
    """A collection of optimization recommendations."""
    id: str
    name: str
    description: str
    recommendations: List[OptimizationRecommendation]
    total_savings: float
    total_current_cost: float
    savings_percentage: float
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "proposed"                # proposed, approved, in_progress, completed


@dataclass
class OptimizationConfig:
    """Configuration for optimization engine.

    Usage:
        config = OptimizationConfig(
            target_savings_percentage=30.0,
            min_value_score_to_keep=20.0,
            auto_apply_low_risk=True,
        )
    """
    target_savings_percentage: float = 20.0     # Target savings
    min_value_score_to_keep: float = 10.0       # Below this = consider dropping
    pareto_threshold: float = 0.8               # 80/20 threshold
    min_savings_for_recommendation: float = 50.0  # Min monthly savings
    auto_apply_low_risk: bool = False
    max_auto_apply_savings: float = 100.0       # Max auto-apply per action
    unused_threshold_days: int = 30             # Days without access = unused


class TelemetryOptimizer:
    """80/20 optimization engine for telemetry budget.

    Usage:
        optimizer = TelemetryOptimizer(namespace="consciousness")

        # Register value scores
        optimizer.register_value_score(TelemetryValueScore(
            source="http_requests_total",
            service="api",
            telemetry_type="metrics",
            value_score=85.0,
            cost_per_unit=0.001,
            monthly_cost=500.0,
            usage_frequency=1000,
            alert_dependency=5,
            dashboard_usage=10,
            slo_dependency=True,
        ))

        # Generate optimization plan
        plan = optimizer.generate_optimization_plan()

        # Apply recommendations
        for rec in plan.recommendations:
            if rec.auto_applicable:
                optimizer.apply_recommendation(rec.id)
    """

    def __init__(
        self,
        namespace: str = "consciousness",
        config: Optional[OptimizationConfig] = None,
    ):
        self.namespace = namespace
        self.config = config or OptimizationConfig()
        self._lock = threading.Lock()

        # Value tracking
        self._value_scores: Dict[str, TelemetryValueScore] = {}
        self._cost_history: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)

        # Recommendations
        self._recommendations: Dict[str, OptimizationRecommendation] = {}
        self._plans: Dict[str, OptimizationPlan] = {}
        self._applied_recommendations: Set[str] = set()

        # Callbacks
        self._optimization_callbacks: List[Callable[[OptimizationRecommendation], None]] = []

        # Statistics
        self._total_analyzed: int = 0
        self._total_savings_identified: float = 0.0
        self._total_savings_applied: float = 0.0

        # Prometheus metrics
        self.value_score = Gauge(
            f"{namespace}_telemetry_value_score",
            "Telemetry value score (0-100)",
            ["source", "service"],
        )

        self.optimization_savings = Gauge(
            f"{namespace}_optimization_savings_identified",
            "Total identified monthly savings",
        )

        self.applied_savings = Gauge(
            f"{namespace}_optimization_savings_applied",
            "Total applied monthly savings",
        )

        self.recommendations_count = Gauge(
            f"{namespace}_optimization_recommendations",
            "Number of active recommendations",
            ["category", "priority"],
        )

        self.pareto_ratio = Gauge(
            f"{namespace}_pareto_efficiency_ratio",
            "Pareto efficiency ratio",
        )

    def register_value_score(self, score: TelemetryValueScore):
        """Register or update a telemetry value score.

        Args:
            score: Value score to register
        """
        key = f"{score.service}:{score.source}"

        with self._lock:
            self._value_scores[key] = score

        # Update metrics
        self.value_score.labels(
            source=score.source,
            service=score.service,
        ).set(score.value_score)

        # Track cost history
        self._cost_history[key].append((datetime.now(), score.monthly_cost))

        # Trim history
        cutoff = datetime.now() - timedelta(days=90)
        self._cost_history[key] = [
            (ts, cost) for ts, cost in self._cost_history[key]
            if ts > cutoff
        ]

    def calculate_value_score(
        self,
        source: str,
        service: str,
        usage_frequency: int = 0,
        alert_dependency: int = 0,
        dashboard_usage: int = 0,
        slo_dependency: bool = False,
        last_accessed: Optional[datetime] = None,
        error_rate_impact: float = 0.0,
    ) -> float:
        """Calculate value score for a telemetry source.

        Args:
            source: Telemetry source name
            service: Service name
            usage_frequency: Query frequency (per day)
            alert_dependency: Number of alerts using this
            dashboard_usage: Number of dashboards using this
            slo_dependency: Whether used in SLOs
            last_accessed: Last access time
            error_rate_impact: Impact on error rate (0-1)

        Returns:
            Value score 0-100
        """
        score = 0.0

        # Usage frequency (max 30 points)
        # 100+ daily queries = 30 points
        usage_score = min(30, (usage_frequency / 100) * 30)
        score += usage_score

        # Alert dependency (max 25 points)
        # Each alert = 5 points, max 25
        alert_score = min(25, alert_dependency * 5)
        score += alert_score

        # Dashboard usage (max 15 points)
        # Each dashboard = 3 points, max 15
        dashboard_score = min(15, dashboard_usage * 3)
        score += dashboard_score

        # SLO dependency (20 points)
        if slo_dependency:
            score += 20

        # Recency bonus (max 10 points)
        if last_accessed:
            days_since = (datetime.now() - last_accessed).days
            if days_since <= 1:
                score += 10
            elif days_since <= 7:
                score += 7
            elif days_since <= 30:
                score += 3
            # Over 30 days = 0 points

        # Error rate impact bonus (max 10 points)
        if error_rate_impact > 0:
            score += min(10, error_rate_impact * 10)

        return min(100, max(0, score))

    def analyze_pareto_distribution(self) -> Dict[str, Any]:
        """Analyze Pareto (80/20) distribution of telemetry value.

        Returns:
            Pareto analysis results
        """
        with self._lock:
            scores = list(self._value_scores.values())

        if not scores:
            return {"error": "No telemetry data to analyze"}

        # Sort by value score descending
        sorted_scores = sorted(scores, key=lambda x: x.value_score, reverse=True)

        # Calculate cumulative value and cost
        total_value = sum(s.value_score for s in scores)
        total_cost = sum(s.monthly_cost for s in scores)

        cumulative_value = 0.0
        cumulative_cost = 0.0
        high_value_sources = []
        low_value_sources = []

        pareto_threshold = self.config.pareto_threshold

        for score in sorted_scores:
            cumulative_value += score.value_score
            cumulative_cost += score.monthly_cost

            value_ratio = cumulative_value / total_value if total_value > 0 else 0
            cost_ratio = cumulative_cost / total_cost if total_cost > 0 else 0

            if value_ratio <= pareto_threshold:
                high_value_sources.append({
                    "source": score.source,
                    "service": score.service,
                    "value_score": score.value_score,
                    "monthly_cost": score.monthly_cost,
                })
            else:
                low_value_sources.append({
                    "source": score.source,
                    "service": score.service,
                    "value_score": score.value_score,
                    "monthly_cost": score.monthly_cost,
                })

        # Calculate Pareto efficiency
        high_value_cost = sum(s["monthly_cost"] for s in high_value_sources)
        low_value_cost = sum(s["monthly_cost"] for s in low_value_sources)

        pareto_efficiency = (
            len(high_value_sources) / len(scores)
            if scores else 0
        )

        self.pareto_ratio.set(pareto_efficiency)

        return {
            "total_sources": len(scores),
            "total_monthly_cost": total_cost,
            "high_value": {
                "count": len(high_value_sources),
                "percentage": len(high_value_sources) / len(scores) * 100,
                "monthly_cost": high_value_cost,
                "cost_percentage": high_value_cost / total_cost * 100 if total_cost > 0 else 0,
                "sources": high_value_sources[:20],  # Top 20
            },
            "low_value": {
                "count": len(low_value_sources),
                "percentage": len(low_value_sources) / len(scores) * 100,
                "monthly_cost": low_value_cost,
                "cost_percentage": low_value_cost / total_cost * 100 if total_cost > 0 else 0,
                "potential_savings": low_value_cost * 0.5,  # Conservative 50% savings
                "sources": low_value_sources[:20],
            },
            "pareto_efficiency": pareto_efficiency,
            "recommendation": self._generate_pareto_recommendation(
                high_value_sources, low_value_sources, total_cost
            ),
        }

    def _generate_pareto_recommendation(
        self,
        high_value: List[Dict],
        low_value: List[Dict],
        total_cost: float,
    ) -> str:
        """Generate recommendation based on Pareto analysis."""
        if len(low_value) == 0:
            return "Telemetry is well-optimized. No low-value sources identified."

        low_value_pct = len(low_value) / (len(high_value) + len(low_value)) * 100
        low_value_cost = sum(s["monthly_cost"] for s in low_value)

        if low_value_pct > 50:
            return (
                f"High optimization opportunity: {low_value_pct:.0f}% of telemetry "
                f"provides low value. Potential monthly savings: ${low_value_cost:.2f}"
            )
        elif low_value_pct > 20:
            return (
                f"Moderate optimization opportunity: {low_value_pct:.0f}% of telemetry "
                f"could be reduced. Review low-value sources for ${low_value_cost:.2f} savings."
            )
        else:
            return (
                f"Telemetry is reasonably optimized. {low_value_pct:.0f}% is low-value "
                f"with ${low_value_cost:.2f} potential savings."
            )

    def generate_optimization_plan(
        self,
        target_savings: Optional[float] = None,
    ) -> OptimizationPlan:
        """Generate comprehensive optimization plan.

        Args:
            target_savings: Target monthly savings (None = use config)

        Returns:
            OptimizationPlan with recommendations
        """
        self._total_analyzed += 1
        target = target_savings or (
            sum(s.monthly_cost for s in self._value_scores.values()) *
            (self.config.target_savings_percentage / 100)
        )

        recommendations = []

        # 1. Identify unused telemetry
        unused_recs = self._find_unused_telemetry()
        recommendations.extend(unused_recs)

        # 2. Identify low-value high-cost telemetry
        pareto_recs = self._find_pareto_opportunities()
        recommendations.extend(pareto_recs)

        # 3. Identify cardinality issues
        cardinality_recs = self._find_cardinality_issues()
        recommendations.extend(cardinality_recs)

        # 4. Identify redundant telemetry
        redundancy_recs = self._find_redundancies()
        recommendations.extend(redundancy_recs)

        # 5. Identify sampling opportunities
        sampling_recs = self._find_sampling_opportunities()
        recommendations.extend(sampling_recs)

        # Sort by savings and priority
        recommendations.sort(
            key=lambda r: (
                r.priority.value,
                -r.estimated_savings,
            )
        )

        # Store recommendations
        with self._lock:
            for rec in recommendations:
                self._recommendations[rec.id] = rec

        total_savings = sum(r.estimated_savings for r in recommendations)
        total_current = sum(r.current_cost for r in recommendations)

        self._total_savings_identified = total_savings
        self.optimization_savings.set(total_savings)

        # Update recommendation counts
        for category in OptimizationCategory:
            for priority in OptimizationPriority:
                count = len([
                    r for r in recommendations
                    if r.category == category and r.priority == priority
                ])
                self.recommendations_count.labels(
                    category=category.value,
                    priority=priority.value,
                ).set(count)

        plan_id = f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        plan = OptimizationPlan(
            id=plan_id,
            name="Telemetry Optimization Plan",
            description=f"Plan to achieve ${target:.2f} monthly savings",
            recommendations=recommendations,
            total_savings=total_savings,
            total_current_cost=total_current,
            savings_percentage=(
                total_savings / total_current * 100 if total_current > 0 else 0
            ),
        )

        with self._lock:
            self._plans[plan_id] = plan

        logger.info(
            f"Generated optimization plan: {len(recommendations)} recommendations, "
            f"${total_savings:.2f} potential savings"
        )

        return plan

    def _find_unused_telemetry(self) -> List[OptimizationRecommendation]:
        """Find telemetry not accessed within threshold days."""
        recommendations = []
        cutoff = datetime.now() - timedelta(days=self.config.unused_threshold_days)
        rec_id = 0

        for key, score in self._value_scores.items():
            if score.last_accessed and score.last_accessed < cutoff:
                days_unused = (datetime.now() - score.last_accessed).days

                if score.monthly_cost >= self.config.min_savings_for_recommendation:
                    rec_id += 1
                    recommendations.append(OptimizationRecommendation(
                        id=f"unused_{rec_id}",
                        title=f"Remove unused telemetry: {score.source}",
                        description=(
                            f"This telemetry hasn't been accessed in {days_unused} days. "
                            f"Consider removing or archiving to save ${score.monthly_cost:.2f}/month."
                        ),
                        category=OptimizationCategory.FILTERING,
                        priority=OptimizationPriority.HIGH if days_unused > 60 else OptimizationPriority.MEDIUM,
                        target=score.source,
                        estimated_savings=score.monthly_cost,
                        current_cost=score.monthly_cost,
                        new_cost=0.0,
                        savings_percentage=100.0,
                        implementation_effort="low",
                        implementation_steps=[
                            f"Verify {score.source} is not used in any alerts or dashboards",
                            "Add to filter rules to drop this telemetry",
                            "Monitor for any issues after removal",
                        ],
                        auto_applicable=score.monthly_cost <= self.config.max_auto_apply_savings,
                        risk_level="low" if days_unused > 90 else "medium",
                    ))

        return recommendations

    def _find_pareto_opportunities(self) -> List[OptimizationRecommendation]:
        """Find high-cost low-value telemetry."""
        recommendations = []
        rec_id = 0

        for key, score in self._value_scores.items():
            if score.value_score < self.config.min_value_score_to_keep:
                # Low value source
                if score.monthly_cost >= self.config.min_savings_for_recommendation:
                    # Calculate potential savings (50% reduction via sampling)
                    savings = score.monthly_cost * 0.5

                    rec_id += 1
                    recommendations.append(OptimizationRecommendation(
                        id=f"pareto_{rec_id}",
                        title=f"Reduce low-value telemetry: {score.source}",
                        description=(
                            f"Value score: {score.value_score:.1f}/100. "
                            f"Cost: ${score.monthly_cost:.2f}/month. "
                            f"Consider reducing collection or increasing sampling."
                        ),
                        category=OptimizationCategory.SAMPLING,
                        priority=self._get_pareto_priority(score),
                        target=score.source,
                        estimated_savings=savings,
                        current_cost=score.monthly_cost,
                        new_cost=score.monthly_cost - savings,
                        savings_percentage=50.0,
                        implementation_effort="low",
                        implementation_steps=[
                            f"Add sampling rule for {score.source} at 10% rate",
                            "Update dashboards to use aggregated views",
                            "Monitor query performance after change",
                        ],
                        auto_applicable=self.config.auto_apply_low_risk,
                        risk_level="low",
                    ))

        return recommendations

    def _get_pareto_priority(self, score: TelemetryValueScore) -> OptimizationPriority:
        """Determine priority based on value score and cost."""
        if score.value_score < 5 and score.monthly_cost > 500:
            return OptimizationPriority.CRITICAL
        elif score.value_score < 10 and score.monthly_cost > 200:
            return OptimizationPriority.HIGH
        elif score.value_score < 20:
            return OptimizationPriority.MEDIUM
        else:
            return OptimizationPriority.LOW

    def _find_cardinality_issues(self) -> List[OptimizationRecommendation]:
        """Find high-cardinality telemetry."""
        recommendations = []
        rec_id = 0

        # Look for sources with high cost per unit (indicates high cardinality)
        for key, score in self._value_scores.items():
            cost_efficiency = score.value_score / (score.monthly_cost + 0.01)

            if cost_efficiency < 0.1 and score.monthly_cost > 100:
                # Poor cost efficiency suggests cardinality issues
                savings = score.monthly_cost * 0.6  # 60% reduction estimate

                rec_id += 1
                recommendations.append(OptimizationRecommendation(
                    id=f"cardinality_{rec_id}",
                    title=f"Reduce cardinality: {score.source}",
                    description=(
                        f"This telemetry has poor cost efficiency (value/cost ratio). "
                        f"Consider reducing label cardinality to save ${savings:.2f}/month."
                    ),
                    category=OptimizationCategory.CARDINALITY,
                    priority=OptimizationPriority.HIGH,
                    target=score.source,
                    estimated_savings=savings,
                    current_cost=score.monthly_cost,
                    new_cost=score.monthly_cost - savings,
                    savings_percentage=60.0,
                    implementation_effort="medium",
                    implementation_steps=[
                        "Analyze label cardinality to identify high-cardinality labels",
                        "Remove or aggregate high-cardinality labels (user_id, request_id, etc.)",
                        "Update queries to use aggregated labels",
                        "Apply cardinality limits to prevent future issues",
                    ],
                    risk_level="medium",
                ))

        return recommendations

    def _find_redundancies(self) -> List[OptimizationRecommendation]:
        """Find redundant/duplicate telemetry."""
        recommendations = []

        # Group by service
        by_service: Dict[str, List[TelemetryValueScore]] = defaultdict(list)
        for score in self._value_scores.values():
            by_service[score.service].append(score)

        rec_id = 0
        for service, scores in by_service.items():
            if len(scores) < 2:
                continue

            # Look for similar sources (might be duplicates)
            for i, s1 in enumerate(scores):
                for s2 in scores[i+1:]:
                    # Check for similar names (basic similarity)
                    if self._is_similar(s1.source, s2.source):
                        lower_value = s1 if s1.value_score < s2.value_score else s2
                        savings = lower_value.monthly_cost

                        if savings >= self.config.min_savings_for_recommendation:
                            rec_id += 1
                            recommendations.append(OptimizationRecommendation(
                                id=f"redundancy_{rec_id}",
                                title=f"Consolidate similar metrics in {service}",
                                description=(
                                    f"'{s1.source}' and '{s2.source}' appear similar. "
                                    f"Consider consolidating to save ${savings:.2f}/month."
                                ),
                                category=OptimizationCategory.REDUNDANCY,
                                priority=OptimizationPriority.MEDIUM,
                                target=lower_value.source,
                                estimated_savings=savings,
                                current_cost=s1.monthly_cost + s2.monthly_cost,
                                new_cost=max(s1.monthly_cost, s2.monthly_cost),
                                savings_percentage=savings / (s1.monthly_cost + s2.monthly_cost) * 100,
                                implementation_effort="medium",
                                implementation_steps=[
                                    f"Verify {s1.source} and {s2.source} serve similar purposes",
                                    "Migrate dashboards and alerts to use single source",
                                    "Remove redundant telemetry collection",
                                ],
                                risk_level="medium",
                            ))

        return recommendations

    def _is_similar(self, name1: str, name2: str) -> bool:
        """Check if two names are similar (potential duplicates)."""
        # Simple similarity check - could be enhanced with fuzzy matching
        parts1 = set(name1.lower().replace("_", " ").replace("-", " ").split())
        parts2 = set(name2.lower().replace("_", " ").replace("-", " ").split())

        if not parts1 or not parts2:
            return False

        intersection = len(parts1 & parts2)
        union = len(parts1 | parts2)

        return (intersection / union) > 0.7  # 70% similarity

    def _find_sampling_opportunities(self) -> List[OptimizationRecommendation]:
        """Find high-volume telemetry that could be sampled."""
        recommendations = []
        rec_id = 0

        # Sort by cost to find high-cost items
        sorted_scores = sorted(
            self._value_scores.values(),
            key=lambda x: x.monthly_cost,
            reverse=True,
        )

        for score in sorted_scores[:20]:  # Top 20 by cost
            # Only recommend sampling for high cost, not-critical items
            if (score.monthly_cost > 200 and
                not score.slo_dependency and
                score.alert_dependency < 3):

                # Suggest 10% sampling
                savings = score.monthly_cost * 0.9

                rec_id += 1
                recommendations.append(OptimizationRecommendation(
                    id=f"sampling_{rec_id}",
                    title=f"Sample high-volume: {score.source}",
                    description=(
                        f"High-cost telemetry (${score.monthly_cost:.2f}/month) "
                        f"without critical dependencies. Consider 10% sampling."
                    ),
                    category=OptimizationCategory.SAMPLING,
                    priority=OptimizationPriority.MEDIUM,
                    target=score.source,
                    estimated_savings=savings,
                    current_cost=score.monthly_cost,
                    new_cost=score.monthly_cost * 0.1,
                    savings_percentage=90.0,
                    implementation_effort="low",
                    implementation_steps=[
                        f"Add tail-based sampling rule for {score.source}",
                        "Ensure errors and slow requests are always sampled",
                        "Update dashboards to account for sampling",
                        "Monitor for any missing insights",
                    ],
                    auto_applicable=self.config.auto_apply_low_risk,
                    risk_level="low" if score.value_score < 50 else "medium",
                ))

        return recommendations

    def apply_recommendation(
        self,
        recommendation_id: str,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Apply an optimization recommendation.

        Args:
            recommendation_id: Recommendation ID
            dry_run: If True, simulate only

        Returns:
            Application result
        """
        with self._lock:
            rec = self._recommendations.get(recommendation_id)
            if not rec:
                return {"error": f"Recommendation {recommendation_id} not found"}

            if recommendation_id in self._applied_recommendations:
                return {"error": "Recommendation already applied"}

        if dry_run:
            return {
                "status": "dry_run",
                "recommendation": rec.id,
                "would_save": rec.estimated_savings,
            }

        # In a real implementation, this would trigger actual changes
        # Here we just track the application

        with self._lock:
            self._applied_recommendations.add(recommendation_id)
            self._total_savings_applied += rec.estimated_savings

        self.applied_savings.set(self._total_savings_applied)

        # Notify callbacks
        for callback in self._optimization_callbacks:
            try:
                callback(rec)
            except Exception as e:
                logger.error(f"Optimization callback error: {e}")

        logger.info(
            f"Applied optimization: {rec.id} - saving ${rec.estimated_savings:.2f}/month"
        )

        return {
            "status": "applied",
            "recommendation": rec.id,
            "savings": rec.estimated_savings,
            "total_applied_savings": self._total_savings_applied,
        }

    def on_optimization(self, callback: Callable[[OptimizationRecommendation], None]):
        """Register callback for applied optimizations.

        Args:
            callback: Function to call when optimization is applied
        """
        self._optimization_callbacks.append(callback)

    def get_quick_wins(self, max_count: int = 5) -> List[OptimizationRecommendation]:
        """Get top quick-win recommendations.

        Args:
            max_count: Maximum recommendations to return

        Returns:
            List of quick-win recommendations
        """
        with self._lock:
            recs = [
                r for r in self._recommendations.values()
                if r.implementation_effort == "low" and
                   r.risk_level == "low" and
                   r.id not in self._applied_recommendations
            ]

        # Sort by savings
        recs.sort(key=lambda r: r.estimated_savings, reverse=True)

        return recs[:max_count]

    def get_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics.

        Returns:
            Statistics dictionary
        """
        with self._lock:
            total_sources = len(self._value_scores)
            total_recs = len(self._recommendations)
            applied_count = len(self._applied_recommendations)

        return {
            "total_sources_analyzed": total_sources,
            "total_recommendations": total_recs,
            "recommendations_applied": applied_count,
            "total_savings_identified": self._total_savings_identified,
            "total_savings_applied": self._total_savings_applied,
            "analyses_performed": self._total_analyzed,
        }
