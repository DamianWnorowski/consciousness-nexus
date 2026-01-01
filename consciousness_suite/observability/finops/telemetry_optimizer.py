"""Telemetry Optimizer

Optimizes telemetry data collection for cost efficiency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import threading
import logging
import random

from prometheus_client import Counter, Gauge

logger = logging.getLogger(__name__)


class SamplingStrategy(str, Enum):
    """Telemetry sampling strategies."""
    NONE = "none"               # No sampling
    RANDOM = "random"           # Random sampling
    RATE_LIMITING = "rate_limiting"  # Rate-based sampling
    PRIORITY = "priority"       # Priority-based (keep important)
    ADAPTIVE = "adaptive"       # Adaptive based on load
    TAIL_BASED = "tail_based"   # Sample based on outcome


@dataclass
class SamplingRule:
    """Rule for telemetry sampling.

    Usage:
        rule = SamplingRule(
            name="high-frequency-logs",
            target_pattern="*.debug",
            sample_rate=0.01,  # Keep 1%
            strategy=SamplingStrategy.RANDOM,
        )
    """
    name: str
    target_pattern: str
    sample_rate: float  # 0.0 to 1.0
    strategy: SamplingStrategy = SamplingStrategy.RANDOM
    priority_tags: List[str] = field(default_factory=list)
    enabled: bool = True
    max_rate_per_second: Optional[int] = None


@dataclass
class OptimizationRecommendation:
    """Recommendation for telemetry optimization."""
    target: str
    recommendation_type: str
    description: str
    estimated_savings_usd: float
    estimated_data_reduction_pct: float
    confidence: float
    implementation_effort: str  # low, medium, high
    priority: int  # 1-5, 1 being highest


@dataclass
class CardinalityReport:
    """Report on metric cardinality."""
    metric_name: str
    unique_label_combinations: int
    label_values: Dict[str, int]  # label -> unique values
    estimated_storage_bytes: int
    high_cardinality_labels: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


class TelemetryOptimizer:
    """Optimizes telemetry collection and storage costs.

    Usage:
        optimizer = TelemetryOptimizer()

        # Add sampling rules
        optimizer.add_rule(SamplingRule(
            name="reduce-debug-logs",
            target_pattern="*.debug",
            sample_rate=0.01,
        ))

        # Check if event should be sampled
        if optimizer.should_sample("app.debug.verbose"):
            send_telemetry(event)

        # Get optimization recommendations
        recommendations = optimizer.get_recommendations()
    """

    def __init__(self, namespace: str = "consciousness"):
        self.namespace = namespace
        self._rules: Dict[str, SamplingRule] = {}
        self._metrics_data: Dict[str, Dict[str, Any]] = {}
        self._rate_counters: Dict[str, List[datetime]] = {}
        self._lock = threading.Lock()

        # Tracking
        self._total_events = 0
        self._sampled_events = 0
        self._dropped_events = 0

        # Prometheus metrics
        self.sampling_rate = Gauge(
            f"{namespace}_telemetry_sampling_rate",
            "Current sampling rate",
            ["rule"],
        )

        self.events_sampled = Counter(
            f"{namespace}_telemetry_events_sampled_total",
            "Events that passed sampling",
            ["rule"],
        )

        self.events_dropped = Counter(
            f"{namespace}_telemetry_events_dropped_total",
            "Events dropped by sampling",
            ["rule"],
        )

        self.cardinality = Gauge(
            f"{namespace}_telemetry_cardinality",
            "Metric cardinality",
            ["metric"],
        )

        self.estimated_savings = Gauge(
            f"{namespace}_telemetry_estimated_savings_dollars",
            "Estimated monthly savings",
        )

    def add_rule(self, rule: SamplingRule):
        """Add a sampling rule.

        Args:
            rule: Sampling rule
        """
        with self._lock:
            self._rules[rule.name] = rule

        self.sampling_rate.labels(rule=rule.name).set(rule.sample_rate)
        logger.info(f"Added sampling rule: {rule.name} ({rule.sample_rate * 100}%)")

    def remove_rule(self, rule_name: str):
        """Remove a sampling rule.

        Args:
            rule_name: Rule name
        """
        with self._lock:
            if rule_name in self._rules:
                del self._rules[rule_name]

    def should_sample(
        self,
        event_name: str,
        tags: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Determine if an event should be sampled (kept).

        Args:
            event_name: Event name
            tags: Event tags

        Returns:
            True if event should be kept
        """
        self._total_events += 1

        with self._lock:
            # Find matching rule
            matching_rule = None
            for rule in self._rules.values():
                if self._matches_pattern(event_name, rule.target_pattern):
                    matching_rule = rule
                    break

        if not matching_rule or not matching_rule.enabled:
            self._sampled_events += 1
            return True

        # Check priority tags
        if tags and matching_rule.priority_tags:
            if any(t in tags.values() for t in matching_rule.priority_tags):
                self._sampled_events += 1
                self.events_sampled.labels(rule=matching_rule.name).inc()
                return True

        # Apply sampling strategy
        should_keep = False

        if matching_rule.strategy == SamplingStrategy.NONE:
            should_keep = True

        elif matching_rule.strategy == SamplingStrategy.RANDOM:
            should_keep = random.random() < matching_rule.sample_rate

        elif matching_rule.strategy == SamplingStrategy.RATE_LIMITING:
            should_keep = self._check_rate_limit(
                matching_rule.name,
                matching_rule.max_rate_per_second or 100,
            )

        elif matching_rule.strategy == SamplingStrategy.ADAPTIVE:
            # Adjust rate based on current load
            load_factor = self._get_load_factor()
            adjusted_rate = matching_rule.sample_rate * (1 / load_factor)
            should_keep = random.random() < min(adjusted_rate, 1.0)

        elif matching_rule.strategy == SamplingStrategy.PRIORITY:
            # Already handled above
            should_keep = random.random() < matching_rule.sample_rate

        if should_keep:
            self._sampled_events += 1
            self.events_sampled.labels(rule=matching_rule.name).inc()
        else:
            self._dropped_events += 1
            self.events_dropped.labels(rule=matching_rule.name).inc()

        return should_keep

    def _matches_pattern(self, name: str, pattern: str) -> bool:
        """Check if name matches pattern."""
        import fnmatch
        return fnmatch.fnmatch(name, pattern)

    def _check_rate_limit(self, rule_name: str, max_per_second: int) -> bool:
        """Check rate limit for a rule."""
        now = datetime.now()
        window = timedelta(seconds=1)

        with self._lock:
            if rule_name not in self._rate_counters:
                self._rate_counters[rule_name] = []

            # Clean old entries
            self._rate_counters[rule_name] = [
                ts for ts in self._rate_counters[rule_name]
                if ts > now - window
            ]

            # Check limit
            if len(self._rate_counters[rule_name]) < max_per_second:
                self._rate_counters[rule_name].append(now)
                return True

        return False

    def _get_load_factor(self) -> float:
        """Get current load factor for adaptive sampling."""
        # Simple implementation - could be more sophisticated
        if self._total_events > 0:
            recent_rate = self._total_events  # events per period
            baseline = 1000  # Expected baseline
            return max(1.0, recent_rate / baseline)
        return 1.0

    def track_metric_cardinality(
        self,
        metric_name: str,
        label_values: Dict[str, str],
    ):
        """Track metric label cardinality.

        Args:
            metric_name: Metric name
            label_values: Current label values
        """
        with self._lock:
            if metric_name not in self._metrics_data:
                self._metrics_data[metric_name] = {
                    "combinations": set(),
                    "label_values": {},
                }

            data = self._metrics_data[metric_name]

            # Track combination
            combo = tuple(sorted(label_values.items()))
            data["combinations"].add(combo)

            # Track individual label values
            for label, value in label_values.items():
                if label not in data["label_values"]:
                    data["label_values"][label] = set()
                data["label_values"][label].add(value)

        # Update metric
        cardinality = len(self._metrics_data[metric_name]["combinations"])
        self.cardinality.labels(metric=metric_name).set(cardinality)

    def get_cardinality_report(self, metric_name: str) -> Optional[CardinalityReport]:
        """Get cardinality report for a metric.

        Args:
            metric_name: Metric name

        Returns:
            CardinalityReport or None
        """
        with self._lock:
            if metric_name not in self._metrics_data:
                return None

            data = self._metrics_data[metric_name]

        unique_combos = len(data["combinations"])
        label_counts = {
            label: len(values)
            for label, values in data["label_values"].items()
        }

        # Identify high cardinality labels (> 100 values)
        high_cardinality = [
            label for label, count in label_counts.items()
            if count > 100
        ]

        # Estimate storage (rough: 100 bytes per combination)
        storage_estimate = unique_combos * 100

        return CardinalityReport(
            metric_name=metric_name,
            unique_label_combinations=unique_combos,
            label_values=label_counts,
            estimated_storage_bytes=storage_estimate,
            high_cardinality_labels=high_cardinality,
        )

    def get_recommendations(self) -> List[OptimizationRecommendation]:
        """Get optimization recommendations.

        Returns:
            List of recommendations
        """
        recommendations = []

        # Check high cardinality metrics
        for metric_name in self._metrics_data:
            report = self.get_cardinality_report(metric_name)
            if report and report.high_cardinality_labels:
                for label in report.high_cardinality_labels:
                    recommendations.append(OptimizationRecommendation(
                        target=f"{metric_name}:{label}",
                        recommendation_type="reduce_cardinality",
                        description=f"Label '{label}' has high cardinality ({report.label_values.get(label, 0)} values). Consider aggregating or removing.",
                        estimated_savings_usd=report.estimated_storage_bytes / 1000 * 0.01,  # $0.01 per KB/month
                        estimated_data_reduction_pct=50.0,
                        confidence=0.8,
                        implementation_effort="medium",
                        priority=2,
                    ))

        # Check sampling opportunities
        if self._total_events > 0:
            drop_rate = self._dropped_events / self._total_events

            if drop_rate < 0.1 and self._total_events > 10000:
                recommendations.append(OptimizationRecommendation(
                    target="telemetry",
                    recommendation_type="increase_sampling",
                    description="Consider increasing sampling for high-volume events",
                    estimated_savings_usd=self._total_events * 0.0001,  # $0.0001 per event
                    estimated_data_reduction_pct=30.0,
                    confidence=0.7,
                    implementation_effort="low",
                    priority=3,
                ))

        return recommendations

    def get_statistics(self) -> Dict[str, Any]:
        """Get telemetry optimization statistics.

        Returns:
            Statistics dictionary
        """
        with self._lock:
            rules = list(self._rules.keys())
            metrics = list(self._metrics_data.keys())

        return {
            "total_events": self._total_events,
            "sampled_events": self._sampled_events,
            "dropped_events": self._dropped_events,
            "sample_rate": self._sampled_events / self._total_events if self._total_events > 0 else 1.0,
            "drop_rate": self._dropped_events / self._total_events if self._total_events > 0 else 0.0,
            "active_rules": len(rules),
            "tracked_metrics": len(metrics),
            "estimated_monthly_savings": self._dropped_events * 0.0001,  # Rough estimate
        }
