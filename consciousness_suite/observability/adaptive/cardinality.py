"""Cardinality Management and Explosion Prevention

Monitors, limits, and manages metric label cardinality to prevent
runaway costs and performance degradation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import threading
import logging
import hashlib
import time

from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)


class CardinalityAction(str, Enum):
    """Actions to take on high cardinality."""
    ALLOW = "allow"             # Allow the label combination
    DROP = "drop"               # Drop the metric entirely
    AGGREGATE = "aggregate"     # Aggregate into "other" bucket
    TRUNCATE = "truncate"       # Truncate high-cardinality label values
    HASH = "hash"               # Hash label values
    SAMPLE = "sample"           # Sample high-cardinality values


class CardinalityLevel(str, Enum):
    """Cardinality severity levels."""
    LOW = "low"                 # < 100 unique combinations
    MEDIUM = "medium"           # 100-1000 combinations
    HIGH = "high"               # 1000-10000 combinations
    CRITICAL = "critical"       # > 10000 combinations
    EXPLOSION = "explosion"     # Growing unbounded


@dataclass
class CardinalityLimit:
    """Limit configuration for a metric.

    Usage:
        limit = CardinalityLimit(
            metric_pattern="http_requests_*",
            max_cardinality=1000,
            max_label_values={"path": 100, "user_id": 0},  # 0 = drop label
            action=CardinalityAction.AGGREGATE,
        )
    """
    metric_pattern: str
    max_cardinality: int = 1000             # Max unique label combinations
    max_label_values: Dict[str, int] = field(default_factory=dict)  # Per-label limits
    action: CardinalityAction = CardinalityAction.AGGREGATE
    aggregate_label: str = "__aggregated__"  # Label value for aggregated bucket
    truncate_length: int = 32               # Max label value length
    warning_threshold: float = 0.8          # Warn at 80% of limit
    enabled: bool = True


@dataclass
class CardinalityReport:
    """Report on metric cardinality status."""
    metric_name: str
    total_combinations: int
    label_cardinalities: Dict[str, int]     # label -> unique value count
    level: CardinalityLevel
    growth_rate_per_hour: float
    estimated_time_to_limit: Optional[timedelta]
    high_cardinality_labels: List[str]
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class LabelValueSet:
    """Tracked label values for a metric."""
    values: Set[str] = field(default_factory=set)
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    count: int = 0
    overflow_count: int = 0                 # Values dropped due to limits


class CardinalityManager:
    """Manages and controls metric cardinality.

    Usage:
        manager = CardinalityManager(namespace="consciousness")

        # Set cardinality limit
        manager.set_limit(CardinalityLimit(
            metric_pattern="http_*",
            max_cardinality=1000,
            max_label_values={"path": 100},
        ))

        # Check/transform labels before recording
        result = manager.check_labels(
            metric_name="http_requests",
            labels={"method": "GET", "path": "/users/123", "status": "200"},
        )

        if result.allowed:
            record_metric(metric_name, result.labels)  # May be transformed
    """

    def __init__(
        self,
        namespace: str = "consciousness",
        default_max_cardinality: int = 10000,
        cleanup_interval_seconds: int = 300,
    ):
        self.namespace = namespace
        self.default_max_cardinality = default_max_cardinality
        self.cleanup_interval = cleanup_interval_seconds
        self._lock = threading.Lock()

        # Configuration
        self._limits: Dict[str, CardinalityLimit] = {}

        # Tracking state
        self._metric_labels: Dict[str, Dict[str, LabelValueSet]] = defaultdict(dict)
        self._metric_combinations: Dict[str, Set[Tuple]] = defaultdict(set)
        self._cardinality_history: Dict[str, List[Tuple[datetime, int]]] = defaultdict(list)
        self._last_cleanup: datetime = datetime.now()

        # Alert callbacks
        self._alert_callbacks: List[Callable[[CardinalityReport], None]] = []

        # Statistics
        self._total_checked: int = 0
        self._total_allowed: int = 0
        self._total_transformed: int = 0
        self._total_dropped: int = 0

        # Prometheus metrics
        self.cardinality_total = Gauge(
            f"{namespace}_cardinality_total",
            "Total label combinations",
            ["metric"],
        )

        self.cardinality_limit = Gauge(
            f"{namespace}_cardinality_limit",
            "Cardinality limit",
            ["metric"],
        )

        self.cardinality_ratio = Gauge(
            f"{namespace}_cardinality_ratio",
            "Cardinality usage ratio (0-1)",
            ["metric"],
        )

        self.label_cardinality = Gauge(
            f"{namespace}_label_cardinality",
            "Per-label cardinality",
            ["metric", "label"],
        )

        self.cardinality_actions = Counter(
            f"{namespace}_cardinality_actions_total",
            "Cardinality management actions",
            ["action", "metric"],
        )

        self.cardinality_alerts = Counter(
            f"{namespace}_cardinality_alerts_total",
            "Cardinality alerts triggered",
            ["level", "metric"],
        )

    def set_limit(self, limit: CardinalityLimit):
        """Set cardinality limit for a metric pattern.

        Args:
            limit: CardinalityLimit configuration
        """
        with self._lock:
            self._limits[limit.metric_pattern] = limit

        self.cardinality_limit.labels(metric=limit.metric_pattern).set(
            limit.max_cardinality
        )

        logger.info(
            f"Set cardinality limit: {limit.metric_pattern} "
            f"(max: {limit.max_cardinality}, action: {limit.action.value})"
        )

    def remove_limit(self, metric_pattern: str):
        """Remove cardinality limit.

        Args:
            metric_pattern: Pattern to remove
        """
        with self._lock:
            if metric_pattern in self._limits:
                del self._limits[metric_pattern]

    def check_labels(
        self,
        metric_name: str,
        labels: Dict[str, str],
    ) -> "CardinalityCheckResult":
        """Check labels against cardinality limits.

        Args:
            metric_name: Metric name
            labels: Label dictionary

        Returns:
            CardinalityCheckResult with allowed status and possibly transformed labels
        """
        self._total_checked += 1

        # Find matching limit
        limit = self._find_matching_limit(metric_name)

        if not limit or not limit.enabled:
            self._total_allowed += 1
            return CardinalityCheckResult(
                allowed=True,
                labels=labels,
                action=CardinalityAction.ALLOW,
            )

        # Apply per-label limits first
        transformed_labels = self._apply_label_limits(metric_name, labels, limit)

        # Track and check cardinality
        combination_key = tuple(sorted(transformed_labels.items()))

        with self._lock:
            current_cardinality = len(self._metric_combinations[metric_name])

            # Check if this is a new combination
            is_new = combination_key not in self._metric_combinations[metric_name]

            if is_new:
                # Check if we're at the limit
                if current_cardinality >= limit.max_cardinality:
                    # Apply action
                    result = self._apply_cardinality_action(
                        metric_name, transformed_labels, limit
                    )
                    return result

                # Add new combination
                self._metric_combinations[metric_name].add(combination_key)

                # Track label values
                for label, value in transformed_labels.items():
                    if label not in self._metric_labels[metric_name]:
                        self._metric_labels[metric_name][label] = LabelValueSet()
                    label_set = self._metric_labels[metric_name][label]
                    label_set.values.add(value)
                    label_set.last_seen = datetime.now()
                    label_set.count += 1

            # Update metrics
            new_cardinality = len(self._metric_combinations[metric_name])

        self._update_metrics(metric_name, limit, new_cardinality)

        # Check warning threshold
        if new_cardinality >= limit.max_cardinality * limit.warning_threshold:
            self._check_and_alert(metric_name, new_cardinality, limit)

        # Cleanup periodically
        self._maybe_cleanup()

        if transformed_labels != labels:
            self._total_transformed += 1
            self.cardinality_actions.labels(
                action="transform",
                metric=metric_name,
            ).inc()

            return CardinalityCheckResult(
                allowed=True,
                labels=transformed_labels,
                action=CardinalityAction.TRUNCATE,
                original_labels=labels,
            )

        self._total_allowed += 1
        return CardinalityCheckResult(
            allowed=True,
            labels=labels,
            action=CardinalityAction.ALLOW,
        )

    def _find_matching_limit(self, metric_name: str) -> Optional[CardinalityLimit]:
        """Find matching cardinality limit for metric."""
        import fnmatch

        for pattern, limit in self._limits.items():
            if fnmatch.fnmatch(metric_name, pattern):
                return limit

        return None

    def _apply_label_limits(
        self,
        metric_name: str,
        labels: Dict[str, str],
        limit: CardinalityLimit,
    ) -> Dict[str, str]:
        """Apply per-label cardinality limits."""
        result = {}

        for label, value in labels.items():
            # Check if label should be dropped
            if limit.max_label_values.get(label, -1) == 0:
                continue

            # Check per-label limit
            label_limit = limit.max_label_values.get(label)
            if label_limit:
                with self._lock:
                    if label in self._metric_labels[metric_name]:
                        label_set = self._metric_labels[metric_name][label]
                        if len(label_set.values) >= label_limit and value not in label_set.values:
                            # Limit reached, aggregate or hash
                            if limit.action == CardinalityAction.HASH:
                                value = self._hash_value(value)
                            else:
                                value = limit.aggregate_label
                                label_set.overflow_count += 1

            # Apply truncation if needed
            if len(value) > limit.truncate_length:
                value = value[:limit.truncate_length]

            result[label] = value

        return result

    def _apply_cardinality_action(
        self,
        metric_name: str,
        labels: Dict[str, str],
        limit: CardinalityLimit,
    ) -> "CardinalityCheckResult":
        """Apply action when cardinality limit reached."""
        action = limit.action

        self.cardinality_actions.labels(
            action=action.value,
            metric=metric_name,
        ).inc()

        if action == CardinalityAction.DROP:
            self._total_dropped += 1
            return CardinalityCheckResult(
                allowed=False,
                labels=labels,
                action=action,
                reason="cardinality_limit_reached",
            )

        elif action == CardinalityAction.AGGREGATE:
            # Aggregate all labels into "other" bucket
            aggregated_labels = {
                label: limit.aggregate_label
                for label in labels
            }
            self._total_transformed += 1
            return CardinalityCheckResult(
                allowed=True,
                labels=aggregated_labels,
                action=action,
                original_labels=labels,
            )

        elif action == CardinalityAction.HASH:
            # Hash label values to reduce cardinality
            hashed_labels = {
                label: self._hash_value(value)
                for label, value in labels.items()
            }
            self._total_transformed += 1
            return CardinalityCheckResult(
                allowed=True,
                labels=hashed_labels,
                action=action,
                original_labels=labels,
            )

        elif action == CardinalityAction.SAMPLE:
            # Probabilistically drop
            import random
            cardinality = len(self._metric_combinations[metric_name])
            keep_probability = limit.max_cardinality / (cardinality + 1)

            if random.random() > keep_probability:
                self._total_dropped += 1
                return CardinalityCheckResult(
                    allowed=False,
                    labels=labels,
                    action=action,
                    reason="sampled_out",
                )

            self._total_allowed += 1
            return CardinalityCheckResult(
                allowed=True,
                labels=labels,
                action=action,
            )

        else:
            # Default: allow with truncation
            truncated = {
                label: value[:limit.truncate_length]
                for label, value in labels.items()
            }
            return CardinalityCheckResult(
                allowed=True,
                labels=truncated,
                action=CardinalityAction.TRUNCATE,
                original_labels=labels,
            )

    def _hash_value(self, value: str, length: int = 8) -> str:
        """Hash a label value to fixed length."""
        return hashlib.md5(value.encode()).hexdigest()[:length]

    def _update_metrics(
        self,
        metric_name: str,
        limit: CardinalityLimit,
        cardinality: int,
    ):
        """Update Prometheus metrics."""
        self.cardinality_total.labels(metric=metric_name).set(cardinality)
        self.cardinality_ratio.labels(metric=metric_name).set(
            cardinality / limit.max_cardinality
        )

        # Update per-label cardinality
        for label, label_set in self._metric_labels.get(metric_name, {}).items():
            self.label_cardinality.labels(
                metric=metric_name,
                label=label,
            ).set(len(label_set.values))

        # Track history
        self._cardinality_history[metric_name].append((datetime.now(), cardinality))

        # Trim history
        cutoff = datetime.now() - timedelta(hours=24)
        self._cardinality_history[metric_name] = [
            (ts, c) for ts, c in self._cardinality_history[metric_name]
            if ts > cutoff
        ]

    def _check_and_alert(
        self,
        metric_name: str,
        cardinality: int,
        limit: CardinalityLimit,
    ):
        """Check conditions and trigger alerts."""
        level = self._get_cardinality_level(cardinality, limit)

        if level in (CardinalityLevel.HIGH, CardinalityLevel.CRITICAL, CardinalityLevel.EXPLOSION):
            report = self.get_cardinality_report(metric_name)

            if report:
                self.cardinality_alerts.labels(
                    level=level.value,
                    metric=metric_name,
                ).inc()

                for callback in self._alert_callbacks:
                    try:
                        callback(report)
                    except Exception as e:
                        logger.error(f"Alert callback error: {e}")

                logger.warning(
                    f"Cardinality alert: {metric_name} at {level.value} "
                    f"({cardinality}/{limit.max_cardinality})"
                )

    def _get_cardinality_level(
        self,
        cardinality: int,
        limit: CardinalityLimit,
    ) -> CardinalityLevel:
        """Determine cardinality severity level."""
        ratio = cardinality / limit.max_cardinality

        if ratio >= 1.0:
            # Check growth rate for explosion detection
            return CardinalityLevel.EXPLOSION
        elif ratio >= 0.8:
            return CardinalityLevel.CRITICAL
        elif ratio >= 0.5:
            return CardinalityLevel.HIGH
        elif ratio >= 0.1:
            return CardinalityLevel.MEDIUM
        else:
            return CardinalityLevel.LOW

    def _maybe_cleanup(self):
        """Periodically cleanup old data."""
        now = datetime.now()
        if (now - self._last_cleanup).total_seconds() < self.cleanup_interval:
            return

        with self._lock:
            # Remove old history entries
            cutoff = now - timedelta(hours=24)
            for metric_name in list(self._cardinality_history.keys()):
                self._cardinality_history[metric_name] = [
                    (ts, c) for ts, c in self._cardinality_history[metric_name]
                    if ts > cutoff
                ]

            self._last_cleanup = now

    def get_cardinality_report(self, metric_name: str) -> Optional[CardinalityReport]:
        """Get detailed cardinality report for a metric.

        Args:
            metric_name: Metric name

        Returns:
            CardinalityReport or None
        """
        with self._lock:
            if metric_name not in self._metric_combinations:
                return None

            cardinality = len(self._metric_combinations[metric_name])

            label_cardinalities = {
                label: len(label_set.values)
                for label, label_set in self._metric_labels.get(metric_name, {}).items()
            }

        # Find limit
        limit = self._find_matching_limit(metric_name)
        max_cardinality = limit.max_cardinality if limit else self.default_max_cardinality

        # Calculate growth rate
        growth_rate = self._calculate_growth_rate(metric_name)

        # Estimate time to limit
        time_to_limit = None
        if growth_rate > 0 and cardinality < max_cardinality:
            remaining = max_cardinality - cardinality
            hours = remaining / growth_rate
            time_to_limit = timedelta(hours=hours)

        # Identify high cardinality labels
        high_cardinality_labels = [
            label for label, count in label_cardinalities.items()
            if count > 100
        ]

        # Generate recommendations
        recommendations = []
        if high_cardinality_labels:
            recommendations.append(
                f"Consider aggregating or dropping labels: {', '.join(high_cardinality_labels)}"
            )
        if growth_rate > 100:
            recommendations.append(
                "High growth rate detected. Consider applying stricter limits."
            )
        if cardinality > max_cardinality * 0.8:
            recommendations.append(
                "Approaching cardinality limit. Take action to prevent data loss."
            )

        level = self._get_cardinality_level(
            cardinality,
            limit or CardinalityLimit(metric_pattern=metric_name, max_cardinality=max_cardinality),
        )

        return CardinalityReport(
            metric_name=metric_name,
            total_combinations=cardinality,
            label_cardinalities=label_cardinalities,
            level=level,
            growth_rate_per_hour=growth_rate,
            estimated_time_to_limit=time_to_limit,
            high_cardinality_labels=high_cardinality_labels,
            recommendations=recommendations,
        )

    def _calculate_growth_rate(self, metric_name: str) -> float:
        """Calculate cardinality growth rate per hour."""
        history = self._cardinality_history.get(metric_name, [])

        if len(history) < 2:
            return 0.0

        # Use linear regression on last hour
        cutoff = datetime.now() - timedelta(hours=1)
        recent = [(ts, c) for ts, c in history if ts > cutoff]

        if len(recent) < 2:
            return 0.0

        # Simple linear regression
        first_ts, first_c = recent[0]
        last_ts, last_c = recent[-1]

        hours = (last_ts - first_ts).total_seconds() / 3600
        if hours <= 0:
            return 0.0

        return (last_c - first_c) / hours

    def get_all_reports(self) -> Dict[str, CardinalityReport]:
        """Get cardinality reports for all tracked metrics.

        Returns:
            Dictionary of metric name to report
        """
        with self._lock:
            metrics = list(self._metric_combinations.keys())

        return {
            metric: report
            for metric in metrics
            if (report := self.get_cardinality_report(metric)) is not None
        }

    def on_alert(self, callback: Callable[[CardinalityReport], None]):
        """Register callback for cardinality alerts.

        Args:
            callback: Function to call on alert
        """
        self._alert_callbacks.append(callback)

    def get_statistics(self) -> Dict[str, Any]:
        """Get cardinality management statistics.

        Returns:
            Statistics dictionary
        """
        with self._lock:
            metrics_tracked = len(self._metric_combinations)
            total_combinations = sum(
                len(combos) for combos in self._metric_combinations.values()
            )

        return {
            "total_checked": self._total_checked,
            "total_allowed": self._total_allowed,
            "total_transformed": self._total_transformed,
            "total_dropped": self._total_dropped,
            "transform_rate": (
                self._total_transformed / self._total_checked
                if self._total_checked > 0 else 0.0
            ),
            "drop_rate": (
                self._total_dropped / self._total_checked
                if self._total_checked > 0 else 0.0
            ),
            "metrics_tracked": metrics_tracked,
            "total_combinations": total_combinations,
            "limits_configured": len(self._limits),
        }

    def reset_metric(self, metric_name: str):
        """Reset tracking for a metric.

        Args:
            metric_name: Metric name to reset
        """
        with self._lock:
            if metric_name in self._metric_combinations:
                del self._metric_combinations[metric_name]
            if metric_name in self._metric_labels:
                del self._metric_labels[metric_name]
            if metric_name in self._cardinality_history:
                del self._cardinality_history[metric_name]


@dataclass
class CardinalityCheckResult:
    """Result of cardinality check."""
    allowed: bool
    labels: Dict[str, str]
    action: CardinalityAction
    original_labels: Optional[Dict[str, str]] = None
    reason: Optional[str] = None
