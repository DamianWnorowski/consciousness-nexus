"""Pre-Aggregation for High-Cardinality Metrics

Aggregates metrics before export to reduce cardinality explosion and costs.
Implements time-windowed aggregation, percentile sketches, and rollups.
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
import time
import heapq

from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)


class AggregationType(str, Enum):
    """Types of metric aggregation."""
    SUM = "sum"                     # Sum of values
    COUNT = "count"                 # Count of observations
    AVERAGE = "average"             # Mean value
    MIN = "min"                     # Minimum value
    MAX = "max"                     # Maximum value
    LAST = "last"                   # Last observed value
    RATE = "rate"                   # Rate per second
    PERCENTILE = "percentile"       # Percentile values
    HISTOGRAM = "histogram"         # Histogram buckets
    UNIQUE = "unique"               # Unique value count (HyperLogLog)


class RollupPeriod(str, Enum):
    """Rollup aggregation periods."""
    SECOND = "1s"
    MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    HOUR = "1h"
    DAY = "1d"


@dataclass
class AggregationRule:
    """Rule for metric aggregation.

    Usage:
        rule = AggregationRule(
            name="api_latency_rollup",
            metric_pattern="http_request_duration_*",
            aggregation_type=AggregationType.PERCENTILE,
            period=RollupPeriod.MINUTE,
            preserve_labels=["service", "method"],
            drop_labels=["instance", "pod"],
            percentiles=[50, 90, 95, 99],
        )
    """
    name: str
    metric_pattern: str
    aggregation_type: AggregationType
    period: RollupPeriod = RollupPeriod.MINUTE
    preserve_labels: List[str] = field(default_factory=list)
    drop_labels: List[str] = field(default_factory=list)
    percentiles: List[float] = field(default_factory=lambda: [50, 90, 95, 99])
    histogram_buckets: List[float] = field(default_factory=list)
    max_cardinality: int = 1000             # Max unique label combinations
    enabled: bool = True


@dataclass
class AggregatedMetric:
    """An aggregated metric value."""
    metric_name: str
    labels: Dict[str, str]
    aggregation_type: AggregationType
    value: float
    count: int
    period_start: datetime
    period_end: datetime
    additional_values: Dict[str, float] = field(default_factory=dict)


@dataclass
class MetricWindow:
    """A time window of metric observations."""
    values: List[float] = field(default_factory=list)
    count: int = 0
    sum: float = 0.0
    min_value: float = float('inf')
    max_value: float = float('-inf')
    last_value: float = 0.0
    first_timestamp: Optional[datetime] = None
    last_timestamp: Optional[datetime] = None
    unique_values: Set[str] = field(default_factory=set)


class TDigest:
    """T-Digest for approximate percentile calculation.

    A streaming algorithm for accurate percentile estimation with bounded memory.
    """

    def __init__(self, compression: float = 100.0):
        self.compression = compression
        self.centroids: List[Tuple[float, float]] = []  # (mean, weight)
        self.total_weight: float = 0.0
        self._buffer: List[float] = []
        self._buffer_size = 500

    def add(self, value: float, weight: float = 1.0):
        """Add a value to the digest."""
        self._buffer.append(value)
        if len(self._buffer) >= self._buffer_size:
            self._flush_buffer()

    def _flush_buffer(self):
        """Flush buffer and merge with centroids."""
        if not self._buffer:
            return

        # Sort buffer and merge
        self._buffer.sort()
        for value in self._buffer:
            self._add_centroid(value, 1.0)
        self._buffer.clear()

        # Compress centroids
        self._compress()

    def _add_centroid(self, mean: float, weight: float):
        """Add a centroid."""
        self.centroids.append((mean, weight))
        self.total_weight += weight

    def _compress(self):
        """Compress centroids to maintain accuracy bounds."""
        if len(self.centroids) < 2:
            return

        # Sort by mean
        self.centroids.sort(key=lambda x: x[0])

        # Merge nearby centroids
        new_centroids = []
        current_mean, current_weight = self.centroids[0]

        for mean, weight in self.centroids[1:]:
            q = (current_weight + weight / 2) / self.total_weight
            max_weight = 4 * self.total_weight * q * (1 - q) / self.compression

            if current_weight + weight <= max_weight:
                # Merge
                total = current_weight + weight
                current_mean = (current_mean * current_weight + mean * weight) / total
                current_weight = total
            else:
                new_centroids.append((current_mean, current_weight))
                current_mean, current_weight = mean, weight

        new_centroids.append((current_mean, current_weight))
        self.centroids = new_centroids

    def percentile(self, q: float) -> float:
        """Get percentile value (0-100)."""
        self._flush_buffer()

        if not self.centroids:
            return 0.0

        if len(self.centroids) == 1:
            return self.centroids[0][0]

        target = q / 100 * self.total_weight
        cumulative = 0.0

        for i, (mean, weight) in enumerate(self.centroids):
            if cumulative + weight >= target:
                # Interpolate
                if i == 0:
                    return mean
                prev_mean, prev_weight = self.centroids[i - 1]
                prev_cumulative = cumulative - prev_weight / 2
                curr_cumulative = cumulative + weight / 2
                if curr_cumulative - prev_cumulative == 0:
                    return mean
                ratio = (target - prev_cumulative) / (curr_cumulative - prev_cumulative)
                return prev_mean + ratio * (mean - prev_mean)
            cumulative += weight

        return self.centroids[-1][0]

    def merge(self, other: TDigest):
        """Merge another TDigest into this one."""
        other._flush_buffer()
        for mean, weight in other.centroids:
            self._add_centroid(mean, weight)
        self._compress()


class MetricAggregator:
    """Pre-aggregates high-cardinality metrics.

    Usage:
        aggregator = MetricAggregator(namespace="consciousness")

        # Add aggregation rule
        aggregator.add_rule(AggregationRule(
            name="api_latency",
            metric_pattern="http_request_duration",
            aggregation_type=AggregationType.PERCENTILE,
            period=RollupPeriod.MINUTE,
            preserve_labels=["service", "method"],
        ))

        # Record metric value
        aggregator.record(
            metric_name="http_request_duration",
            value=0.125,
            labels={"service": "api", "method": "GET", "path": "/users/123"},
        )

        # Get aggregated metrics
        aggregated = aggregator.flush()
    """

    def __init__(
        self,
        namespace: str = "consciousness",
        max_windows: int = 100,
        flush_interval_seconds: float = 60.0,
    ):
        self.namespace = namespace
        self.max_windows = max_windows
        self.flush_interval = flush_interval_seconds
        self._lock = threading.Lock()

        # Rules and windows
        self._rules: Dict[str, AggregationRule] = {}
        self._windows: Dict[str, Dict[Tuple, MetricWindow]] = defaultdict(dict)
        self._digests: Dict[str, Dict[Tuple, TDigest]] = defaultdict(dict)
        self._current_period: Dict[str, datetime] = {}
        self._last_flush: datetime = datetime.now()

        # Statistics
        self._total_recorded: int = 0
        self._total_aggregated: int = 0
        self._cardinality_exceeded: int = 0

        # Prometheus metrics
        self.records_total = Counter(
            f"{namespace}_aggregator_records_total",
            "Total records aggregated",
            ["metric", "rule"],
        )

        self.aggregated_metrics = Counter(
            f"{namespace}_aggregator_aggregated_total",
            "Total aggregated metrics produced",
            ["rule"],
        )

        self.cardinality = Gauge(
            f"{namespace}_aggregator_cardinality",
            "Current label cardinality",
            ["metric", "rule"],
        )

        self.window_size = Gauge(
            f"{namespace}_aggregator_window_size",
            "Current window observation count",
            ["metric", "rule"],
        )

        self.aggregation_latency = Histogram(
            f"{namespace}_aggregator_latency_seconds",
            "Aggregation latency",
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
        )

    def add_rule(self, rule: AggregationRule):
        """Add an aggregation rule.

        Args:
            rule: Aggregation rule
        """
        with self._lock:
            self._rules[rule.name] = rule
            self._windows[rule.name] = {}
            self._digests[rule.name] = {}
            self._current_period[rule.name] = self._get_period_start(rule.period)

        logger.info(
            f"Added aggregation rule: {rule.name} "
            f"({rule.aggregation_type.value}, {rule.period.value})"
        )

    def remove_rule(self, rule_name: str):
        """Remove an aggregation rule.

        Args:
            rule_name: Rule name to remove
        """
        with self._lock:
            if rule_name in self._rules:
                del self._rules[rule_name]
                del self._windows[rule_name]
                del self._digests[rule_name]
                del self._current_period[rule_name]

    def record(
        self,
        metric_name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None,
    ):
        """Record a metric observation.

        Args:
            metric_name: Metric name
            value: Metric value
            labels: Metric labels
            timestamp: Observation timestamp
        """
        labels = labels or {}
        timestamp = timestamp or datetime.now()
        self._total_recorded += 1

        with self._lock:
            for rule_name, rule in self._rules.items():
                if not rule.enabled:
                    continue

                if not self._matches_pattern(metric_name, rule.metric_pattern):
                    continue

                # Check if period rolled over
                period_start = self._get_period_start(rule.period)
                if period_start > self._current_period[rule_name]:
                    # Period changed, flush previous window
                    self._flush_rule_window(rule_name)
                    self._current_period[rule_name] = period_start

                # Get aggregation key (preserved labels only)
                agg_key = self._get_aggregation_key(labels, rule)

                # Check cardinality limit
                if len(self._windows[rule_name]) >= rule.max_cardinality:
                    if agg_key not in self._windows[rule_name]:
                        self._cardinality_exceeded += 1
                        continue

                # Get or create window
                if agg_key not in self._windows[rule_name]:
                    self._windows[rule_name][agg_key] = MetricWindow()
                    if rule.aggregation_type == AggregationType.PERCENTILE:
                        self._digests[rule_name][agg_key] = TDigest()

                window = self._windows[rule_name][agg_key]

                # Update window
                window.count += 1
                window.sum += value
                window.min_value = min(window.min_value, value)
                window.max_value = max(window.max_value, value)
                window.last_value = value
                window.last_timestamp = timestamp

                if window.first_timestamp is None:
                    window.first_timestamp = timestamp

                # For percentile, add to digest
                if rule.aggregation_type == AggregationType.PERCENTILE:
                    self._digests[rule_name][agg_key].add(value)

                # For histogram, track values
                if rule.aggregation_type == AggregationType.HISTOGRAM:
                    window.values.append(value)
                    # Limit memory
                    if len(window.values) > 10000:
                        window.values = window.values[-5000:]

                # For unique count
                if rule.aggregation_type == AggregationType.UNIQUE:
                    window.unique_values.add(str(value))

                self.records_total.labels(metric=metric_name, rule=rule_name).inc()
                self.cardinality.labels(metric=metric_name, rule=rule_name).set(
                    len(self._windows[rule_name])
                )
                self.window_size.labels(metric=metric_name, rule=rule_name).set(
                    window.count
                )

    def _matches_pattern(self, name: str, pattern: str) -> bool:
        """Check if metric name matches pattern."""
        import fnmatch
        return fnmatch.fnmatch(name, pattern)

    def _get_aggregation_key(
        self,
        labels: Dict[str, str],
        rule: AggregationRule,
    ) -> Tuple:
        """Get aggregation key from labels."""
        preserved = {}

        for label, value in labels.items():
            if label in rule.drop_labels:
                continue
            if rule.preserve_labels and label not in rule.preserve_labels:
                continue
            preserved[label] = value

        return tuple(sorted(preserved.items()))

    def _get_period_start(self, period: RollupPeriod) -> datetime:
        """Get start of current period."""
        now = datetime.now()

        if period == RollupPeriod.SECOND:
            return now.replace(microsecond=0)
        elif period == RollupPeriod.MINUTE:
            return now.replace(second=0, microsecond=0)
        elif period == RollupPeriod.FIVE_MINUTES:
            minute = (now.minute // 5) * 5
            return now.replace(minute=minute, second=0, microsecond=0)
        elif period == RollupPeriod.FIFTEEN_MINUTES:
            minute = (now.minute // 15) * 15
            return now.replace(minute=minute, second=0, microsecond=0)
        elif period == RollupPeriod.HOUR:
            return now.replace(minute=0, second=0, microsecond=0)
        elif period == RollupPeriod.DAY:
            return now.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            return now.replace(second=0, microsecond=0)

    def _get_period_duration(self, period: RollupPeriod) -> timedelta:
        """Get duration of a period."""
        durations = {
            RollupPeriod.SECOND: timedelta(seconds=1),
            RollupPeriod.MINUTE: timedelta(minutes=1),
            RollupPeriod.FIVE_MINUTES: timedelta(minutes=5),
            RollupPeriod.FIFTEEN_MINUTES: timedelta(minutes=15),
            RollupPeriod.HOUR: timedelta(hours=1),
            RollupPeriod.DAY: timedelta(days=1),
        }
        return durations.get(period, timedelta(minutes=1))

    def _flush_rule_window(self, rule_name: str) -> List[AggregatedMetric]:
        """Flush a rule's window and produce aggregated metrics."""
        rule = self._rules[rule_name]
        results = []

        period_start = self._current_period[rule_name]
        period_end = period_start + self._get_period_duration(rule.period)

        for agg_key, window in self._windows[rule_name].items():
            if window.count == 0:
                continue

            labels = dict(agg_key)

            # Calculate aggregated value based on type
            if rule.aggregation_type == AggregationType.SUM:
                aggregated = AggregatedMetric(
                    metric_name=rule.metric_pattern,
                    labels=labels,
                    aggregation_type=rule.aggregation_type,
                    value=window.sum,
                    count=window.count,
                    period_start=period_start,
                    period_end=period_end,
                )

            elif rule.aggregation_type == AggregationType.COUNT:
                aggregated = AggregatedMetric(
                    metric_name=rule.metric_pattern,
                    labels=labels,
                    aggregation_type=rule.aggregation_type,
                    value=float(window.count),
                    count=window.count,
                    period_start=period_start,
                    period_end=period_end,
                )

            elif rule.aggregation_type == AggregationType.AVERAGE:
                avg = window.sum / window.count if window.count > 0 else 0.0
                aggregated = AggregatedMetric(
                    metric_name=rule.metric_pattern,
                    labels=labels,
                    aggregation_type=rule.aggregation_type,
                    value=avg,
                    count=window.count,
                    period_start=period_start,
                    period_end=period_end,
                )

            elif rule.aggregation_type == AggregationType.MIN:
                aggregated = AggregatedMetric(
                    metric_name=rule.metric_pattern,
                    labels=labels,
                    aggregation_type=rule.aggregation_type,
                    value=window.min_value if window.min_value != float('inf') else 0.0,
                    count=window.count,
                    period_start=period_start,
                    period_end=period_end,
                )

            elif rule.aggregation_type == AggregationType.MAX:
                aggregated = AggregatedMetric(
                    metric_name=rule.metric_pattern,
                    labels=labels,
                    aggregation_type=rule.aggregation_type,
                    value=window.max_value if window.max_value != float('-inf') else 0.0,
                    count=window.count,
                    period_start=period_start,
                    period_end=period_end,
                )

            elif rule.aggregation_type == AggregationType.LAST:
                aggregated = AggregatedMetric(
                    metric_name=rule.metric_pattern,
                    labels=labels,
                    aggregation_type=rule.aggregation_type,
                    value=window.last_value,
                    count=window.count,
                    period_start=period_start,
                    period_end=period_end,
                )

            elif rule.aggregation_type == AggregationType.RATE:
                duration = self._get_period_duration(rule.period).total_seconds()
                rate = window.count / duration if duration > 0 else 0.0
                aggregated = AggregatedMetric(
                    metric_name=rule.metric_pattern,
                    labels=labels,
                    aggregation_type=rule.aggregation_type,
                    value=rate,
                    count=window.count,
                    period_start=period_start,
                    period_end=period_end,
                )

            elif rule.aggregation_type == AggregationType.PERCENTILE:
                digest = self._digests[rule_name].get(agg_key)
                if digest:
                    percentile_values = {
                        f"p{int(p)}": digest.percentile(p)
                        for p in rule.percentiles
                    }
                else:
                    percentile_values = {}

                aggregated = AggregatedMetric(
                    metric_name=rule.metric_pattern,
                    labels=labels,
                    aggregation_type=rule.aggregation_type,
                    value=percentile_values.get("p50", 0.0),
                    count=window.count,
                    period_start=period_start,
                    period_end=period_end,
                    additional_values=percentile_values,
                )

            elif rule.aggregation_type == AggregationType.HISTOGRAM:
                buckets = rule.histogram_buckets or [
                    0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
                ]
                bucket_counts = {f"le_{b}": 0.0 for b in buckets}
                bucket_counts["le_inf"] = float(len(window.values))

                for value in window.values:
                    for bucket in buckets:
                        if value <= bucket:
                            bucket_counts[f"le_{bucket}"] += 1

                aggregated = AggregatedMetric(
                    metric_name=rule.metric_pattern,
                    labels=labels,
                    aggregation_type=rule.aggregation_type,
                    value=window.sum,
                    count=window.count,
                    period_start=period_start,
                    period_end=period_end,
                    additional_values=bucket_counts,
                )

            elif rule.aggregation_type == AggregationType.UNIQUE:
                aggregated = AggregatedMetric(
                    metric_name=rule.metric_pattern,
                    labels=labels,
                    aggregation_type=rule.aggregation_type,
                    value=float(len(window.unique_values)),
                    count=window.count,
                    period_start=period_start,
                    period_end=period_end,
                )

            else:
                continue

            results.append(aggregated)
            self._total_aggregated += 1

        # Clear window
        self._windows[rule_name] = {}
        self._digests[rule_name] = {}

        self.aggregated_metrics.labels(rule=rule_name).inc(len(results))

        return results

    def flush(self, force: bool = False) -> List[AggregatedMetric]:
        """Flush all windows and get aggregated metrics.

        Args:
            force: Force flush regardless of time

        Returns:
            List of aggregated metrics
        """
        start = time.perf_counter()
        results = []

        with self._lock:
            now = datetime.now()

            # Check if flush interval elapsed
            if not force and (now - self._last_flush).total_seconds() < self.flush_interval:
                return []

            for rule_name in list(self._rules.keys()):
                results.extend(self._flush_rule_window(rule_name))
                self._current_period[rule_name] = self._get_period_start(
                    self._rules[rule_name].period
                )

            self._last_flush = now

        duration = time.perf_counter() - start
        self.aggregation_latency.observe(duration)

        if results:
            logger.debug(f"Flushed {len(results)} aggregated metrics")

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregator statistics.

        Returns:
            Statistics dictionary
        """
        with self._lock:
            rule_stats = {}
            for rule_name, windows in self._windows.items():
                total_observations = sum(w.count for w in windows.values())
                rule_stats[rule_name] = {
                    "cardinality": len(windows),
                    "observations": total_observations,
                }

        return {
            "total_recorded": self._total_recorded,
            "total_aggregated": self._total_aggregated,
            "cardinality_exceeded": self._cardinality_exceeded,
            "rules": len(self._rules),
            "rule_stats": rule_stats,
        }


class RollupAggregator:
    """Multi-level rollup aggregator for hierarchical aggregation.

    Usage:
        rollup = RollupAggregator(
            levels=[
                RollupPeriod.MINUTE,
                RollupPeriod.HOUR,
                RollupPeriod.DAY,
            ]
        )
    """

    def __init__(
        self,
        namespace: str = "consciousness",
        levels: Optional[List[RollupPeriod]] = None,
    ):
        self.namespace = namespace
        self.levels = levels or [
            RollupPeriod.MINUTE,
            RollupPeriod.HOUR,
            RollupPeriod.DAY,
        ]

        # Create aggregator for each level
        self._aggregators: Dict[RollupPeriod, MetricAggregator] = {}
        for level in self.levels:
            flush_interval = self._get_flush_interval(level)
            self._aggregators[level] = MetricAggregator(
                namespace=f"{namespace}_{level.value}",
                flush_interval_seconds=flush_interval,
            )

    def _get_flush_interval(self, period: RollupPeriod) -> float:
        """Get flush interval for a period."""
        intervals = {
            RollupPeriod.SECOND: 1.0,
            RollupPeriod.MINUTE: 60.0,
            RollupPeriod.FIVE_MINUTES: 300.0,
            RollupPeriod.FIFTEEN_MINUTES: 900.0,
            RollupPeriod.HOUR: 3600.0,
            RollupPeriod.DAY: 86400.0,
        }
        return intervals.get(period, 60.0)

    def add_rule(self, rule: AggregationRule, levels: Optional[List[RollupPeriod]] = None):
        """Add a rule to specified levels.

        Args:
            rule: Aggregation rule
            levels: Levels to add to (None = all levels)
        """
        target_levels = levels or self.levels

        for level in target_levels:
            if level in self._aggregators:
                level_rule = AggregationRule(
                    name=f"{rule.name}_{level.value}",
                    metric_pattern=rule.metric_pattern,
                    aggregation_type=rule.aggregation_type,
                    period=level,
                    preserve_labels=rule.preserve_labels,
                    drop_labels=rule.drop_labels,
                    percentiles=rule.percentiles,
                    histogram_buckets=rule.histogram_buckets,
                    max_cardinality=rule.max_cardinality,
                    enabled=rule.enabled,
                )
                self._aggregators[level].add_rule(level_rule)

    def record(
        self,
        metric_name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ):
        """Record to finest granularity level.

        Args:
            metric_name: Metric name
            value: Metric value
            labels: Metric labels
        """
        if self.levels:
            finest_level = self.levels[0]
            self._aggregators[finest_level].record(metric_name, value, labels)

    def flush_all(self) -> Dict[RollupPeriod, List[AggregatedMetric]]:
        """Flush all levels and cascade aggregations.

        Returns:
            Dictionary of level to aggregated metrics
        """
        results = {}

        for level in self.levels:
            results[level] = self._aggregators[level].flush(force=True)

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics for all levels."""
        return {
            level.value: aggregator.get_statistics()
            for level, aggregator in self._aggregators.items()
        }
