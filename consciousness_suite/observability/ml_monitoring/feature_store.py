"""Feature Store Monitoring

Provides monitoring for feature stores:
- Feature freshness tracking
- Feature quality metrics
- Feature usage analytics
- Feature drift alerts
"""

from __future__ import annotations

import time
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)


class FeatureType(str, Enum):
    """Types of features."""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    TEXT = "text"
    EMBEDDING = "embedding"
    TIMESTAMP = "timestamp"
    ARRAY = "array"


class FreshnessStatus(str, Enum):
    """Freshness status levels."""
    FRESH = "fresh"
    STALE = "stale"
    EXPIRED = "expired"
    UNKNOWN = "unknown"


class HealthStatus(str, Enum):
    """Feature health status."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class FeatureMetadata:
    """Metadata for a feature."""
    name: str
    feature_type: FeatureType
    entity: str
    description: str = ""
    owner: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    freshness_sla_seconds: int = 3600  # Default 1 hour SLA
    expected_min: Optional[float] = None
    expected_max: Optional[float] = None
    expected_null_rate: float = 0.0
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureFreshness:
    """Freshness information for a feature."""
    feature_name: str
    last_updated: datetime
    age_seconds: float
    sla_seconds: int
    status: FreshnessStatus
    staleness_ratio: float  # age / sla
    within_sla: bool


@dataclass
class FeatureHealth:
    """Health information for a feature."""
    feature_name: str
    status: HealthStatus
    null_rate: float
    out_of_range_rate: float
    freshness: FeatureFreshness
    quality_score: float
    issues: List[str] = field(default_factory=list)
    last_checked: datetime = field(default_factory=datetime.now)


@dataclass
class FeatureStats:
    """Statistical summary of a feature."""
    feature_name: str
    count: int
    null_count: int
    null_rate: float
    unique_count: int
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean_value: Optional[float] = None
    std_value: Optional[float] = None
    median_value: Optional[float] = None
    most_common: Optional[Any] = None
    timestamp: datetime = field(default_factory=datetime.now)


class FeatureStoreMonitor:
    """Monitor for feature store health and quality.

    Usage:
        monitor = FeatureStoreMonitor(store_name="main-features")

        # Register feature
        monitor.register_feature(FeatureMetadata(
            name="user_age",
            feature_type=FeatureType.NUMERICAL,
            entity="user",
            freshness_sla_seconds=3600,
        ))

        # Record feature update
        monitor.record_update("user_age")

        # Check feature health
        health = monitor.get_feature_health("user_age")

        # Record feature serving
        with monitor.track_serving("user_age"):
            features = get_feature("user_age")
    """

    def __init__(
        self,
        store_name: str,
        namespace: str = "consciousness",
        default_sla_seconds: int = 3600,
    ):
        self.store_name = store_name
        self.namespace = namespace
        self.default_sla_seconds = default_sla_seconds

        self._features: Dict[str, FeatureMetadata] = {}
        self._last_updates: Dict[str, datetime] = {}
        self._feature_stats: Dict[str, FeatureStats] = {}
        self._serving_history: Dict[str, List[Dict[str, Any]]] = {}
        self._alerts: List[Dict[str, Any]] = []
        self._callbacks: List[Callable[[FeatureHealth], None]] = []
        self._lock = threading.Lock()
        self._max_history = 10000

        # Prometheus metrics
        self.feature_freshness_seconds = Gauge(
            f"{namespace}_feature_freshness_seconds",
            "Feature freshness in seconds",
            ["store", "feature", "entity"],
        )

        self.feature_staleness_ratio = Gauge(
            f"{namespace}_feature_staleness_ratio",
            "Feature staleness ratio (age/sla)",
            ["store", "feature"],
        )

        self.feature_null_rate = Gauge(
            f"{namespace}_feature_null_rate",
            "Feature null rate",
            ["store", "feature"],
        )

        self.feature_quality_score = Gauge(
            f"{namespace}_feature_quality_score",
            "Feature quality score (0-1)",
            ["store", "feature"],
        )

        self.feature_updates = Counter(
            f"{namespace}_feature_updates_total",
            "Total feature updates",
            ["store", "feature"],
        )

        self.feature_serves = Counter(
            f"{namespace}_feature_serves_total",
            "Total feature serving requests",
            ["store", "feature", "status"],
        )

        self.feature_serving_latency = Histogram(
            f"{namespace}_feature_serving_duration_seconds",
            "Feature serving latency",
            ["store", "feature"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
        )

        self.stale_features_gauge = Gauge(
            f"{namespace}_feature_store_stale_features",
            "Number of stale features",
            ["store"],
        )

        self.total_features_gauge = Gauge(
            f"{namespace}_feature_store_total_features",
            "Total registered features",
            ["store"],
        )

    def register_feature(self, metadata: FeatureMetadata):
        """Register a feature for monitoring.

        Args:
            metadata: Feature metadata
        """
        with self._lock:
            self._features[metadata.name] = metadata
            self._last_updates[metadata.name] = datetime.now()
            self._serving_history[metadata.name] = []

        self.total_features_gauge.labels(store=self.store_name).set(
            len(self._features)
        )

        logger.info(f"Registered feature: {metadata.name}")

    def unregister_feature(self, feature_name: str):
        """Unregister a feature.

        Args:
            feature_name: Feature to unregister
        """
        with self._lock:
            if feature_name in self._features:
                del self._features[feature_name]
                self._last_updates.pop(feature_name, None)
                self._feature_stats.pop(feature_name, None)
                self._serving_history.pop(feature_name, None)

        self.total_features_gauge.labels(store=self.store_name).set(
            len(self._features)
        )

    def record_update(
        self,
        feature_name: str,
        stats: Optional[FeatureStats] = None,
    ):
        """Record a feature update.

        Args:
            feature_name: Feature that was updated
            stats: Optional statistics about the update
        """
        now = datetime.now()

        with self._lock:
            self._last_updates[feature_name] = now

            if stats:
                self._feature_stats[feature_name] = stats
                self.feature_null_rate.labels(
                    store=self.store_name,
                    feature=feature_name,
                ).set(stats.null_rate)

        self.feature_updates.labels(
            store=self.store_name,
            feature=feature_name,
        ).inc()

        self.feature_freshness_seconds.labels(
            store=self.store_name,
            feature=feature_name,
            entity=self._features.get(feature_name, FeatureMetadata(
                name=feature_name,
                feature_type=FeatureType.NUMERICAL,
                entity="unknown",
            )).entity,
        ).set(0)

    def record_stats(
        self,
        feature_name: str,
        values: List[Any],
        feature_type: Optional[FeatureType] = None,
    ):
        """Record feature statistics from values.

        Args:
            feature_name: Feature name
            values: List of feature values
            feature_type: Optional feature type override
        """
        if not values:
            return

        null_count = sum(1 for v in values if v is None)
        non_null_values = [v for v in values if v is not None]

        stats = FeatureStats(
            feature_name=feature_name,
            count=len(values),
            null_count=null_count,
            null_rate=null_count / len(values) if values else 0,
            unique_count=len(set(str(v) for v in non_null_values)),
        )

        # Calculate numerical stats if applicable
        ftype = feature_type
        if feature_name in self._features:
            ftype = self._features[feature_name].feature_type

        if ftype == FeatureType.NUMERICAL and non_null_values:
            numeric_values = [float(v) for v in non_null_values if self._is_numeric(v)]
            if numeric_values:
                stats.min_value = min(numeric_values)
                stats.max_value = max(numeric_values)
                stats.mean_value = sum(numeric_values) / len(numeric_values)

                if len(numeric_values) > 1:
                    mean = stats.mean_value
                    variance = sum((x - mean) ** 2 for x in numeric_values) / len(numeric_values)
                    stats.std_value = variance ** 0.5

                sorted_vals = sorted(numeric_values)
                n = len(sorted_vals)
                stats.median_value = sorted_vals[n // 2]

        with self._lock:
            self._feature_stats[feature_name] = stats

        self.feature_null_rate.labels(
            store=self.store_name,
            feature=feature_name,
        ).set(stats.null_rate)

    def _is_numeric(self, value: Any) -> bool:
        """Check if value is numeric."""
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False

    def track_serving(self, feature_name: str):
        """Context manager for tracking feature serving.

        Args:
            feature_name: Feature being served

        Returns:
            Context manager
        """
        return _ServingTracker(self, feature_name)

    def record_serving(
        self,
        feature_name: str,
        latency_seconds: float,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Record a feature serving event.

        Args:
            feature_name: Feature served
            latency_seconds: Serving latency
            success: Whether serving was successful
            metadata: Additional metadata
        """
        status = "success" if success else "error"

        self.feature_serves.labels(
            store=self.store_name,
            feature=feature_name,
            status=status,
        ).inc()

        self.feature_serving_latency.labels(
            store=self.store_name,
            feature=feature_name,
        ).observe(latency_seconds)

        with self._lock:
            if feature_name in self._serving_history:
                self._serving_history[feature_name].append({
                    "timestamp": datetime.now(),
                    "latency": latency_seconds,
                    "success": success,
                    "metadata": metadata or {},
                })

                # Trim history
                if len(self._serving_history[feature_name]) > self._max_history:
                    self._serving_history[feature_name] = (
                        self._serving_history[feature_name][-self._max_history // 2:]
                    )

    def get_feature_freshness(self, feature_name: str) -> FeatureFreshness:
        """Get freshness status for a feature.

        Args:
            feature_name: Feature name

        Returns:
            FeatureFreshness
        """
        now = datetime.now()

        with self._lock:
            metadata = self._features.get(feature_name)
            last_update = self._last_updates.get(feature_name)

        if not last_update:
            return FeatureFreshness(
                feature_name=feature_name,
                last_updated=datetime.min,
                age_seconds=float("inf"),
                sla_seconds=self.default_sla_seconds,
                status=FreshnessStatus.UNKNOWN,
                staleness_ratio=float("inf"),
                within_sla=False,
            )

        sla = metadata.freshness_sla_seconds if metadata else self.default_sla_seconds
        age = (now - last_update).total_seconds()
        staleness_ratio = age / sla if sla > 0 else 0

        if staleness_ratio <= 1.0:
            status = FreshnessStatus.FRESH
        elif staleness_ratio <= 2.0:
            status = FreshnessStatus.STALE
        else:
            status = FreshnessStatus.EXPIRED

        freshness = FeatureFreshness(
            feature_name=feature_name,
            last_updated=last_update,
            age_seconds=age,
            sla_seconds=sla,
            status=status,
            staleness_ratio=staleness_ratio,
            within_sla=staleness_ratio <= 1.0,
        )

        # Update metrics
        self.feature_freshness_seconds.labels(
            store=self.store_name,
            feature=feature_name,
            entity=metadata.entity if metadata else "unknown",
        ).set(age)

        self.feature_staleness_ratio.labels(
            store=self.store_name,
            feature=feature_name,
        ).set(staleness_ratio)

        return freshness

    def get_feature_health(self, feature_name: str) -> FeatureHealth:
        """Get health status for a feature.

        Args:
            feature_name: Feature name

        Returns:
            FeatureHealth
        """
        freshness = self.get_feature_freshness(feature_name)

        with self._lock:
            metadata = self._features.get(feature_name)
            stats = self._feature_stats.get(feature_name)

        issues: List[str] = []
        quality_score = 1.0

        # Check freshness
        if freshness.status == FreshnessStatus.EXPIRED:
            issues.append(f"Feature expired: {freshness.age_seconds:.0f}s old (SLA: {freshness.sla_seconds}s)")
            quality_score -= 0.3
        elif freshness.status == FreshnessStatus.STALE:
            issues.append(f"Feature stale: {freshness.age_seconds:.0f}s old (SLA: {freshness.sla_seconds}s)")
            quality_score -= 0.1

        # Check null rate
        null_rate = 0.0
        if stats:
            null_rate = stats.null_rate
            expected_null_rate = metadata.expected_null_rate if metadata else 0.0

            if null_rate > expected_null_rate + 0.1:
                issues.append(f"High null rate: {null_rate:.2%} (expected: {expected_null_rate:.2%})")
                quality_score -= 0.2

        # Check value ranges
        out_of_range_rate = 0.0
        if stats and metadata and stats.count > 0:
            if metadata.expected_min is not None and stats.min_value is not None:
                if stats.min_value < metadata.expected_min:
                    issues.append(f"Values below minimum: min={stats.min_value} (expected >= {metadata.expected_min})")
                    quality_score -= 0.1

            if metadata.expected_max is not None and stats.max_value is not None:
                if stats.max_value > metadata.expected_max:
                    issues.append(f"Values above maximum: max={stats.max_value} (expected <= {metadata.expected_max})")
                    quality_score -= 0.1

        # Determine status
        quality_score = max(0, min(1, quality_score))

        if quality_score >= 0.8 and freshness.status == FreshnessStatus.FRESH:
            status = HealthStatus.HEALTHY
        elif quality_score >= 0.5:
            status = HealthStatus.WARNING
        else:
            status = HealthStatus.CRITICAL

        health = FeatureHealth(
            feature_name=feature_name,
            status=status,
            null_rate=null_rate,
            out_of_range_rate=out_of_range_rate,
            freshness=freshness,
            quality_score=quality_score,
            issues=issues,
        )

        self.feature_quality_score.labels(
            store=self.store_name,
            feature=feature_name,
        ).set(quality_score)

        # Trigger callbacks for unhealthy features
        if status != HealthStatus.HEALTHY:
            for callback in self._callbacks:
                try:
                    callback(health)
                except Exception as e:
                    logger.error(f"Feature health callback error: {e}")

        return health

    def get_all_features_health(self) -> Dict[str, FeatureHealth]:
        """Get health status for all features.

        Returns:
            Dictionary mapping feature names to health
        """
        with self._lock:
            feature_names = list(self._features.keys())

        return {name: self.get_feature_health(name) for name in feature_names}

    def get_stale_features(self) -> List[FeatureFreshness]:
        """Get list of stale features.

        Returns:
            List of stale features
        """
        with self._lock:
            feature_names = list(self._features.keys())

        stale = []
        for name in feature_names:
            freshness = self.get_feature_freshness(name)
            if freshness.status in [FreshnessStatus.STALE, FreshnessStatus.EXPIRED]:
                stale.append(freshness)

        self.stale_features_gauge.labels(store=self.store_name).set(len(stale))

        return stale

    def get_serving_metrics(
        self,
        feature_name: str,
        since: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Get serving metrics for a feature.

        Args:
            feature_name: Feature name
            since: Only consider events after this time

        Returns:
            Serving metrics dictionary
        """
        with self._lock:
            history = list(self._serving_history.get(feature_name, []))

        if since:
            history = [h for h in history if h["timestamp"] >= since]

        if not history:
            return {
                "total_serves": 0,
                "success_rate": 0,
                "avg_latency": 0,
                "p50_latency": 0,
                "p95_latency": 0,
                "p99_latency": 0,
            }

        latencies = sorted([h["latency"] for h in history])
        success_count = sum(1 for h in history if h["success"])

        n = len(latencies)
        return {
            "total_serves": n,
            "success_rate": success_count / n,
            "avg_latency": sum(latencies) / n,
            "p50_latency": latencies[n // 2],
            "p95_latency": latencies[int(n * 0.95)] if n >= 20 else latencies[-1],
            "p99_latency": latencies[int(n * 0.99)] if n >= 100 else latencies[-1],
        }

    def on_health_issue(self, callback: Callable[[FeatureHealth], None]):
        """Register callback for health issues.

        Args:
            callback: Function to call on health issues
        """
        self._callbacks.append(callback)

    def get_feature_dependencies(
        self,
        feature_name: str,
    ) -> Dict[str, List[str]]:
        """Get feature dependencies.

        Args:
            feature_name: Feature name

        Returns:
            Dictionary with upstream and downstream dependencies
        """
        with self._lock:
            metadata = self._features.get(feature_name)
            all_features = dict(self._features)

        upstream = metadata.dependencies if metadata else []

        downstream = [
            name for name, meta in all_features.items()
            if feature_name in meta.dependencies
        ]

        return {
            "upstream": upstream,
            "downstream": downstream,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get feature store summary.

        Returns:
            Summary dictionary
        """
        all_health = self.get_all_features_health()

        healthy_count = sum(1 for h in all_health.values() if h.status == HealthStatus.HEALTHY)
        warning_count = sum(1 for h in all_health.values() if h.status == HealthStatus.WARNING)
        critical_count = sum(1 for h in all_health.values() if h.status == HealthStatus.CRITICAL)

        stale_features = self.get_stale_features()

        return {
            "store_name": self.store_name,
            "total_features": len(self._features),
            "healthy_features": healthy_count,
            "warning_features": warning_count,
            "critical_features": critical_count,
            "stale_features": len(stale_features),
            "avg_quality_score": (
                sum(h.quality_score for h in all_health.values()) / len(all_health)
                if all_health else 1.0
            ),
            "callbacks_registered": len(self._callbacks),
        }


class _ServingTracker:
    """Context manager for tracking feature serving."""

    def __init__(self, monitor: FeatureStoreMonitor, feature_name: str):
        self.monitor = monitor
        self.feature_name = feature_name
        self.start_time: float = 0
        self.success = True

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        latency = time.time() - self.start_time
        self.success = exc_type is None
        self.monitor.record_serving(
            self.feature_name,
            latency,
            success=self.success,
        )
        return False

    def set_error(self):
        """Mark the serving as failed."""
        self.success = False
