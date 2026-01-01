"""Data and Concept Drift Detection

Provides statistical methods for detecting drift:
- Kolmogorov-Smirnov (KS) test for distribution drift
- Population Stability Index (PSI) for feature stability
- Chi-square test for categorical features
- Jensen-Shannon divergence for probability distributions
"""

from __future__ import annotations

import time
import math
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)


class DriftType(str, Enum):
    """Types of drift that can be detected."""
    DATA_DRIFT = "data_drift"           # Input distribution change
    CONCEPT_DRIFT = "concept_drift"     # Relationship between features and target changes
    PREDICTION_DRIFT = "prediction_drift"  # Model output distribution change
    LABEL_DRIFT = "label_drift"         # Target variable distribution change
    FEATURE_DRIFT = "feature_drift"     # Individual feature distribution change


class DriftSeverity(str, Enum):
    """Severity levels for detected drift."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DriftResult:
    """Result of a drift detection test."""
    drift_detected: bool
    drift_type: DriftType
    severity: DriftSeverity
    test_statistic: float
    p_value: float
    threshold: float
    feature_name: Optional[str] = None
    reference_mean: Optional[float] = None
    current_mean: Optional[float] = None
    reference_std: Optional[float] = None
    current_std: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureDrift:
    """Drift information for a single feature."""
    feature_name: str
    drift_score: float
    drift_detected: bool
    severity: DriftSeverity
    test_method: str
    p_value: float
    reference_distribution: Dict[str, float] = field(default_factory=dict)
    current_distribution: Dict[str, float] = field(default_factory=dict)


@dataclass
class DatasetDrift:
    """Aggregated drift information for a dataset."""
    model_id: str
    timestamp: datetime
    overall_drift_detected: bool
    overall_severity: DriftSeverity
    drift_share: float  # Percentage of features with drift
    feature_drifts: List[FeatureDrift] = field(default_factory=list)
    dataset_drift_score: float = 0.0
    samples_analyzed: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class DriftDetector:
    """Detects data and concept drift using statistical methods.

    Usage:
        detector = DriftDetector(model_id="my-model")

        # Set reference distribution
        detector.set_reference(reference_data)

        # Check for drift
        result = detector.detect_drift(current_data)

        if result.overall_drift_detected:
            print(f"Drift detected with severity: {result.overall_severity}")

        # Register callback for drift alerts
        detector.on_drift(handle_drift_alert)
    """

    def __init__(
        self,
        model_id: str,
        namespace: str = "consciousness",
        drift_threshold: float = 0.05,
        psi_threshold: float = 0.25,
        feature_drift_share_threshold: float = 0.5,
    ):
        self.model_id = model_id
        self.namespace = namespace
        self.drift_threshold = drift_threshold  # p-value threshold
        self.psi_threshold = psi_threshold      # PSI threshold
        self.feature_drift_share_threshold = feature_drift_share_threshold

        self._reference_data: Dict[str, List[float]] = {}
        self._reference_stats: Dict[str, Dict[str, float]] = {}
        self._reference_bins: Dict[str, List[Tuple[float, float]]] = {}
        self._drift_history: List[DatasetDrift] = []
        self._callbacks: List[Callable[[DatasetDrift], None]] = []
        self._lock = threading.Lock()
        self._max_history = 1000

        # Prometheus metrics
        self.drift_detected_counter = Counter(
            f"{namespace}_ml_drift_detected_total",
            "Total drift detections",
            ["model_id", "drift_type", "severity"],
        )

        self.drift_score_gauge = Gauge(
            f"{namespace}_ml_drift_score",
            "Current drift score",
            ["model_id", "feature"],
        )

        self.psi_gauge = Gauge(
            f"{namespace}_ml_psi_score",
            "Population Stability Index",
            ["model_id", "feature"],
        )

        self.ks_statistic_gauge = Gauge(
            f"{namespace}_ml_ks_statistic",
            "KS test statistic",
            ["model_id", "feature"],
        )

        self.feature_drift_share = Gauge(
            f"{namespace}_ml_feature_drift_share",
            "Share of features with detected drift",
            ["model_id"],
        )

        self.drift_check_latency = Histogram(
            f"{namespace}_ml_drift_check_duration_seconds",
            "Time to perform drift check",
            ["model_id"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
        )

    def set_reference(
        self,
        reference_data: Dict[str, List[float]],
        num_bins: int = 10,
    ):
        """Set reference distribution for drift detection.

        Args:
            reference_data: Dictionary mapping feature names to values
            num_bins: Number of bins for PSI calculation
        """
        with self._lock:
            self._reference_data = reference_data
            self._reference_stats = {}
            self._reference_bins = {}

            for feature_name, values in reference_data.items():
                if not values:
                    continue

                # Calculate statistics
                mean = sum(values) / len(values)
                variance = sum((x - mean) ** 2 for x in values) / len(values)
                std = math.sqrt(variance) if variance > 0 else 0

                self._reference_stats[feature_name] = {
                    "mean": mean,
                    "std": std,
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }

                # Create bins for PSI
                min_val = min(values)
                max_val = max(values)
                if min_val != max_val:
                    bin_width = (max_val - min_val) / num_bins
                    bins = [
                        (min_val + i * bin_width, min_val + (i + 1) * bin_width)
                        for i in range(num_bins)
                    ]
                    self._reference_bins[feature_name] = bins

        logger.info(
            f"Set reference for model {self.model_id} with "
            f"{len(reference_data)} features"
        )

    def detect_drift(
        self,
        current_data: Dict[str, List[float]],
        test_methods: Optional[List[str]] = None,
    ) -> DatasetDrift:
        """Detect drift between reference and current data.

        Args:
            current_data: Current data to compare against reference
            test_methods: List of test methods to use (ks, psi, zscore)

        Returns:
            DatasetDrift with detailed results
        """
        start_time = time.time()
        test_methods = test_methods or ["ks", "psi"]

        feature_drifts: List[FeatureDrift] = []
        features_with_drift = 0
        total_features = 0

        with self._lock:
            reference_data = self._reference_data.copy()
            reference_stats = self._reference_stats.copy()
            reference_bins = self._reference_bins.copy()

        for feature_name, current_values in current_data.items():
            if feature_name not in reference_data:
                continue

            reference_values = reference_data[feature_name]
            if not current_values or not reference_values:
                continue

            total_features += 1

            # Run drift tests
            drift_scores: List[float] = []
            p_values: List[float] = []
            test_used = ""

            if "ks" in test_methods:
                ks_stat, ks_pvalue = self._ks_test(reference_values, current_values)
                drift_scores.append(ks_stat)
                p_values.append(ks_pvalue)
                test_used = "ks"

                self.ks_statistic_gauge.labels(
                    model_id=self.model_id,
                    feature=feature_name,
                ).set(ks_stat)

            if "psi" in test_methods and feature_name in reference_bins:
                psi_score = self._calculate_psi(
                    reference_values,
                    current_values,
                    reference_bins[feature_name],
                )
                drift_scores.append(psi_score)
                test_used = "psi" if not test_used else f"{test_used}+psi"

                self.psi_gauge.labels(
                    model_id=self.model_id,
                    feature=feature_name,
                ).set(psi_score)

            if "zscore" in test_methods and feature_name in reference_stats:
                zscore_drift = self._zscore_drift(
                    current_values,
                    reference_stats[feature_name],
                )
                drift_scores.append(zscore_drift)
                test_used = "zscore" if not test_used else f"{test_used}+zscore"

            # Determine drift
            max_score = max(drift_scores) if drift_scores else 0
            min_pvalue = min(p_values) if p_values else 1.0

            drift_detected = False
            if p_values and min_pvalue < self.drift_threshold:
                drift_detected = True
            if "psi" in test_methods and max_score > self.psi_threshold:
                drift_detected = True

            severity = self._calculate_severity(max_score, min_pvalue)

            if drift_detected:
                features_with_drift += 1

            feature_drift = FeatureDrift(
                feature_name=feature_name,
                drift_score=max_score,
                drift_detected=drift_detected,
                severity=severity,
                test_method=test_used,
                p_value=min_pvalue,
                reference_distribution=self._get_distribution_stats(reference_values),
                current_distribution=self._get_distribution_stats(current_values),
            )
            feature_drifts.append(feature_drift)

            self.drift_score_gauge.labels(
                model_id=self.model_id,
                feature=feature_name,
            ).set(max_score)

        # Calculate overall drift
        drift_share = features_with_drift / total_features if total_features > 0 else 0
        overall_drift_detected = drift_share >= self.feature_drift_share_threshold
        overall_severity = self._calculate_overall_severity(feature_drifts)
        dataset_drift_score = (
            sum(f.drift_score for f in feature_drifts) / len(feature_drifts)
            if feature_drifts else 0
        )

        self.feature_drift_share.labels(model_id=self.model_id).set(drift_share)

        dataset_drift = DatasetDrift(
            model_id=self.model_id,
            timestamp=datetime.now(),
            overall_drift_detected=overall_drift_detected,
            overall_severity=overall_severity,
            drift_share=drift_share,
            feature_drifts=feature_drifts,
            dataset_drift_score=dataset_drift_score,
            samples_analyzed=sum(len(v) for v in current_data.values()),
        )

        # Record history and metrics
        with self._lock:
            self._drift_history.append(dataset_drift)
            if len(self._drift_history) > self._max_history:
                self._drift_history = self._drift_history[-self._max_history // 2:]

        if overall_drift_detected:
            self.drift_detected_counter.labels(
                model_id=self.model_id,
                drift_type=DriftType.DATA_DRIFT.value,
                severity=overall_severity.value,
            ).inc()

            for callback in self._callbacks:
                try:
                    callback(dataset_drift)
                except Exception as e:
                    logger.error(f"Drift callback error: {e}")

        duration = time.time() - start_time
        self.drift_check_latency.labels(model_id=self.model_id).observe(duration)

        return dataset_drift

    def _ks_test(
        self,
        reference: List[float],
        current: List[float],
    ) -> Tuple[float, float]:
        """Perform Kolmogorov-Smirnov two-sample test.

        Args:
            reference: Reference sample
            current: Current sample

        Returns:
            (statistic, p_value)
        """
        # Sort both samples
        ref_sorted = sorted(reference)
        cur_sorted = sorted(current)
        n1 = len(ref_sorted)
        n2 = len(cur_sorted)

        if n1 == 0 or n2 == 0:
            return 0.0, 1.0

        # Compute empirical CDFs
        all_values = sorted(set(ref_sorted + cur_sorted))

        ref_idx = 0
        cur_idx = 0
        max_diff = 0.0

        for val in all_values:
            # Advance reference index
            while ref_idx < n1 and ref_sorted[ref_idx] <= val:
                ref_idx += 1
            # Advance current index
            while cur_idx < n2 and cur_sorted[cur_idx] <= val:
                cur_idx += 1

            cdf_ref = ref_idx / n1
            cdf_cur = cur_idx / n2
            diff = abs(cdf_ref - cdf_cur)
            max_diff = max(max_diff, diff)

        # Approximate p-value using asymptotic formula
        en = math.sqrt(n1 * n2 / (n1 + n2))
        lambda_val = (en + 0.12 + 0.11 / en) * max_diff

        # Approximation of Kolmogorov distribution
        if lambda_val < 0.001:
            p_value = 1.0
        elif lambda_val > 3:
            p_value = 0.0
        else:
            # Series approximation
            p_value = 2 * sum(
                ((-1) ** (i - 1)) * math.exp(-2 * (i ** 2) * (lambda_val ** 2))
                for i in range(1, 101)
            )
            p_value = max(0, min(1, p_value))

        return max_diff, p_value

    def _calculate_psi(
        self,
        reference: List[float],
        current: List[float],
        bins: List[Tuple[float, float]],
    ) -> float:
        """Calculate Population Stability Index.

        Args:
            reference: Reference sample
            current: Current sample
            bins: Bin edges

        Returns:
            PSI score (>0.25 indicates significant drift)
        """
        if not bins:
            return 0.0

        def get_bin_counts(values: List[float]) -> List[int]:
            counts = [0] * len(bins)
            for val in values:
                for i, (low, high) in enumerate(bins):
                    if low <= val < high or (i == len(bins) - 1 and val == high):
                        counts[i] += 1
                        break
            return counts

        ref_counts = get_bin_counts(reference)
        cur_counts = get_bin_counts(current)

        ref_total = sum(ref_counts)
        cur_total = sum(cur_counts)

        if ref_total == 0 or cur_total == 0:
            return 0.0

        psi = 0.0
        epsilon = 0.0001  # Avoid log(0)

        for ref_count, cur_count in zip(ref_counts, cur_counts):
            ref_pct = (ref_count / ref_total) + epsilon
            cur_pct = (cur_count / cur_total) + epsilon
            psi += (cur_pct - ref_pct) * math.log(cur_pct / ref_pct)

        return psi

    def _zscore_drift(
        self,
        current: List[float],
        reference_stats: Dict[str, float],
    ) -> float:
        """Detect drift using z-score of mean difference.

        Args:
            current: Current sample
            reference_stats: Reference statistics

        Returns:
            Absolute z-score
        """
        if not current:
            return 0.0

        current_mean = sum(current) / len(current)
        ref_mean = reference_stats.get("mean", 0)
        ref_std = reference_stats.get("std", 1)

        if ref_std == 0:
            return 0.0

        zscore = abs(current_mean - ref_mean) / ref_std
        return zscore

    def _calculate_severity(
        self,
        drift_score: float,
        p_value: float,
    ) -> DriftSeverity:
        """Calculate drift severity based on score and p-value."""
        if p_value >= self.drift_threshold and drift_score < self.psi_threshold:
            return DriftSeverity.NONE

        if drift_score < 0.1 or p_value > 0.01:
            return DriftSeverity.LOW
        elif drift_score < 0.25 or p_value > 0.001:
            return DriftSeverity.MEDIUM
        elif drift_score < 0.5:
            return DriftSeverity.HIGH
        else:
            return DriftSeverity.CRITICAL

    def _calculate_overall_severity(
        self,
        feature_drifts: List[FeatureDrift],
    ) -> DriftSeverity:
        """Calculate overall severity from feature drifts."""
        if not feature_drifts:
            return DriftSeverity.NONE

        severity_order = [
            DriftSeverity.NONE,
            DriftSeverity.LOW,
            DriftSeverity.MEDIUM,
            DriftSeverity.HIGH,
            DriftSeverity.CRITICAL,
        ]

        max_severity = DriftSeverity.NONE
        for fd in feature_drifts:
            if severity_order.index(fd.severity) > severity_order.index(max_severity):
                max_severity = fd.severity

        return max_severity

    def _get_distribution_stats(
        self,
        values: List[float],
    ) -> Dict[str, float]:
        """Get distribution statistics for a list of values."""
        if not values:
            return {}

        sorted_vals = sorted(values)
        n = len(sorted_vals)
        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / n

        return {
            "mean": mean,
            "std": math.sqrt(variance) if variance > 0 else 0,
            "min": sorted_vals[0],
            "max": sorted_vals[-1],
            "median": sorted_vals[n // 2],
            "p25": sorted_vals[n // 4] if n >= 4 else sorted_vals[0],
            "p75": sorted_vals[3 * n // 4] if n >= 4 else sorted_vals[-1],
            "count": n,
        }

    def detect_feature_drift(
        self,
        feature_name: str,
        current_values: List[float],
        test_method: str = "ks",
    ) -> DriftResult:
        """Detect drift for a single feature.

        Args:
            feature_name: Feature name
            current_values: Current feature values
            test_method: Test method (ks, psi, zscore)

        Returns:
            DriftResult for the feature
        """
        with self._lock:
            if feature_name not in self._reference_data:
                return DriftResult(
                    drift_detected=False,
                    drift_type=DriftType.FEATURE_DRIFT,
                    severity=DriftSeverity.NONE,
                    test_statistic=0.0,
                    p_value=1.0,
                    threshold=self.drift_threshold,
                    feature_name=feature_name,
                )

            reference_values = self._reference_data[feature_name]
            reference_stats = self._reference_stats.get(feature_name, {})
            reference_bins = self._reference_bins.get(feature_name, [])

        if test_method == "ks":
            statistic, p_value = self._ks_test(reference_values, current_values)
            drift_detected = p_value < self.drift_threshold
        elif test_method == "psi":
            statistic = self._calculate_psi(
                reference_values, current_values, reference_bins
            )
            p_value = 0.0 if statistic > self.psi_threshold else 1.0
            drift_detected = statistic > self.psi_threshold
        elif test_method == "zscore":
            statistic = self._zscore_drift(current_values, reference_stats)
            p_value = 0.0 if statistic > 3 else 1.0  # z > 3 is significant
            drift_detected = statistic > 3
        else:
            statistic, p_value = 0.0, 1.0
            drift_detected = False

        severity = self._calculate_severity(statistic, p_value)

        current_stats = self._get_distribution_stats(current_values)

        return DriftResult(
            drift_detected=drift_detected,
            drift_type=DriftType.FEATURE_DRIFT,
            severity=severity,
            test_statistic=statistic,
            p_value=p_value,
            threshold=self.drift_threshold,
            feature_name=feature_name,
            reference_mean=reference_stats.get("mean"),
            current_mean=current_stats.get("mean"),
            reference_std=reference_stats.get("std"),
            current_std=current_stats.get("std"),
        )

    def detect_prediction_drift(
        self,
        reference_predictions: List[float],
        current_predictions: List[float],
    ) -> DriftResult:
        """Detect drift in model predictions.

        Args:
            reference_predictions: Reference prediction values
            current_predictions: Current prediction values

        Returns:
            DriftResult for predictions
        """
        statistic, p_value = self._ks_test(
            reference_predictions, current_predictions
        )
        drift_detected = p_value < self.drift_threshold
        severity = self._calculate_severity(statistic, p_value)

        ref_stats = self._get_distribution_stats(reference_predictions)
        cur_stats = self._get_distribution_stats(current_predictions)

        return DriftResult(
            drift_detected=drift_detected,
            drift_type=DriftType.PREDICTION_DRIFT,
            severity=severity,
            test_statistic=statistic,
            p_value=p_value,
            threshold=self.drift_threshold,
            feature_name="predictions",
            reference_mean=ref_stats.get("mean"),
            current_mean=cur_stats.get("mean"),
            reference_std=ref_stats.get("std"),
            current_std=cur_stats.get("std"),
        )

    def on_drift(self, callback: Callable[[DatasetDrift], None]):
        """Register a callback for drift detection.

        Args:
            callback: Function to call when drift is detected
        """
        self._callbacks.append(callback)

    def get_drift_history(
        self,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[DatasetDrift]:
        """Get drift detection history.

        Args:
            since: Only return results after this time
            limit: Maximum results to return

        Returns:
            List of DatasetDrift records
        """
        with self._lock:
            history = list(self._drift_history)

        if since:
            history = [d for d in history if d.timestamp >= since]

        return history[-limit:]

    def get_summary(self) -> Dict[str, Any]:
        """Get drift detection summary.

        Returns:
            Summary dictionary
        """
        with self._lock:
            history = list(self._drift_history)
            feature_count = len(self._reference_data)

        recent = [d for d in history if d.timestamp >= datetime.now() - timedelta(hours=24)]
        drift_events = sum(1 for d in recent if d.overall_drift_detected)

        return {
            "model_id": self.model_id,
            "reference_features": feature_count,
            "total_checks": len(history),
            "recent_checks_24h": len(recent),
            "drift_events_24h": drift_events,
            "drift_rate_24h": drift_events / len(recent) if recent else 0,
            "avg_drift_score": (
                sum(d.dataset_drift_score for d in recent) / len(recent)
                if recent else 0
            ),
            "callbacks_registered": len(self._callbacks),
        }

    def reset(self):
        """Reset detector state."""
        with self._lock:
            self._reference_data.clear()
            self._reference_stats.clear()
            self._reference_bins.clear()
            self._drift_history.clear()
