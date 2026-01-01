"""Prediction Logging and Analysis

Provides comprehensive prediction tracking:
- Prediction logging with metadata
- Ground truth reconciliation
- Prediction distribution analysis
- Performance monitoring
"""

from __future__ import annotations

import time
import math
import hashlib
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from prometheus_client import Counter, Gauge, Histogram, Summary

logger = logging.getLogger(__name__)


class PredictionType(str, Enum):
    """Types of predictions."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    RANKING = "ranking"
    PROBABILITY = "probability"
    EMBEDDING = "embedding"
    MULTI_LABEL = "multi_label"


@dataclass
class PredictionRecord:
    """Record of a single prediction."""
    prediction_id: str
    model_id: str
    model_version: str
    prediction_type: PredictionType
    prediction: Any
    probability: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None
    features: Optional[Dict[str, Any]] = None
    ground_truth: Optional[Any] = None
    timestamp: datetime = field(default_factory=datetime.now)
    latency_ms: float = 0.0
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_correct(self) -> Optional[bool]:
        """Check if prediction matches ground truth."""
        if self.ground_truth is None:
            return None
        return self.prediction == self.ground_truth


@dataclass
class PredictionDistribution:
    """Distribution of predictions."""
    model_id: str
    prediction_type: PredictionType
    period_start: datetime
    period_end: datetime
    total_predictions: int
    class_distribution: Dict[str, int] = field(default_factory=dict)
    probability_histogram: List[int] = field(default_factory=list)
    value_stats: Dict[str, float] = field(default_factory=dict)
    top_predictions: List[Tuple[Any, int]] = field(default_factory=list)


@dataclass
class PredictionAnalytics:
    """Analytics for predictions over a time period."""
    model_id: str
    model_version: str
    period_start: datetime
    period_end: datetime
    total_predictions: int
    predictions_with_ground_truth: int
    accuracy: Optional[float] = None
    precision_by_class: Dict[str, float] = field(default_factory=dict)
    recall_by_class: Dict[str, float] = field(default_factory=dict)
    confusion_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    prediction_distribution: PredictionDistribution = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PredictionLogger:
    """Logger for model predictions with analysis capabilities.

    Usage:
        logger = PredictionLogger(model_id="fraud-detector")

        # Log prediction
        record = logger.log_prediction(
            prediction="fraud",
            probability=0.92,
            features={"amount": 1500, "country": "US"},
            latency_ms=12.5,
        )

        # Add ground truth later
        logger.add_ground_truth(record.prediction_id, "fraud")

        # Get analytics
        analytics = logger.get_analytics(since=datetime.now() - timedelta(hours=1))

        # Get distribution
        dist = logger.get_prediction_distribution()
    """

    def __init__(
        self,
        model_id: str,
        model_version: str = "unknown",
        namespace: str = "consciousness",
        max_history: int = 100000,
        prediction_type: PredictionType = PredictionType.CLASSIFICATION,
    ):
        self.model_id = model_id
        self.model_version = model_version
        self.namespace = namespace
        self.prediction_type = prediction_type
        self._max_history = max_history

        self._predictions: List[PredictionRecord] = []
        self._predictions_by_id: Dict[str, PredictionRecord] = {}
        self._ground_truth_pending: Dict[str, Any] = {}
        self._callbacks: List[Callable[[PredictionRecord], None]] = []
        self._lock = threading.Lock()
        self._prediction_counter = 0

        # Aggregated counters
        self._class_counts: Dict[str, int] = defaultdict(int)
        self._correct_counts: Dict[str, int] = defaultdict(int)
        self._total_latency = 0.0
        self._latencies: List[float] = []

        # Prometheus metrics
        self.predictions_total = Counter(
            f"{namespace}_predictions_total",
            "Total predictions made",
            ["model_id", "model_version", "prediction"],
        )

        self.predictions_correct = Counter(
            f"{namespace}_predictions_correct_total",
            "Correct predictions",
            ["model_id", "model_version", "prediction"],
        )

        self.prediction_latency = Histogram(
            f"{namespace}_prediction_latency_ms",
            "Prediction latency in milliseconds",
            ["model_id", "model_version"],
            buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000],
        )

        self.prediction_probability = Histogram(
            f"{namespace}_prediction_probability",
            "Prediction probability distribution",
            ["model_id", "model_version"],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
        )

        self.accuracy_gauge = Gauge(
            f"{namespace}_prediction_accuracy",
            "Rolling prediction accuracy",
            ["model_id", "model_version"],
        )

        self.ground_truth_backlog = Gauge(
            f"{namespace}_prediction_ground_truth_pending",
            "Predictions awaiting ground truth",
            ["model_id"],
        )

        self.predictions_per_second = Gauge(
            f"{namespace}_predictions_per_second",
            "Predictions per second (1min avg)",
            ["model_id"],
        )

    def _generate_prediction_id(self) -> str:
        """Generate unique prediction ID."""
        with self._lock:
            self._prediction_counter += 1
            content = f"{self.model_id}:{time.time()}:{self._prediction_counter}"
            return hashlib.sha256(content.encode()).hexdigest()[:16]

    def log_prediction(
        self,
        prediction: Any,
        probability: Optional[float] = None,
        probabilities: Optional[Dict[str, float]] = None,
        features: Optional[Dict[str, Any]] = None,
        ground_truth: Optional[Any] = None,
        latency_ms: float = 0.0,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PredictionRecord:
        """Log a prediction.

        Args:
            prediction: The prediction value
            probability: Confidence probability
            probabilities: Class probabilities
            features: Input features
            ground_truth: Ground truth (if known immediately)
            latency_ms: Prediction latency
            request_id: Request identifier
            user_id: User identifier
            metadata: Additional metadata

        Returns:
            PredictionRecord
        """
        prediction_id = self._generate_prediction_id()

        record = PredictionRecord(
            prediction_id=prediction_id,
            model_id=self.model_id,
            model_version=self.model_version,
            prediction_type=self.prediction_type,
            prediction=prediction,
            probability=probability,
            probabilities=probabilities,
            features=features,
            ground_truth=ground_truth,
            latency_ms=latency_ms,
            request_id=request_id,
            user_id=user_id,
            metadata=metadata or {},
        )

        with self._lock:
            self._predictions.append(record)
            self._predictions_by_id[prediction_id] = record

            # Trim history
            if len(self._predictions) > self._max_history:
                removed = self._predictions[:-self._max_history // 2]
                self._predictions = self._predictions[-self._max_history // 2:]

                for r in removed:
                    self._predictions_by_id.pop(r.prediction_id, None)

            # Update aggregates
            pred_str = str(prediction)
            self._class_counts[pred_str] += 1
            self._total_latency += latency_ms
            self._latencies.append(latency_ms)

            if len(self._latencies) > 10000:
                self._latencies = self._latencies[-5000:]

            if ground_truth is not None and prediction == ground_truth:
                self._correct_counts[pred_str] += 1

        # Update Prometheus metrics
        self.predictions_total.labels(
            model_id=self.model_id,
            model_version=self.model_version,
            prediction=pred_str[:50],  # Limit label length
        ).inc()

        self.prediction_latency.labels(
            model_id=self.model_id,
            model_version=self.model_version,
        ).observe(latency_ms)

        if probability is not None:
            self.prediction_probability.labels(
                model_id=self.model_id,
                model_version=self.model_version,
            ).observe(probability)

        if ground_truth is not None and prediction == ground_truth:
            self.predictions_correct.labels(
                model_id=self.model_id,
                model_version=self.model_version,
                prediction=pred_str[:50],
            ).inc()

        # Update accuracy gauge
        self._update_accuracy_gauge()

        # Trigger callbacks
        for callback in self._callbacks:
            try:
                callback(record)
            except Exception as e:
                logger.error(f"Prediction callback error: {e}")

        return record

    def add_ground_truth(
        self,
        prediction_id: str,
        ground_truth: Any,
    ) -> bool:
        """Add ground truth to a prediction.

        Args:
            prediction_id: Prediction ID
            ground_truth: Ground truth value

        Returns:
            True if prediction found and updated
        """
        with self._lock:
            record = self._predictions_by_id.get(prediction_id)

            if not record:
                # Store for later if prediction not yet logged
                self._ground_truth_pending[prediction_id] = ground_truth
                return False

            record.ground_truth = ground_truth

            if record.prediction == ground_truth:
                pred_str = str(record.prediction)
                self._correct_counts[pred_str] += 1

                self.predictions_correct.labels(
                    model_id=self.model_id,
                    model_version=self.model_version,
                    prediction=pred_str[:50],
                ).inc()

        self._update_accuracy_gauge()
        self.ground_truth_backlog.labels(model_id=self.model_id).set(
            len(self._ground_truth_pending)
        )

        return True

    def add_ground_truth_batch(
        self,
        ground_truths: Dict[str, Any],
    ) -> int:
        """Add ground truth for multiple predictions.

        Args:
            ground_truths: Mapping of prediction_id to ground_truth

        Returns:
            Number of predictions updated
        """
        updated = 0
        for prediction_id, ground_truth in ground_truths.items():
            if self.add_ground_truth(prediction_id, ground_truth):
                updated += 1
        return updated

    def _update_accuracy_gauge(self):
        """Update the accuracy gauge."""
        with self._lock:
            total_correct = sum(self._correct_counts.values())
            total = sum(self._class_counts.values())

        if total > 0:
            accuracy = total_correct / total
            self.accuracy_gauge.labels(
                model_id=self.model_id,
                model_version=self.model_version,
            ).set(accuracy)

    def get_prediction(self, prediction_id: str) -> Optional[PredictionRecord]:
        """Get a prediction by ID.

        Args:
            prediction_id: Prediction ID

        Returns:
            PredictionRecord or None
        """
        with self._lock:
            return self._predictions_by_id.get(prediction_id)

    def get_recent_predictions(
        self,
        limit: int = 100,
        prediction_type: Optional[Any] = None,
        with_ground_truth: Optional[bool] = None,
    ) -> List[PredictionRecord]:
        """Get recent predictions.

        Args:
            limit: Maximum predictions to return
            prediction_type: Filter by prediction value
            with_ground_truth: Filter by ground truth presence

        Returns:
            List of predictions
        """
        with self._lock:
            predictions = list(self._predictions)

        if prediction_type is not None:
            predictions = [p for p in predictions if p.prediction == prediction_type]

        if with_ground_truth is not None:
            predictions = [
                p for p in predictions
                if (p.ground_truth is not None) == with_ground_truth
            ]

        return predictions[-limit:]

    def get_prediction_distribution(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> PredictionDistribution:
        """Get prediction distribution.

        Args:
            since: Start time
            until: End time

        Returns:
            PredictionDistribution
        """
        now = datetime.now()
        since = since or now - timedelta(hours=1)
        until = until or now

        with self._lock:
            predictions = [
                p for p in self._predictions
                if since <= p.timestamp <= until
            ]

        if not predictions:
            return PredictionDistribution(
                model_id=self.model_id,
                prediction_type=self.prediction_type,
                period_start=since,
                period_end=until,
                total_predictions=0,
            )

        # Count class distribution
        class_counts: Dict[str, int] = defaultdict(int)
        probabilities: List[float] = []

        for p in predictions:
            class_counts[str(p.prediction)] += 1
            if p.probability is not None:
                probabilities.append(p.probability)

        # Create probability histogram
        prob_hist = [0] * 10
        for prob in probabilities:
            bin_idx = min(9, int(prob * 10))
            prob_hist[bin_idx] += 1

        # Get value stats for regression
        value_stats = {}
        if self.prediction_type == PredictionType.REGRESSION:
            values = [float(p.prediction) for p in predictions if self._is_numeric(p.prediction)]
            if values:
                value_stats = {
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values),
                    "median": sorted(values)[len(values) // 2],
                }

        # Top predictions
        top_predictions = sorted(
            class_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]

        return PredictionDistribution(
            model_id=self.model_id,
            prediction_type=self.prediction_type,
            period_start=since,
            period_end=until,
            total_predictions=len(predictions),
            class_distribution=dict(class_counts),
            probability_histogram=prob_hist,
            value_stats=value_stats,
            top_predictions=top_predictions,
        )

    def _is_numeric(self, value: Any) -> bool:
        """Check if value is numeric."""
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False

    def get_analytics(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> PredictionAnalytics:
        """Get prediction analytics.

        Args:
            since: Start time
            until: End time

        Returns:
            PredictionAnalytics
        """
        now = datetime.now()
        since = since or now - timedelta(hours=1)
        until = until or now

        with self._lock:
            predictions = [
                p for p in self._predictions
                if since <= p.timestamp <= until
            ]

        if not predictions:
            return PredictionAnalytics(
                model_id=self.model_id,
                model_version=self.model_version,
                period_start=since,
                period_end=until,
                total_predictions=0,
                predictions_with_ground_truth=0,
            )

        # Basic counts
        with_ground_truth = [p for p in predictions if p.ground_truth is not None]

        # Calculate accuracy
        accuracy = None
        if with_ground_truth:
            correct = sum(1 for p in with_ground_truth if p.is_correct)
            accuracy = correct / len(with_ground_truth)

        # Build confusion matrix
        confusion_matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for p in with_ground_truth:
            pred_str = str(p.prediction)
            truth_str = str(p.ground_truth)
            confusion_matrix[truth_str][pred_str] += 1

        # Calculate precision/recall per class
        precision_by_class: Dict[str, float] = {}
        recall_by_class: Dict[str, float] = {}

        all_classes = set(str(p.prediction) for p in with_ground_truth)
        all_classes.update(str(p.ground_truth) for p in with_ground_truth)

        for cls in all_classes:
            # Precision: TP / (TP + FP)
            tp = confusion_matrix[cls][cls]
            fp = sum(confusion_matrix[actual][cls] for actual in all_classes if actual != cls)

            if tp + fp > 0:
                precision_by_class[cls] = tp / (tp + fp)

            # Recall: TP / (TP + FN)
            fn = sum(confusion_matrix[cls][pred] for pred in all_classes if pred != cls)

            if tp + fn > 0:
                recall_by_class[cls] = tp / (tp + fn)

        # Latency stats
        latencies = sorted([p.latency_ms for p in predictions])
        n = len(latencies)

        avg_latency = sum(latencies) / n
        p50_latency = latencies[n // 2]
        p95_latency = latencies[int(n * 0.95)] if n >= 20 else latencies[-1]
        p99_latency = latencies[int(n * 0.99)] if n >= 100 else latencies[-1]

        # Get distribution
        distribution = self.get_prediction_distribution(since, until)

        return PredictionAnalytics(
            model_id=self.model_id,
            model_version=self.model_version,
            period_start=since,
            period_end=until,
            total_predictions=len(predictions),
            predictions_with_ground_truth=len(with_ground_truth),
            accuracy=accuracy,
            precision_by_class=precision_by_class,
            recall_by_class=recall_by_class,
            confusion_matrix={k: dict(v) for k, v in confusion_matrix.items()},
            avg_latency_ms=avg_latency,
            p50_latency_ms=p50_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            prediction_distribution=distribution,
        )

    def get_predictions_by_feature(
        self,
        feature_name: str,
        feature_value: Any,
        limit: int = 100,
    ) -> List[PredictionRecord]:
        """Get predictions with specific feature value.

        Args:
            feature_name: Feature name
            feature_value: Feature value to match
            limit: Maximum results

        Returns:
            Matching predictions
        """
        with self._lock:
            predictions = [
                p for p in self._predictions
                if p.features and p.features.get(feature_name) == feature_value
            ]

        return predictions[-limit:]

    def get_incorrect_predictions(
        self,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[PredictionRecord]:
        """Get incorrect predictions for analysis.

        Args:
            since: Start time
            limit: Maximum results

        Returns:
            Incorrect predictions
        """
        since = since or datetime.min

        with self._lock:
            incorrect = [
                p for p in self._predictions
                if p.timestamp >= since
                and p.ground_truth is not None
                and p.prediction != p.ground_truth
            ]

        return incorrect[-limit:]

    def get_low_confidence_predictions(
        self,
        threshold: float = 0.5,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[PredictionRecord]:
        """Get low confidence predictions.

        Args:
            threshold: Probability threshold
            since: Start time
            limit: Maximum results

        Returns:
            Low confidence predictions
        """
        since = since or datetime.min

        with self._lock:
            low_conf = [
                p for p in self._predictions
                if p.timestamp >= since
                and p.probability is not None
                and p.probability < threshold
            ]

        return low_conf[-limit:]

    def on_prediction(self, callback: Callable[[PredictionRecord], None]):
        """Register callback for new predictions.

        Args:
            callback: Function to call on new prediction
        """
        self._callbacks.append(callback)

    def get_throughput(
        self,
        window_seconds: int = 60,
    ) -> float:
        """Get current throughput (predictions per second).

        Args:
            window_seconds: Time window in seconds

        Returns:
            Predictions per second
        """
        now = datetime.now()
        window_start = now - timedelta(seconds=window_seconds)

        with self._lock:
            count = sum(1 for p in self._predictions if p.timestamp >= window_start)

        throughput = count / window_seconds
        self.predictions_per_second.labels(model_id=self.model_id).set(throughput)

        return throughput

    def get_summary(self) -> Dict[str, Any]:
        """Get prediction logger summary.

        Returns:
            Summary dictionary
        """
        with self._lock:
            total = len(self._predictions)
            with_ground_truth = sum(1 for p in self._predictions if p.ground_truth is not None)
            total_correct = sum(self._correct_counts.values())
            latencies = list(self._latencies)

        accuracy = total_correct / with_ground_truth if with_ground_truth > 0 else None

        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "prediction_type": self.prediction_type.value,
            "total_predictions": total,
            "predictions_with_ground_truth": with_ground_truth,
            "accuracy": accuracy,
            "class_distribution": dict(self._class_counts),
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
            "ground_truth_pending": len(self._ground_truth_pending),
            "callbacks_registered": len(self._callbacks),
            "throughput_1min": self.get_throughput(60),
        }

    def reset(self):
        """Reset all prediction data."""
        with self._lock:
            self._predictions.clear()
            self._predictions_by_id.clear()
            self._ground_truth_pending.clear()
            self._class_counts.clear()
            self._correct_counts.clear()
            self._total_latency = 0.0
            self._latencies.clear()
