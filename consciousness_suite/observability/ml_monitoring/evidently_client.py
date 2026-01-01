"""Evidently AI Integration for ML Monitoring

Provides integration with Evidently AI for:
- Data quality reports
- Data drift reports
- Model performance reports
- Target drift analysis
"""

from __future__ import annotations

import time
import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)

# Check for evidently availability
try:
    import evidently
    from evidently.report import Report
    from evidently.metric_preset import (
        DataDriftPreset,
        DataQualityPreset,
        TargetDriftPreset,
    )
    from evidently.metrics import (
        ColumnDriftMetric,
        DatasetDriftMetric,
        DatasetMissingValuesMetric,
    )
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    evidently = None
    Report = None


class ReportType(str, Enum):
    """Types of Evidently reports."""
    DATA_DRIFT = "data_drift"
    DATA_QUALITY = "data_quality"
    TARGET_DRIFT = "target_drift"
    MODEL_PERFORMANCE = "model_performance"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CUSTOM = "custom"


@dataclass
class MetricResult:
    """Result from an Evidently metric."""
    metric_name: str
    value: Any
    drift_detected: Optional[bool] = None
    drift_score: Optional[float] = None
    threshold: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidentlyReport:
    """Container for an Evidently report result."""
    report_id: str
    report_type: ReportType
    model_id: str
    timestamp: datetime
    metrics: List[MetricResult]
    drift_detected: bool
    drift_share: float
    quality_score: float
    html_report: Optional[str] = None
    json_report: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class EvidentlyClient:
    """Client for Evidently AI ML monitoring integration.

    Usage:
        client = EvidentlyClient(model_id="my-model")

        # Generate drift report
        report = client.create_drift_report(
            reference_data=ref_df,
            current_data=cur_df,
        )

        if report.drift_detected:
            print(f"Drift share: {report.drift_share}")

        # Generate data quality report
        quality_report = client.create_quality_report(current_data=cur_df)
    """

    def __init__(
        self,
        model_id: str,
        namespace: str = "consciousness",
        drift_share_threshold: float = 0.5,
        stattest_threshold: float = 0.05,
    ):
        self.model_id = model_id
        self.namespace = namespace
        self.drift_share_threshold = drift_share_threshold
        self.stattest_threshold = stattest_threshold

        self._reports: List[EvidentlyReport] = []
        self._callbacks: List[Callable[[EvidentlyReport], None]] = []
        self._lock = threading.Lock()
        self._max_reports = 100
        self._report_counter = 0

        # Prometheus metrics
        self.report_generated = Counter(
            f"{namespace}_evidently_reports_total",
            "Total Evidently reports generated",
            ["model_id", "report_type"],
        )

        self.report_latency = Histogram(
            f"{namespace}_evidently_report_duration_seconds",
            "Time to generate Evidently report",
            ["model_id", "report_type"],
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
        )

        self.drift_share_gauge = Gauge(
            f"{namespace}_evidently_drift_share",
            "Share of drifted features",
            ["model_id"],
        )

        self.quality_score_gauge = Gauge(
            f"{namespace}_evidently_quality_score",
            "Data quality score",
            ["model_id"],
        )

        self.missing_values_gauge = Gauge(
            f"{namespace}_evidently_missing_values_share",
            "Share of missing values",
            ["model_id"],
        )

    def _generate_report_id(self) -> str:
        """Generate unique report ID."""
        with self._lock:
            self._report_counter += 1
            return f"{self.model_id}-{int(time.time())}-{self._report_counter}"

    def create_drift_report(
        self,
        reference_data: Any,
        current_data: Any,
        column_mapping: Optional[Dict[str, Any]] = None,
        include_html: bool = False,
    ) -> EvidentlyReport:
        """Create a data drift report.

        Args:
            reference_data: Reference DataFrame
            current_data: Current DataFrame to compare
            column_mapping: Evidently column mapping
            include_html: Include HTML report in result

        Returns:
            EvidentlyReport with drift analysis
        """
        start_time = time.time()

        if not EVIDENTLY_AVAILABLE:
            return self._create_fallback_report(
                ReportType.DATA_DRIFT,
                reference_data,
                current_data,
            )

        try:
            report = Report(metrics=[DataDriftPreset()])
            report.run(
                reference_data=reference_data,
                current_data=current_data,
                column_mapping=column_mapping,
            )

            json_result = report.as_dict()
            metrics = self._extract_drift_metrics(json_result)

            drift_detected = any(m.drift_detected for m in metrics if m.drift_detected is not None)
            drifted_count = sum(1 for m in metrics if m.drift_detected)
            drift_share = drifted_count / len(metrics) if metrics else 0

            html_report = None
            if include_html:
                html_report = report.get_html()

        except Exception as e:
            logger.error(f"Failed to create Evidently drift report: {e}")
            return self._create_fallback_report(
                ReportType.DATA_DRIFT,
                reference_data,
                current_data,
            )

        evidently_report = EvidentlyReport(
            report_id=self._generate_report_id(),
            report_type=ReportType.DATA_DRIFT,
            model_id=self.model_id,
            timestamp=datetime.now(),
            metrics=metrics,
            drift_detected=drift_share >= self.drift_share_threshold,
            drift_share=drift_share,
            quality_score=1.0 - drift_share,
            html_report=html_report,
            json_report=json_result,
        )

        self._store_report(evidently_report)
        self._update_metrics(evidently_report, time.time() - start_time)

        return evidently_report

    def create_quality_report(
        self,
        current_data: Any,
        reference_data: Optional[Any] = None,
        column_mapping: Optional[Dict[str, Any]] = None,
        include_html: bool = False,
    ) -> EvidentlyReport:
        """Create a data quality report.

        Args:
            current_data: Current DataFrame
            reference_data: Optional reference DataFrame
            column_mapping: Evidently column mapping
            include_html: Include HTML report in result

        Returns:
            EvidentlyReport with quality analysis
        """
        start_time = time.time()

        if not EVIDENTLY_AVAILABLE:
            return self._create_fallback_quality_report(current_data)

        try:
            report = Report(metrics=[DataQualityPreset()])
            report.run(
                reference_data=reference_data,
                current_data=current_data,
                column_mapping=column_mapping,
            )

            json_result = report.as_dict()
            metrics = self._extract_quality_metrics(json_result)

            # Calculate quality score
            missing_share = self._get_missing_share(json_result)
            quality_score = 1.0 - missing_share

            html_report = None
            if include_html:
                html_report = report.get_html()

        except Exception as e:
            logger.error(f"Failed to create Evidently quality report: {e}")
            return self._create_fallback_quality_report(current_data)

        evidently_report = EvidentlyReport(
            report_id=self._generate_report_id(),
            report_type=ReportType.DATA_QUALITY,
            model_id=self.model_id,
            timestamp=datetime.now(),
            metrics=metrics,
            drift_detected=False,
            drift_share=0.0,
            quality_score=quality_score,
            html_report=html_report,
            json_report=json_result,
        )

        self._store_report(evidently_report)
        self._update_metrics(evidently_report, time.time() - start_time)

        return evidently_report

    def create_target_drift_report(
        self,
        reference_data: Any,
        current_data: Any,
        column_mapping: Optional[Dict[str, Any]] = None,
        include_html: bool = False,
    ) -> EvidentlyReport:
        """Create a target drift report.

        Args:
            reference_data: Reference DataFrame
            current_data: Current DataFrame
            column_mapping: Evidently column mapping with target column
            include_html: Include HTML report in result

        Returns:
            EvidentlyReport with target drift analysis
        """
        start_time = time.time()

        if not EVIDENTLY_AVAILABLE:
            return self._create_fallback_report(
                ReportType.TARGET_DRIFT,
                reference_data,
                current_data,
            )

        try:
            report = Report(metrics=[TargetDriftPreset()])
            report.run(
                reference_data=reference_data,
                current_data=current_data,
                column_mapping=column_mapping,
            )

            json_result = report.as_dict()
            metrics = self._extract_target_metrics(json_result)

            drift_detected = any(m.drift_detected for m in metrics if m.drift_detected is not None)

            html_report = None
            if include_html:
                html_report = report.get_html()

        except Exception as e:
            logger.error(f"Failed to create Evidently target drift report: {e}")
            return self._create_fallback_report(
                ReportType.TARGET_DRIFT,
                reference_data,
                current_data,
            )

        evidently_report = EvidentlyReport(
            report_id=self._generate_report_id(),
            report_type=ReportType.TARGET_DRIFT,
            model_id=self.model_id,
            timestamp=datetime.now(),
            metrics=metrics,
            drift_detected=drift_detected,
            drift_share=1.0 if drift_detected else 0.0,
            quality_score=0.0 if drift_detected else 1.0,
            html_report=html_report,
            json_report=json_result,
        )

        self._store_report(evidently_report)
        self._update_metrics(evidently_report, time.time() - start_time)

        return evidently_report

    def create_custom_report(
        self,
        metrics: List[Any],
        reference_data: Any,
        current_data: Any,
        column_mapping: Optional[Dict[str, Any]] = None,
        include_html: bool = False,
    ) -> EvidentlyReport:
        """Create a custom report with specified metrics.

        Args:
            metrics: List of Evidently metrics
            reference_data: Reference DataFrame
            current_data: Current DataFrame
            column_mapping: Evidently column mapping
            include_html: Include HTML report in result

        Returns:
            EvidentlyReport with custom metrics
        """
        start_time = time.time()

        if not EVIDENTLY_AVAILABLE:
            return self._create_fallback_report(
                ReportType.CUSTOM,
                reference_data,
                current_data,
            )

        try:
            report = Report(metrics=metrics)
            report.run(
                reference_data=reference_data,
                current_data=current_data,
                column_mapping=column_mapping,
            )

            json_result = report.as_dict()
            metric_results = self._extract_custom_metrics(json_result)

            html_report = None
            if include_html:
                html_report = report.get_html()

        except Exception as e:
            logger.error(f"Failed to create custom Evidently report: {e}")
            return self._create_fallback_report(
                ReportType.CUSTOM,
                reference_data,
                current_data,
            )

        evidently_report = EvidentlyReport(
            report_id=self._generate_report_id(),
            report_type=ReportType.CUSTOM,
            model_id=self.model_id,
            timestamp=datetime.now(),
            metrics=metric_results,
            drift_detected=False,
            drift_share=0.0,
            quality_score=1.0,
            html_report=html_report,
            json_report=json_result,
        )

        self._store_report(evidently_report)
        self._update_metrics(evidently_report, time.time() - start_time)

        return evidently_report

    def _extract_drift_metrics(
        self,
        json_result: Dict[str, Any],
    ) -> List[MetricResult]:
        """Extract drift metrics from Evidently JSON result."""
        metrics = []

        try:
            for metric_data in json_result.get("metrics", []):
                metric_name = metric_data.get("metric", "unknown")
                result = metric_data.get("result", {})

                if "drift_by_columns" in result:
                    for col_name, col_data in result["drift_by_columns"].items():
                        metrics.append(MetricResult(
                            metric_name=f"column_drift_{col_name}",
                            value=col_data.get("drift_score", 0),
                            drift_detected=col_data.get("drift_detected", False),
                            drift_score=col_data.get("drift_score"),
                            threshold=col_data.get("threshold"),
                            details=col_data,
                        ))

                elif "drift_share" in result:
                    metrics.append(MetricResult(
                        metric_name="dataset_drift",
                        value=result.get("drift_share", 0),
                        drift_detected=result.get("dataset_drift", False),
                        drift_score=result.get("drift_share"),
                        details=result,
                    ))

        except Exception as e:
            logger.warning(f"Error extracting drift metrics: {e}")

        return metrics

    def _extract_quality_metrics(
        self,
        json_result: Dict[str, Any],
    ) -> List[MetricResult]:
        """Extract quality metrics from Evidently JSON result."""
        metrics = []

        try:
            for metric_data in json_result.get("metrics", []):
                metric_name = metric_data.get("metric", "unknown")
                result = metric_data.get("result", {})

                if "current" in result:
                    current = result["current"]
                    metrics.append(MetricResult(
                        metric_name=metric_name,
                        value=current,
                        details=result,
                    ))
                else:
                    metrics.append(MetricResult(
                        metric_name=metric_name,
                        value=result,
                        details=result,
                    ))

        except Exception as e:
            logger.warning(f"Error extracting quality metrics: {e}")

        return metrics

    def _extract_target_metrics(
        self,
        json_result: Dict[str, Any],
    ) -> List[MetricResult]:
        """Extract target drift metrics from Evidently JSON result."""
        metrics = []

        try:
            for metric_data in json_result.get("metrics", []):
                metric_name = metric_data.get("metric", "unknown")
                result = metric_data.get("result", {})

                drift_detected = result.get("drift_detected", False)
                drift_score = result.get("drift_score", result.get("stattest_threshold", 0))

                metrics.append(MetricResult(
                    metric_name=metric_name,
                    value=drift_score,
                    drift_detected=drift_detected,
                    drift_score=drift_score,
                    threshold=result.get("stattest_threshold"),
                    details=result,
                ))

        except Exception as e:
            logger.warning(f"Error extracting target metrics: {e}")

        return metrics

    def _extract_custom_metrics(
        self,
        json_result: Dict[str, Any],
    ) -> List[MetricResult]:
        """Extract custom metrics from Evidently JSON result."""
        metrics = []

        try:
            for metric_data in json_result.get("metrics", []):
                metric_name = metric_data.get("metric", "unknown")
                result = metric_data.get("result", {})

                metrics.append(MetricResult(
                    metric_name=metric_name,
                    value=result,
                    details=result,
                ))

        except Exception as e:
            logger.warning(f"Error extracting custom metrics: {e}")

        return metrics

    def _get_missing_share(self, json_result: Dict[str, Any]) -> float:
        """Get share of missing values from quality report."""
        try:
            for metric_data in json_result.get("metrics", []):
                result = metric_data.get("result", {})
                if "current" in result and "share_of_missing_values" in result["current"]:
                    return result["current"]["share_of_missing_values"]
        except Exception:
            pass
        return 0.0

    def _create_fallback_report(
        self,
        report_type: ReportType,
        reference_data: Any,
        current_data: Any,
    ) -> EvidentlyReport:
        """Create a fallback report when Evidently is not available."""
        logger.warning(
            "Evidently not available, creating fallback report. "
            "Install with: pip install evidently"
        )

        return EvidentlyReport(
            report_id=self._generate_report_id(),
            report_type=report_type,
            model_id=self.model_id,
            timestamp=datetime.now(),
            metrics=[
                MetricResult(
                    metric_name="fallback_notice",
                    value="Evidently not installed",
                    details={"error": "evidently package not available"},
                )
            ],
            drift_detected=False,
            drift_share=0.0,
            quality_score=1.0,
            metadata={"fallback": True},
        )

    def _create_fallback_quality_report(self, data: Any) -> EvidentlyReport:
        """Create a fallback quality report."""
        return self._create_fallback_report(ReportType.DATA_QUALITY, None, data)

    def _store_report(self, report: EvidentlyReport):
        """Store report in history."""
        with self._lock:
            self._reports.append(report)
            if len(self._reports) > self._max_reports:
                self._reports = self._reports[-self._max_reports // 2:]

        # Trigger callbacks if drift detected
        if report.drift_detected:
            for callback in self._callbacks:
                try:
                    callback(report)
                except Exception as e:
                    logger.error(f"Evidently callback error: {e}")

    def _update_metrics(self, report: EvidentlyReport, duration: float):
        """Update Prometheus metrics."""
        self.report_generated.labels(
            model_id=self.model_id,
            report_type=report.report_type.value,
        ).inc()

        self.report_latency.labels(
            model_id=self.model_id,
            report_type=report.report_type.value,
        ).observe(duration)

        self.drift_share_gauge.labels(model_id=self.model_id).set(report.drift_share)
        self.quality_score_gauge.labels(model_id=self.model_id).set(report.quality_score)

    def on_drift(self, callback: Callable[[EvidentlyReport], None]):
        """Register callback for drift detection.

        Args:
            callback: Function to call when drift is detected
        """
        self._callbacks.append(callback)

    def get_reports(
        self,
        report_type: Optional[ReportType] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[EvidentlyReport]:
        """Get report history.

        Args:
            report_type: Filter by report type
            since: Only reports after this time
            limit: Maximum reports to return

        Returns:
            List of reports
        """
        with self._lock:
            reports = list(self._reports)

        if report_type:
            reports = [r for r in reports if r.report_type == report_type]

        if since:
            reports = [r for r in reports if r.timestamp >= since]

        return reports[-limit:]

    def get_latest_report(
        self,
        report_type: Optional[ReportType] = None,
    ) -> Optional[EvidentlyReport]:
        """Get the most recent report.

        Args:
            report_type: Filter by report type

        Returns:
            Latest report or None
        """
        reports = self.get_reports(report_type=report_type, limit=1)
        return reports[0] if reports else None

    def get_summary(self) -> Dict[str, Any]:
        """Get Evidently monitoring summary.

        Returns:
            Summary dictionary
        """
        with self._lock:
            reports = list(self._reports)

        recent = [r for r in reports if r.timestamp >= datetime.now() - timedelta(hours=24)]
        drift_reports = [r for r in recent if r.drift_detected]

        return {
            "model_id": self.model_id,
            "evidently_available": EVIDENTLY_AVAILABLE,
            "total_reports": len(reports),
            "reports_24h": len(recent),
            "drift_detected_24h": len(drift_reports),
            "avg_drift_share": (
                sum(r.drift_share for r in recent) / len(recent)
                if recent else 0
            ),
            "avg_quality_score": (
                sum(r.quality_score for r in recent) / len(recent)
                if recent else 1.0
            ),
            "callbacks_registered": len(self._callbacks),
        }
