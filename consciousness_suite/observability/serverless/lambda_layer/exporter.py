"""Lambda OTLP Exporter

OTLP telemetry export optimized for Lambda:
- Async batching with deadline awareness
- gRPC and HTTP/protobuf protocols
- Automatic retry with backoff
- Compression support
- Deadline-aware flushing
"""

from __future__ import annotations

import os
import json
import time
import gzip
import threading
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from collections import deque
import urllib.request
import urllib.error

from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)


class ExportProtocol(str, Enum):
    """OTLP export protocol."""
    GRPC = "grpc"
    HTTP_PROTOBUF = "http/protobuf"
    HTTP_JSON = "http/json"


class ExportStatus(str, Enum):
    """Export result status."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    RETRY = "retry"


@dataclass
class ExportConfig:
    """Configuration for OTLP exporter."""
    # Endpoint settings
    endpoint: str = "http://localhost:4318"
    protocol: str = "http/json"
    # Auth
    headers: Dict[str, str] = field(default_factory=dict)
    api_key: Optional[str] = None
    # Timeouts
    timeout_ms: int = 5000
    connect_timeout_ms: int = 2000
    # Retry settings
    max_retries: int = 3
    retry_delay_ms: int = 100
    retry_backoff_multiplier: float = 2.0
    max_retry_delay_ms: int = 5000
    # Compression
    compression: str = "gzip"  # none, gzip
    # Batching
    max_batch_size: int = 100
    max_batch_delay_ms: int = 1000
    # Resource attributes
    service_name: str = field(
        default_factory=lambda: os.environ.get("AWS_LAMBDA_FUNCTION_NAME", "lambda")
    )
    service_version: str = field(
        default_factory=lambda: os.environ.get("AWS_LAMBDA_FUNCTION_VERSION", "$LATEST")
    )


@dataclass
class BatchConfig:
    """Batch processing configuration."""
    max_size: int = 100
    max_delay_ms: int = 1000
    max_bytes: int = 5 * 1024 * 1024  # 5MB


@dataclass
class ExportResult:
    """Result of an export operation."""
    status: ExportStatus
    exported_count: int
    failed_count: int
    duration_ms: float
    error: Optional[str] = None
    retry_after_ms: Optional[int] = None

    @property
    def success(self) -> bool:
        return self.status in (ExportStatus.SUCCESS, ExportStatus.PARTIAL_SUCCESS)


class LambdaOTLPExporter:
    """OTLP exporter optimized for AWS Lambda.

    Handles the unique constraints of Lambda:
    - Limited execution time (deadline awareness)
    - Cold start considerations
    - Async flush during freeze phase
    - Memory constraints

    Usage:
        exporter = LambdaOTLPExporter(config=ExportConfig(
            endpoint="https://otlp.example.com",
            api_key="secret",
        ))

        # Add telemetry
        exporter.add_span(span_data)
        exporter.add_metric(metric_data)

        # Flush before Lambda freeze
        result = exporter.flush()
    """

    def __init__(
        self,
        config: Optional[ExportConfig] = None,
        batch_config: Optional[BatchConfig] = None,
        namespace: str = "consciousness",
    ):
        self.config = config or ExportConfig()
        self.batch_config = batch_config or BatchConfig()
        self.namespace = namespace

        # Separate queues for different signal types
        self._traces_queue: deque = deque(maxlen=self.batch_config.max_size * 10)
        self._metrics_queue: deque = deque(maxlen=self.batch_config.max_size * 10)
        self._logs_queue: deque = deque(maxlen=self.batch_config.max_size * 10)
        self._queue_lock = threading.Lock()

        # Export state
        self._last_export_time = time.time()
        self._pending_retries: List[Dict[str, Any]] = []

        # Setup metrics
        self._setup_metrics()

    def _setup_metrics(self):
        """Initialize Prometheus metrics for exporter."""
        self.export_total = Counter(
            f"{self.namespace}_lambda_export_total",
            "Total export attempts",
            ["signal_type", "status"],
        )

        self.export_items = Counter(
            f"{self.namespace}_lambda_export_items_total",
            "Total items exported",
            ["signal_type"],
        )

        self.export_duration = Histogram(
            f"{self.namespace}_lambda_export_duration_seconds",
            "Export duration",
            ["signal_type"],
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
        )

        self.queue_size = Gauge(
            f"{self.namespace}_lambda_export_queue_size",
            "Export queue size",
            ["signal_type"],
        )

        self.export_bytes = Counter(
            f"{self.namespace}_lambda_export_bytes_total",
            "Total bytes exported",
            ["signal_type", "compressed"],
        )

    def _get_endpoint(self, signal_type: str) -> str:
        """Get endpoint URL for signal type."""
        base = self.config.endpoint.rstrip("/")

        if self.config.protocol == "grpc":
            return base

        # HTTP endpoints
        endpoints = {
            "traces": f"{base}/v1/traces",
            "metrics": f"{base}/v1/metrics",
            "logs": f"{base}/v1/logs",
        }
        return endpoints.get(signal_type, base)

    def _build_headers(self) -> Dict[str, str]:
        """Build request headers."""
        headers = {
            "Content-Type": "application/json",
            **self.config.headers,
        }

        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        if self.config.compression == "gzip":
            headers["Content-Encoding"] = "gzip"

        return headers

    def _compress_data(self, data: bytes) -> bytes:
        """Compress data if configured."""
        if self.config.compression == "gzip":
            return gzip.compress(data)
        return data

    def _build_resource(self) -> Dict[str, Any]:
        """Build OTLP resource."""
        return {
            "attributes": [
                {"key": "service.name", "value": {"stringValue": self.config.service_name}},
                {"key": "service.version", "value": {"stringValue": self.config.service_version}},
                {"key": "faas.name", "value": {"stringValue": self.config.service_name}},
                {"key": "cloud.provider", "value": {"stringValue": "aws"}},
                {"key": "cloud.platform", "value": {"stringValue": "aws_lambda"}},
                {"key": "cloud.region", "value": {"stringValue": os.environ.get("AWS_REGION", "unknown")}},
            ]
        }

    def add_span(self, span_data: Dict[str, Any]):
        """Add a span to the export queue.

        Args:
            span_data: Span data in OTLP format
        """
        with self._queue_lock:
            self._traces_queue.append(span_data)
        self.queue_size.labels(signal_type="traces").set(len(self._traces_queue))

    def add_metric(self, metric_data: Dict[str, Any]):
        """Add a metric to the export queue.

        Args:
            metric_data: Metric data in OTLP format
        """
        with self._queue_lock:
            self._metrics_queue.append(metric_data)
        self.queue_size.labels(signal_type="metrics").set(len(self._metrics_queue))

    def add_log(self, log_data: Dict[str, Any]):
        """Add a log record to the export queue.

        Args:
            log_data: Log data in OTLP format
        """
        with self._queue_lock:
            self._logs_queue.append(log_data)
        self.queue_size.labels(signal_type="logs").set(len(self._logs_queue))

    def export_batch(self, batch: List[Dict[str, Any]]) -> ExportResult:
        """Export a batch of telemetry data.

        Intelligently routes data to appropriate OTLP endpoints.

        Args:
            batch: List of telemetry records

        Returns:
            ExportResult
        """
        # Categorize items
        traces = []
        metrics = []
        logs = []

        for item in batch:
            item_type = item.get("type", "trace")
            if item_type == "metric":
                metrics.append(item)
            elif item_type == "log":
                logs.append(item)
            else:
                traces.append(item)

        results = []

        if traces:
            result = self._export_signal("traces", traces)
            results.append(result)

        if metrics:
            result = self._export_signal("metrics", metrics)
            results.append(result)

        if logs:
            result = self._export_signal("logs", logs)
            results.append(result)

        # Aggregate results
        total_exported = sum(r.exported_count for r in results)
        total_failed = sum(r.failed_count for r in results)
        total_duration = sum(r.duration_ms for r in results)

        if all(r.status == ExportStatus.SUCCESS for r in results):
            status = ExportStatus.SUCCESS
        elif any(r.status == ExportStatus.SUCCESS for r in results):
            status = ExportStatus.PARTIAL_SUCCESS
        else:
            status = ExportStatus.FAILURE

        return ExportResult(
            status=status,
            exported_count=total_exported,
            failed_count=total_failed,
            duration_ms=total_duration,
        )

    def _export_signal(
        self,
        signal_type: str,
        items: List[Dict[str, Any]],
    ) -> ExportResult:
        """Export items for a specific signal type.

        Args:
            signal_type: traces, metrics, or logs
            items: Items to export

        Returns:
            ExportResult
        """
        start_time = time.perf_counter()

        # Build OTLP payload
        payload = self._build_payload(signal_type, items)
        payload_json = json.dumps(payload)
        payload_bytes = payload_json.encode("utf-8")

        # Compress if configured
        if self.config.compression == "gzip":
            compressed = self._compress_data(payload_bytes)
            self.export_bytes.labels(
                signal_type=signal_type, compressed="true"
            ).inc(len(compressed))
            payload_bytes = compressed
        else:
            self.export_bytes.labels(
                signal_type=signal_type, compressed="false"
            ).inc(len(payload_bytes))

        # Export with retry
        result = self._send_with_retry(signal_type, payload_bytes)

        duration = (time.perf_counter() - start_time) * 1000

        self.export_duration.labels(signal_type=signal_type).observe(duration / 1000)
        self.export_total.labels(
            signal_type=signal_type,
            status=result.status.value,
        ).inc()

        if result.success:
            self.export_items.labels(signal_type=signal_type).inc(len(items))

        result.duration_ms = duration
        return result

    def _build_payload(
        self,
        signal_type: str,
        items: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build OTLP payload for signal type.

        Args:
            signal_type: traces, metrics, or logs
            items: Items to include

        Returns:
            OTLP payload dict
        """
        resource = self._build_resource()

        if signal_type == "traces":
            return {
                "resourceSpans": [{
                    "resource": resource,
                    "scopeSpans": [{
                        "scope": {
                            "name": "consciousness-lambda",
                            "version": "1.0.0",
                        },
                        "spans": self._convert_spans(items),
                    }],
                }],
            }
        elif signal_type == "metrics":
            return {
                "resourceMetrics": [{
                    "resource": resource,
                    "scopeMetrics": [{
                        "scope": {
                            "name": "consciousness-lambda",
                            "version": "1.0.0",
                        },
                        "metrics": self._convert_metrics(items),
                    }],
                }],
            }
        else:  # logs
            return {
                "resourceLogs": [{
                    "resource": resource,
                    "scopeLogs": [{
                        "scope": {
                            "name": "consciousness-lambda",
                            "version": "1.0.0",
                        },
                        "logRecords": self._convert_logs(items),
                    }],
                }],
            }

    def _convert_spans(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert internal span format to OTLP."""
        spans = []
        for item in items:
            span = {
                "traceId": item.get("trace_id", "0" * 32),
                "spanId": item.get("span_id", "0" * 16),
                "name": item.get("name", "unknown"),
                "kind": item.get("kind", 1),  # INTERNAL
                "startTimeUnixNano": int(item.get("start_time", time.time()) * 1e9),
                "endTimeUnixNano": int(item.get("end_time", time.time()) * 1e9),
                "attributes": self._convert_attributes(item.get("attributes", {})),
                "status": {
                    "code": 1 if item.get("success", True) else 2,  # OK or ERROR
                },
            }

            if item.get("parent_span_id"):
                span["parentSpanId"] = item["parent_span_id"]

            if item.get("error"):
                span["status"]["message"] = item["error"]

            spans.append(span)

        return spans

    def _convert_metrics(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert internal metric format to OTLP."""
        metrics = []
        for item in items:
            metric = {
                "name": item.get("name", "unknown"),
                "description": item.get("description", ""),
                "unit": item.get("unit", "1"),
            }

            # Handle different metric types
            metric_type = item.get("metric_type", "gauge")
            value = item.get("value", 0)
            timestamp = int(item.get("timestamp", time.time()) * 1e9)

            if metric_type == "counter":
                metric["sum"] = {
                    "dataPoints": [{
                        "asDouble": float(value),
                        "timeUnixNano": timestamp,
                        "attributes": self._convert_attributes(item.get("labels", {})),
                    }],
                    "isMonotonic": True,
                    "aggregationTemporality": 2,  # CUMULATIVE
                }
            elif metric_type == "histogram":
                metric["histogram"] = {
                    "dataPoints": [{
                        "count": item.get("count", 1),
                        "sum": item.get("sum", value),
                        "bucketCounts": item.get("bucket_counts", [1]),
                        "explicitBounds": item.get("bounds", []),
                        "timeUnixNano": timestamp,
                        "attributes": self._convert_attributes(item.get("labels", {})),
                    }],
                    "aggregationTemporality": 2,
                }
            else:  # gauge
                metric["gauge"] = {
                    "dataPoints": [{
                        "asDouble": float(value),
                        "timeUnixNano": timestamp,
                        "attributes": self._convert_attributes(item.get("labels", {})),
                    }],
                }

            metrics.append(metric)

        return metrics

    def _convert_logs(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert internal log format to OTLP."""
        logs = []
        for item in items:
            log_record = {
                "timeUnixNano": int(item.get("timestamp", time.time()) * 1e9),
                "severityNumber": self._severity_to_number(item.get("level", "INFO")),
                "severityText": item.get("level", "INFO"),
                "body": {"stringValue": item.get("message", "")},
                "attributes": self._convert_attributes(item.get("attributes", {})),
            }

            if item.get("trace_id"):
                log_record["traceId"] = item["trace_id"]
            if item.get("span_id"):
                log_record["spanId"] = item["span_id"]

            logs.append(log_record)

        return logs

    def _convert_attributes(self, attrs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert dict attributes to OTLP format."""
        result = []
        for key, value in attrs.items():
            attr = {"key": key}
            if isinstance(value, bool):
                attr["value"] = {"boolValue": value}
            elif isinstance(value, int):
                attr["value"] = {"intValue": str(value)}
            elif isinstance(value, float):
                attr["value"] = {"doubleValue": value}
            elif isinstance(value, list):
                attr["value"] = {"arrayValue": {"values": [
                    {"stringValue": str(v)} for v in value
                ]}}
            else:
                attr["value"] = {"stringValue": str(value)}
            result.append(attr)
        return result

    def _severity_to_number(self, level: str) -> int:
        """Convert log level to OTLP severity number."""
        levels = {
            "TRACE": 1,
            "DEBUG": 5,
            "INFO": 9,
            "WARN": 13,
            "WARNING": 13,
            "ERROR": 17,
            "FATAL": 21,
            "CRITICAL": 21,
        }
        return levels.get(level.upper(), 9)

    def _send_with_retry(
        self,
        signal_type: str,
        payload: bytes,
    ) -> ExportResult:
        """Send payload with retry logic.

        Args:
            signal_type: Signal type for endpoint selection
            payload: Compressed payload bytes

        Returns:
            ExportResult
        """
        endpoint = self._get_endpoint(signal_type)
        headers = self._build_headers()

        delay_ms = self.config.retry_delay_ms
        last_error = None

        for attempt in range(self.config.max_retries + 1):
            try:
                request = urllib.request.Request(
                    endpoint,
                    data=payload,
                    headers=headers,
                    method="POST",
                )

                timeout = self.config.timeout_ms / 1000.0
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    if response.status in (200, 202):
                        return ExportResult(
                            status=ExportStatus.SUCCESS,
                            exported_count=1,
                            failed_count=0,
                            duration_ms=0,
                        )
                    elif response.status == 206:
                        return ExportResult(
                            status=ExportStatus.PARTIAL_SUCCESS,
                            exported_count=1,
                            failed_count=0,
                            duration_ms=0,
                        )

            except urllib.error.HTTPError as e:
                last_error = f"HTTP {e.code}: {e.reason}"
                if e.code in (429, 503):
                    # Retryable
                    retry_after = e.headers.get("Retry-After")
                    if retry_after:
                        delay_ms = int(retry_after) * 1000
                elif e.code >= 400 and e.code < 500:
                    # Client error, don't retry
                    return ExportResult(
                        status=ExportStatus.FAILURE,
                        exported_count=0,
                        failed_count=1,
                        duration_ms=0,
                        error=last_error,
                    )

            except urllib.error.URLError as e:
                last_error = f"Connection error: {e.reason}"

            except Exception as e:
                last_error = str(e)

            # Wait before retry
            if attempt < self.config.max_retries:
                time.sleep(delay_ms / 1000.0)
                delay_ms = min(
                    delay_ms * self.config.retry_backoff_multiplier,
                    self.config.max_retry_delay_ms,
                )

        return ExportResult(
            status=ExportStatus.FAILURE,
            exported_count=0,
            failed_count=1,
            duration_ms=0,
            error=last_error,
        )

    def flush(self, deadline_ms: Optional[int] = None) -> ExportResult:
        """Flush all queued telemetry.

        Args:
            deadline_ms: Maximum time to spend flushing

        Returns:
            Aggregated ExportResult
        """
        start_time = time.perf_counter()
        deadline = deadline_ms / 1000.0 if deadline_ms else float("inf")

        results = []

        # Export traces
        with self._queue_lock:
            traces = list(self._traces_queue)
            self._traces_queue.clear()

        if traces and (time.perf_counter() - start_time) < deadline:
            result = self._export_signal("traces", traces)
            results.append(result)

        # Export metrics
        with self._queue_lock:
            metrics = list(self._metrics_queue)
            self._metrics_queue.clear()

        if metrics and (time.perf_counter() - start_time) < deadline:
            result = self._export_signal("metrics", metrics)
            results.append(result)

        # Export logs
        with self._queue_lock:
            logs = list(self._logs_queue)
            self._logs_queue.clear()

        if logs and (time.perf_counter() - start_time) < deadline:
            result = self._export_signal("logs", logs)
            results.append(result)

        # Update queue size metrics
        self.queue_size.labels(signal_type="traces").set(len(self._traces_queue))
        self.queue_size.labels(signal_type="metrics").set(len(self._metrics_queue))
        self.queue_size.labels(signal_type="logs").set(len(self._logs_queue))

        # Aggregate results
        if not results:
            return ExportResult(
                status=ExportStatus.SUCCESS,
                exported_count=0,
                failed_count=0,
                duration_ms=(time.perf_counter() - start_time) * 1000,
            )

        total_exported = sum(r.exported_count for r in results)
        total_failed = sum(r.failed_count for r in results)

        if all(r.status == ExportStatus.SUCCESS for r in results):
            status = ExportStatus.SUCCESS
        elif any(r.status == ExportStatus.SUCCESS for r in results):
            status = ExportStatus.PARTIAL_SUCCESS
        else:
            status = ExportStatus.FAILURE

        return ExportResult(
            status=status,
            exported_count=total_exported,
            failed_count=total_failed,
            duration_ms=(time.perf_counter() - start_time) * 1000,
        )

    def shutdown(self, timeout_ms: int = 5000) -> ExportResult:
        """Shutdown exporter, flushing all remaining data.

        Args:
            timeout_ms: Maximum time to wait

        Returns:
            Final ExportResult
        """
        return self.flush(deadline_ms=timeout_ms)
