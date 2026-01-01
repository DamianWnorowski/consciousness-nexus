"""
Prometheus Exporters

Provides:
- Custom metrics exporters
- Push Gateway integration for batch jobs
- Remote write support
"""

from typing import Any, Dict, List, Optional
import logging
import time
from dataclasses import dataclass

from prometheus_client import (
    CollectorRegistry,
    generate_latest,
    push_to_gateway,
    REGISTRY,
)
from prometheus_client.exposition import basic_auth_handler

logger = logging.getLogger(__name__)


@dataclass
class MetricSample:
    """Container for a metric sample."""
    name: str
    labels: Dict[str, str]
    value: float
    timestamp: Optional[float] = None


class MetricsExporter:
    """
    Base class for exporting metrics.

    Provides common functionality for collecting and
    exporting metrics in various formats.
    """

    def __init__(
        self,
        registry: Optional[CollectorRegistry] = None,
        prefix: str = "",
    ):
        self.registry = registry or REGISTRY
        self.prefix = prefix
        self._samples: List[MetricSample] = []

    def add_sample(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        timestamp: Optional[float] = None,
    ):
        """Add a metric sample for export."""
        full_name = f"{self.prefix}_{name}" if self.prefix else name
        self._samples.append(MetricSample(
            name=full_name,
            labels=labels or {},
            value=value,
            timestamp=timestamp or time.time(),
        ))

    def clear_samples(self):
        """Clear pending samples."""
        self._samples.clear()

    def get_prometheus_format(self) -> bytes:
        """Get metrics in Prometheus exposition format."""
        return generate_latest(self.registry)

    def get_openmetrics_format(self) -> str:
        """Get metrics in OpenMetrics format."""
        lines = []
        for sample in self._samples:
            label_str = ",".join(
                f'{k}="{v}"' for k, v in sample.labels.items()
            )
            if label_str:
                metric_line = f"{sample.name}{{{label_str}}} {sample.value} {int(sample.timestamp * 1000)}"
            else:
                metric_line = f"{sample.name} {sample.value} {int(sample.timestamp * 1000)}"
            lines.append(metric_line)
        return "\n".join(lines)


class PushGatewayExporter(MetricsExporter):
    """
    Exporter for Prometheus Push Gateway.

    Used for batch jobs and short-lived processes that
    can't be scraped directly.

    Usage:
        exporter = PushGatewayExporter(
            gateway_url="http://pushgateway:9091",
            job_name="batch_process",
        )

        # Record metrics
        exporter.add_sample("batch_items_processed", 1000)
        exporter.add_sample("batch_duration_seconds", 120.5)

        # Push to gateway
        exporter.push()
    """

    def __init__(
        self,
        gateway_url: str,
        job_name: str,
        registry: Optional[CollectorRegistry] = None,
        prefix: str = "consciousness",
        grouping_key: Optional[Dict[str, str]] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        super().__init__(registry, prefix)
        self.gateway_url = gateway_url
        self.job_name = job_name
        self.grouping_key = grouping_key or {}
        self.username = username
        self.password = password

    def _get_auth_handler(self):
        """Create authentication handler if credentials provided."""
        if self.username and self.password:
            return basic_auth_handler(self.username, self.password)
        return None

    def push(self, timeout: int = 30) -> bool:
        """
        Push metrics to the Push Gateway.

        Args:
            timeout: Request timeout in seconds

        Returns:
            True if push succeeded, False otherwise
        """
        try:
            push_to_gateway(
                self.gateway_url,
                job=self.job_name,
                registry=self.registry,
                grouping_key=self.grouping_key,
                handler=self._get_auth_handler(),
                timeout=timeout,
            )
            logger.info(f"Metrics pushed to gateway: {self.gateway_url}")
            return True
        except Exception as e:
            logger.error(f"Failed to push metrics to gateway: {e}")
            return False

    def push_add(self, timeout: int = 30) -> bool:
        """
        Push metrics using push_add (doesn't replace existing metrics).

        Args:
            timeout: Request timeout in seconds

        Returns:
            True if push succeeded, False otherwise
        """
        try:
            from prometheus_client import pushadd_to_gateway
            pushadd_to_gateway(
                self.gateway_url,
                job=self.job_name,
                registry=self.registry,
                grouping_key=self.grouping_key,
                handler=self._get_auth_handler(),
                timeout=timeout,
            )
            logger.info(f"Metrics pushed (add) to gateway: {self.gateway_url}")
            return True
        except Exception as e:
            logger.error(f"Failed to push_add metrics to gateway: {e}")
            return False

    def delete_from_gateway(self, timeout: int = 30) -> bool:
        """
        Delete metrics from the Push Gateway.

        Args:
            timeout: Request timeout in seconds

        Returns:
            True if delete succeeded, False otherwise
        """
        try:
            from prometheus_client import delete_from_gateway
            delete_from_gateway(
                self.gateway_url,
                job=self.job_name,
                grouping_key=self.grouping_key,
                handler=self._get_auth_handler(),
                timeout=timeout,
            )
            logger.info(f"Metrics deleted from gateway: {self.gateway_url}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete metrics from gateway: {e}")
            return False


class RemoteWriteExporter(MetricsExporter):
    """
    Exporter for Prometheus Remote Write protocol.

    Supports pushing metrics to:
    - Prometheus with remote_write enabled
    - Cortex
    - Thanos
    - Mimir
    - VictoriaMetrics
    """

    def __init__(
        self,
        endpoint: str,
        registry: Optional[CollectorRegistry] = None,
        prefix: str = "consciousness",
        headers: Optional[Dict[str, str]] = None,
        batch_size: int = 500,
        flush_interval_seconds: float = 10.0,
    ):
        super().__init__(registry, prefix)
        self.endpoint = endpoint
        self.headers = headers or {}
        self.batch_size = batch_size
        self.flush_interval_seconds = flush_interval_seconds
        self._last_flush = time.time()

    def _should_flush(self) -> bool:
        """Check if we should flush based on batch size or time."""
        if len(self._samples) >= self.batch_size:
            return True
        if time.time() - self._last_flush >= self.flush_interval_seconds:
            return True
        return False

    def _build_write_request(self) -> bytes:
        """
        Build a Prometheus Remote Write request.

        Uses the Prometheus protobuf format (snappy compressed).
        """
        # This would normally use the prometheus_remote_write library
        # For now, we'll use a simplified JSON format for demonstration
        import json
        data = {
            "timeseries": [
                {
                    "labels": [
                        {"name": "__name__", "value": s.name},
                        *[{"name": k, "value": v} for k, v in s.labels.items()]
                    ],
                    "samples": [
                        {"value": s.value, "timestamp": int(s.timestamp * 1000)}
                    ]
                }
                for s in self._samples
            ]
        }
        return json.dumps(data).encode()

    async def flush_async(self) -> bool:
        """
        Flush pending metrics to remote write endpoint (async).

        Returns:
            True if flush succeeded, False otherwise
        """
        if not self._samples:
            return True

        try:
            import httpx
            data = self._build_write_request()

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.endpoint,
                    content=data,
                    headers={
                        "Content-Type": "application/x-protobuf",
                        "Content-Encoding": "snappy",
                        **self.headers,
                    },
                    timeout=30.0,
                )
                response.raise_for_status()

            self.clear_samples()
            self._last_flush = time.time()
            logger.debug(f"Flushed {len(self._samples)} metrics to {self.endpoint}")
            return True

        except Exception as e:
            logger.error(f"Failed to flush metrics: {e}")
            return False

    def flush_sync(self) -> bool:
        """
        Flush pending metrics to remote write endpoint (sync).

        Returns:
            True if flush succeeded, False otherwise
        """
        if not self._samples:
            return True

        try:
            import httpx
            data = self._build_write_request()

            with httpx.Client() as client:
                response = client.post(
                    self.endpoint,
                    content=data,
                    headers={
                        "Content-Type": "application/x-protobuf",
                        "Content-Encoding": "snappy",
                        **self.headers,
                    },
                    timeout=30.0,
                )
                response.raise_for_status()

            sample_count = len(self._samples)
            self.clear_samples()
            self._last_flush = time.time()
            logger.debug(f"Flushed {sample_count} metrics to {self.endpoint}")
            return True

        except Exception as e:
            logger.error(f"Failed to flush metrics: {e}")
            return False
