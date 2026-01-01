"""Flow Data Export

Export network flow data to various backends (Kafka, Elasticsearch, S3, etc.).
"""

from __future__ import annotations

import json
import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type, Union
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import logging
import queue
import gzip
import hashlib

from prometheus_client import Counter, Gauge, Histogram

from .cilium.flows import FlowRecord, FlowAggregation

logger = logging.getLogger(__name__)


class ExportBackend(str, Enum):
    """Supported export backends."""
    KAFKA = "kafka"
    ELASTICSEARCH = "elasticsearch"
    S3 = "s3"
    AZURE_BLOB = "azure_blob"
    GCS = "gcs"
    FILE = "file"
    WEBHOOK = "webhook"
    STDOUT = "stdout"


@dataclass
class ExportConfig:
    """Configuration for flow export.

    Attributes:
        backend: Export backend type
        batch_size: Number of records per batch
        flush_interval_seconds: Flush interval
        compression: Enable compression
        include_raw_flow: Include raw flow data
        field_mapping: Custom field mapping
        retry_attempts: Number of retry attempts
        retry_delay_seconds: Delay between retries
        dead_letter_queue: Enable dead letter queue
    """
    backend: ExportBackend = ExportBackend.FILE
    batch_size: int = 100
    flush_interval_seconds: float = 10.0
    compression: bool = True
    include_raw_flow: bool = False
    field_mapping: Dict[str, str] = field(default_factory=dict)
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    dead_letter_queue: bool = True

    # Backend-specific config
    endpoint: Optional[str] = None
    auth_config: Dict[str, Any] = field(default_factory=dict)
    extra_config: Dict[str, Any] = field(default_factory=dict)


class BaseExporter(ABC):
    """Base class for flow exporters."""

    def __init__(self, config: ExportConfig, namespace: str = "consciousness"):
        self.config = config
        self.namespace = namespace
        self._lock = threading.Lock()
        self._buffer: List[Dict[str, Any]] = []
        self._dead_letter: List[Dict[str, Any]] = []
        self._running = False
        self._flush_thread: Optional[threading.Thread] = None

        # Metrics
        self.records_exported = Counter(
            f"{namespace}_flow_export_records_total",
            "Total records exported",
            ["backend", "status"],
        )

        self.export_batch_size = Histogram(
            f"{namespace}_flow_export_batch_size",
            "Export batch size",
            ["backend"],
            buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000],
        )

        self.export_latency = Histogram(
            f"{namespace}_flow_export_latency_seconds",
            "Export latency",
            ["backend"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        )

        self.buffer_size = Gauge(
            f"{namespace}_flow_export_buffer_size",
            "Current buffer size",
            ["backend"],
        )

    @abstractmethod
    def _do_export(self, records: List[Dict[str, Any]]) -> bool:
        """Perform the actual export. Returns True on success."""
        pass

    def export_flow(self, record: FlowRecord):
        """Export a single flow record.

        Args:
            record: FlowRecord to export
        """
        data = self._serialize_flow(record)

        with self._lock:
            self._buffer.append(data)
            self.buffer_size.labels(backend=self.config.backend.value).set(len(self._buffer))

        if len(self._buffer) >= self.config.batch_size:
            self._flush()

    def export_flows(self, records: List[FlowRecord]):
        """Export multiple flow records.

        Args:
            records: List of FlowRecords to export
        """
        data_list = [self._serialize_flow(r) for r in records]

        with self._lock:
            self._buffer.extend(data_list)
            self.buffer_size.labels(backend=self.config.backend.value).set(len(self._buffer))

        if len(self._buffer) >= self.config.batch_size:
            self._flush()

    def export_aggregation(self, aggregation: FlowAggregation):
        """Export a flow aggregation.

        Args:
            aggregation: FlowAggregation to export
        """
        data = self._serialize_aggregation(aggregation)

        with self._lock:
            self._buffer.append(data)
            self.buffer_size.labels(backend=self.config.backend.value).set(len(self._buffer))

        if len(self._buffer) >= self.config.batch_size:
            self._flush()

    def _serialize_flow(self, record: FlowRecord) -> Dict[str, Any]:
        """Serialize a flow record to dict."""
        flow = record.flow

        data = {
            "@timestamp": flow.time.isoformat(),
            "flow_id": flow.flow_id,
            "verdict": flow.verdict,
            "flow_type": record.flow_type.value,
            "source": {
                "ip": flow.source_ip,
                "port": flow.source_port,
                "pod": flow.source_pod,
                "namespace": flow.source_namespace,
                "service": record.source_service,
            },
            "destination": {
                "ip": flow.destination_ip,
                "port": flow.destination_port,
                "pod": flow.destination_pod,
                "namespace": flow.destination_namespace,
                "service": record.destination_service,
            },
            "network": {
                "protocol": flow.protocol,
                "l7_protocol": flow.l7_protocol,
                "direction": flow.traffic_direction,
            },
            "is_error": record.is_error,
            "error_category": record.error_category,
            "slo_relevant": record.slo_relevant,
            "tags": record.tags,
        }

        if record.latency_ms is not None:
            data["latency_ms"] = record.latency_ms

        if flow.http_info:
            data["http"] = flow.http_info

        if flow.dns_info:
            data["dns"] = flow.dns_info

        if self.config.include_raw_flow:
            data["raw"] = flow.raw_data

        # Apply field mapping
        if self.config.field_mapping:
            data = self._apply_field_mapping(data)

        return data

    def _serialize_aggregation(self, agg: FlowAggregation) -> Dict[str, Any]:
        """Serialize a flow aggregation to dict."""
        return {
            "@timestamp": datetime.now().isoformat(),
            "type": "aggregation",
            "source_service": agg.source_service,
            "destination_service": agg.destination_service,
            "protocol": agg.protocol,
            "l7_protocol": agg.l7_protocol,
            "flow_count": agg.flow_count,
            "bytes_sent": agg.bytes_sent,
            "bytes_received": agg.bytes_received,
            "error_count": agg.error_count,
            "dropped_count": agg.dropped_count,
            "latency": {
                "avg_ms": agg.avg_latency_ms,
                "p50_ms": agg.p50_latency_ms,
                "p95_ms": agg.p95_latency_ms,
                "p99_ms": agg.p99_latency_ms,
            },
            "first_seen": agg.first_seen.isoformat() if agg.first_seen else None,
            "last_seen": agg.last_seen.isoformat() if agg.last_seen else None,
        }

    def _apply_field_mapping(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply custom field mapping."""
        for src, dst in self.config.field_mapping.items():
            parts = src.split(".")
            value = data
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    value = None
                    break

            if value is not None:
                data[dst] = value

        return data

    def _flush(self):
        """Flush buffered records to backend."""
        with self._lock:
            if not self._buffer:
                return
            records = list(self._buffer)
            self._buffer.clear()
            self.buffer_size.labels(backend=self.config.backend.value).set(0)

        start_time = time.perf_counter()
        success = False

        for attempt in range(self.config.retry_attempts):
            try:
                success = self._do_export(records)
                if success:
                    break
            except Exception as e:
                logger.warning(f"Export attempt {attempt + 1} failed: {e}")
                if attempt < self.config.retry_attempts - 1:
                    time.sleep(self.config.retry_delay_seconds)

        duration = time.perf_counter() - start_time

        if success:
            self.records_exported.labels(
                backend=self.config.backend.value,
                status="success",
            ).inc(len(records))
        else:
            self.records_exported.labels(
                backend=self.config.backend.value,
                status="failed",
            ).inc(len(records))

            if self.config.dead_letter_queue:
                with self._lock:
                    self._dead_letter.extend(records)
                logger.warning(f"Moved {len(records)} records to dead letter queue")

        self.export_batch_size.labels(backend=self.config.backend.value).observe(len(records))
        self.export_latency.labels(backend=self.config.backend.value).observe(duration)

    def start(self):
        """Start the background flush thread."""
        self._running = True

        def flush_worker():
            while self._running:
                time.sleep(self.config.flush_interval_seconds)
                self._flush()

        self._flush_thread = threading.Thread(target=flush_worker, daemon=True)
        self._flush_thread.start()

    def stop(self):
        """Stop the exporter and flush remaining records."""
        self._running = False
        if self._flush_thread:
            self._flush_thread.join(timeout=5.0)
        self._flush()

    def get_dead_letter_records(self) -> List[Dict[str, Any]]:
        """Get records from dead letter queue."""
        with self._lock:
            records = list(self._dead_letter)
            self._dead_letter.clear()
        return records

    def retry_dead_letter(self) -> int:
        """Retry exporting dead letter records.

        Returns:
            Number of records successfully exported
        """
        records = self.get_dead_letter_records()
        if not records:
            return 0

        success = self._do_export(records)
        if success:
            return len(records)

        with self._lock:
            self._dead_letter.extend(records)
        return 0


class KafkaExporter(BaseExporter):
    """Export flows to Kafka.

    Usage:
        config = ExportConfig(
            backend=ExportBackend.KAFKA,
            endpoint="localhost:9092",
            extra_config={"topic": "network-flows"},
        )
        exporter = KafkaExporter(config)
        exporter.start()

        for flow in flows:
            exporter.export_flow(flow)

        exporter.stop()
    """

    def __init__(self, config: ExportConfig, namespace: str = "consciousness"):
        super().__init__(config, namespace)
        self._producer = None
        self._topic = config.extra_config.get("topic", "network-flows")

    def _get_producer(self):
        """Get or create Kafka producer."""
        if self._producer:
            return self._producer

        try:
            from kafka import KafkaProducer

            bootstrap_servers = self.config.endpoint or "localhost:9092"

            producer_config = {
                "bootstrap_servers": bootstrap_servers,
                "value_serializer": lambda v: json.dumps(v).encode("utf-8"),
                "acks": "all",
                "retries": 3,
            }

            if self.config.compression:
                producer_config["compression_type"] = "gzip"

            # Add auth config
            if self.config.auth_config:
                producer_config.update(self.config.auth_config)

            self._producer = KafkaProducer(**producer_config)
            return self._producer

        except ImportError:
            logger.error("kafka-python package not installed")
            raise
        except Exception as e:
            logger.error(f"Failed to create Kafka producer: {e}")
            raise

    def _do_export(self, records: List[Dict[str, Any]]) -> bool:
        """Export records to Kafka."""
        try:
            producer = self._get_producer()

            for record in records:
                # Use flow_id or generate a key
                key = record.get("flow_id", "").encode("utf-8") if record.get("flow_id") else None
                producer.send(self._topic, value=record, key=key)

            producer.flush()
            return True

        except Exception as e:
            logger.error(f"Kafka export failed: {e}")
            return False


class ElasticsearchExporter(BaseExporter):
    """Export flows to Elasticsearch.

    Usage:
        config = ExportConfig(
            backend=ExportBackend.ELASTICSEARCH,
            endpoint="http://localhost:9200",
            extra_config={"index_prefix": "network-flows"},
        )
        exporter = ElasticsearchExporter(config)
        exporter.start()

        for flow in flows:
            exporter.export_flow(flow)

        exporter.stop()
    """

    def __init__(self, config: ExportConfig, namespace: str = "consciousness"):
        super().__init__(config, namespace)
        self._client = None
        self._index_prefix = config.extra_config.get("index_prefix", "network-flows")

    def _get_client(self):
        """Get or create Elasticsearch client."""
        if self._client:
            return self._client

        try:
            from elasticsearch import Elasticsearch

            hosts = self.config.endpoint or "http://localhost:9200"

            client_config = {
                "hosts": [hosts] if isinstance(hosts, str) else hosts,
            }

            # Add auth
            if "username" in self.config.auth_config:
                client_config["basic_auth"] = (
                    self.config.auth_config["username"],
                    self.config.auth_config.get("password", ""),
                )

            if "api_key" in self.config.auth_config:
                client_config["api_key"] = self.config.auth_config["api_key"]

            self._client = Elasticsearch(**client_config)
            return self._client

        except ImportError:
            logger.error("elasticsearch package not installed")
            raise
        except Exception as e:
            logger.error(f"Failed to create Elasticsearch client: {e}")
            raise

    def _do_export(self, records: List[Dict[str, Any]]) -> bool:
        """Export records to Elasticsearch."""
        try:
            from elasticsearch import helpers

            client = self._get_client()

            # Generate index name with date
            date_suffix = datetime.now().strftime("%Y.%m.%d")
            index_name = f"{self._index_prefix}-{date_suffix}"

            # Prepare bulk actions
            actions = []
            for record in records:
                action = {
                    "_index": index_name,
                    "_source": record,
                }
                if record.get("flow_id"):
                    action["_id"] = record["flow_id"]
                actions.append(action)

            # Bulk insert
            success, failed = helpers.bulk(
                client,
                actions,
                raise_on_error=False,
                stats_only=True,
            )

            if failed:
                logger.warning(f"Elasticsearch bulk insert: {success} success, {failed} failed")

            return failed == 0

        except Exception as e:
            logger.error(f"Elasticsearch export failed: {e}")
            return False


class S3Exporter(BaseExporter):
    """Export flows to S3.

    Usage:
        config = ExportConfig(
            backend=ExportBackend.S3,
            extra_config={
                "bucket": "my-network-flows",
                "prefix": "flows/",
                "region": "us-east-1",
            },
        )
        exporter = S3Exporter(config)
        exporter.start()

        for flow in flows:
            exporter.export_flow(flow)

        exporter.stop()
    """

    def __init__(self, config: ExportConfig, namespace: str = "consciousness"):
        super().__init__(config, namespace)
        self._client = None
        self._bucket = config.extra_config.get("bucket", "network-flows")
        self._prefix = config.extra_config.get("prefix", "flows/")
        self._region = config.extra_config.get("region", "us-east-1")

    def _get_client(self):
        """Get or create S3 client."""
        if self._client:
            return self._client

        try:
            import boto3

            client_config = {
                "region_name": self._region,
            }

            if "aws_access_key_id" in self.config.auth_config:
                client_config["aws_access_key_id"] = self.config.auth_config["aws_access_key_id"]
                client_config["aws_secret_access_key"] = self.config.auth_config.get("aws_secret_access_key", "")

            if self.config.endpoint:
                client_config["endpoint_url"] = self.config.endpoint

            self._client = boto3.client("s3", **client_config)
            return self._client

        except ImportError:
            logger.error("boto3 package not installed")
            raise
        except Exception as e:
            logger.error(f"Failed to create S3 client: {e}")
            raise

    def _do_export(self, records: List[Dict[str, Any]]) -> bool:
        """Export records to S3."""
        try:
            client = self._get_client()

            # Create file content
            content = "\n".join(json.dumps(r) for r in records)

            # Compress if enabled
            if self.config.compression:
                content = gzip.compress(content.encode("utf-8"))
                ext = "json.gz"
            else:
                content = content.encode("utf-8")
                ext = "json"

            # Generate object key
            timestamp = datetime.now()
            date_path = timestamp.strftime("%Y/%m/%d/%H")
            file_hash = hashlib.md5(content).hexdigest()[:8]
            key = f"{self._prefix}{date_path}/flows_{timestamp.strftime('%Y%m%d%H%M%S')}_{file_hash}.{ext}"

            # Upload
            client.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=content,
                ContentType="application/gzip" if self.config.compression else "application/json",
            )

            logger.debug(f"Exported {len(records)} records to s3://{self._bucket}/{key}")
            return True

        except Exception as e:
            logger.error(f"S3 export failed: {e}")
            return False


class FlowExporter:
    """Main flow exporter with multi-backend support.

    Usage:
        exporter = FlowExporter()

        # Add backends
        exporter.add_backend(KafkaExporter(kafka_config))
        exporter.add_backend(ElasticsearchExporter(es_config))

        # Start exporting
        exporter.start()

        for flow in flows:
            exporter.export(flow)

        exporter.stop()
    """

    def __init__(self, namespace: str = "consciousness"):
        self.namespace = namespace
        self._backends: List[BaseExporter] = []
        self._lock = threading.Lock()
        self._running = False

    def add_backend(self, backend: BaseExporter):
        """Add an export backend.

        Args:
            backend: BaseExporter instance
        """
        with self._lock:
            self._backends.append(backend)
        logger.info(f"Added export backend: {backend.config.backend.value}")

    def remove_backend(self, backend_type: ExportBackend):
        """Remove backends of a specific type.

        Args:
            backend_type: ExportBackend type to remove
        """
        with self._lock:
            self._backends = [
                b for b in self._backends
                if b.config.backend != backend_type
            ]

    def export(self, record: Union[FlowRecord, FlowAggregation]):
        """Export a flow record or aggregation to all backends.

        Args:
            record: FlowRecord or FlowAggregation to export
        """
        with self._lock:
            backends = list(self._backends)

        for backend in backends:
            try:
                if isinstance(record, FlowRecord):
                    backend.export_flow(record)
                elif isinstance(record, FlowAggregation):
                    backend.export_aggregation(record)
            except Exception as e:
                logger.error(f"Export to {backend.config.backend.value} failed: {e}")

    def export_batch(self, records: List[FlowRecord]):
        """Export multiple flow records to all backends.

        Args:
            records: List of FlowRecords to export
        """
        with self._lock:
            backends = list(self._backends)

        for backend in backends:
            try:
                backend.export_flows(records)
            except Exception as e:
                logger.error(f"Batch export to {backend.config.backend.value} failed: {e}")

    def start(self):
        """Start all backends."""
        self._running = True
        for backend in self._backends:
            backend.start()
        logger.info(f"Started {len(self._backends)} export backends")

    def stop(self):
        """Stop all backends."""
        self._running = False
        for backend in self._backends:
            backend.stop()
        logger.info("Stopped all export backends")

    def get_stats(self) -> Dict[str, Any]:
        """Get export statistics.

        Returns:
            Dict with export statistics
        """
        with self._lock:
            backends = list(self._backends)

        return {
            "backends": [
                {
                    "type": b.config.backend.value,
                    "buffer_size": len(b._buffer),
                    "dead_letter_size": len(b._dead_letter),
                }
                for b in backends
            ],
            "total_backends": len(backends),
            "running": self._running,
        }

    @staticmethod
    def create_backend(config: ExportConfig) -> BaseExporter:
        """Create a backend from config.

        Args:
            config: ExportConfig

        Returns:
            Appropriate BaseExporter instance
        """
        backend_map: Dict[ExportBackend, Type[BaseExporter]] = {
            ExportBackend.KAFKA: KafkaExporter,
            ExportBackend.ELASTICSEARCH: ElasticsearchExporter,
            ExportBackend.S3: S3Exporter,
        }

        backend_class = backend_map.get(config.backend)
        if not backend_class:
            raise ValueError(f"Unsupported backend: {config.backend}")

        return backend_class(config)
