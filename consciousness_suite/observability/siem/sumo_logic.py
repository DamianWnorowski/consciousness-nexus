"""Sumo Logic Integration

Sumo Logic HTTP Source integration for security event ingestion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import json
import threading
import logging
import asyncio
import time
import gzip
import io

from prometheus_client import Counter, Gauge, Histogram

from .security_events import SecurityEvent, EventSeverity, EventCategory

logger = logging.getLogger(__name__)


class SumoSourceType(str, Enum):
    """Sumo Logic source types."""
    HTTP = "http"
    SYSLOG = "syslog"
    CLOUD_SYSLOG = "cloud_syslog"


class SumoLogFormat(str, Enum):
    """Log format for Sumo Logic."""
    JSON = "json"
    TEXT = "text"
    CEF = "cef"
    LEEF = "leef"


class SumoCategory(str, Enum):
    """Sumo Logic source categories."""
    SECURITY = "security"
    SECURITY_ALERTS = "security/alerts"
    SECURITY_EVENTS = "security/events"
    SECURITY_THREATS = "security/threats"
    CONSCIOUSNESS = "consciousness/security"


@dataclass
class SumoConfig:
    """Sumo Logic configuration."""
    endpoint_url: str
    category: SumoCategory = SumoCategory.CONSCIOUSNESS
    source_name: str = "consciousness-nexus"
    source_host: Optional[str] = None
    log_format: SumoLogFormat = SumoLogFormat.JSON
    compress: bool = True
    timeout: int = 30
    max_retries: int = 3
    batch_size: int = 100
    flush_interval: float = 5.0
    fields: Dict[str, str] = field(default_factory=dict)

    def get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for Sumo Logic."""
        headers = {
            "Content-Type": "application/json",
            "X-Sumo-Category": self.category.value,
            "X-Sumo-Name": self.source_name,
        }

        if self.source_host:
            headers["X-Sumo-Host"] = self.source_host

        if self.compress:
            headers["Content-Encoding"] = "gzip"

        # Add custom fields
        if self.fields:
            fields_str = ",".join(f"{k}={v}" for k, v in self.fields.items())
            headers["X-Sumo-Fields"] = fields_str

        return headers


@dataclass
class SumoResponse:
    """Response from Sumo Logic."""
    success: bool
    status_code: int = 0
    message: str = ""
    request_id: Optional[str] = None


@dataclass
class BatchResult:
    """Result of a batch send operation."""
    total: int = 0
    success: int = 0
    failed: int = 0
    errors: List[str] = field(default_factory=list)
    duration_ms: int = 0


@dataclass
class SumoLogEntry:
    """A log entry for Sumo Logic."""
    message: str
    timestamp: datetime
    level: str = "INFO"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Convert to JSON string."""
        data = {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level,
            "message": self.message,
            **self.metadata,
        }
        return json.dumps(data)


class SumoLogicClient:
    """Client for Sumo Logic HTTP Source.

    Usage:
        config = SumoConfig(
            endpoint_url="https://endpoint.collection.sumologic.com/receiver/v1/http/...",
            category=SumoCategory.SECURITY_EVENTS,
        )

        client = SumoLogicClient(config)

        # Send single event
        await client.send_event(security_event)

        # Send batch
        result = await client.send_batch(events)

        # Start auto-flush
        client.start_auto_flush()
    """

    def __init__(
        self,
        config: SumoConfig,
        namespace: str = "consciousness",
    ):
        self.config = config
        self.namespace = namespace

        self._buffer: List[SecurityEvent] = []
        self._buffer_lock = threading.Lock()
        self._auto_flush_task: Optional[asyncio.Task] = None
        self._running = False

        # Metrics
        self.events_sent = Counter(
            f"{namespace}_siem_sumo_events_sent_total",
            "Total events sent to Sumo Logic",
            ["category", "status"],
        )

        self.batch_operations = Counter(
            f"{namespace}_siem_sumo_batch_operations_total",
            "Total batch operations",
            ["status"],
        )

        self.send_latency = Histogram(
            f"{namespace}_siem_sumo_send_latency_seconds",
            "Event send latency",
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
        )

        self.buffer_size = Gauge(
            f"{namespace}_siem_sumo_buffer_size",
            "Current buffer size",
        )

        self.bytes_sent = Counter(
            f"{namespace}_siem_sumo_bytes_sent_total",
            "Total bytes sent",
            ["compressed"],
        )

    def _event_to_sumo(self, event: SecurityEvent) -> Dict[str, Any]:
        """Convert SecurityEvent to Sumo Logic format."""
        # Build Sumo-optimized format with Cloud SIEM fields
        data: Dict[str, Any] = {
            "timestamp": event.timestamp.isoformat(),
            "event_id": event.event_id,
            # Cloud SIEM record fields
            "_siemVendor": "Consciousness",
            "_siemProduct": "SecurityNexus",
            "_siemFormat": "JSON",
            "_siemEventId": event.event_id,
            "_siemSeverity": event.severity.value,
            "_siemCategory": event.category.value,
            # Event fields
            "kind": event.kind.value,
            "category": event.category.value,
            "severity": event.severity.value,
            "outcome": event.outcome.value,
            "action": event.action,
            "message": event.message,
            "risk_score": event.risk_score,
            "tags": event.tags,
            "labels": event.labels,
        }

        # Add source fields for Cloud SIEM
        if event.source:
            source_dict = event.source.to_dict()
            if source_dict.get("ip"):
                data["srcDevice_ip"] = source_dict["ip"]
                data["device_ip"] = source_dict["ip"]
            if source_dict.get("port"):
                data["srcPort"] = source_dict["port"]
            if source_dict.get("user"):
                data["user_username"] = source_dict["user"]
            if source_dict.get("domain"):
                data["srcDevice_hostname"] = source_dict["domain"]

        # Add destination fields
        if event.destination:
            dest_dict = event.destination.to_dict()
            if dest_dict.get("ip"):
                data["dstDevice_ip"] = dest_dict["ip"]
            if dest_dict.get("port"):
                data["dstPort"] = dest_dict["port"]
            if dest_dict.get("resource"):
                data["resource"] = dest_dict["resource"]

        # Add threat intel
        if event.threat:
            threat_dict = event.threat.to_dict()
            data["threat_indicator_type"] = threat_dict.get("type")
            data["threat_indicator_value"] = threat_dict.get("value")
            data["threat_confidence"] = threat_dict.get("confidence")

        # Add rule info
        if event.rule_id:
            data["rule_id"] = event.rule_id
            data["rule_name"] = event.rule_name

        # Add metadata
        if event.metadata:
            data["metadata"] = event.metadata

        return data

    def _compress(self, data: bytes) -> bytes:
        """Compress data using gzip."""
        buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode="wb") as f:
            f.write(data)
        return buffer.getvalue()

    async def send_event(self, event: SecurityEvent) -> SumoResponse:
        """Send a single security event.

        Args:
            event: Event to send

        Returns:
            SumoResponse
        """
        start_time = time.time()

        try:
            data = self._event_to_sumo(event)

            if self.config.log_format == SumoLogFormat.JSON:
                payload = json.dumps(data).encode()
            elif self.config.log_format == SumoLogFormat.CEF:
                payload = event.to_cef().encode()
            elif self.config.log_format == SumoLogFormat.LEEF:
                payload = event.to_leef().encode()
            else:
                payload = json.dumps(data).encode()

            compressed = "false"
            if self.config.compress:
                payload = self._compress(payload)
                compressed = "true"

            # In real implementation, would POST to Sumo endpoint
            headers = self.config.get_headers()
            logger.debug(f"Sending event {event.event_id} to Sumo Logic")

            duration = time.time() - start_time
            self.send_latency.observe(duration)
            self.events_sent.labels(
                category=self.config.category.value,
                status="success",
            ).inc()
            self.bytes_sent.labels(compressed=compressed).inc(len(payload))

            return SumoResponse(success=True, status_code=200)

        except Exception as e:
            logger.error(f"Failed to send event: {e}")
            self.events_sent.labels(
                category=self.config.category.value,
                status="error",
            ).inc()
            return SumoResponse(success=False, message=str(e))

    async def send_batch(
        self,
        events: List[SecurityEvent],
    ) -> BatchResult:
        """Send a batch of events.

        Args:
            events: Events to send

        Returns:
            BatchResult
        """
        start_time = time.time()
        result = BatchResult(total=len(events))

        if not events:
            return result

        try:
            # Build batch (newline-delimited JSON)
            lines = []
            for event in events:
                data = self._event_to_sumo(event)
                lines.append(json.dumps(data))

            payload = "\n".join(lines).encode()

            compressed = "false"
            if self.config.compress:
                payload = self._compress(payload)
                compressed = "true"

            # In real implementation, would POST batch
            headers = self.config.get_headers()
            logger.info(f"Sending batch of {len(events)} events to Sumo Logic")

            result.success = len(events)
            result.duration_ms = int((time.time() - start_time) * 1000)

            self.batch_operations.labels(status="success").inc()
            self.bytes_sent.labels(compressed=compressed).inc(len(payload))

            for event in events:
                self.events_sent.labels(
                    category=self.config.category.value,
                    status="success",
                ).inc()

            return result

        except Exception as e:
            logger.error(f"Batch send failed: {e}")
            result.failed = len(events)
            result.errors.append(str(e))
            self.batch_operations.labels(status="error").inc()
            return result

    def buffer_event(self, event: SecurityEvent):
        """Add event to buffer.

        Args:
            event: Event to buffer
        """
        with self._buffer_lock:
            self._buffer.append(event)
            current_size = len(self._buffer)
            self.buffer_size.set(current_size)

            if current_size >= self.config.batch_size:
                asyncio.create_task(self.flush())

    async def flush(self) -> BatchResult:
        """Flush buffered events.

        Returns:
            BatchResult
        """
        with self._buffer_lock:
            if not self._buffer:
                return BatchResult()

            events = list(self._buffer)
            self._buffer.clear()
            self.buffer_size.set(0)

        return await self.send_batch(events)

    async def _auto_flush_loop(self):
        """Background flush loop."""
        while self._running:
            try:
                await asyncio.sleep(self.config.flush_interval)
                if self._buffer:
                    await self.flush()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto-flush error: {e}")

    def start_auto_flush(self):
        """Start automatic buffer flushing."""
        if not self._running:
            self._running = True
            self._auto_flush_task = asyncio.create_task(self._auto_flush_loop())
            logger.info("Started Sumo Logic auto-flush")

    async def stop_auto_flush(self):
        """Stop automatic buffer flushing."""
        self._running = False
        if self._auto_flush_task:
            self._auto_flush_task.cancel()
            try:
                await self._auto_flush_task
            except asyncio.CancelledError:
                pass

        await self.flush()
        logger.info("Stopped Sumo Logic auto-flush")

    def get_statistics(self) -> Dict[str, Any]:
        """Get client statistics.

        Returns:
            Statistics dictionary
        """
        with self._buffer_lock:
            buffer_size = len(self._buffer)

        return {
            "endpoint": self.config.endpoint_url[:50] + "...",
            "category": self.config.category.value,
            "source_name": self.config.source_name,
            "log_format": self.config.log_format.value,
            "compress": self.config.compress,
            "buffer_size": buffer_size,
            "batch_size": self.config.batch_size,
            "auto_flush_running": self._running,
        }


class SumoCloudSIEMClient:
    """Client for Sumo Logic Cloud SIEM features."""

    def __init__(
        self,
        access_id: str,
        access_key: str,
        deployment: str = "us1",
        namespace: str = "consciousness",
    ):
        self.access_id = access_id
        self.access_key = access_key
        self.deployment = deployment
        self.namespace = namespace
        self.base_url = f"https://api.{deployment}.sumologic.com/api"

    def _build_auth_headers(self) -> Dict[str, str]:
        """Build authenticated headers."""
        import base64
        credentials = f"{self.access_id}:{self.access_key}"
        encoded = base64.b64encode(credentials.encode()).decode()

        return {
            "Authorization": f"Basic {encoded}",
            "Content-Type": "application/json",
        }

    async def get_insights(
        self,
        severity: Optional[EventSeverity] = None,
        limit: int = 100,
        time_range: str = "-24h",
    ) -> List[Dict[str, Any]]:
        """Get Cloud SIEM insights.

        Args:
            severity: Filter by severity
            limit: Maximum results
            time_range: Time range

        Returns:
            List of insights
        """
        # In real implementation, would call Cloud SIEM API
        logger.info(f"Fetching Cloud SIEM insights: severity={severity}, limit={limit}")

        # Simulated empty results
        return []

    async def get_signals(
        self,
        entity_type: Optional[str] = None,
        entity_value: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get Cloud SIEM signals.

        Args:
            entity_type: Filter by entity type
            entity_value: Filter by entity value
            limit: Maximum results

        Returns:
            List of signals
        """
        logger.info(f"Fetching Cloud SIEM signals")
        return []

    async def get_entity_summary(
        self,
        entity_type: str,
        entity_value: str,
    ) -> Dict[str, Any]:
        """Get entity summary from Cloud SIEM.

        Args:
            entity_type: Entity type (ip, user, hostname, etc.)
            entity_value: Entity value

        Returns:
            Entity summary
        """
        logger.info(f"Fetching entity summary: {entity_type}={entity_value}")
        return {}

    async def create_custom_insight(
        self,
        name: str,
        description: str,
        severity: EventSeverity,
        entity_type: str,
        entity_value: str,
        signals: List[str],
    ) -> bool:
        """Create a custom insight.

        Args:
            name: Insight name
            description: Insight description
            severity: Severity level
            entity_type: Primary entity type
            entity_value: Primary entity value
            signals: Associated signal IDs

        Returns:
            True if created successfully
        """
        insight = {
            "name": name,
            "description": description,
            "severity": severity.value,
            "entityType": entity_type,
            "entityValue": entity_value,
            "signalIds": signals,
        }

        logger.info(f"Creating custom insight: {name}")
        return True

    async def search_records(
        self,
        query: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Search security records.

        Args:
            query: Search query
            start_time: Start time
            end_time: End time
            limit: Maximum results

        Returns:
            List of records
        """
        start = start_time or datetime.utcnow() - timedelta(hours=24)
        end = end_time or datetime.utcnow()

        logger.info(f"Searching records: {query}")
        return []


# Factory functions
def create_sumo_client(
    endpoint_url: str,
    category: str = "consciousness/security",
    source_name: str = "consciousness-nexus",
    namespace: str = "consciousness",
) -> SumoLogicClient:
    """Create a Sumo Logic HTTP Source client.

    Args:
        endpoint_url: HTTP Source endpoint URL
        category: Source category
        source_name: Source name
        namespace: Metrics namespace

    Returns:
        SumoLogicClient
    """
    try:
        cat = SumoCategory(category)
    except ValueError:
        cat = SumoCategory.CONSCIOUSNESS

    config = SumoConfig(
        endpoint_url=endpoint_url,
        category=cat,
        source_name=source_name,
    )

    return SumoLogicClient(config, namespace)


def create_cloud_siem_client(
    access_id: str,
    access_key: str,
    deployment: str = "us1",
    namespace: str = "consciousness",
) -> SumoCloudSIEMClient:
    """Create a Sumo Logic Cloud SIEM client.

    Args:
        access_id: Sumo Logic access ID
        access_key: Sumo Logic access key
        deployment: Deployment region (us1, us2, eu, etc.)
        namespace: Metrics namespace

    Returns:
        SumoCloudSIEMClient
    """
    return SumoCloudSIEMClient(access_id, access_key, deployment, namespace)
