"""Splunk HEC Integration

Splunk HTTP Event Collector (HEC) integration for security event ingestion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
from enum import Enum
import json
import threading
import logging
import asyncio
import time
import hashlib

from prometheus_client import Counter, Gauge, Histogram

from .security_events import SecurityEvent, EventSeverity

logger = logging.getLogger(__name__)


class HECEndpoint(str, Enum):
    """HEC endpoint types."""
    RAW = "/services/collector/raw"
    EVENT = "/services/collector/event"
    HEALTH = "/services/collector/health"


class SourceType(str, Enum):
    """Splunk source types for security events."""
    SYSLOG = "syslog"
    JSON = "_json"
    CEF = "cef"
    SECURITY = "security:event"
    ACCESS_COMBINED = "access_combined"
    CONSCIOUSNESS = "consciousness:security"


@dataclass
class HECConfig:
    """Splunk HEC configuration."""
    url: str
    token: str
    index: str = "main"
    source: str = "consciousness-nexus"
    sourcetype: SourceType = SourceType.CONSCIOUSNESS
    host: Optional[str] = None
    verify_ssl: bool = True
    timeout: int = 30
    max_retries: int = 3
    batch_size: int = 100
    flush_interval: float = 5.0
    use_raw_endpoint: bool = False
    channel: Optional[str] = None  # Indexer acknowledgment channel
    enable_ack: bool = False

    def get_endpoint(self) -> str:
        """Get the appropriate HEC endpoint URL."""
        base = self.url.rstrip("/")
        endpoint = HECEndpoint.RAW if self.use_raw_endpoint else HECEndpoint.EVENT
        return f"{base}{endpoint.value}"


@dataclass
class HECResponse:
    """Response from HEC endpoint."""
    success: bool
    text: str = ""
    code: int = 0
    ack_id: Optional[int] = None
    invalid_event_number: Optional[int] = None


@dataclass
class BatchResult:
    """Result of a batch send operation."""
    total: int = 0
    success: int = 0
    failed: int = 0
    ack_ids: List[int] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    duration_ms: int = 0


@dataclass
class HECEvent:
    """A Splunk HEC event."""
    event: Dict[str, Any]
    time: Optional[float] = None
    host: Optional[str] = None
    source: Optional[str] = None
    sourcetype: Optional[str] = None
    index: Optional[str] = None
    fields: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to HEC event format."""
        result: Dict[str, Any] = {"event": self.event}

        if self.time:
            result["time"] = self.time

        if self.host:
            result["host"] = self.host

        if self.source:
            result["source"] = self.source

        if self.sourcetype:
            result["sourcetype"] = self.sourcetype

        if self.index:
            result["index"] = self.index

        if self.fields:
            result["fields"] = self.fields

        return result


class SplunkHECClient:
    """Client for Splunk HTTP Event Collector.

    Usage:
        config = HECConfig(
            url="https://splunk-hec.example.com:8088",
            token="your-hec-token",
            index="security",
        )

        client = SplunkHECClient(config)

        # Send single event
        await client.send_event(security_event)

        # Buffer events for batch sending
        client.buffer_event(event1)
        client.buffer_event(event2)
        await client.flush()

        # Start background flusher
        client.start_auto_flush()
    """

    def __init__(
        self,
        config: HECConfig,
        namespace: str = "consciousness",
    ):
        self.config = config
        self.namespace = namespace

        self._buffer: List[SecurityEvent] = []
        self._buffer_lock = threading.Lock()
        self._auto_flush_task: Optional[asyncio.Task] = None
        self._running = False
        self._pending_acks: Dict[int, List[SecurityEvent]] = {}

        # Metrics
        self.events_sent = Counter(
            f"{namespace}_siem_splunk_events_sent_total",
            "Total events sent to Splunk",
            ["index", "status"],
        )

        self.batch_operations = Counter(
            f"{namespace}_siem_splunk_batch_operations_total",
            "Total batch operations",
            ["status"],
        )

        self.send_latency = Histogram(
            f"{namespace}_siem_splunk_send_latency_seconds",
            "Event send latency",
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
        )

        self.buffer_size = Gauge(
            f"{namespace}_siem_splunk_buffer_size",
            "Current buffer size",
        )

        self.connection_status = Gauge(
            f"{namespace}_siem_splunk_connected",
            "Connection status (1=healthy, 0=unhealthy)",
        )

        self.ack_pending = Gauge(
            f"{namespace}_siem_splunk_ack_pending",
            "Pending acknowledgments",
        )

    def _build_headers(self) -> Dict[str, str]:
        """Build request headers."""
        headers = {
            "Authorization": f"Splunk {self.config.token}",
            "Content-Type": "application/json",
        }

        if self.config.channel:
            headers["X-Splunk-Request-Channel"] = self.config.channel

        return headers

    def _event_to_hec(self, event: SecurityEvent) -> HECEvent:
        """Convert SecurityEvent to HEC format."""
        # Build event data in CIM-compatible format
        event_data = {
            "timestamp": event.timestamp.isoformat(),
            "event_id": event.event_id,
            "event_kind": event.kind.value,
            "event_category": event.category.value,
            "event_severity": event.severity.value,
            "event_outcome": event.outcome.value,
            "action": event.action,
            "message": event.message,
            "risk_score": event.risk_score,
            "tags": event.tags,
        }

        # Add source fields
        if event.source:
            source_dict = event.source.to_dict()
            if source_dict.get("ip"):
                event_data["src_ip"] = source_dict["ip"]
            if source_dict.get("port"):
                event_data["src_port"] = source_dict["port"]
            if source_dict.get("user"):
                event_data["src_user"] = source_dict["user"]
            if source_dict.get("domain"):
                event_data["src_domain"] = source_dict["domain"]

        # Add destination fields
        if event.destination:
            dest_dict = event.destination.to_dict()
            if dest_dict.get("ip"):
                event_data["dest_ip"] = dest_dict["ip"]
            if dest_dict.get("port"):
                event_data["dest_port"] = dest_dict["port"]
            if dest_dict.get("user"):
                event_data["dest_user"] = dest_dict["user"]

        # Add threat intel
        if event.threat:
            event_data["threat_indicator"] = event.threat.to_dict()

        # Add rule info
        if event.rule_id:
            event_data["rule_id"] = event.rule_id
            event_data["rule_name"] = event.rule_name

        # Add metadata
        if event.metadata:
            event_data["metadata"] = event.metadata

        # Build HEC event
        return HECEvent(
            event=event_data,
            time=event.timestamp.timestamp(),
            host=self.config.host,
            source=self.config.source,
            sourcetype=self.config.sourcetype.value,
            index=self.config.index,
            fields={
                "severity": event.severity.value,
                "category": event.category.value,
            },
        )

    async def check_health(self) -> bool:
        """Check HEC endpoint health.

        Returns:
            True if healthy
        """
        try:
            # In real implementation, would check /services/collector/health
            health_url = f"{self.config.url.rstrip('/')}{HECEndpoint.HEALTH.value}"
            logger.debug(f"Checking HEC health: {health_url}")

            # Simulated health check
            self.connection_status.set(1)
            return True

        except Exception as e:
            logger.error(f"HEC health check failed: {e}")
            self.connection_status.set(0)
            return False

    async def send_event(
        self,
        event: SecurityEvent,
        wait_for_ack: bool = False,
    ) -> HECResponse:
        """Send a single security event.

        Args:
            event: Event to send
            wait_for_ack: Wait for indexer acknowledgment

        Returns:
            HECResponse
        """
        start_time = time.time()

        try:
            hec_event = self._event_to_hec(event)
            payload = json.dumps(hec_event.to_dict())

            # In real implementation, would POST to HEC endpoint
            endpoint = self.config.get_endpoint()
            headers = self._build_headers()

            logger.debug(f"Sending event {event.event_id} to {endpoint}")

            # Simulated successful response
            response = HECResponse(
                success=True,
                text='{"text":"Success","code":0}',
                code=0,
            )

            duration = time.time() - start_time
            self.send_latency.observe(duration)
            self.events_sent.labels(
                index=self.config.index,
                status="success",
            ).inc()

            return response

        except Exception as e:
            logger.error(f"Failed to send event: {e}")
            self.events_sent.labels(
                index=self.config.index,
                status="error",
            ).inc()

            return HECResponse(
                success=False,
                text=str(e),
                code=-1,
            )

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
            # Build batch payload (newline-delimited JSON)
            lines = []
            for event in events:
                hec_event = self._event_to_hec(event)
                lines.append(json.dumps(hec_event.to_dict()))

            payload = "\n".join(lines)

            # In real implementation, would POST batch to HEC
            endpoint = self.config.get_endpoint()
            logger.info(f"Sending batch of {len(events)} events to {endpoint}")

            # Simulated success
            result.success = len(events)
            result.duration_ms = int((time.time() - start_time) * 1000)

            self.batch_operations.labels(status="success").inc()

            for event in events:
                self.events_sent.labels(
                    index=self.config.index,
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

            # Auto-flush if buffer is full
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
            logger.info("Started Splunk HEC auto-flush")

    async def stop_auto_flush(self):
        """Stop automatic buffer flushing."""
        self._running = False
        if self._auto_flush_task:
            self._auto_flush_task.cancel()
            try:
                await self._auto_flush_task
            except asyncio.CancelledError:
                pass

        # Final flush
        await self.flush()
        logger.info("Stopped Splunk HEC auto-flush")

    async def check_acks(self) -> Dict[int, bool]:
        """Check status of pending acknowledgments.

        Returns:
            Dict mapping ack_id to success status
        """
        if not self.config.enable_ack or not self._pending_acks:
            return {}

        # In real implementation, would check /services/collector/ack
        results = {}
        for ack_id in list(self._pending_acks.keys()):
            # Simulated ack check
            results[ack_id] = True
            del self._pending_acks[ack_id]

        self.ack_pending.set(len(self._pending_acks))
        return results

    def send_raw(self, data: str) -> HECResponse:
        """Send raw data to HEC.

        Args:
            data: Raw data string

        Returns:
            HECResponse
        """
        try:
            endpoint = f"{self.config.url.rstrip('/')}{HECEndpoint.RAW.value}"
            headers = self._build_headers()
            headers["Content-Type"] = "text/plain"

            # In real implementation, would POST to raw endpoint
            logger.debug(f"Sending raw data to {endpoint}")

            return HECResponse(success=True, code=0)

        except Exception as e:
            logger.error(f"Raw send failed: {e}")
            return HECResponse(success=False, text=str(e), code=-1)

    def send_cef(self, event: SecurityEvent) -> HECResponse:
        """Send event in CEF format.

        Args:
            event: Event to send

        Returns:
            HECResponse
        """
        cef_string = event.to_cef()
        return self.send_raw(cef_string)

    def get_statistics(self) -> Dict[str, Any]:
        """Get client statistics.

        Returns:
            Statistics dictionary
        """
        with self._buffer_lock:
            buffer_size = len(self._buffer)

        return {
            "url": self.config.url,
            "index": self.config.index,
            "source": self.config.source,
            "sourcetype": self.config.sourcetype.value,
            "buffer_size": buffer_size,
            "batch_size": self.config.batch_size,
            "flush_interval": self.config.flush_interval,
            "auto_flush_running": self._running,
            "pending_acks": len(self._pending_acks),
            "enable_ack": self.config.enable_ack,
        }


class SplunkSearchClient:
    """Client for Splunk Search API (for querying events)."""

    def __init__(
        self,
        base_url: str,
        token: str,
        namespace: str = "consciousness",
    ):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.namespace = namespace

    def _build_headers(self) -> Dict[str, str]:
        """Build request headers."""
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    async def search(
        self,
        query: str,
        earliest_time: str = "-24h",
        latest_time: str = "now",
        max_results: int = 100,
    ) -> List[Dict[str, Any]]:
        """Execute a Splunk search.

        Args:
            query: SPL search query
            earliest_time: Start time
            latest_time: End time
            max_results: Maximum results

        Returns:
            List of search results
        """
        # In real implementation, would use Splunk REST API
        search_query = f"search {query} | head {max_results}"
        logger.info(f"Executing Splunk search: {search_query}")

        # Simulated empty results
        return []

    async def get_notable_events(
        self,
        severity: Optional[EventSeverity] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get notable events from Splunk ES.

        Args:
            severity: Filter by severity
            limit: Maximum results

        Returns:
            List of notable events
        """
        query = 'index="notable"'
        if severity:
            query += f' severity="{severity.value}"'

        return await self.search(query, max_results=limit)


# Factory functions
def create_hec_client(
    url: str,
    token: str,
    index: str = "main",
    sourcetype: str = "consciousness:security",
    namespace: str = "consciousness",
) -> SplunkHECClient:
    """Create a Splunk HEC client.

    Args:
        url: HEC endpoint URL
        token: HEC token
        index: Target index
        sourcetype: Source type
        namespace: Metrics namespace

    Returns:
        SplunkHECClient
    """
    config = HECConfig(
        url=url,
        token=token,
        index=index,
        sourcetype=SourceType(sourcetype) if sourcetype in [s.value for s in SourceType] else SourceType.CONSCIOUSNESS,
    )

    return SplunkHECClient(config, namespace)


def create_cloud_hec_client(
    stack_name: str,
    token: str,
    region: str = "us",
    namespace: str = "consciousness",
) -> SplunkHECClient:
    """Create a Splunk Cloud HEC client.

    Args:
        stack_name: Splunk Cloud stack name
        token: HEC token
        region: Cloud region
        namespace: Metrics namespace

    Returns:
        SplunkHECClient
    """
    url = f"https://http-inputs-{stack_name}.splunkcloud.com:443"

    config = HECConfig(
        url=url,
        token=token,
        index="main",
        verify_ssl=True,
    )

    return SplunkHECClient(config, namespace)
