"""Elastic SIEM Integration

Elasticsearch SIEM integration for security event ingestion and querying.
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
from urllib.parse import urljoin

from prometheus_client import Counter, Gauge, Histogram

from .security_events import SecurityEvent, EventSeverity

logger = logging.getLogger(__name__)


class ElasticAuthType(str, Enum):
    """Elasticsearch authentication types."""
    BASIC = "basic"
    API_KEY = "api_key"
    CLOUD_ID = "cloud_id"
    BEARER = "bearer"


class IndexStrategy(str, Enum):
    """Index naming strategies."""
    DAILY = "daily"  # security-events-YYYY.MM.DD
    MONTHLY = "monthly"  # security-events-YYYY.MM
    SINGLE = "single"  # security-events


@dataclass
class ElasticConfig:
    """Elasticsearch connection configuration."""
    hosts: List[str] = field(default_factory=lambda: ["http://localhost:9200"])
    auth_type: ElasticAuthType = ElasticAuthType.BASIC
    username: Optional[str] = None
    password: Optional[str] = None
    api_key: Optional[str] = None
    api_key_id: Optional[str] = None
    cloud_id: Optional[str] = None
    bearer_token: Optional[str] = None
    index_prefix: str = "security-events"
    index_strategy: IndexStrategy = IndexStrategy.DAILY
    verify_certs: bool = True
    ca_certs: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    retry_on_timeout: bool = True
    sniff_on_start: bool = False
    sniff_on_connection_fail: bool = False

    def get_index_name(self, timestamp: Optional[datetime] = None) -> str:
        """Get index name based on strategy."""
        ts = timestamp or datetime.utcnow()

        if self.index_strategy == IndexStrategy.DAILY:
            return f"{self.index_prefix}-{ts.strftime('%Y.%m.%d')}"
        elif self.index_strategy == IndexStrategy.MONTHLY:
            return f"{self.index_prefix}-{ts.strftime('%Y.%m')}"
        else:
            return self.index_prefix


@dataclass
class BulkResult:
    """Result of a bulk indexing operation."""
    success: int = 0
    failed: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=list)
    took_ms: int = 0


@dataclass
class SearchResult:
    """Result of an Elasticsearch search."""
    total: int = 0
    hits: List[Dict[str, Any]] = field(default_factory=list)
    took_ms: int = 0
    aggregations: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertRule:
    """Elastic SIEM detection rule."""
    rule_id: str
    name: str
    description: str
    query: str
    severity: EventSeverity
    interval: str = "5m"
    enabled: bool = True
    risk_score: int = 50
    tags: List[str] = field(default_factory=list)
    threat: List[Dict[str, Any]] = field(default_factory=list)

    def to_rule_dict(self) -> Dict[str, Any]:
        """Convert to Elastic detection rule format."""
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "description": self.description,
            "type": "query",
            "query": self.query,
            "severity": self.severity.value,
            "risk_score": self.risk_score,
            "enabled": self.enabled,
            "interval": self.interval,
            "tags": self.tags,
            "threat": self.threat,
        }


class ElasticSIEMClient:
    """Client for Elastic SIEM integration.

    Usage:
        config = ElasticConfig(
            hosts=["https://elasticsearch:9200"],
            username="elastic",
            password="changeme",
        )

        client = ElasticSIEMClient(config)

        # Send event
        await client.index_event(security_event)

        # Bulk send events
        result = await client.bulk_index(events)

        # Search events
        results = await client.search(
            query="event.severity:critical",
            size=100,
        )

        # Create detection rule
        await client.create_detection_rule(rule)
    """

    def __init__(
        self,
        config: ElasticConfig,
        namespace: str = "consciousness",
    ):
        self.config = config
        self.namespace = namespace

        self._buffer: List[SecurityEvent] = []
        self._buffer_lock = threading.Lock()
        self._flush_interval = 5.0  # seconds
        self._max_buffer_size = 1000
        self._connected = False

        # Metrics
        self.events_indexed = Counter(
            f"{namespace}_siem_elastic_events_indexed_total",
            "Total events indexed to Elastic",
            ["index", "status"],
        )

        self.bulk_operations = Counter(
            f"{namespace}_siem_elastic_bulk_operations_total",
            "Total bulk operations",
            ["status"],
        )

        self.index_latency = Histogram(
            f"{namespace}_siem_elastic_index_latency_seconds",
            "Index operation latency",
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
        )

        self.buffer_size = Gauge(
            f"{namespace}_siem_elastic_buffer_size",
            "Current buffer size",
        )

        self.connection_status = Gauge(
            f"{namespace}_siem_elastic_connected",
            "Connection status (1=connected, 0=disconnected)",
        )

        self.search_latency = Histogram(
            f"{namespace}_siem_elastic_search_latency_seconds",
            "Search operation latency",
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
        )

    def _build_headers(self) -> Dict[str, str]:
        """Build request headers with authentication."""
        headers = {"Content-Type": "application/json"}

        if self.config.auth_type == ElasticAuthType.BASIC:
            import base64
            if self.config.username and self.config.password:
                credentials = f"{self.config.username}:{self.config.password}"
                encoded = base64.b64encode(credentials.encode()).decode()
                headers["Authorization"] = f"Basic {encoded}"

        elif self.config.auth_type == ElasticAuthType.API_KEY:
            if self.config.api_key_id and self.config.api_key:
                import base64
                credentials = f"{self.config.api_key_id}:{self.config.api_key}"
                encoded = base64.b64encode(credentials.encode()).decode()
                headers["Authorization"] = f"ApiKey {encoded}"

        elif self.config.auth_type == ElasticAuthType.BEARER:
            if self.config.bearer_token:
                headers["Authorization"] = f"Bearer {self.config.bearer_token}"

        return headers

    async def connect(self) -> bool:
        """Test connection to Elasticsearch.

        Returns:
            True if connected successfully
        """
        try:
            # In real implementation, would use aiohttp or elasticsearch-py
            logger.info(f"Connecting to Elastic: {self.config.hosts}")
            self._connected = True
            self.connection_status.set(1)
            return True

        except Exception as e:
            logger.error(f"Elastic connection failed: {e}")
            self._connected = False
            self.connection_status.set(0)
            return False

    async def disconnect(self):
        """Disconnect from Elasticsearch."""
        # Flush any remaining buffered events
        await self.flush()
        self._connected = False
        self.connection_status.set(0)

    def _event_to_document(self, event: SecurityEvent) -> Dict[str, Any]:
        """Convert SecurityEvent to Elasticsearch document."""
        doc = event.to_ecs()

        # Add Elastic-specific fields
        doc["ecs"] = {"version": "8.11.0"}
        doc["observer"] = {
            "type": "consciousness-nexus",
            "name": self.namespace,
            "version": "1.0.0",
        }

        return doc

    async def index_event(
        self,
        event: SecurityEvent,
        index: Optional[str] = None,
        pipeline: Optional[str] = None,
    ) -> bool:
        """Index a single security event.

        Args:
            event: Event to index
            index: Optional index name override
            pipeline: Optional ingest pipeline

        Returns:
            True if indexed successfully
        """
        import time
        start_time = time.time()

        try:
            target_index = index or self.config.get_index_name(event.timestamp)
            doc = self._event_to_document(event)

            # In real implementation, would send to Elasticsearch
            logger.debug(f"Indexing event {event.event_id} to {target_index}")

            duration = time.time() - start_time
            self.index_latency.observe(duration)
            self.events_indexed.labels(index=target_index, status="success").inc()

            return True

        except Exception as e:
            logger.error(f"Failed to index event: {e}")
            self.events_indexed.labels(
                index=index or "unknown",
                status="error",
            ).inc()
            return False

    async def bulk_index(
        self,
        events: List[SecurityEvent],
        index: Optional[str] = None,
    ) -> BulkResult:
        """Bulk index multiple events.

        Args:
            events: Events to index
            index: Optional index name override

        Returns:
            BulkResult with success/failure counts
        """
        import time
        start_time = time.time()

        result = BulkResult()

        if not events:
            return result

        try:
            # Build bulk request body
            bulk_body = []
            for event in events:
                target_index = index or self.config.get_index_name(event.timestamp)

                # Action line
                action = {
                    "index": {
                        "_index": target_index,
                        "_id": event.event_id,
                    }
                }
                bulk_body.append(json.dumps(action))

                # Document line
                doc = self._event_to_document(event)
                bulk_body.append(json.dumps(doc))

            # In real implementation, would send bulk request
            logger.info(f"Bulk indexing {len(events)} events")

            result.success = len(events)
            result.took_ms = int((time.time() - start_time) * 1000)

            self.bulk_operations.labels(status="success").inc()

            for event in events:
                target_index = index or self.config.get_index_name(event.timestamp)
                self.events_indexed.labels(index=target_index, status="success").inc()

            return result

        except Exception as e:
            logger.error(f"Bulk index failed: {e}")
            result.failed = len(events)
            result.errors.append({"error": str(e)})
            self.bulk_operations.labels(status="error").inc()
            return result

    def buffer_event(self, event: SecurityEvent):
        """Add event to buffer for batch indexing.

        Args:
            event: Event to buffer
        """
        with self._buffer_lock:
            self._buffer.append(event)
            self.buffer_size.set(len(self._buffer))

            # Auto-flush if buffer is full
            if len(self._buffer) >= self._max_buffer_size:
                asyncio.create_task(self.flush())

    async def flush(self) -> BulkResult:
        """Flush buffered events to Elasticsearch.

        Returns:
            BulkResult
        """
        with self._buffer_lock:
            if not self._buffer:
                return BulkResult()

            events = list(self._buffer)
            self._buffer.clear()
            self.buffer_size.set(0)

        return await self.bulk_index(events)

    async def search(
        self,
        query: str,
        index: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        size: int = 100,
        sort: Optional[List[Dict[str, str]]] = None,
        aggs: Optional[Dict[str, Any]] = None,
    ) -> SearchResult:
        """Search for security events.

        Args:
            query: Lucene query string
            index: Index pattern to search
            start_time: Start of time range
            end_time: End of time range
            size: Maximum results
            sort: Sort specification
            aggs: Aggregations

        Returns:
            SearchResult
        """
        import time
        search_start = time.time()

        result = SearchResult()

        try:
            target_index = index or f"{self.config.index_prefix}-*"

            # Build query body
            body: Dict[str, Any] = {
                "query": {
                    "bool": {
                        "must": [
                            {"query_string": {"query": query}},
                        ],
                    }
                },
                "size": size,
            }

            # Add time range filter
            if start_time or end_time:
                time_range: Dict[str, Any] = {}
                if start_time:
                    time_range["gte"] = start_time.isoformat()
                if end_time:
                    time_range["lte"] = end_time.isoformat()

                body["query"]["bool"]["filter"] = [
                    {"range": {"@timestamp": time_range}}
                ]

            # Add sort
            if sort:
                body["sort"] = sort
            else:
                body["sort"] = [{"@timestamp": {"order": "desc"}}]

            # Add aggregations
            if aggs:
                body["aggs"] = aggs

            # In real implementation, would execute search
            logger.debug(f"Searching {target_index}: {query}")

            result.took_ms = int((time.time() - search_start) * 1000)
            self.search_latency.observe(result.took_ms / 1000)

            return result

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return result

    async def get_event_counts(
        self,
        interval: str = "1h",
        start_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Get event counts over time.

        Args:
            interval: Time bucket interval
            start_time: Start time

        Returns:
            Aggregation results
        """
        start = start_time or datetime.utcnow() - timedelta(hours=24)

        aggs = {
            "events_over_time": {
                "date_histogram": {
                    "field": "@timestamp",
                    "fixed_interval": interval,
                }
            },
            "by_severity": {
                "terms": {"field": "event.severity"}
            },
            "by_category": {
                "terms": {"field": "event.category"}
            },
        }

        result = await self.search(
            query="*",
            start_time=start,
            size=0,
            aggs=aggs,
        )

        return result.aggregations

    async def create_detection_rule(
        self,
        rule: AlertRule,
    ) -> bool:
        """Create an Elastic SIEM detection rule.

        Args:
            rule: Detection rule to create

        Returns:
            True if created successfully
        """
        try:
            rule_dict = rule.to_rule_dict()

            # In real implementation, would call Detection Engine API
            logger.info(f"Creating detection rule: {rule.name}")

            return True

        except Exception as e:
            logger.error(f"Failed to create detection rule: {e}")
            return False

    async def get_alerts(
        self,
        status: Optional[str] = None,
        severity: Optional[EventSeverity] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get SIEM alerts.

        Args:
            status: Filter by status (open, acknowledged, closed)
            severity: Filter by severity
            limit: Maximum alerts

        Returns:
            List of alerts
        """
        query_parts = ["signal.status:*"]

        if status:
            query_parts.append(f"signal.status:{status}")

        if severity:
            query_parts.append(f"signal.rule.severity:{severity.value}")

        query = " AND ".join(query_parts)

        result = await self.search(
            query=query,
            index=".siem-signals-*",
            size=limit,
        )

        return result.hits

    async def create_index_template(self) -> bool:
        """Create index template for security events.

        Returns:
            True if created successfully
        """
        template = {
            "index_patterns": [f"{self.config.index_prefix}-*"],
            "template": {
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 1,
                    "index.lifecycle.name": "security-events-policy",
                    "index.lifecycle.rollover_alias": self.config.index_prefix,
                },
                "mappings": {
                    "properties": {
                        "@timestamp": {"type": "date"},
                        "message": {"type": "text"},
                        "tags": {"type": "keyword"},
                        "labels": {"type": "object"},
                        "event": {
                            "properties": {
                                "id": {"type": "keyword"},
                                "kind": {"type": "keyword"},
                                "category": {"type": "keyword"},
                                "type": {"type": "keyword"},
                                "outcome": {"type": "keyword"},
                                "severity": {"type": "integer"},
                                "risk_score": {"type": "float"},
                            }
                        },
                        "source": {
                            "properties": {
                                "ip": {"type": "ip"},
                                "port": {"type": "integer"},
                                "domain": {"type": "keyword"},
                                "user": {"type": "keyword"},
                            }
                        },
                        "destination": {
                            "properties": {
                                "ip": {"type": "ip"},
                                "port": {"type": "integer"},
                                "domain": {"type": "keyword"},
                            }
                        },
                        "threat": {
                            "properties": {
                                "indicator": {
                                    "properties": {
                                        "type": {"type": "keyword"},
                                        "value": {"type": "keyword"},
                                        "provider": {"type": "keyword"},
                                        "confidence": {"type": "float"},
                                    }
                                }
                            }
                        },
                        "rule": {
                            "properties": {
                                "id": {"type": "keyword"},
                                "name": {"type": "keyword"},
                            }
                        },
                    }
                },
            },
        }

        try:
            # In real implementation, would create template via API
            logger.info(f"Creating index template: {self.config.index_prefix}")
            return True

        except Exception as e:
            logger.error(f"Failed to create index template: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get client statistics.

        Returns:
            Statistics dictionary
        """
        with self._buffer_lock:
            buffer_size = len(self._buffer)

        return {
            "connected": self._connected,
            "hosts": self.config.hosts,
            "index_prefix": self.config.index_prefix,
            "index_strategy": self.config.index_strategy.value,
            "buffer_size": buffer_size,
            "max_buffer_size": self._max_buffer_size,
        }


# Factory function for common configurations
def create_elastic_cloud_client(
    cloud_id: str,
    api_key: str,
    namespace: str = "consciousness",
) -> ElasticSIEMClient:
    """Create client for Elastic Cloud.

    Args:
        cloud_id: Elastic Cloud deployment ID
        api_key: API key for authentication
        namespace: Metrics namespace

    Returns:
        ElasticSIEMClient
    """
    # Parse API key (format: id:key)
    parts = api_key.split(":")
    api_key_id = parts[0] if len(parts) > 1 else ""
    api_key_value = parts[1] if len(parts) > 1 else api_key

    config = ElasticConfig(
        cloud_id=cloud_id,
        auth_type=ElasticAuthType.API_KEY,
        api_key_id=api_key_id,
        api_key=api_key_value,
        verify_certs=True,
    )

    return ElasticSIEMClient(config, namespace)


def create_local_client(
    host: str = "http://localhost:9200",
    username: str = "elastic",
    password: str = "changeme",
    namespace: str = "consciousness",
) -> ElasticSIEMClient:
    """Create client for local Elasticsearch.

    Args:
        host: Elasticsearch host
        username: Username
        password: Password
        namespace: Metrics namespace

    Returns:
        ElasticSIEMClient
    """
    config = ElasticConfig(
        hosts=[host],
        auth_type=ElasticAuthType.BASIC,
        username=username,
        password=password,
        verify_certs=False,
    )

    return ElasticSIEMClient(config, namespace)
