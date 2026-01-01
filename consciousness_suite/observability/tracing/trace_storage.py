"""
Trace Storage - Unified Persistence Abstraction

Provides a unified trace storage layer that can persist traces to:
- In-memory store with LRU eviction
- Jaeger backend
- Tempo backend
- File-based storage

Thread-safe with indexing, querying, and Prometheus metrics.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import hashlib

from prometheus_client import Counter, Gauge, Histogram

from .jaeger_client import JaegerClient, JaegerConfig, JaegerTrace, JaegerSpan
from .tempo_client import TempoClient, TempoConfig, TempoTrace, TempoSpan

logger = logging.getLogger(__name__)


class StorageBackend(str, Enum):
    """Available storage backends."""
    MEMORY = "memory"
    JAEGER = "jaeger"
    TEMPO = "tempo"
    FILE = "file"


@dataclass
class StoredSpan:
    """Unified span representation for storage."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    service_name: str
    start_time: datetime
    duration_ms: float
    status: str = "OK"
    status_message: str = ""
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    is_error: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "operation_name": self.operation_name,
            "service_name": self.service_name,
            "start_time": self.start_time.isoformat(),
            "duration_ms": self.duration_ms,
            "status": self.status,
            "status_message": self.status_message,
            "tags": self.tags,
            "logs": self.logs,
            "is_error": self.is_error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StoredSpan:
        return cls(
            trace_id=data["trace_id"],
            span_id=data["span_id"],
            parent_span_id=data.get("parent_span_id"),
            operation_name=data["operation_name"],
            service_name=data["service_name"],
            start_time=datetime.fromisoformat(data["start_time"]),
            duration_ms=data["duration_ms"],
            status=data.get("status", "OK"),
            status_message=data.get("status_message", ""),
            tags=data.get("tags", {}),
            logs=data.get("logs", []),
            is_error=data.get("is_error", False),
        )

    @classmethod
    def from_jaeger_span(cls, span: JaegerSpan) -> StoredSpan:
        tags = {t.key: t.value for t in span.tags}
        logs = [
            {
                "timestamp": log.datetime.isoformat(),
                "fields": {f.key: f.value for f in log.fields},
            }
            for log in span.logs
        ]
        return cls(
            trace_id=span.trace_id,
            span_id=span.span_id,
            parent_span_id=span.parent_span_id,
            operation_name=span.operation_name,
            service_name=span.process.service_name,
            start_time=span.start_datetime,
            duration_ms=span.duration_ms,
            status="ERROR" if span.is_error else "OK",
            tags=tags,
            logs=logs,
            is_error=span.is_error,
        )

    @classmethod
    def from_tempo_span(cls, span: TempoSpan) -> StoredSpan:
        tags = {a.key: a.value for a in span.attributes}
        logs = [
            {
                "timestamp": e.datetime.isoformat(),
                "name": e.name,
                "fields": {a.key: a.value for a in e.attributes},
            }
            for e in span.events
        ]
        return cls(
            trace_id=span.trace_id,
            span_id=span.span_id,
            parent_span_id=span.parent_span_id,
            operation_name=span.name,
            service_name=span.service_name,
            start_time=span.start_time,
            duration_ms=span.duration_ms,
            status=span.status.value,
            status_message=span.status_message,
            tags=tags,
            logs=logs,
            is_error=span.is_error,
        )


@dataclass
class StoredTrace:
    """Unified trace representation for storage."""
    trace_id: str
    spans: List[StoredSpan] = field(default_factory=list)
    stored_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def root_span(self) -> Optional[StoredSpan]:
        for span in self.spans:
            if span.parent_span_id is None:
                return span
        return self.spans[0] if self.spans else None

    @property
    def service_name(self) -> str:
        root = self.root_span
        return root.service_name if root else "unknown"

    @property
    def operation_name(self) -> str:
        root = self.root_span
        return root.operation_name if root else "unknown"

    @property
    def start_time(self) -> Optional[datetime]:
        if not self.spans:
            return None
        return min(s.start_time for s in self.spans)

    @property
    def duration_ms(self) -> float:
        if not self.spans:
            return 0.0
        start = min(s.start_time for s in self.spans)
        end_times = [s.start_time.timestamp() * 1000 + s.duration_ms for s in self.spans]
        end = max(end_times)
        return end - start.timestamp() * 1000

    @property
    def span_count(self) -> int:
        return len(self.spans)

    @property
    def error_count(self) -> int:
        return sum(1 for s in self.spans if s.is_error)

    @property
    def has_errors(self) -> bool:
        return self.error_count > 0

    def get_spans_by_service(self, service: str) -> List[StoredSpan]:
        return [s for s in self.spans if s.service_name == service]

    def get_spans_by_operation(self, operation: str) -> List[StoredSpan]:
        return [s for s in self.spans if s.operation_name == operation]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "spans": [s.to_dict() for s in self.spans],
            "stored_at": self.stored_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StoredTrace:
        return cls(
            trace_id=data["trace_id"],
            spans=[StoredSpan.from_dict(s) for s in data.get("spans", [])],
            stored_at=datetime.fromisoformat(data.get("stored_at", datetime.now().isoformat())),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_jaeger_trace(cls, trace: JaegerTrace) -> StoredTrace:
        return cls(
            trace_id=trace.trace_id,
            spans=[StoredSpan.from_jaeger_span(s) for s in trace.spans],
        )

    @classmethod
    def from_tempo_trace(cls, trace: TempoTrace) -> StoredTrace:
        return cls(
            trace_id=trace.trace_id,
            spans=[StoredSpan.from_tempo_span(s) for s in trace.spans],
        )


@dataclass
class TraceQuery:
    """Query for retrieving traces."""
    services: Optional[List[str]] = None
    operations: Optional[List[str]] = None
    trace_ids: Optional[List[str]] = None
    tags: Optional[Dict[str, Any]] = None
    from_time: Optional[datetime] = None
    to_time: Optional[datetime] = None
    min_duration_ms: Optional[float] = None
    max_duration_ms: Optional[float] = None
    has_errors: Optional[bool] = None
    limit: int = 100
    offset: int = 0
    order_by: str = "start_time"
    order_desc: bool = True


@dataclass
class TraceIndex:
    """Index for fast trace lookups."""
    name: str
    field: str
    data: Dict[str, List[str]] = field(default_factory=dict)  # value -> trace_ids


@dataclass
class TraceStorageConfig:
    """Trace storage configuration."""
    backend: StorageBackend = StorageBackend.MEMORY
    max_traces: int = 10000
    retention_hours: int = 24
    jaeger_config: Optional[JaegerConfig] = None
    tempo_config: Optional[TempoConfig] = None
    file_path: Optional[str] = None
    namespace: str = "consciousness"
    enable_indexing: bool = True
    index_fields: List[str] = field(default_factory=lambda: ["service", "operation", "has_errors"])


class TraceStorageBackend(ABC):
    """Abstract base for trace storage backends."""

    @abstractmethod
    def store(self, trace: StoredTrace) -> bool:
        """Store a trace."""
        pass

    @abstractmethod
    def get(self, trace_id: str) -> Optional[StoredTrace]:
        """Get trace by ID."""
        pass

    @abstractmethod
    def query(self, query: TraceQuery) -> List[StoredTrace]:
        """Query traces."""
        pass

    @abstractmethod
    def delete(self, trace_id: str) -> bool:
        """Delete a trace."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Get total trace count."""
        pass

    @abstractmethod
    def cleanup_expired(self, cutoff: datetime) -> int:
        """Remove traces older than cutoff."""
        pass


class MemoryStorageBackend(TraceStorageBackend):
    """In-memory trace storage with LRU eviction."""

    def __init__(self, max_traces: int = 10000, enable_indexing: bool = True):
        self.max_traces = max_traces
        self.enable_indexing = enable_indexing
        self._traces: Dict[str, StoredTrace] = {}
        self._access_order: List[str] = []  # LRU tracking
        self._indexes: Dict[str, TraceIndex] = {}
        self._lock = threading.Lock()

        if enable_indexing:
            self._create_indexes()

    def _create_indexes(self):
        """Create default indexes."""
        self._indexes["service"] = TraceIndex(name="service_index", field="service_name")
        self._indexes["operation"] = TraceIndex(name="operation_index", field="operation_name")
        self._indexes["has_errors"] = TraceIndex(name="error_index", field="has_errors")

    def _update_indexes(self, trace: StoredTrace):
        """Update indexes with trace data."""
        if not self.enable_indexing:
            return

        # Service index
        service = trace.service_name
        idx = self._indexes["service"]
        if service not in idx.data:
            idx.data[service] = []
        if trace.trace_id not in idx.data[service]:
            idx.data[service].append(trace.trace_id)

        # Operation index
        operation = trace.operation_name
        idx = self._indexes["operation"]
        if operation not in idx.data:
            idx.data[operation] = []
        if trace.trace_id not in idx.data[operation]:
            idx.data[operation].append(trace.trace_id)

        # Error index
        error_key = "true" if trace.has_errors else "false"
        idx = self._indexes["has_errors"]
        if error_key not in idx.data:
            idx.data[error_key] = []
        if trace.trace_id not in idx.data[error_key]:
            idx.data[error_key].append(trace.trace_id)

    def _remove_from_indexes(self, trace: StoredTrace):
        """Remove trace from indexes."""
        if not self.enable_indexing:
            return

        for idx in self._indexes.values():
            for trace_ids in idx.data.values():
                if trace.trace_id in trace_ids:
                    trace_ids.remove(trace.trace_id)

    def _evict_lru(self):
        """Evict least recently used traces."""
        while len(self._traces) > self.max_traces:
            if not self._access_order:
                break
            lru_id = self._access_order.pop(0)
            if lru_id in self._traces:
                trace = self._traces.pop(lru_id)
                self._remove_from_indexes(trace)

    def store(self, trace: StoredTrace) -> bool:
        with self._lock:
            self._traces[trace.trace_id] = trace
            if trace.trace_id in self._access_order:
                self._access_order.remove(trace.trace_id)
            self._access_order.append(trace.trace_id)
            self._update_indexes(trace)
            self._evict_lru()
        return True

    def get(self, trace_id: str) -> Optional[StoredTrace]:
        with self._lock:
            trace = self._traces.get(trace_id)
            if trace and trace_id in self._access_order:
                # Update access order (LRU)
                self._access_order.remove(trace_id)
                self._access_order.append(trace_id)
            return trace

    def query(self, query: TraceQuery) -> List[StoredTrace]:
        with self._lock:
            candidates = self._get_candidates(query)
            results = []

            for trace_id in candidates:
                trace = self._traces.get(trace_id)
                if trace and self._matches_query(trace, query):
                    results.append(trace)

        # Sort
        if query.order_by == "start_time":
            results.sort(key=lambda t: t.start_time or datetime.min, reverse=query.order_desc)
        elif query.order_by == "duration":
            results.sort(key=lambda t: t.duration_ms, reverse=query.order_desc)

        # Apply offset and limit
        return results[query.offset:query.offset + query.limit]

    def _get_candidates(self, query: TraceQuery) -> set:
        """Get candidate trace IDs using indexes."""
        if query.trace_ids:
            return set(query.trace_ids)

        candidates = None

        # Use service index
        if query.services and self.enable_indexing:
            service_candidates = set()
            for service in query.services:
                if service in self._indexes["service"].data:
                    service_candidates.update(self._indexes["service"].data[service])
            candidates = service_candidates

        # Use operation index
        if query.operations and self.enable_indexing:
            op_candidates = set()
            for op in query.operations:
                if op in self._indexes["operation"].data:
                    op_candidates.update(self._indexes["operation"].data[op])
            if candidates is None:
                candidates = op_candidates
            else:
                candidates &= op_candidates

        # Use error index
        if query.has_errors is not None and self.enable_indexing:
            error_key = "true" if query.has_errors else "false"
            error_candidates = set(self._indexes["has_errors"].data.get(error_key, []))
            if candidates is None:
                candidates = error_candidates
            else:
                candidates &= error_candidates

        if candidates is None:
            return set(self._traces.keys())

        return candidates

    def _matches_query(self, trace: StoredTrace, query: TraceQuery) -> bool:
        """Check if trace matches query criteria."""
        # Time filters
        if query.from_time and trace.start_time and trace.start_time < query.from_time:
            return False
        if query.to_time and trace.start_time and trace.start_time > query.to_time:
            return False

        # Duration filters
        if query.min_duration_ms and trace.duration_ms < query.min_duration_ms:
            return False
        if query.max_duration_ms and trace.duration_ms > query.max_duration_ms:
            return False

        # Tag filters
        if query.tags:
            root = trace.root_span
            if root:
                for key, value in query.tags.items():
                    if root.tags.get(key) != value:
                        return False

        return True

    def delete(self, trace_id: str) -> bool:
        with self._lock:
            if trace_id in self._traces:
                trace = self._traces.pop(trace_id)
                self._remove_from_indexes(trace)
                if trace_id in self._access_order:
                    self._access_order.remove(trace_id)
                return True
        return False

    def count(self) -> int:
        with self._lock:
            return len(self._traces)

    def cleanup_expired(self, cutoff: datetime) -> int:
        removed = 0
        with self._lock:
            to_remove = []
            for trace_id, trace in self._traces.items():
                if trace.stored_at < cutoff:
                    to_remove.append(trace_id)

            for trace_id in to_remove:
                trace = self._traces.pop(trace_id)
                self._remove_from_indexes(trace)
                if trace_id in self._access_order:
                    self._access_order.remove(trace_id)
                removed += 1

        return removed


class FileStorageBackend(TraceStorageBackend):
    """File-based trace storage with JSON persistence."""

    def __init__(self, file_path: str, max_traces: int = 10000):
        self.file_path = Path(file_path)
        self.max_traces = max_traces
        self._memory = MemoryStorageBackend(max_traces=max_traces)
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        """Load traces from file."""
        if self.file_path.exists():
            try:
                with open(self.file_path, "r") as f:
                    data = json.load(f)
                    for trace_data in data.get("traces", []):
                        trace = StoredTrace.from_dict(trace_data)
                        self._memory.store(trace)
                logger.info(f"Loaded {self._memory.count()} traces from {self.file_path}")
            except Exception as e:
                logger.warning(f"Failed to load traces from file: {e}")

    def _save(self):
        """Save traces to file."""
        with self._lock:
            try:
                self.file_path.parent.mkdir(parents=True, exist_ok=True)
                traces = self._memory.query(TraceQuery(limit=self.max_traces))
                data = {"traces": [t.to_dict() for t in traces]}
                with open(self.file_path, "w") as f:
                    json.dump(data, f)
            except Exception as e:
                logger.warning(f"Failed to save traces to file: {e}")

    def store(self, trace: StoredTrace) -> bool:
        result = self._memory.store(trace)
        self._save()
        return result

    def get(self, trace_id: str) -> Optional[StoredTrace]:
        return self._memory.get(trace_id)

    def query(self, query: TraceQuery) -> List[StoredTrace]:
        return self._memory.query(query)

    def delete(self, trace_id: str) -> bool:
        result = self._memory.delete(trace_id)
        if result:
            self._save()
        return result

    def count(self) -> int:
        return self._memory.count()

    def cleanup_expired(self, cutoff: datetime) -> int:
        removed = self._memory.cleanup_expired(cutoff)
        if removed > 0:
            self._save()
        return removed


class TraceStorage:
    """Unified trace storage with multiple backend support.

    Thread-safe storage abstraction with indexing, querying,
    and Prometheus metrics instrumentation.

    Usage:
        config = TraceStorageConfig(
            backend=StorageBackend.MEMORY,
            max_traces=10000,
        )
        storage = TraceStorage(config)

        # Store trace
        storage.store(trace)

        # Query traces
        traces = storage.query(TraceQuery(
            services=["api-gateway"],
            has_errors=True,
            limit=20,
        ))

        # Get by ID
        trace = storage.get("abc123")

        # Fetch from Jaeger and store
        storage.fetch_and_store_from_jaeger("abc123")
    """

    def __init__(self, config: Optional[TraceStorageConfig] = None):
        self.config = config or TraceStorageConfig()
        self._backend: TraceStorageBackend
        self._jaeger_client: Optional[JaegerClient] = None
        self._tempo_client: Optional[TempoClient] = None
        self._lock = threading.Lock()

        # Initialize backend
        self._init_backend()

        # Initialize clients
        if self.config.jaeger_config:
            self._jaeger_client = JaegerClient(self.config.jaeger_config)
        if self.config.tempo_config:
            self._tempo_client = TempoClient(self.config.tempo_config)

        # Metrics
        ns = self.config.namespace
        self.traces_stored = Counter(
            f"{ns}_trace_storage_stored_total",
            "Total traces stored",
        )
        self.traces_fetched = Counter(
            f"{ns}_trace_storage_fetched_total",
            "Total traces fetched",
        )
        self.storage_size = Gauge(
            f"{ns}_trace_storage_size",
            "Current storage size",
        )
        self.query_latency = Histogram(
            f"{ns}_trace_storage_query_latency_seconds",
            "Query latency",
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
        )
        self.traces_expired = Counter(
            f"{ns}_trace_storage_expired_total",
            "Traces expired",
        )

    def _init_backend(self):
        """Initialize storage backend."""
        if self.config.backend == StorageBackend.MEMORY:
            self._backend = MemoryStorageBackend(
                max_traces=self.config.max_traces,
                enable_indexing=self.config.enable_indexing,
            )
        elif self.config.backend == StorageBackend.FILE:
            if not self.config.file_path:
                raise ValueError("file_path required for FILE backend")
            self._backend = FileStorageBackend(
                file_path=self.config.file_path,
                max_traces=self.config.max_traces,
            )
        else:
            # Default to memory for JAEGER/TEMPO (they are query-through)
            self._backend = MemoryStorageBackend(
                max_traces=self.config.max_traces,
                enable_indexing=self.config.enable_indexing,
            )

    def store(self, trace: StoredTrace) -> bool:
        """Store a trace.

        Args:
            trace: Trace to store

        Returns:
            True if stored successfully
        """
        result = self._backend.store(trace)
        if result:
            self.traces_stored.inc()
            self.storage_size.set(self._backend.count())
        return result

    def get(self, trace_id: str) -> Optional[StoredTrace]:
        """Get trace by ID.

        Checks local storage first, then queries backends.

        Args:
            trace_id: Trace ID

        Returns:
            StoredTrace or None
        """
        # Check local storage first
        trace = self._backend.get(trace_id)
        if trace:
            return trace

        # Try Jaeger
        if self._jaeger_client:
            jaeger_trace = self._jaeger_client.get_trace(trace_id)
            if jaeger_trace:
                trace = StoredTrace.from_jaeger_trace(jaeger_trace)
                self._backend.store(trace)
                self.traces_fetched.inc()
                return trace

        # Try Tempo
        if self._tempo_client:
            tempo_trace = self._tempo_client.get_trace(trace_id)
            if tempo_trace:
                trace = StoredTrace.from_tempo_trace(tempo_trace)
                self._backend.store(trace)
                self.traces_fetched.inc()
                return trace

        return None

    def query(self, query: TraceQuery) -> List[StoredTrace]:
        """Query traces.

        Args:
            query: Query parameters

        Returns:
            List of matching traces
        """
        start_time = time.time()
        results = self._backend.query(query)
        self.query_latency.observe(time.time() - start_time)
        return results

    def delete(self, trace_id: str) -> bool:
        """Delete a trace.

        Args:
            trace_id: Trace ID

        Returns:
            True if deleted
        """
        result = self._backend.delete(trace_id)
        if result:
            self.storage_size.set(self._backend.count())
        return result

    def count(self) -> int:
        """Get total trace count."""
        return self._backend.count()

    def fetch_and_store_from_jaeger(
        self,
        service: str,
        operation: Optional[str] = None,
        limit: int = 100,
    ) -> int:
        """Fetch traces from Jaeger and store locally.

        Args:
            service: Service name
            operation: Operation name filter
            limit: Maximum traces

        Returns:
            Number of traces stored
        """
        if not self._jaeger_client:
            logger.warning("Jaeger client not configured")
            return 0

        traces = self._jaeger_client.search_traces(
            service=service,
            operation=operation,
            limit=limit,
        )

        stored = 0
        for jaeger_trace in traces:
            trace = StoredTrace.from_jaeger_trace(jaeger_trace)
            if self._backend.store(trace):
                stored += 1

        self.traces_fetched.inc(stored)
        self.storage_size.set(self._backend.count())
        return stored

    def fetch_and_store_from_tempo(
        self,
        query: str,
        limit: int = 100,
    ) -> int:
        """Fetch traces from Tempo using TraceQL and store locally.

        Args:
            query: TraceQL query
            limit: Maximum traces

        Returns:
            Number of traces stored
        """
        if not self._tempo_client:
            logger.warning("Tempo client not configured")
            return 0

        result = self._tempo_client.search_traceql(query=query, limit=limit)

        stored = 0
        for tempo_trace in result.traces:
            # Fetch full trace if search only returned summary
            if len(tempo_trace.spans) <= 1:
                full_trace = self._tempo_client.get_trace(tempo_trace.trace_id)
                if full_trace:
                    tempo_trace = full_trace

            trace = StoredTrace.from_tempo_trace(tempo_trace)
            if self._backend.store(trace):
                stored += 1

        self.traces_fetched.inc(stored)
        self.storage_size.set(self._backend.count())
        return stored

    def cleanup_expired(self) -> int:
        """Remove expired traces based on retention.

        Returns:
            Number of traces removed
        """
        cutoff = datetime.now() - timedelta(hours=self.config.retention_hours)
        removed = self._backend.cleanup_expired(cutoff)
        self.traces_expired.inc(removed)
        self.storage_size.set(self._backend.count())
        logger.info(f"Cleaned up {removed} expired traces")
        return removed

    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics.

        Returns:
            Statistics dictionary
        """
        traces = self._backend.query(TraceQuery(limit=self.config.max_traces))

        by_service: Dict[str, int] = {}
        by_operation: Dict[str, int] = {}
        error_count = 0
        total_duration = 0.0

        for trace in traces:
            svc = trace.service_name
            by_service[svc] = by_service.get(svc, 0) + 1

            op = trace.operation_name
            by_operation[op] = by_operation.get(op, 0) + 1

            if trace.has_errors:
                error_count += 1

            total_duration += trace.duration_ms

        return {
            "total_traces": len(traces),
            "max_traces": self.config.max_traces,
            "utilization_percent": len(traces) / self.config.max_traces * 100,
            "by_service": by_service,
            "by_operation": by_operation,
            "error_count": error_count,
            "error_rate": error_count / len(traces) if traces else 0,
            "average_duration_ms": total_duration / len(traces) if traces else 0,
            "retention_hours": self.config.retention_hours,
            "backend": self.config.backend.value,
        }

    def close(self):
        """Close storage and clients."""
        if self._jaeger_client:
            self._jaeger_client.close()
        if self._tempo_client:
            self._tempo_client.close()
