"""Event Store

Event persistence and querying for observability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import threading
import logging
import json

from prometheus_client import Gauge, Counter

from .cloudevents import CloudEvent

logger = logging.getLogger(__name__)


class IndexType(str, Enum):
    """Types of event indexes."""
    TYPE = "type"
    SOURCE = "source"
    SUBJECT = "subject"
    TIME = "time"
    EXTENSION = "extension"


@dataclass
class EventIndex:
    """Index for fast event lookups."""
    name: str
    index_type: IndexType
    field: str
    data: Dict[str, List[str]] = field(default_factory=dict)  # value -> event_ids


@dataclass
class StoredEvent:
    """A stored event with metadata."""
    event: CloudEvent
    stored_at: datetime = field(default_factory=datetime.now)
    partition: str = "default"
    sequence: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def event_id(self) -> str:
        return self.event.id


@dataclass
class EventQuery:
    """Query for retrieving events."""
    event_types: Optional[List[str]] = None
    sources: Optional[List[str]] = None
    subjects: Optional[List[str]] = None
    from_time: Optional[datetime] = None
    to_time: Optional[datetime] = None
    extensions: Optional[Dict[str, Any]] = None
    data_filters: Optional[Dict[str, Any]] = None
    limit: int = 100
    offset: int = 0
    order_by: str = "time"
    order_desc: bool = True


class EventStore:
    """Stores and queries events.

    Usage:
        store = EventStore()

        # Store event
        store.store(event)

        # Query events
        events = store.query(EventQuery(
            event_types=["io.consciousness.alert.*"],
            from_time=datetime.now() - timedelta(hours=1),
        ))

        # Get by ID
        event = store.get_by_id("event-123")
    """

    def __init__(
        self,
        namespace: str = "consciousness",
        max_events: int = 100000,
        retention_days: int = 7,
    ):
        self.namespace = namespace
        self.max_events = max_events
        self.retention_days = retention_days

        self._events: Dict[str, StoredEvent] = {}
        self._sequence = 0
        self._indexes: Dict[str, EventIndex] = {}
        self._lock = threading.Lock()

        # Create default indexes
        self._create_default_indexes()

        # Metrics
        self.events_stored = Counter(
            f"{namespace}_event_store_events_total",
            "Total events stored",
            ["partition"],
        )

        self.store_size = Gauge(
            f"{namespace}_event_store_size",
            "Current store size",
        )

        self.events_expired = Counter(
            f"{namespace}_event_store_events_expired_total",
            "Events expired",
        )

        self.query_latency = Gauge(
            f"{namespace}_event_store_query_latency_seconds",
            "Query latency",
        )

    def _create_default_indexes(self):
        """Create default event indexes."""
        self._indexes["type"] = EventIndex(
            name="type_index",
            index_type=IndexType.TYPE,
            field="type",
        )
        self._indexes["source"] = EventIndex(
            name="source_index",
            index_type=IndexType.SOURCE,
            field="source",
        )
        self._indexes["subject"] = EventIndex(
            name="subject_index",
            index_type=IndexType.SUBJECT,
            field="subject",
        )

    def store(
        self,
        event: CloudEvent,
        partition: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StoredEvent:
        """Store an event.

        Args:
            event: Event to store
            partition: Storage partition
            metadata: Additional metadata

        Returns:
            StoredEvent
        """
        with self._lock:
            self._sequence += 1

            stored = StoredEvent(
                event=event,
                partition=partition,
                sequence=self._sequence,
                metadata=metadata or {},
            )

            self._events[event.id] = stored

            # Update indexes
            self._index_event(event)

            # Enforce limits
            if len(self._events) > self.max_events:
                self._evict_oldest()

        self.events_stored.labels(partition=partition).inc()
        self.store_size.set(len(self._events))

        return stored

    def _index_event(self, event: CloudEvent):
        """Add event to indexes."""
        # Type index
        if event.type:
            idx = self._indexes["type"]
            if event.type not in idx.data:
                idx.data[event.type] = []
            idx.data[event.type].append(event.id)

        # Source index
        if event.source:
            idx = self._indexes["source"]
            if event.source not in idx.data:
                idx.data[event.source] = []
            idx.data[event.source].append(event.id)

        # Subject index
        if event.subject:
            idx = self._indexes["subject"]
            if event.subject not in idx.data:
                idx.data[event.subject] = []
            idx.data[event.subject].append(event.id)

    def _evict_oldest(self):
        """Evict oldest events to maintain size limit."""
        # Sort by sequence and remove oldest
        sorted_events = sorted(
            self._events.values(),
            key=lambda e: e.sequence,
        )

        to_remove = len(self._events) - (self.max_events // 2)

        for stored in sorted_events[:to_remove]:
            self._remove_from_indexes(stored.event)
            del self._events[stored.event.id]
            self.events_expired.inc()

    def _remove_from_indexes(self, event: CloudEvent):
        """Remove event from all indexes."""
        for idx in self._indexes.values():
            for value, event_ids in idx.data.items():
                if event.id in event_ids:
                    event_ids.remove(event.id)

    def get_by_id(self, event_id: str) -> Optional[StoredEvent]:
        """Get event by ID.

        Args:
            event_id: Event ID

        Returns:
            StoredEvent or None
        """
        with self._lock:
            return self._events.get(event_id)

    def query(self, query: EventQuery) -> List[StoredEvent]:
        """Query events.

        Args:
            query: Query parameters

        Returns:
            List of matching events
        """
        import time
        start_time = time.time()

        with self._lock:
            # Start with all events or use index
            candidate_ids = self._get_candidates(query)

            # Filter candidates
            results = []
            for event_id in candidate_ids:
                stored = self._events.get(event_id)
                if stored and self._matches_query(stored, query):
                    results.append(stored)

        # Sort
        if query.order_by == "time":
            results.sort(
                key=lambda e: e.event.time or e.stored_at,
                reverse=query.order_desc,
            )
        elif query.order_by == "sequence":
            results.sort(key=lambda e: e.sequence, reverse=query.order_desc)

        # Apply offset and limit
        results = results[query.offset:query.offset + query.limit]

        duration = time.time() - start_time
        self.query_latency.set(duration)

        return results

    def _get_candidates(self, query: EventQuery) -> set:
        """Get candidate event IDs using indexes."""
        candidates = None

        # Use type index
        if query.event_types:
            type_candidates = set()
            for event_type in query.event_types:
                if event_type in self._indexes["type"].data:
                    type_candidates.update(self._indexes["type"].data[event_type])
            candidates = type_candidates

        # Use source index
        if query.sources:
            source_candidates = set()
            for source in query.sources:
                if source in self._indexes["source"].data:
                    source_candidates.update(self._indexes["source"].data[source])
            if candidates is None:
                candidates = source_candidates
            else:
                candidates &= source_candidates

        # Use subject index
        if query.subjects:
            subject_candidates = set()
            for subject in query.subjects:
                if subject in self._indexes["subject"].data:
                    subject_candidates.update(self._indexes["subject"].data[subject])
            if candidates is None:
                candidates = subject_candidates
            else:
                candidates &= subject_candidates

        # If no index used, return all
        if candidates is None:
            return set(self._events.keys())

        return candidates

    def _matches_query(self, stored: StoredEvent, query: EventQuery) -> bool:
        """Check if event matches query."""
        event = stored.event

        # Time filter
        if query.from_time:
            event_time = event.time or stored.stored_at
            if event_time < query.from_time:
                return False

        if query.to_time:
            event_time = event.time or stored.stored_at
            if event_time > query.to_time:
                return False

        # Extension filters
        if query.extensions:
            for key, expected in query.extensions.items():
                if event.extensions.get(key) != expected:
                    return False

        # Data filters
        if query.data_filters and isinstance(event.data, dict):
            for key, expected in query.data_filters.items():
                if event.data.get(key) != expected:
                    return False

        return True

    def count(self, query: Optional[EventQuery] = None) -> int:
        """Count events matching query.

        Args:
            query: Optional query

        Returns:
            Event count
        """
        if query is None:
            with self._lock:
                return len(self._events)

        # Use query with high limit
        query.limit = self.max_events
        query.offset = 0
        results = self.query(query)
        return len(results)

    def delete(self, event_id: str) -> bool:
        """Delete an event.

        Args:
            event_id: Event ID

        Returns:
            True if deleted
        """
        with self._lock:
            stored = self._events.get(event_id)
            if stored:
                self._remove_from_indexes(stored.event)
                del self._events[event_id]
                self.store_size.set(len(self._events))
                return True
        return False

    def clear(self, partition: Optional[str] = None):
        """Clear events.

        Args:
            partition: Clear only this partition
        """
        with self._lock:
            if partition:
                to_delete = [
                    eid for eid, stored in self._events.items()
                    if stored.partition == partition
                ]
                for eid in to_delete:
                    stored = self._events[eid]
                    self._remove_from_indexes(stored.event)
                    del self._events[eid]
            else:
                self._events.clear()
                for idx in self._indexes.values():
                    idx.data.clear()

        self.store_size.set(len(self._events))

    def cleanup_expired(self) -> int:
        """Remove expired events.

        Returns:
            Number of events removed
        """
        cutoff = datetime.now() - timedelta(days=self.retention_days)
        removed = 0

        with self._lock:
            to_delete = []
            for event_id, stored in self._events.items():
                event_time = stored.event.time or stored.stored_at
                if event_time < cutoff:
                    to_delete.append(event_id)

            for event_id in to_delete:
                stored = self._events[event_id]
                self._remove_from_indexes(stored.event)
                del self._events[event_id]
                removed += 1
                self.events_expired.inc()

        self.store_size.set(len(self._events))
        logger.info(f"Cleaned up {removed} expired events")

        return removed

    def get_statistics(self) -> Dict[str, Any]:
        """Get store statistics.

        Returns:
            Statistics dictionary
        """
        with self._lock:
            events = list(self._events.values())

        # By type
        by_type: Dict[str, int] = {}
        for stored in events:
            t = stored.event.type
            by_type[t] = by_type.get(t, 0) + 1

        # By partition
        by_partition: Dict[str, int] = {}
        for stored in events:
            p = stored.partition
            by_partition[p] = by_partition.get(p, 0) + 1

        # Time range
        times = [s.event.time or s.stored_at for s in events]
        oldest = min(times) if times else None
        newest = max(times) if times else None

        return {
            "total_events": len(events),
            "max_events": self.max_events,
            "utilization_percent": len(events) / self.max_events * 100,
            "by_type": by_type,
            "by_partition": by_partition,
            "oldest_event": oldest.isoformat() if oldest else None,
            "newest_event": newest.isoformat() if newest else None,
            "retention_days": self.retention_days,
            "index_count": len(self._indexes),
        }
