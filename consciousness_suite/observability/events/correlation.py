"""Event-Trace Correlation

Correlates events with distributed traces for unified observability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import threading
import logging
import re

from prometheus_client import Counter, Gauge, Histogram

from .cloudevents import CloudEvent

logger = logging.getLogger(__name__)


class CorrelationType(str, Enum):
    """Types of correlation."""
    TRACE = "trace"
    SPAN = "span"
    TRANSACTION = "transaction"
    SESSION = "session"
    DEPLOYMENT = "deployment"
    INCIDENT = "incident"
    CUSTOM = "custom"


@dataclass
class TraceEventLink:
    """Link between a trace and an event."""
    trace_id: str
    span_id: Optional[str] = None
    event_id: str = ""
    event_type: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    relationship: str = "caused_by"  # caused_by, follows, parent_of
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "relationship": self.relationship,
            "metadata": self.metadata,
        }


@dataclass
class CorrelationRule:
    """Rule for correlating events."""
    rule_id: str
    name: str
    correlation_type: CorrelationType
    event_types: List[str] = field(default_factory=list)
    time_window_seconds: int = 300
    key_extractor: Optional[Callable[[CloudEvent], str]] = None
    match_fn: Optional[Callable[[CloudEvent, CloudEvent], bool]] = None
    priority: int = 0
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CorrelatedEvents:
    """A group of correlated events."""
    correlation_id: str
    correlation_type: CorrelationType
    key: str
    events: List[CloudEvent] = field(default_factory=list)
    trace_links: List[TraceEventLink] = field(default_factory=list)
    first_event_time: Optional[datetime] = None
    last_event_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_event(self, event: CloudEvent):
        """Add event to correlation group."""
        self.events.append(event)

        event_time = event.time or datetime.now()

        if self.first_event_time is None or event_time < self.first_event_time:
            self.first_event_time = event_time

        if self.last_event_time is None or event_time > self.last_event_time:
            self.last_event_time = event_time

    @property
    def duration_seconds(self) -> float:
        """Get duration of correlated events."""
        if self.first_event_time and self.last_event_time:
            return (self.last_event_time - self.first_event_time).total_seconds()
        return 0.0

    @property
    def event_count(self) -> int:
        return len(self.events)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "correlation_id": self.correlation_id,
            "correlation_type": self.correlation_type.value,
            "key": self.key,
            "event_count": self.event_count,
            "first_event_time": self.first_event_time.isoformat() if self.first_event_time else None,
            "last_event_time": self.last_event_time.isoformat() if self.last_event_time else None,
            "duration_seconds": self.duration_seconds,
            "trace_links": [link.to_dict() for link in self.trace_links],
            "events": [{"id": e.id, "type": e.type, "time": e.time.isoformat() if e.time else None} for e in self.events],
            "metadata": self.metadata,
        }


class EventCorrelator:
    """Correlates events with traces and other events.

    Usage:
        correlator = EventCorrelator()

        # Add correlation rule
        correlator.add_rule(CorrelationRule(
            rule_id="deployment-events",
            name="Deployment Event Correlation",
            correlation_type=CorrelationType.DEPLOYMENT,
            event_types=["io.consciousness.deployment.*"],
        ))

        # Correlate event
        correlated = correlator.correlate(event)

        # Get correlated events
        group = correlator.get_correlation("deployment-123")
    """

    def __init__(self, namespace: str = "consciousness"):
        self.namespace = namespace

        self._rules: Dict[str, CorrelationRule] = {}
        self._correlations: Dict[str, CorrelatedEvents] = {}
        self._trace_index: Dict[str, List[str]] = {}  # trace_id -> correlation_ids
        self._lock = threading.Lock()

        self._max_correlations = 10000
        self._default_window = timedelta(minutes=5)

        # Create default rules
        self._create_default_rules()

        # Metrics
        self.correlations_created = Counter(
            f"{namespace}_event_correlations_created_total",
            "Total correlation groups created",
            ["type"],
        )

        self.events_correlated = Counter(
            f"{namespace}_events_correlated_total",
            "Total events correlated",
            ["type"],
        )

        self.correlation_group_size = Histogram(
            f"{namespace}_event_correlation_group_size",
            "Correlation group sizes",
            buckets=[1, 2, 5, 10, 20, 50, 100],
        )

        self.active_correlations = Gauge(
            f"{namespace}_event_active_correlations",
            "Active correlation groups",
            ["type"],
        )

    def _create_default_rules(self):
        """Create default correlation rules."""
        # Trace-based correlation
        self._rules["trace"] = CorrelationRule(
            rule_id="trace",
            name="Trace ID Correlation",
            correlation_type=CorrelationType.TRACE,
            key_extractor=self._extract_trace_id,
            priority=100,
        )

        # Deployment correlation
        self._rules["deployment"] = CorrelationRule(
            rule_id="deployment",
            name="Deployment Correlation",
            correlation_type=CorrelationType.DEPLOYMENT,
            event_types=[
                "io.consciousness.deployment.started",
                "io.consciousness.deployment.completed",
                "io.consciousness.deployment.failed",
                "io.consciousness.deployment.rolledback",
            ],
            key_extractor=self._extract_deployment_id,
            time_window_seconds=3600,
            priority=90,
        )

        # Incident correlation
        self._rules["incident"] = CorrelationRule(
            rule_id="incident",
            name="Incident Correlation",
            correlation_type=CorrelationType.INCIDENT,
            event_types=[
                "io.consciousness.incident.created",
                "io.consciousness.incident.acknowledged",
                "io.consciousness.incident.resolved",
                "io.consciousness.alert.triggered",
                "io.consciousness.alert.resolved",
            ],
            key_extractor=self._extract_incident_id,
            time_window_seconds=86400,  # 24 hours
            priority=80,
        )

    def _extract_trace_id(self, event: CloudEvent) -> Optional[str]:
        """Extract trace ID from event."""
        # Check traceparent extension
        traceparent = event.extensions.get("traceparent")
        if traceparent:
            # Parse W3C traceparent: version-trace_id-span_id-flags
            parts = traceparent.split("-")
            if len(parts) >= 2:
                return parts[1]

        # Check trace_id extension
        trace_id = event.extensions.get("trace_id")
        if trace_id:
            return trace_id

        # Check data for trace_id
        if isinstance(event.data, dict):
            return event.data.get("trace_id")

        return None

    def _extract_deployment_id(self, event: CloudEvent) -> Optional[str]:
        """Extract deployment ID from event."""
        if isinstance(event.data, dict):
            return event.data.get("deployment_id") or event.data.get("deploy_id")
        return event.subject

    def _extract_incident_id(self, event: CloudEvent) -> Optional[str]:
        """Extract incident ID from event."""
        if isinstance(event.data, dict):
            return event.data.get("incident_id") or event.data.get("alert_id")
        return event.subject

    def add_rule(self, rule: CorrelationRule):
        """Add a correlation rule.

        Args:
            rule: Correlation rule
        """
        with self._lock:
            self._rules[rule.rule_id] = rule

        logger.info(f"Added correlation rule: {rule.rule_id}")

    def remove_rule(self, rule_id: str):
        """Remove a correlation rule.

        Args:
            rule_id: Rule ID
        """
        with self._lock:
            self._rules.pop(rule_id, None)

    def correlate(self, event: CloudEvent) -> Optional[CorrelatedEvents]:
        """Correlate an event with existing events.

        Args:
            event: Event to correlate

        Returns:
            CorrelatedEvents group if correlated
        """
        # Sort rules by priority
        with self._lock:
            rules = sorted(
                [r for r in self._rules.values() if r.enabled],
                key=lambda r: r.priority,
                reverse=True,
            )

        for rule in rules:
            # Check event type match
            if rule.event_types:
                if not self._matches_event_types(event.type, rule.event_types):
                    continue

            # Extract correlation key
            key = None
            if rule.key_extractor:
                key = rule.key_extractor(event)

            if not key:
                continue

            # Find or create correlation group
            correlation_id = f"{rule.correlation_type.value}:{key}"

            with self._lock:
                if correlation_id in self._correlations:
                    correlated = self._correlations[correlation_id]

                    # Check time window
                    if correlated.last_event_time:
                        window = timedelta(seconds=rule.time_window_seconds)
                        event_time = event.time or datetime.now()
                        if event_time - correlated.last_event_time > window:
                            # Create new correlation
                            correlated = self._create_correlation(
                                correlation_id, rule.correlation_type, key
                            )
                else:
                    correlated = self._create_correlation(
                        correlation_id, rule.correlation_type, key
                    )

                # Add event to correlation
                correlated.add_event(event)

                # Create trace link if available
                trace_id = self._extract_trace_id(event)
                if trace_id:
                    link = TraceEventLink(
                        trace_id=trace_id,
                        event_id=event.id,
                        event_type=event.type,
                        timestamp=event.time or datetime.now(),
                    )
                    correlated.trace_links.append(link)

                    # Update trace index
                    if trace_id not in self._trace_index:
                        self._trace_index[trace_id] = []
                    if correlation_id not in self._trace_index[trace_id]:
                        self._trace_index[trace_id].append(correlation_id)

            # Update metrics
            self.events_correlated.labels(type=rule.correlation_type.value).inc()

            return correlated

        return None

    def _matches_event_types(self, event_type: str, patterns: List[str]) -> bool:
        """Check if event type matches any pattern."""
        for pattern in patterns:
            if pattern.endswith(".*"):
                prefix = pattern[:-2]
                if event_type.startswith(prefix):
                    return True
            elif pattern == event_type:
                return True
        return False

    def _create_correlation(
        self,
        correlation_id: str,
        correlation_type: CorrelationType,
        key: str,
    ) -> CorrelatedEvents:
        """Create a new correlation group."""
        correlated = CorrelatedEvents(
            correlation_id=correlation_id,
            correlation_type=correlation_type,
            key=key,
        )

        self._correlations[correlation_id] = correlated

        # Enforce limits
        if len(self._correlations) > self._max_correlations:
            self._evict_oldest()

        # Update metrics
        self.correlations_created.labels(type=correlation_type.value).inc()
        self._update_active_metrics()

        return correlated

    def _evict_oldest(self):
        """Evict oldest correlations."""
        sorted_correlations = sorted(
            self._correlations.items(),
            key=lambda x: x[1].last_event_time or datetime.min,
        )

        to_remove = len(self._correlations) - (self._max_correlations // 2)

        for correlation_id, _ in sorted_correlations[:to_remove]:
            del self._correlations[correlation_id]

    def _update_active_metrics(self):
        """Update active correlation metrics."""
        by_type: Dict[str, int] = {}
        for correlated in self._correlations.values():
            t = correlated.correlation_type.value
            by_type[t] = by_type.get(t, 0) + 1

        for t, count in by_type.items():
            self.active_correlations.labels(type=t).set(count)

    def get_correlation(self, correlation_id: str) -> Optional[CorrelatedEvents]:
        """Get a correlation group by ID.

        Args:
            correlation_id: Correlation ID

        Returns:
            CorrelatedEvents or None
        """
        with self._lock:
            return self._correlations.get(correlation_id)

    def get_by_trace(self, trace_id: str) -> List[CorrelatedEvents]:
        """Get correlation groups by trace ID.

        Args:
            trace_id: Trace ID

        Returns:
            List of correlated event groups
        """
        with self._lock:
            correlation_ids = self._trace_index.get(trace_id, [])
            return [
                self._correlations[cid]
                for cid in correlation_ids
                if cid in self._correlations
            ]

    def link_trace_to_event(
        self,
        trace_id: str,
        event_id: str,
        event_type: str,
        span_id: Optional[str] = None,
        relationship: str = "caused_by",
    ) -> TraceEventLink:
        """Create a link between a trace and an event.

        Args:
            trace_id: Trace ID
            event_id: Event ID
            event_type: Event type
            span_id: Optional span ID
            relationship: Relationship type

        Returns:
            TraceEventLink
        """
        link = TraceEventLink(
            trace_id=trace_id,
            span_id=span_id,
            event_id=event_id,
            event_type=event_type,
            relationship=relationship,
        )

        # Find correlation for this trace
        with self._lock:
            correlation_ids = self._trace_index.get(trace_id, [])
            for cid in correlation_ids:
                if cid in self._correlations:
                    self._correlations[cid].trace_links.append(link)

        return link

    def find_related_events(
        self,
        event: CloudEvent,
        time_window_seconds: int = 300,
        limit: int = 100,
    ) -> List[CloudEvent]:
        """Find events related to a given event.

        Args:
            event: Reference event
            time_window_seconds: Time window in seconds
            limit: Maximum results

        Returns:
            List of related events
        """
        related = []
        event_time = event.time or datetime.now()
        window = timedelta(seconds=time_window_seconds)

        # Check by trace
        trace_id = self._extract_trace_id(event)
        if trace_id:
            for correlated in self.get_by_trace(trace_id):
                for e in correlated.events:
                    if e.id != event.id:
                        related.append(e)

        # Check by subject
        if event.subject:
            with self._lock:
                for correlated in self._correlations.values():
                    for e in correlated.events:
                        if e.subject == event.subject and e.id != event.id:
                            e_time = e.time or datetime.now()
                            if abs((e_time - event_time).total_seconds()) <= time_window_seconds:
                                related.append(e)

        # Deduplicate and sort
        seen = set()
        unique_related = []
        for e in related:
            if e.id not in seen:
                seen.add(e.id)
                unique_related.append(e)

        # Sort by time proximity
        unique_related.sort(
            key=lambda e: abs((e.time or datetime.now()) - event_time).total_seconds()
        )

        return unique_related[:limit]

    def cleanup_expired(self, max_age_hours: int = 24) -> int:
        """Remove expired correlations.

        Args:
            max_age_hours: Maximum age in hours

        Returns:
            Number of correlations removed
        """
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        removed = 0

        with self._lock:
            to_remove = []
            for correlation_id, correlated in self._correlations.items():
                if correlated.last_event_time and correlated.last_event_time < cutoff:
                    to_remove.append(correlation_id)

            for correlation_id in to_remove:
                del self._correlations[correlation_id]
                removed += 1

            # Clean trace index
            self._trace_index = {
                trace_id: [
                    cid for cid in cids
                    if cid in self._correlations
                ]
                for trace_id, cids in self._trace_index.items()
            }
            self._trace_index = {
                k: v for k, v in self._trace_index.items() if v
            }

        self._update_active_metrics()
        logger.info(f"Cleaned up {removed} expired correlations")

        return removed

    def get_statistics(self) -> Dict[str, Any]:
        """Get correlator statistics.

        Returns:
            Statistics dictionary
        """
        with self._lock:
            correlations = list(self._correlations.values())

        # By type
        by_type: Dict[str, int] = {}
        for c in correlations:
            t = c.correlation_type.value
            by_type[t] = by_type.get(t, 0) + 1

        # Group sizes
        sizes = [c.event_count for c in correlations]
        avg_size = sum(sizes) / len(sizes) if sizes else 0

        # Duration stats
        durations = [c.duration_seconds for c in correlations if c.duration_seconds > 0]
        avg_duration = sum(durations) / len(durations) if durations else 0

        return {
            "total_correlations": len(correlations),
            "by_type": by_type,
            "total_rules": len(self._rules),
            "trace_index_size": len(self._trace_index),
            "average_group_size": avg_size,
            "average_duration_seconds": avg_duration,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get correlator summary.

        Returns:
            Summary dictionary
        """
        stats = self.get_statistics()

        with self._lock:
            rules = [
                {
                    "id": r.rule_id,
                    "name": r.name,
                    "type": r.correlation_type.value,
                    "enabled": r.enabled,
                    "priority": r.priority,
                }
                for r in self._rules.values()
            ]

        return {
            **stats,
            "rules": rules,
        }
