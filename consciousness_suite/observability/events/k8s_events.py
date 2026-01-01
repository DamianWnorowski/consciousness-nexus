"""Kubernetes Events Collector

Collects and processes Kubernetes events for observability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
from enum import Enum
import threading
import logging

from prometheus_client import Gauge, Counter

logger = logging.getLogger(__name__)


class K8sEventType(str, Enum):
    """Kubernetes event types."""
    NORMAL = "Normal"
    WARNING = "Warning"


class EventReason(str, Enum):
    """Common Kubernetes event reasons."""
    # Pod events
    SCHEDULED = "Scheduled"
    PULLING = "Pulling"
    PULLED = "Pulled"
    CREATED = "Created"
    STARTED = "Started"
    KILLING = "Killing"
    FAILED = "Failed"
    BACK_OFF = "BackOff"
    EXCEEDED_GRACE_PERIOD = "ExceededGracePeriod"

    # Node events
    NODE_READY = "NodeReady"
    NODE_NOT_READY = "NodeNotReady"
    STARTING = "Starting"
    REBOOTED = "Rebooted"

    # Deployment events
    SCALED_UP = "ScaledUpReplicaSet"
    SCALED_DOWN = "ScaledDownReplicaSet"

    # Other
    UNHEALTHY = "Unhealthy"
    PROBE_FAILED = "ProbesFailed"
    OOM_KILLED = "OOMKilled"


@dataclass
class K8sEvent:
    """A Kubernetes event."""
    uid: str
    name: str
    namespace: str
    kind: str  # Pod, Node, Deployment, etc.
    event_type: K8sEventType
    reason: str
    message: str
    first_timestamp: datetime
    last_timestamp: datetime
    count: int = 1
    source_component: str = ""
    source_host: str = ""
    involved_object_name: str = ""
    involved_object_uid: str = ""
    involved_object_kind: str = ""
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)

    @property
    def is_warning(self) -> bool:
        return self.event_type == K8sEventType.WARNING

    @property
    def age_seconds(self) -> float:
        return (datetime.now() - self.first_timestamp).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uid": self.uid,
            "name": self.name,
            "namespace": self.namespace,
            "kind": self.kind,
            "type": self.event_type.value,
            "reason": self.reason,
            "message": self.message,
            "first_timestamp": self.first_timestamp.isoformat(),
            "last_timestamp": self.last_timestamp.isoformat(),
            "count": self.count,
            "source": f"{self.source_component}/{self.source_host}",
            "involved_object": f"{self.involved_object_kind}/{self.involved_object_name}",
        }


class K8sEventCollector:
    """Collects Kubernetes events.

    Usage:
        collector = K8sEventCollector()

        # Process incoming events
        collector.process_event(k8s_event)

        # Get recent events
        events = collector.get_recent_events(namespace="production")

        # Get warnings
        warnings = collector.get_warnings()
    """

    def __init__(self, namespace: str = "consciousness"):
        self.prom_namespace = namespace

        self._events: List[K8sEvent] = []
        self._events_by_object: Dict[str, List[K8sEvent]] = {}
        self._lock = threading.Lock()
        self._max_events = 10000

        self._callbacks: List[Callable[[K8sEvent], None]] = []

        # Prometheus metrics
        self.k8s_events = Counter(
            f"{namespace}_k8s_events_total",
            "Total Kubernetes events",
            ["namespace", "type", "reason"],
        )

        self.k8s_warnings = Gauge(
            f"{namespace}_k8s_warnings_active",
            "Active Kubernetes warnings",
            ["namespace", "kind"],
        )

        self.k8s_event_rate = Gauge(
            f"{namespace}_k8s_event_rate_per_minute",
            "Event rate per minute",
            ["namespace"],
        )

        self.k8s_pod_events = Counter(
            f"{namespace}_k8s_pod_events_total",
            "Pod events",
            ["namespace", "pod", "reason"],
        )

    def process_event(self, event: K8sEvent):
        """Process a Kubernetes event.

        Args:
            event: K8sEvent to process
        """
        with self._lock:
            self._events.append(event)

            # Index by involved object
            obj_key = f"{event.involved_object_kind}/{event.namespace}/{event.involved_object_name}"
            if obj_key not in self._events_by_object:
                self._events_by_object[obj_key] = []
            self._events_by_object[obj_key].append(event)

            # Trim old events
            if len(self._events) > self._max_events:
                self._events = self._events[-self._max_events // 2:]

        # Update metrics
        self.k8s_events.labels(
            namespace=event.namespace,
            type=event.event_type.value,
            reason=event.reason,
        ).inc()

        if event.involved_object_kind == "Pod":
            self.k8s_pod_events.labels(
                namespace=event.namespace,
                pod=event.involved_object_name[:50],
                reason=event.reason,
            ).inc()

        # Update warning count
        if event.is_warning:
            warning_count = sum(
                1 for e in self._events
                if e.is_warning and e.namespace == event.namespace
                and e.kind == event.kind
            )
            self.k8s_warnings.labels(
                namespace=event.namespace,
                kind=event.kind,
            ).set(warning_count)

        # Calculate event rate
        from datetime import timedelta
        now = datetime.now()
        recent = [e for e in self._events if now - e.last_timestamp < timedelta(minutes=1)]
        ns_rate = sum(1 for e in recent if e.namespace == event.namespace)
        self.k8s_event_rate.labels(namespace=event.namespace).set(ns_rate)

        # Trigger callbacks
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")

    def on_event(self, callback: Callable[[K8sEvent], None]):
        """Register event callback.

        Args:
            callback: Function to call with K8sEvent
        """
        self._callbacks.append(callback)

    def get_recent_events(
        self,
        namespace: Optional[str] = None,
        kind: Optional[str] = None,
        limit: int = 100,
    ) -> List[K8sEvent]:
        """Get recent events.

        Args:
            namespace: Filter by namespace
            kind: Filter by object kind
            limit: Maximum results

        Returns:
            List of events
        """
        with self._lock:
            events = list(self._events)

        if namespace:
            events = [e for e in events if e.namespace == namespace]

        if kind:
            events = [e for e in events if e.kind == kind or e.involved_object_kind == kind]

        # Sort by last timestamp
        events.sort(key=lambda e: e.last_timestamp, reverse=True)

        return events[:limit]

    def get_warnings(
        self,
        namespace: Optional[str] = None,
        minutes: int = 60,
    ) -> List[K8sEvent]:
        """Get warning events.

        Args:
            namespace: Filter by namespace
            minutes: Time window

        Returns:
            List of warning events
        """
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(minutes=minutes)

        with self._lock:
            events = [
                e for e in self._events
                if e.is_warning and e.last_timestamp >= cutoff
            ]

        if namespace:
            events = [e for e in events if e.namespace == namespace]

        events.sort(key=lambda e: e.last_timestamp, reverse=True)

        return events

    def get_events_for_object(
        self,
        kind: str,
        namespace: str,
        name: str,
    ) -> List[K8sEvent]:
        """Get events for a specific object.

        Args:
            kind: Object kind
            namespace: Object namespace
            name: Object name

        Returns:
            List of events
        """
        obj_key = f"{kind}/{namespace}/{name}"

        with self._lock:
            events = self._events_by_object.get(obj_key, [])

        return sorted(events, key=lambda e: e.last_timestamp, reverse=True)

    def get_pod_timeline(
        self,
        namespace: str,
        pod_name: str,
    ) -> List[Dict[str, Any]]:
        """Get pod event timeline.

        Args:
            namespace: Pod namespace
            pod_name: Pod name

        Returns:
            Timeline of pod events
        """
        events = self.get_events_for_object("Pod", namespace, pod_name)

        timeline = []
        for e in sorted(events, key=lambda x: x.first_timestamp):
            timeline.append({
                "time": e.first_timestamp.isoformat(),
                "reason": e.reason,
                "message": e.message[:200],
                "type": e.event_type.value,
                "count": e.count,
            })

        return timeline

    def get_summary(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """Get events summary.

        Args:
            namespace: Filter by namespace

        Returns:
            Summary dictionary
        """
        with self._lock:
            events = list(self._events)

        if namespace:
            events = [e for e in events if e.namespace == namespace]

        # By type
        normal = sum(1 for e in events if e.event_type == K8sEventType.NORMAL)
        warning = sum(1 for e in events if e.event_type == K8sEventType.WARNING)

        # By reason
        reasons: Dict[str, int] = {}
        for e in events:
            reasons[e.reason] = reasons.get(e.reason, 0) + 1

        # By namespace
        namespaces: Dict[str, int] = {}
        for e in events:
            namespaces[e.namespace] = namespaces.get(e.namespace, 0) + 1

        # Top reasons
        top_reasons = sorted(reasons.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "total_events": len(events),
            "normal_events": normal,
            "warning_events": warning,
            "unique_reasons": len(reasons),
            "top_reasons": [{"reason": r, "count": c} for r, c in top_reasons],
            "by_namespace": namespaces,
        }
