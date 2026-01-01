"""Flux Reconciliation Tracking

Monitors Flux CD reconciliation loops and status.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import threading
import logging

from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)


class ReconciliationStatus(str, Enum):
    """Flux reconciliation status."""
    READY = "Ready"
    RECONCILING = "Reconciling"
    STALLED = "Stalled"
    FAILED = "Failed"
    SUSPENDED = "Suspended"
    UNKNOWN = "Unknown"


class SourceType(str, Enum):
    """Flux source types."""
    GIT_REPOSITORY = "GitRepository"
    HELM_REPOSITORY = "HelmRepository"
    BUCKET = "Bucket"
    OCI_REPOSITORY = "OCIRepository"


class WorkloadType(str, Enum):
    """Flux workload types."""
    KUSTOMIZATION = "Kustomization"
    HELM_RELEASE = "HelmRelease"


@dataclass
class FluxCondition:
    """Flux resource condition."""
    condition_type: str
    status: str
    reason: str
    message: str
    last_transition_time: Optional[datetime] = None


@dataclass
class SourceRef:
    """Reference to a Flux source."""
    kind: SourceType
    name: str
    namespace: str = "flux-system"


@dataclass
class FluxSource:
    """Flux source representation."""
    name: str
    namespace: str
    source_type: SourceType
    url: str
    branch: str = "main"
    revision: str = ""
    status: ReconciliationStatus = ReconciliationStatus.UNKNOWN
    last_applied_revision: str = ""
    interval_seconds: int = 60
    conditions: List[FluxCondition] = field(default_factory=list)
    suspended: bool = False
    last_reconcile_time: Optional[datetime] = None
    artifact_checksum: str = ""

    @property
    def is_ready(self) -> bool:
        return self.status == ReconciliationStatus.READY

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "namespace": self.namespace,
            "source_type": self.source_type.value,
            "url": self.url,
            "branch": self.branch,
            "revision": self.revision[:12] if self.revision else "",
            "status": self.status.value,
            "is_ready": self.is_ready,
            "interval_seconds": self.interval_seconds,
            "suspended": self.suspended,
            "last_reconcile_time": (
                self.last_reconcile_time.isoformat()
                if self.last_reconcile_time else None
            ),
        }


@dataclass
class FluxWorkload:
    """Flux workload (Kustomization or HelmRelease)."""
    name: str
    namespace: str
    workload_type: WorkloadType
    source_ref: SourceRef
    path: str = ""
    target_namespace: str = ""
    status: ReconciliationStatus = ReconciliationStatus.UNKNOWN
    revision: str = ""
    interval_seconds: int = 60
    timeout_seconds: int = 300
    retry_attempts: int = 0
    conditions: List[FluxCondition] = field(default_factory=list)
    suspended: bool = False
    last_applied_revision: str = ""
    last_reconcile_time: Optional[datetime] = None
    last_attempted_revision: str = ""
    health_checks: List[Dict[str, str]] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    prune: bool = False

    @property
    def is_ready(self) -> bool:
        return self.status == ReconciliationStatus.READY

    @property
    def is_stalled(self) -> bool:
        return self.status == ReconciliationStatus.STALLED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "namespace": self.namespace,
            "workload_type": self.workload_type.value,
            "source_ref": f"{self.source_ref.kind.value}/{self.source_ref.name}",
            "path": self.path,
            "target_namespace": self.target_namespace,
            "status": self.status.value,
            "is_ready": self.is_ready,
            "revision": self.revision[:12] if self.revision else "",
            "interval_seconds": self.interval_seconds,
            "suspended": self.suspended,
            "dependencies": self.dependencies,
            "last_reconcile_time": (
                self.last_reconcile_time.isoformat()
                if self.last_reconcile_time else None
            ),
        }


@dataclass
class ReconciliationEvent:
    """Flux reconciliation event."""
    event_id: str
    workload_name: str
    workload_namespace: str
    workload_type: WorkloadType
    event_type: str  # Normal, Warning
    reason: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    revision: str = ""
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "workload": f"{self.workload_namespace}/{self.workload_name}",
            "workload_type": self.workload_type.value,
            "event_type": self.event_type,
            "reason": self.reason,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "revision": self.revision[:12] if self.revision else "",
        }


@dataclass
class ReconciliationMetrics:
    """Reconciliation metrics for a workload."""
    workload_name: str
    total_reconciliations: int = 0
    successful_reconciliations: int = 0
    failed_reconciliations: int = 0
    average_duration_seconds: float = 0.0
    last_reconcile_duration_seconds: float = 0.0
    reconciliation_errors: List[str] = field(default_factory=list)
    first_reconcile_time: Optional[datetime] = None
    last_successful_time: Optional[datetime] = None


class FluxReconciliationTracker:
    """Tracks Flux CD reconciliation status and events.

    Usage:
        tracker = FluxReconciliationTracker()

        # Register sources and workloads
        tracker.register_source(source)
        tracker.register_workload(workload)

        # Start tracking
        tracker.start()

        # Get reconciliation status
        status = tracker.get_workload_status("my-kustomization")

        # Get events
        events = tracker.get_events("my-kustomization")

        # Register callbacks
        tracker.on_reconciliation_complete(handle_reconcile)
        tracker.on_reconciliation_failed(handle_failure)
    """

    def __init__(
        self,
        namespace: str = "consciousness",
        check_interval_seconds: int = 15,
    ):
        self.namespace = namespace
        self.check_interval = check_interval_seconds
        self._lock = threading.Lock()

        # State
        self._sources: Dict[str, FluxSource] = {}
        self._workloads: Dict[str, FluxWorkload] = {}
        self._events: Dict[str, List[ReconciliationEvent]] = {}
        self._metrics: Dict[str, ReconciliationMetrics] = {}

        # Callbacks
        self._complete_callbacks: List[Callable[[FluxWorkload], None]] = []
        self._failed_callbacks: List[Callable[[FluxWorkload, str], None]] = []
        self._stalled_callbacks: List[Callable[[FluxWorkload], None]] = []

        # Tracking
        self._running = False
        self._check_thread: Optional[threading.Thread] = None
        self._max_events = 1000

        # Prometheus metrics
        self.reconciliation_status = Gauge(
            f"{namespace}_flux_reconciliation_status",
            "Workload reconciliation status (1=ready, 0=not ready)",
            ["workload", "namespace", "type"],
        )

        self.reconciliation_total = Counter(
            f"{namespace}_flux_reconciliations_total",
            "Total reconciliations",
            ["workload", "namespace", "result"],
        )

        self.reconciliation_duration = Histogram(
            f"{namespace}_flux_reconciliation_duration_seconds",
            "Reconciliation duration",
            ["workload", "type"],
            buckets=[1, 5, 10, 30, 60, 120, 300, 600],
        )

        self.source_status = Gauge(
            f"{namespace}_flux_source_status",
            "Source status (1=ready, 0=not ready)",
            ["source", "namespace", "type"],
        )

        self.suspended_workloads = Gauge(
            f"{namespace}_flux_suspended_workloads",
            "Number of suspended workloads",
        )

        self.stalled_workloads = Gauge(
            f"{namespace}_flux_stalled_workloads",
            "Number of stalled workloads",
        )

        self.time_since_last_reconcile = Gauge(
            f"{namespace}_flux_time_since_last_reconcile_seconds",
            "Seconds since last reconciliation",
            ["workload", "namespace"],
        )

    def start(self):
        """Start reconciliation tracking."""
        if self._running:
            return

        self._running = True
        self._check_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
        )
        self._check_thread.start()
        logger.info("Flux reconciliation tracker started")

    def stop(self):
        """Stop reconciliation tracking."""
        self._running = False
        if self._check_thread:
            self._check_thread.join(timeout=5)
        logger.info("Flux reconciliation tracker stopped")

    def _monitor_loop(self):
        """Background monitoring loop."""
        import time

        while self._running:
            try:
                self._check_reconciliation_status()
                self._update_time_metrics()
            except Exception as e:
                logger.error(f"Flux tracker error: {e}")

            time.sleep(self.check_interval)

    def _check_reconciliation_status(self):
        """Check reconciliation status of all workloads."""
        with self._lock:
            workloads = list(self._workloads.values())

        suspended_count = 0
        stalled_count = 0

        for workload in workloads:
            # Update status metrics
            self.reconciliation_status.labels(
                workload=workload.name,
                namespace=workload.namespace,
                type=workload.workload_type.value,
            ).set(1 if workload.is_ready else 0)

            if workload.suspended:
                suspended_count += 1

            if workload.is_stalled:
                stalled_count += 1
                for callback in self._stalled_callbacks:
                    try:
                        callback(workload)
                    except Exception as e:
                        logger.error(f"Stalled callback error: {e}")

        self.suspended_workloads.set(suspended_count)
        self.stalled_workloads.set(stalled_count)

        # Check sources
        with self._lock:
            sources = list(self._sources.values())

        for source in sources:
            self.source_status.labels(
                source=source.name,
                namespace=source.namespace,
                type=source.source_type.value,
            ).set(1 if source.is_ready else 0)

    def _update_time_metrics(self):
        """Update time-based metrics."""
        now = datetime.now()

        with self._lock:
            for workload in self._workloads.values():
                if workload.last_reconcile_time:
                    seconds = (now - workload.last_reconcile_time).total_seconds()
                    self.time_since_last_reconcile.labels(
                        workload=workload.name,
                        namespace=workload.namespace,
                    ).set(seconds)

    def register_source(self, source: FluxSource):
        """Register a Flux source for tracking.

        Args:
            source: Flux source to track
        """
        key = f"{source.namespace}/{source.name}"

        with self._lock:
            self._sources[key] = source

        logger.info(f"Registered Flux source: {key}")

    def register_workload(self, workload: FluxWorkload):
        """Register a Flux workload for tracking.

        Args:
            workload: Flux workload to track
        """
        key = f"{workload.namespace}/{workload.name}"

        with self._lock:
            self._workloads[key] = workload
            if key not in self._metrics:
                self._metrics[key] = ReconciliationMetrics(workload_name=workload.name)

        logger.info(f"Registered Flux workload: {key}")

    def update_workload_status(
        self,
        name: str,
        namespace: str,
        status: ReconciliationStatus,
        revision: str = "",
        message: str = "",
    ):
        """Update workload reconciliation status.

        Args:
            name: Workload name
            namespace: Workload namespace
            status: New status
            revision: Applied revision
            message: Status message
        """
        key = f"{namespace}/{name}"

        with self._lock:
            workload = self._workloads.get(key)
            if not workload:
                return

            old_status = workload.status
            workload.status = status
            workload.last_reconcile_time = datetime.now()

            if revision:
                workload.last_applied_revision = revision

            # Update metrics
            metrics = self._metrics.get(key)
            if metrics:
                metrics.total_reconciliations += 1
                if status == ReconciliationStatus.READY:
                    metrics.successful_reconciliations += 1
                    metrics.last_successful_time = datetime.now()
                elif status == ReconciliationStatus.FAILED:
                    metrics.failed_reconciliations += 1
                    if message:
                        metrics.reconciliation_errors.append(message)
                        # Keep last 20 errors
                        metrics.reconciliation_errors = metrics.reconciliation_errors[-20:]

        # Update Prometheus metrics
        self.reconciliation_status.labels(
            workload=name,
            namespace=namespace,
            type=workload.workload_type.value,
        ).set(1 if status == ReconciliationStatus.READY else 0)

        result = "success" if status == ReconciliationStatus.READY else "failure"
        self.reconciliation_total.labels(
            workload=name,
            namespace=namespace,
            result=result,
        ).inc()

        # Trigger callbacks
        if status == ReconciliationStatus.READY and old_status != ReconciliationStatus.READY:
            for callback in self._complete_callbacks:
                try:
                    callback(workload)
                except Exception as e:
                    logger.error(f"Complete callback error: {e}")

        elif status == ReconciliationStatus.FAILED:
            for callback in self._failed_callbacks:
                try:
                    callback(workload, message)
                except Exception as e:
                    logger.error(f"Failed callback error: {e}")

        logger.info(f"Updated workload status: {key} -> {status.value}")

    def record_event(
        self,
        workload_name: str,
        workload_namespace: str,
        workload_type: WorkloadType,
        event_type: str,
        reason: str,
        message: str,
        revision: str = "",
    ) -> ReconciliationEvent:
        """Record a reconciliation event.

        Args:
            workload_name: Workload name
            workload_namespace: Workload namespace
            workload_type: Type of workload
            event_type: Event type (Normal/Warning)
            reason: Event reason
            message: Event message
            revision: Associated revision

        Returns:
            ReconciliationEvent
        """
        event_id = f"event-{workload_name}-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

        event = ReconciliationEvent(
            event_id=event_id,
            workload_name=workload_name,
            workload_namespace=workload_namespace,
            workload_type=workload_type,
            event_type=event_type,
            reason=reason,
            message=message,
            revision=revision,
        )

        key = f"{workload_namespace}/{workload_name}"

        with self._lock:
            if key not in self._events:
                self._events[key] = []
            self._events[key].append(event)

            # Trim old events
            if len(self._events[key]) > self._max_events:
                self._events[key] = self._events[key][-self._max_events:]

        return event

    def get_workload(self, name: str, namespace: str) -> Optional[FluxWorkload]:
        """Get workload by name and namespace.

        Args:
            name: Workload name
            namespace: Workload namespace

        Returns:
            FluxWorkload or None
        """
        key = f"{namespace}/{name}"
        with self._lock:
            return self._workloads.get(key)

    def get_source(self, name: str, namespace: str) -> Optional[FluxSource]:
        """Get source by name and namespace.

        Args:
            name: Source name
            namespace: Source namespace

        Returns:
            FluxSource or None
        """
        key = f"{namespace}/{name}"
        with self._lock:
            return self._sources.get(key)

    def get_all_workloads(
        self,
        status_filter: Optional[ReconciliationStatus] = None,
    ) -> List[FluxWorkload]:
        """Get all workloads, optionally filtered by status.

        Args:
            status_filter: Filter by status

        Returns:
            List of workloads
        """
        with self._lock:
            workloads = list(self._workloads.values())

        if status_filter:
            workloads = [w for w in workloads if w.status == status_filter]

        return workloads

    def get_all_sources(self) -> List[FluxSource]:
        """Get all sources.

        Returns:
            List of sources
        """
        with self._lock:
            return list(self._sources.values())

    def get_events(
        self,
        workload_name: str,
        workload_namespace: str = "flux-system",
        limit: int = 50,
        event_type: Optional[str] = None,
    ) -> List[ReconciliationEvent]:
        """Get events for a workload.

        Args:
            workload_name: Workload name
            workload_namespace: Workload namespace
            limit: Maximum events to return
            event_type: Filter by event type

        Returns:
            List of events
        """
        key = f"{workload_namespace}/{workload_name}"

        with self._lock:
            events = list(self._events.get(key, []))

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        # Sort by timestamp descending
        events.sort(key=lambda e: e.timestamp, reverse=True)

        return events[:limit]

    def get_workload_metrics(
        self,
        name: str,
        namespace: str,
    ) -> Optional[ReconciliationMetrics]:
        """Get reconciliation metrics for a workload.

        Args:
            name: Workload name
            namespace: Workload namespace

        Returns:
            ReconciliationMetrics or None
        """
        key = f"{namespace}/{name}"
        with self._lock:
            return self._metrics.get(key)

    def get_failed_workloads(self) -> List[FluxWorkload]:
        """Get all failed workloads.

        Returns:
            List of failed workloads
        """
        return self.get_all_workloads(status_filter=ReconciliationStatus.FAILED)

    def get_stalled_workloads(self) -> List[FluxWorkload]:
        """Get all stalled workloads.

        Returns:
            List of stalled workloads
        """
        return self.get_all_workloads(status_filter=ReconciliationStatus.STALLED)

    def get_suspended_workloads(self) -> List[FluxWorkload]:
        """Get all suspended workloads.

        Returns:
            List of suspended workloads
        """
        with self._lock:
            return [w for w in self._workloads.values() if w.suspended]

    def suspend_workload(self, name: str, namespace: str) -> bool:
        """Suspend a workload.

        Args:
            name: Workload name
            namespace: Workload namespace

        Returns:
            True if suspended
        """
        key = f"{namespace}/{name}"

        with self._lock:
            workload = self._workloads.get(key)
            if workload:
                workload.suspended = True
                logger.info(f"Suspended workload: {key}")
                return True

        return False

    def resume_workload(self, name: str, namespace: str) -> bool:
        """Resume a suspended workload.

        Args:
            name: Workload name
            namespace: Workload namespace

        Returns:
            True if resumed
        """
        key = f"{namespace}/{name}"

        with self._lock:
            workload = self._workloads.get(key)
            if workload:
                workload.suspended = False
                logger.info(f"Resumed workload: {key}")
                return True

        return False

    def on_reconciliation_complete(self, callback: Callable[[FluxWorkload], None]):
        """Register callback for successful reconciliation.

        Args:
            callback: Function to call on successful reconciliation
        """
        self._complete_callbacks.append(callback)

    def on_reconciliation_failed(self, callback: Callable[[FluxWorkload, str], None]):
        """Register callback for failed reconciliation.

        Args:
            callback: Function to call on failed reconciliation
        """
        self._failed_callbacks.append(callback)

    def on_workload_stalled(self, callback: Callable[[FluxWorkload], None]):
        """Register callback for stalled workloads.

        Args:
            callback: Function to call when workload stalls
        """
        self._stalled_callbacks.append(callback)

    def get_summary(self) -> Dict[str, Any]:
        """Get tracker summary.

        Returns:
            Summary dictionary
        """
        with self._lock:
            workloads = list(self._workloads.values())
            sources = list(self._sources.values())

        ready = sum(1 for w in workloads if w.is_ready)
        failed = sum(1 for w in workloads if w.status == ReconciliationStatus.FAILED)
        stalled = sum(1 for w in workloads if w.is_stalled)
        suspended = sum(1 for w in workloads if w.suspended)

        sources_ready = sum(1 for s in sources if s.is_ready)

        # By type
        by_type: Dict[str, int] = {}
        for w in workloads:
            t = w.workload_type.value
            by_type[t] = by_type.get(t, 0) + 1

        return {
            "running": self._running,
            "total_workloads": len(workloads),
            "ready_workloads": ready,
            "failed_workloads": failed,
            "stalled_workloads": stalled,
            "suspended_workloads": suspended,
            "total_sources": len(sources),
            "ready_sources": sources_ready,
            "workloads_by_type": by_type,
            "health_percentage": (ready / len(workloads) * 100) if workloads else 100.0,
        }
