"""ArgoCD Sync Status Tracker

Tracks and monitors ArgoCD application sync status with detailed metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import threading
import logging

from prometheus_client import Counter, Gauge, Histogram

from .connector import (
    Application,
    ApplicationSyncStatus,
    SyncOperation,
    SyncOperationPhase,
    ArgoCDConnector,
)

logger = logging.getLogger(__name__)


class SyncDriftType(str, Enum):
    """Types of sync drift."""
    CONFIG_DRIFT = "config_drift"
    IMAGE_DRIFT = "image_drift"
    RESOURCE_DRIFT = "resource_drift"
    SECRET_DRIFT = "secret_drift"
    SCALE_DRIFT = "scale_drift"
    UNKNOWN = "unknown"


class SyncFailureReason(str, Enum):
    """Reasons for sync failure."""
    VALIDATION_ERROR = "validation_error"
    RESOURCE_CONFLICT = "resource_conflict"
    PERMISSION_DENIED = "permission_denied"
    TIMEOUT = "timeout"
    NETWORK_ERROR = "network_error"
    HOOK_FAILURE = "hook_failure"
    UNKNOWN = "unknown"


@dataclass
class SyncDrift:
    """Detected sync drift details."""
    application: str
    drift_type: SyncDriftType
    resource_kind: str
    resource_name: str
    resource_namespace: str
    expected_value: str
    actual_value: str
    detected_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        end = self.resolved_at or datetime.now()
        return (end - self.detected_at).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "application": self.application,
            "drift_type": self.drift_type.value,
            "resource_kind": self.resource_kind,
            "resource_name": self.resource_name,
            "resource_namespace": self.resource_namespace,
            "expected_value": self.expected_value,
            "actual_value": self.actual_value,
            "detected_at": self.detected_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolved": self.resolved,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class SyncStatus:
    """Detailed sync status for an application."""
    application: str
    sync_status: ApplicationSyncStatus
    revision: str
    target_revision: str
    sync_time: Optional[datetime] = None
    last_check_time: datetime = field(default_factory=datetime.now)
    drifts: List[SyncDrift] = field(default_factory=list)
    out_of_sync_resources: int = 0
    total_resources: int = 0
    in_sync_resources: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_synced(self) -> bool:
        return self.sync_status == ApplicationSyncStatus.SYNCED

    @property
    def sync_percentage(self) -> float:
        if self.total_resources == 0:
            return 100.0
        return (self.in_sync_resources / self.total_resources) * 100

    @property
    def active_drifts(self) -> List[SyncDrift]:
        return [d for d in self.drifts if not d.resolved]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "application": self.application,
            "sync_status": self.sync_status.value,
            "is_synced": self.is_synced,
            "revision": self.revision,
            "target_revision": self.target_revision,
            "sync_time": self.sync_time.isoformat() if self.sync_time else None,
            "sync_percentage": self.sync_percentage,
            "out_of_sync_resources": self.out_of_sync_resources,
            "total_resources": self.total_resources,
            "active_drifts": len(self.active_drifts),
        }


@dataclass
class SyncFailure:
    """Sync failure record."""
    application: str
    sync_id: str
    reason: SyncFailureReason
    message: str
    occurred_at: datetime = field(default_factory=datetime.now)
    retry_count: int = 0
    resources_affected: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "application": self.application,
            "sync_id": self.sync_id,
            "reason": self.reason.value,
            "message": self.message,
            "occurred_at": self.occurred_at.isoformat(),
            "retry_count": self.retry_count,
            "resources_affected": self.resources_affected,
        }


class SyncTracker:
    """Tracks ArgoCD sync status with detailed metrics.

    Usage:
        tracker = SyncTracker(connector)

        # Start tracking
        tracker.start()

        # Get sync status
        status = tracker.get_sync_status("my-app")

        # Check for drift
        drifts = tracker.get_active_drifts()

        # Register callbacks
        tracker.on_drift_detected(handle_drift)
        tracker.on_sync_failure(handle_failure)
    """

    def __init__(
        self,
        connector: Optional[ArgoCDConnector] = None,
        namespace: str = "consciousness",
        check_interval_seconds: int = 30,
    ):
        self.connector = connector
        self.namespace = namespace
        self.check_interval = check_interval_seconds
        self._lock = threading.Lock()

        # State
        self._sync_statuses: Dict[str, SyncStatus] = {}
        self._sync_failures: Dict[str, List[SyncFailure]] = {}
        self._active_drifts: Dict[str, List[SyncDrift]] = {}

        # Callbacks
        self._drift_callbacks: List[Callable[[SyncDrift], None]] = []
        self._failure_callbacks: List[Callable[[SyncFailure], None]] = []
        self._sync_callbacks: List[Callable[[SyncStatus], None]] = []

        # Tracking state
        self._running = False
        self._check_thread: Optional[threading.Thread] = None
        self._max_failures = 100
        self._max_drifts = 500

        # Prometheus metrics
        self.sync_status_gauge = Gauge(
            f"{namespace}_gitops_sync_status",
            "Application sync status (1=synced, 0=out_of_sync)",
            ["application", "project"],
        )

        self.sync_percentage = Gauge(
            f"{namespace}_gitops_sync_percentage",
            "Percentage of resources in sync",
            ["application"],
        )

        self.drift_detected = Counter(
            f"{namespace}_gitops_drift_detected_total",
            "Total drifts detected",
            ["application", "drift_type"],
        )

        self.drift_resolved = Counter(
            f"{namespace}_gitops_drift_resolved_total",
            "Total drifts resolved",
            ["application", "drift_type"],
        )

        self.active_drifts_gauge = Gauge(
            f"{namespace}_gitops_active_drifts",
            "Number of active drifts",
            ["application"],
        )

        self.sync_failures_total = Counter(
            f"{namespace}_gitops_sync_failures_total",
            "Total sync failures",
            ["application", "reason"],
        )

        self.time_since_last_sync = Gauge(
            f"{namespace}_gitops_time_since_last_sync_seconds",
            "Time since last successful sync",
            ["application"],
        )

        self.out_of_sync_resources = Gauge(
            f"{namespace}_gitops_out_of_sync_resources",
            "Number of out-of-sync resources",
            ["application"],
        )

    def start(self):
        """Start sync tracking."""
        if self._running:
            return

        self._running = True
        self._check_thread = threading.Thread(
            target=self._check_loop,
            daemon=True,
        )
        self._check_thread.start()
        logger.info("Sync tracker started")

    def stop(self):
        """Stop sync tracking."""
        self._running = False
        if self._check_thread:
            self._check_thread.join(timeout=5)
        logger.info("Sync tracker stopped")

    def _check_loop(self):
        """Background check loop."""
        import time

        while self._running:
            try:
                self._check_all_applications()
            except Exception as e:
                logger.error(f"Sync check error: {e}")

            time.sleep(self.check_interval)

    def _check_all_applications(self):
        """Check sync status of all applications."""
        if not self.connector or not self.connector.is_connected:
            return

        apps = self.connector.list_applications()

        for app in apps:
            self._check_application_sync(app)

    def _check_application_sync(self, app: Application):
        """Check sync status of a single application."""
        # Create sync status
        status = SyncStatus(
            application=app.name,
            sync_status=app.sync_status,
            revision=app.sync_revision,
            target_revision=app.target_revision,
            sync_time=app.synced_at,
            total_resources=len(app.resources),
        )

        # Count in-sync resources
        for resource in app.resources:
            if resource.sync_status == "Synced":
                status.in_sync_resources += 1
            else:
                status.out_of_sync_resources += 1

        # Detect drifts
        self._detect_drifts(app, status)

        # Store status
        with self._lock:
            old_status = self._sync_statuses.get(app.name)
            self._sync_statuses[app.name] = status

        # Update metrics
        self._update_sync_metrics(app, status)

        # Notify callbacks
        if old_status is None or old_status.sync_status != status.sync_status:
            for callback in self._sync_callbacks:
                try:
                    callback(status)
                except Exception as e:
                    logger.error(f"Sync callback error: {e}")

    def _detect_drifts(self, app: Application, status: SyncStatus):
        """Detect configuration drifts."""
        for resource in app.resources:
            if resource.sync_status != "Synced":
                drift = SyncDrift(
                    application=app.name,
                    drift_type=self._classify_drift(resource),
                    resource_kind=resource.kind,
                    resource_name=resource.name,
                    resource_namespace=resource.namespace,
                    expected_value="synced",
                    actual_value=resource.sync_status,
                )

                # Check if drift already exists
                with self._lock:
                    existing = self._find_existing_drift(drift)

                if not existing:
                    self._record_drift(drift)
                    status.drifts.append(drift)

    def _classify_drift(self, resource) -> SyncDriftType:
        """Classify the type of drift based on resource."""
        kind = resource.kind.lower()

        if kind in ("configmap", "secret"):
            return SyncDriftType.CONFIG_DRIFT
        elif kind in ("deployment", "statefulset", "daemonset"):
            if "image" in str(resource.__dict__).lower():
                return SyncDriftType.IMAGE_DRIFT
            return SyncDriftType.RESOURCE_DRIFT
        elif kind == "secret":
            return SyncDriftType.SECRET_DRIFT
        elif kind in ("hpa", "horizontalpodautoscaler"):
            return SyncDriftType.SCALE_DRIFT
        else:
            return SyncDriftType.UNKNOWN

    def _find_existing_drift(self, drift: SyncDrift) -> Optional[SyncDrift]:
        """Find existing unresolved drift matching this one."""
        app_drifts = self._active_drifts.get(drift.application, [])

        for existing in app_drifts:
            if (
                not existing.resolved
                and existing.resource_kind == drift.resource_kind
                and existing.resource_name == drift.resource_name
                and existing.resource_namespace == drift.resource_namespace
            ):
                return existing

        return None

    def _record_drift(self, drift: SyncDrift):
        """Record a new drift."""
        with self._lock:
            if drift.application not in self._active_drifts:
                self._active_drifts[drift.application] = []
            self._active_drifts[drift.application].append(drift)

            # Trim old resolved drifts
            if len(self._active_drifts[drift.application]) > self._max_drifts:
                self._active_drifts[drift.application] = [
                    d for d in self._active_drifts[drift.application]
                    if not d.resolved
                ][-self._max_drifts:]

        # Update metrics
        self.drift_detected.labels(
            application=drift.application,
            drift_type=drift.drift_type.value,
        ).inc()

        self._update_drift_gauge(drift.application)

        # Notify callbacks
        for callback in self._drift_callbacks:
            try:
                callback(drift)
            except Exception as e:
                logger.error(f"Drift callback error: {e}")

        logger.warning(
            f"Drift detected: {drift.application} - {drift.resource_kind}/{drift.resource_name}"
        )

    def _update_sync_metrics(self, app: Application, status: SyncStatus):
        """Update Prometheus metrics for sync status."""
        self.sync_status_gauge.labels(
            application=app.name,
            project=app.project,
        ).set(1 if status.is_synced else 0)

        self.sync_percentage.labels(
            application=app.name,
        ).set(status.sync_percentage)

        self.out_of_sync_resources.labels(
            application=app.name,
        ).set(status.out_of_sync_resources)

        # Time since last sync
        if status.sync_time:
            seconds_since = (datetime.now() - status.sync_time).total_seconds()
            self.time_since_last_sync.labels(
                application=app.name,
            ).set(seconds_since)

    def _update_drift_gauge(self, application: str):
        """Update active drifts gauge."""
        with self._lock:
            drifts = self._active_drifts.get(application, [])
            active = sum(1 for d in drifts if not d.resolved)

        self.active_drifts_gauge.labels(application=application).set(active)

    def get_sync_status(self, application: str) -> Optional[SyncStatus]:
        """Get sync status for an application.

        Args:
            application: Application name

        Returns:
            SyncStatus or None
        """
        with self._lock:
            return self._sync_statuses.get(application)

    def get_all_sync_statuses(self) -> Dict[str, SyncStatus]:
        """Get sync status for all applications.

        Returns:
            Dictionary of application -> SyncStatus
        """
        with self._lock:
            return dict(self._sync_statuses)

    def get_active_drifts(
        self,
        application: Optional[str] = None,
    ) -> List[SyncDrift]:
        """Get active (unresolved) drifts.

        Args:
            application: Filter by application (optional)

        Returns:
            List of active drifts
        """
        with self._lock:
            if application:
                drifts = self._active_drifts.get(application, [])
                return [d for d in drifts if not d.resolved]
            else:
                all_drifts = []
                for app_drifts in self._active_drifts.values():
                    all_drifts.extend([d for d in app_drifts if not d.resolved])
                return all_drifts

    def resolve_drift(
        self,
        application: str,
        resource_kind: str,
        resource_name: str,
    ) -> bool:
        """Mark a drift as resolved.

        Args:
            application: Application name
            resource_kind: Resource kind
            resource_name: Resource name

        Returns:
            True if drift was found and resolved
        """
        with self._lock:
            drifts = self._active_drifts.get(application, [])

            for drift in drifts:
                if (
                    not drift.resolved
                    and drift.resource_kind == resource_kind
                    and drift.resource_name == resource_name
                ):
                    drift.resolved = True
                    drift.resolved_at = datetime.now()

                    # Update metrics
                    self.drift_resolved.labels(
                        application=application,
                        drift_type=drift.drift_type.value,
                    ).inc()

                    self._update_drift_gauge(application)

                    logger.info(
                        f"Drift resolved: {application} - {resource_kind}/{resource_name}"
                    )

                    return True

        return False

    def record_sync_failure(
        self,
        application: str,
        sync_id: str,
        reason: SyncFailureReason,
        message: str,
        resources_affected: Optional[List[str]] = None,
    ) -> SyncFailure:
        """Record a sync failure.

        Args:
            application: Application name
            sync_id: Sync operation ID
            reason: Failure reason
            message: Error message
            resources_affected: List of affected resources

        Returns:
            SyncFailure record
        """
        failure = SyncFailure(
            application=application,
            sync_id=sync_id,
            reason=reason,
            message=message,
            resources_affected=resources_affected or [],
        )

        with self._lock:
            if application not in self._sync_failures:
                self._sync_failures[application] = []
            self._sync_failures[application].append(failure)

            # Trim old failures
            if len(self._sync_failures[application]) > self._max_failures:
                self._sync_failures[application] = self._sync_failures[application][-self._max_failures:]

        # Update metrics
        self.sync_failures_total.labels(
            application=application,
            reason=reason.value,
        ).inc()

        # Notify callbacks
        for callback in self._failure_callbacks:
            try:
                callback(failure)
            except Exception as e:
                logger.error(f"Failure callback error: {e}")

        logger.error(f"Sync failure: {application} - {reason.value}: {message}")

        return failure

    def get_sync_failures(
        self,
        application: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 50,
    ) -> List[SyncFailure]:
        """Get sync failures.

        Args:
            application: Filter by application
            since: Only failures after this time
            limit: Maximum number to return

        Returns:
            List of sync failures
        """
        with self._lock:
            if application:
                failures = self._sync_failures.get(application, [])
            else:
                failures = []
                for app_failures in self._sync_failures.values():
                    failures.extend(app_failures)

        if since:
            failures = [f for f in failures if f.occurred_at >= since]

        # Sort by time descending
        failures.sort(key=lambda f: f.occurred_at, reverse=True)

        return failures[:limit]

    def on_drift_detected(self, callback: Callable[[SyncDrift], None]):
        """Register callback for drift detection.

        Args:
            callback: Function to call when drift is detected
        """
        self._drift_callbacks.append(callback)

    def on_sync_failure(self, callback: Callable[[SyncFailure], None]):
        """Register callback for sync failures.

        Args:
            callback: Function to call on sync failure
        """
        self._failure_callbacks.append(callback)

    def on_sync_status_change(self, callback: Callable[[SyncStatus], None]):
        """Register callback for sync status changes.

        Args:
            callback: Function to call on sync status change
        """
        self._sync_callbacks.append(callback)

    def get_out_of_sync_applications(self) -> List[str]:
        """Get list of out-of-sync applications.

        Returns:
            List of application names
        """
        with self._lock:
            return [
                name for name, status in self._sync_statuses.items()
                if not status.is_synced
            ]

    def get_summary(self) -> Dict[str, Any]:
        """Get sync tracker summary.

        Returns:
            Summary dictionary
        """
        with self._lock:
            statuses = list(self._sync_statuses.values())
            all_drifts = []
            for app_drifts in self._active_drifts.values():
                all_drifts.extend(app_drifts)
            all_failures = []
            for app_failures in self._sync_failures.values():
                all_failures.extend(app_failures)

        synced = sum(1 for s in statuses if s.is_synced)
        active_drifts = sum(1 for d in all_drifts if not d.resolved)

        # Group drifts by type
        drift_by_type: Dict[str, int] = {}
        for d in all_drifts:
            if not d.resolved:
                t = d.drift_type.value
                drift_by_type[t] = drift_by_type.get(t, 0) + 1

        # Recent failures (last 24 hours)
        cutoff = datetime.now() - timedelta(hours=24)
        recent_failures = sum(1 for f in all_failures if f.occurred_at >= cutoff)

        return {
            "running": self._running,
            "total_applications": len(statuses),
            "synced_applications": synced,
            "out_of_sync_applications": len(statuses) - synced,
            "sync_percentage": (synced / len(statuses) * 100) if statuses else 100.0,
            "active_drifts": active_drifts,
            "drifts_by_type": drift_by_type,
            "total_failures": len(all_failures),
            "recent_failures_24h": recent_failures,
        }
