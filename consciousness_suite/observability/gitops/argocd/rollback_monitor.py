"""ArgoCD Rollback Monitor

Detects and monitors rollback events in ArgoCD applications.
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
    SyncOperation,
    SyncOperationPhase,
    ArgoCDConnector,
)

logger = logging.getLogger(__name__)


class RollbackType(str, Enum):
    """Types of rollback."""
    MANUAL = "manual"
    AUTOMATIC = "automatic"
    PROGRESSIVE = "progressive"
    INSTANT = "instant"


class RollbackReason(str, Enum):
    """Reasons for rollback."""
    SYNC_FAILURE = "sync_failure"
    HEALTH_DEGRADED = "health_degraded"
    CANARY_FAILED = "canary_failed"
    USER_INITIATED = "user_initiated"
    POLICY_VIOLATION = "policy_violation"
    PERFORMANCE_REGRESSION = "performance_regression"
    ERROR_RATE_SPIKE = "error_rate_spike"
    UNKNOWN = "unknown"


class RollbackStatus(str, Enum):
    """Rollback operation status."""
    INITIATED = "initiated"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class RevisionInfo:
    """Git revision information."""
    revision: str
    author: str = ""
    message: str = ""
    timestamp: Optional[datetime] = None
    branch: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "revision": self.revision,
            "author": self.author,
            "message": self.message,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "branch": self.branch,
            "tags": self.tags,
        }


@dataclass
class RollbackEvent:
    """Rollback event record."""
    rollback_id: str
    application: str
    project: str
    rollback_type: RollbackType
    reason: RollbackReason
    status: RollbackStatus
    from_revision: RevisionInfo
    to_revision: RevisionInfo
    initiated_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    initiated_by: str = ""
    message: str = ""
    affected_resources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        if self.completed_at:
            return (self.completed_at - self.initiated_at).total_seconds()
        return (datetime.now() - self.initiated_at).total_seconds()

    @property
    def is_in_progress(self) -> bool:
        return self.status in (RollbackStatus.INITIATED, RollbackStatus.IN_PROGRESS)

    @property
    def is_successful(self) -> bool:
        return self.status == RollbackStatus.COMPLETED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rollback_id": self.rollback_id,
            "application": self.application,
            "project": self.project,
            "rollback_type": self.rollback_type.value,
            "reason": self.reason.value,
            "status": self.status.value,
            "from_revision": self.from_revision.to_dict(),
            "to_revision": self.to_revision.to_dict(),
            "initiated_at": self.initiated_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "initiated_by": self.initiated_by,
            "message": self.message,
            "affected_resources": self.affected_resources,
        }


@dataclass
class RollbackPolicy:
    """Rollback policy configuration."""
    policy_id: str
    application_pattern: str  # glob pattern for matching apps
    enabled: bool = True
    auto_rollback_on_failure: bool = True
    max_auto_rollbacks: int = 3
    cooldown_minutes: int = 15
    rollback_window_hours: int = 24
    notify_on_rollback: bool = True
    require_approval: bool = False
    excluded_revisions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RollbackAnalysis:
    """Analysis of rollback patterns."""
    application: str
    total_rollbacks: int
    successful_rollbacks: int
    failed_rollbacks: int
    average_duration_seconds: float
    rollback_frequency_per_week: float
    most_common_reason: Optional[RollbackReason]
    last_rollback: Optional[datetime]
    time_between_rollbacks_hours: float


class RollbackMonitor:
    """Monitors and tracks ArgoCD rollback events.

    Usage:
        monitor = RollbackMonitor(connector)

        # Start monitoring
        monitor.start()

        # Get rollback history
        rollbacks = monitor.get_rollback_history("my-app")

        # Check for active rollbacks
        active = monitor.get_active_rollbacks()

        # Register callbacks
        monitor.on_rollback_started(handle_rollback_start)
        monitor.on_rollback_completed(handle_rollback_complete)
    """

    def __init__(
        self,
        connector: Optional[ArgoCDConnector] = None,
        namespace: str = "consciousness",
        check_interval_seconds: int = 10,
    ):
        self.connector = connector
        self.namespace = namespace
        self.check_interval = check_interval_seconds
        self._lock = threading.Lock()

        # State
        self._rollback_history: Dict[str, List[RollbackEvent]] = {}
        self._active_rollbacks: Dict[str, RollbackEvent] = {}
        self._policies: Dict[str, RollbackPolicy] = {}
        self._revision_history: Dict[str, List[str]] = {}  # app -> [revisions]
        self._auto_rollback_counts: Dict[str, int] = {}
        self._last_rollback_times: Dict[str, datetime] = {}

        # Callbacks
        self._start_callbacks: List[Callable[[RollbackEvent], None]] = []
        self._complete_callbacks: List[Callable[[RollbackEvent], None]] = []

        # Tracking
        self._running = False
        self._check_thread: Optional[threading.Thread] = None
        self._max_history = 500

        # Prometheus metrics
        self.rollbacks_total = Counter(
            f"{namespace}_gitops_rollbacks_total",
            "Total rollback operations",
            ["application", "type", "reason"],
        )

        self.rollbacks_in_progress = Gauge(
            f"{namespace}_gitops_rollbacks_in_progress",
            "Rollbacks currently in progress",
            ["application"],
        )

        self.rollback_duration = Histogram(
            f"{namespace}_gitops_rollback_duration_seconds",
            "Rollback operation duration",
            ["application", "type"],
            buckets=[5, 10, 30, 60, 120, 300, 600, 1800],
        )

        self.rollback_success_rate = Gauge(
            f"{namespace}_gitops_rollback_success_rate",
            "Rollback success rate (0-1)",
            ["application"],
        )

        self.auto_rollbacks = Counter(
            f"{namespace}_gitops_auto_rollbacks_total",
            "Automatic rollbacks triggered",
            ["application", "reason"],
        )

        self.time_since_last_rollback = Gauge(
            f"{namespace}_gitops_time_since_last_rollback_seconds",
            "Seconds since last rollback",
            ["application"],
        )

    def start(self):
        """Start rollback monitoring."""
        if self._running:
            return

        self._running = True
        self._check_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
        )
        self._check_thread.start()
        logger.info("Rollback monitor started")

    def stop(self):
        """Stop rollback monitoring."""
        self._running = False
        if self._check_thread:
            self._check_thread.join(timeout=5)
        logger.info("Rollback monitor stopped")

    def _monitor_loop(self):
        """Background monitoring loop."""
        import time

        while self._running:
            try:
                self._check_for_rollbacks()
                self._update_time_metrics()
            except Exception as e:
                logger.error(f"Rollback monitor error: {e}")

            time.sleep(self.check_interval)

    def _check_for_rollbacks(self):
        """Check for new rollback events."""
        if not self.connector or not self.connector.is_connected:
            return

        apps = self.connector.list_applications()

        for app in apps:
            self._detect_rollback(app)

    def _detect_rollback(self, app: Application):
        """Detect if application has rolled back."""
        with self._lock:
            history = self._revision_history.get(app.name, [])

        current_revision = app.sync_revision

        if not current_revision:
            return

        if not history:
            # First time seeing this app
            with self._lock:
                self._revision_history[app.name] = [current_revision]
            return

        # Check if revision went backwards (rollback indicator)
        if current_revision in history[:-1]:
            # This is a revision we've seen before, but not the latest
            # This indicates a rollback
            previous_revision = history[-1] if history else ""

            if previous_revision != current_revision:
                self._record_rollback_detection(
                    app=app,
                    from_revision=previous_revision,
                    to_revision=current_revision,
                )

        # Update history
        with self._lock:
            if current_revision not in self._revision_history[app.name]:
                self._revision_history[app.name].append(current_revision)
                # Keep last 100 revisions
                if len(self._revision_history[app.name]) > 100:
                    self._revision_history[app.name] = self._revision_history[app.name][-100:]

    def _record_rollback_detection(
        self,
        app: Application,
        from_revision: str,
        to_revision: str,
    ):
        """Record detected rollback."""
        rollback_id = f"rollback-{app.name}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        rollback = RollbackEvent(
            rollback_id=rollback_id,
            application=app.name,
            project=app.project,
            rollback_type=RollbackType.AUTOMATIC,  # Detected, assume automatic
            reason=RollbackReason.UNKNOWN,  # Will be updated if known
            status=RollbackStatus.COMPLETED,  # Already happened
            from_revision=RevisionInfo(revision=from_revision),
            to_revision=RevisionInfo(revision=to_revision),
            completed_at=datetime.now(),
        )

        self._store_rollback(rollback)

    def record_rollback(
        self,
        application: str,
        project: str,
        from_revision: str,
        to_revision: str,
        rollback_type: RollbackType = RollbackType.MANUAL,
        reason: RollbackReason = RollbackReason.USER_INITIATED,
        initiated_by: str = "",
        message: str = "",
    ) -> RollbackEvent:
        """Manually record a rollback event.

        Args:
            application: Application name
            project: Project name
            from_revision: Source revision
            to_revision: Target revision
            rollback_type: Type of rollback
            reason: Reason for rollback
            initiated_by: User/system that initiated
            message: Optional message

        Returns:
            RollbackEvent
        """
        rollback_id = f"rollback-{application}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        rollback = RollbackEvent(
            rollback_id=rollback_id,
            application=application,
            project=project,
            rollback_type=rollback_type,
            reason=reason,
            status=RollbackStatus.INITIATED,
            from_revision=RevisionInfo(revision=from_revision),
            to_revision=RevisionInfo(revision=to_revision),
            initiated_by=initiated_by,
            message=message,
        )

        self._store_rollback(rollback)

        with self._lock:
            self._active_rollbacks[rollback_id] = rollback

        # Update metrics
        self.rollbacks_in_progress.labels(application=application).inc()

        # Notify callbacks
        for callback in self._start_callbacks:
            try:
                callback(rollback)
            except Exception as e:
                logger.error(f"Rollback start callback error: {e}")

        logger.info(
            f"Rollback initiated: {application} from {from_revision[:8]} to {to_revision[:8]}"
        )

        return rollback

    def _store_rollback(self, rollback: RollbackEvent):
        """Store rollback in history."""
        with self._lock:
            if rollback.application not in self._rollback_history:
                self._rollback_history[rollback.application] = []

            self._rollback_history[rollback.application].append(rollback)
            self._last_rollback_times[rollback.application] = rollback.initiated_at

            # Trim history
            if len(self._rollback_history[rollback.application]) > self._max_history:
                self._rollback_history[rollback.application] = (
                    self._rollback_history[rollback.application][-self._max_history:]
                )

        # Update metrics
        self.rollbacks_total.labels(
            application=rollback.application,
            type=rollback.rollback_type.value,
            reason=rollback.reason.value,
        ).inc()

        if rollback.rollback_type == RollbackType.AUTOMATIC:
            self.auto_rollbacks.labels(
                application=rollback.application,
                reason=rollback.reason.value,
            ).inc()

        self._update_success_rate(rollback.application)

    def complete_rollback(
        self,
        rollback_id: str,
        status: RollbackStatus = RollbackStatus.COMPLETED,
        message: str = "",
        affected_resources: Optional[List[str]] = None,
    ) -> Optional[RollbackEvent]:
        """Complete a rollback operation.

        Args:
            rollback_id: Rollback ID
            status: Final status
            message: Completion message
            affected_resources: List of affected resources

        Returns:
            Updated RollbackEvent or None
        """
        with self._lock:
            if rollback_id not in self._active_rollbacks:
                return None

            rollback = self._active_rollbacks.pop(rollback_id)

        rollback.status = status
        rollback.completed_at = datetime.now()
        rollback.message = message
        if affected_resources:
            rollback.affected_resources = affected_resources

        # Update metrics
        self.rollbacks_in_progress.labels(application=rollback.application).dec()

        self.rollback_duration.labels(
            application=rollback.application,
            type=rollback.rollback_type.value,
        ).observe(rollback.duration_seconds)

        self._update_success_rate(rollback.application)

        # Notify callbacks
        for callback in self._complete_callbacks:
            try:
                callback(rollback)
            except Exception as e:
                logger.error(f"Rollback complete callback error: {e}")

        logger.info(
            f"Rollback completed: {rollback.application} - {status.value} "
            f"({rollback.duration_seconds:.1f}s)"
        )

        return rollback

    def _update_success_rate(self, application: str):
        """Update success rate metric."""
        with self._lock:
            history = self._rollback_history.get(application, [])

        if not history:
            return

        successful = sum(1 for r in history if r.status == RollbackStatus.COMPLETED)
        rate = successful / len(history)

        self.rollback_success_rate.labels(application=application).set(rate)

    def _update_time_metrics(self):
        """Update time-based metrics."""
        now = datetime.now()

        with self._lock:
            for app, last_time in self._last_rollback_times.items():
                seconds = (now - last_time).total_seconds()
                self.time_since_last_rollback.labels(application=app).set(seconds)

    def get_active_rollbacks(self) -> List[RollbackEvent]:
        """Get currently active rollbacks.

        Returns:
            List of active rollback events
        """
        with self._lock:
            return list(self._active_rollbacks.values())

    def get_rollback_history(
        self,
        application: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 50,
    ) -> List[RollbackEvent]:
        """Get rollback history.

        Args:
            application: Filter by application
            since: Only rollbacks after this time
            limit: Maximum number to return

        Returns:
            List of rollback events
        """
        with self._lock:
            if application:
                rollbacks = list(self._rollback_history.get(application, []))
            else:
                rollbacks = []
                for app_history in self._rollback_history.values():
                    rollbacks.extend(app_history)

        if since:
            rollbacks = [r for r in rollbacks if r.initiated_at >= since]

        # Sort by time descending
        rollbacks.sort(key=lambda r: r.initiated_at, reverse=True)

        return rollbacks[:limit]

    def add_policy(self, policy: RollbackPolicy):
        """Add a rollback policy.

        Args:
            policy: Rollback policy configuration
        """
        with self._lock:
            self._policies[policy.policy_id] = policy

        logger.info(f"Added rollback policy: {policy.policy_id}")

    def remove_policy(self, policy_id: str):
        """Remove a rollback policy.

        Args:
            policy_id: Policy ID to remove
        """
        with self._lock:
            self._policies.pop(policy_id, None)

    def get_policy(self, application: str) -> Optional[RollbackPolicy]:
        """Get applicable policy for an application.

        Args:
            application: Application name

        Returns:
            RollbackPolicy or None
        """
        import fnmatch

        with self._lock:
            for policy in self._policies.values():
                if policy.enabled and fnmatch.fnmatch(application, policy.application_pattern):
                    return policy
        return None

    def should_auto_rollback(
        self,
        application: str,
        reason: RollbackReason,
    ) -> bool:
        """Check if automatic rollback should be triggered.

        Args:
            application: Application name
            reason: Reason for potential rollback

        Returns:
            True if auto-rollback is allowed
        """
        policy = self.get_policy(application)

        if not policy:
            return False

        if not policy.auto_rollback_on_failure:
            return False

        # Check cooldown
        with self._lock:
            last_rollback = self._last_rollback_times.get(application)
            auto_count = self._auto_rollback_counts.get(application, 0)

        if last_rollback:
            cooldown = timedelta(minutes=policy.cooldown_minutes)
            if datetime.now() - last_rollback < cooldown:
                logger.warning(
                    f"Auto-rollback blocked for {application}: cooldown period"
                )
                return False

        # Check max auto-rollbacks
        if auto_count >= policy.max_auto_rollbacks:
            logger.warning(
                f"Auto-rollback blocked for {application}: max limit reached"
            )
            return False

        return True

    def analyze_rollbacks(self, application: str) -> Optional[RollbackAnalysis]:
        """Analyze rollback patterns for an application.

        Args:
            application: Application name

        Returns:
            RollbackAnalysis or None
        """
        with self._lock:
            history = list(self._rollback_history.get(application, []))

        if not history:
            return None

        total = len(history)
        successful = sum(1 for r in history if r.status == RollbackStatus.COMPLETED)
        failed = sum(1 for r in history if r.status == RollbackStatus.FAILED)

        # Average duration
        completed = [r for r in history if r.completed_at]
        avg_duration = (
            sum(r.duration_seconds for r in completed) / len(completed)
            if completed else 0.0
        )

        # Frequency calculation
        if len(history) >= 2:
            first = history[0].initiated_at
            last = history[-1].initiated_at
            span_days = (last - first).total_seconds() / 86400
            frequency = (total / span_days) * 7 if span_days > 0 else 0
        else:
            frequency = 0

        # Most common reason
        reason_counts: Dict[RollbackReason, int] = {}
        for r in history:
            reason_counts[r.reason] = reason_counts.get(r.reason, 0) + 1

        most_common = max(reason_counts, key=reason_counts.get) if reason_counts else None

        # Time between rollbacks
        if len(history) >= 2:
            gaps = []
            for i in range(1, len(history)):
                gap = (history[i].initiated_at - history[i-1].initiated_at).total_seconds() / 3600
                gaps.append(gap)
            avg_gap = sum(gaps) / len(gaps)
        else:
            avg_gap = 0

        return RollbackAnalysis(
            application=application,
            total_rollbacks=total,
            successful_rollbacks=successful,
            failed_rollbacks=failed,
            average_duration_seconds=avg_duration,
            rollback_frequency_per_week=frequency,
            most_common_reason=most_common,
            last_rollback=history[-1].initiated_at if history else None,
            time_between_rollbacks_hours=avg_gap,
        )

    def on_rollback_started(self, callback: Callable[[RollbackEvent], None]):
        """Register callback for rollback start events.

        Args:
            callback: Function to call when rollback starts
        """
        self._start_callbacks.append(callback)

    def on_rollback_completed(self, callback: Callable[[RollbackEvent], None]):
        """Register callback for rollback completion events.

        Args:
            callback: Function to call when rollback completes
        """
        self._complete_callbacks.append(callback)

    def get_summary(self) -> Dict[str, Any]:
        """Get rollback monitor summary.

        Returns:
            Summary dictionary
        """
        with self._lock:
            all_rollbacks = []
            for history in self._rollback_history.values():
                all_rollbacks.extend(history)
            active_count = len(self._active_rollbacks)

        # Stats from last 24 hours
        cutoff = datetime.now() - timedelta(hours=24)
        recent = [r for r in all_rollbacks if r.initiated_at >= cutoff]

        successful = sum(1 for r in recent if r.status == RollbackStatus.COMPLETED)
        failed = sum(1 for r in recent if r.status == RollbackStatus.FAILED)

        # By reason
        by_reason: Dict[str, int] = {}
        for r in recent:
            by_reason[r.reason.value] = by_reason.get(r.reason.value, 0) + 1

        # By type
        by_type: Dict[str, int] = {}
        for r in recent:
            by_type[r.rollback_type.value] = by_type.get(r.rollback_type.value, 0) + 1

        return {
            "running": self._running,
            "active_rollbacks": active_count,
            "total_rollbacks_24h": len(recent),
            "successful_24h": successful,
            "failed_24h": failed,
            "success_rate_24h": successful / len(recent) if recent else 1.0,
            "by_reason_24h": by_reason,
            "by_type_24h": by_type,
            "total_policies": len(self._policies),
            "applications_tracked": len(self._revision_history),
        }
