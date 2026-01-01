"""Cross-Tool Deployment Tracker

Tracks deployments across multiple GitOps tools (ArgoCD, Flux, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
import threading
import logging
import uuid

from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)


class DeploymentTool(str, Enum):
    """GitOps deployment tool."""
    ARGOCD = "argocd"
    FLUX = "flux"
    HELM = "helm"
    KUBECTL = "kubectl"
    KUSTOMIZE = "kustomize"
    TERRAFORM = "terraform"
    PULUMI = "pulumi"
    OTHER = "other"


class DeploymentPhase(str, Enum):
    """Deployment phase."""
    PENDING = "pending"
    INITIALIZING = "initializing"
    IN_PROGRESS = "in_progress"
    SYNCING = "syncing"
    HEALTH_CHECK = "health_check"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"


class DeploymentStrategy(str, Enum):
    """Deployment strategy."""
    ROLLING = "rolling"
    RECREATE = "recreate"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    A_B_TESTING = "a_b_testing"


@dataclass
class DeploymentResource:
    """Resource affected by deployment."""
    kind: str
    name: str
    namespace: str
    action: str  # create, update, delete
    before_image: str = ""
    after_image: str = ""
    status: str = ""
    message: str = ""


@dataclass
class DeploymentChange:
    """Change in a deployment."""
    change_type: str  # config, image, replica, secret, etc.
    resource: str
    field: str
    old_value: str
    new_value: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Deployment:
    """Unified deployment representation across tools."""
    deployment_id: str
    name: str
    namespace: str
    cluster: str
    environment: str
    tool: DeploymentTool
    phase: DeploymentPhase
    strategy: DeploymentStrategy = DeploymentStrategy.ROLLING
    source_revision: str = ""
    target_revision: str = ""
    source_repo: str = ""
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    initiated_by: str = ""
    message: str = ""
    resources: List[DeploymentResource] = field(default_factory=list)
    changes: List[DeploymentChange] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    health_status: str = ""
    error_message: str = ""

    @property
    def duration_seconds(self) -> float:
        if not self.started_at:
            return 0.0
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds()

    @property
    def is_in_progress(self) -> bool:
        return self.phase in (
            DeploymentPhase.PENDING,
            DeploymentPhase.INITIALIZING,
            DeploymentPhase.IN_PROGRESS,
            DeploymentPhase.SYNCING,
            DeploymentPhase.HEALTH_CHECK,
        )

    @property
    def is_successful(self) -> bool:
        return self.phase == DeploymentPhase.COMPLETED

    @property
    def is_failed(self) -> bool:
        return self.phase in (DeploymentPhase.FAILED, DeploymentPhase.ROLLED_BACK)

    @property
    def resource_count(self) -> int:
        return len(self.resources)

    @property
    def change_count(self) -> int:
        return len(self.changes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deployment_id": self.deployment_id,
            "name": self.name,
            "namespace": self.namespace,
            "cluster": self.cluster,
            "environment": self.environment,
            "tool": self.tool.value,
            "phase": self.phase.value,
            "strategy": self.strategy.value,
            "source_revision": self.source_revision[:12] if self.source_revision else "",
            "target_revision": self.target_revision[:12] if self.target_revision else "",
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "is_successful": self.is_successful,
            "resource_count": self.resource_count,
            "change_count": self.change_count,
            "initiated_by": self.initiated_by,
            "labels": self.labels,
        }


@dataclass
class DeploymentWindow:
    """Deployment window/freeze configuration."""
    window_id: str
    name: str
    start_time: datetime
    end_time: datetime
    environments: List[str] = field(default_factory=list)
    clusters: List[str] = field(default_factory=list)
    allow_emergency: bool = True
    reason: str = ""
    created_by: str = ""

    @property
    def is_active(self) -> bool:
        now = datetime.now()
        return self.start_time <= now <= self.end_time


@dataclass
class DeploymentStats:
    """Deployment statistics."""
    total_deployments: int = 0
    successful_deployments: int = 0
    failed_deployments: int = 0
    rolled_back_deployments: int = 0
    average_duration_seconds: float = 0.0
    deployment_frequency_per_day: float = 0.0
    success_rate: float = 0.0
    mttr_seconds: float = 0.0  # Mean time to recovery
    lead_time_seconds: float = 0.0


class DeploymentTracker:
    """Tracks deployments across multiple GitOps tools.

    Usage:
        tracker = DeploymentTracker()

        # Start a deployment
        deployment = tracker.start_deployment(
            name="my-app",
            namespace="production",
            cluster="prod-1",
            environment="production",
            tool=DeploymentTool.ARGOCD,
        )

        # Update deployment phase
        tracker.update_phase(deployment.deployment_id, DeploymentPhase.SYNCING)

        # Complete deployment
        tracker.complete_deployment(deployment.deployment_id)

        # Get deployment history
        history = tracker.get_deployment_history("my-app")

        # Get statistics
        stats = tracker.get_deployment_stats("production")
    """

    def __init__(
        self,
        namespace: str = "consciousness",
    ):
        self.namespace = namespace
        self._lock = threading.Lock()

        # State
        self._active_deployments: Dict[str, Deployment] = {}
        self._deployment_history: Dict[str, List[Deployment]] = {}
        self._deployment_windows: Dict[str, DeploymentWindow] = {}

        # Callbacks
        self._start_callbacks: List[Callable[[Deployment], None]] = []
        self._complete_callbacks: List[Callable[[Deployment], None]] = []
        self._failed_callbacks: List[Callable[[Deployment], None]] = []
        self._phase_callbacks: List[Callable[[Deployment, DeploymentPhase], None]] = []

        self._max_history = 1000

        # Prometheus metrics
        self.deployments_total = Counter(
            f"{namespace}_gitops_deployments_total",
            "Total deployments",
            ["environment", "cluster", "tool", "result"],
        )

        self.deployments_in_progress = Gauge(
            f"{namespace}_gitops_deployments_in_progress",
            "Deployments currently in progress",
            ["environment", "cluster", "tool"],
        )

        self.deployment_duration = Histogram(
            f"{namespace}_gitops_deployment_duration_seconds",
            "Deployment duration",
            ["environment", "tool", "strategy"],
            buckets=[30, 60, 120, 300, 600, 1200, 1800, 3600],
        )

        self.deployment_resources = Histogram(
            f"{namespace}_gitops_deployment_resources",
            "Resources per deployment",
            ["environment", "tool"],
            buckets=[1, 5, 10, 20, 50, 100, 200],
        )

        self.deployment_changes = Counter(
            f"{namespace}_gitops_deployment_changes_total",
            "Total deployment changes",
            ["environment", "change_type"],
        )

        self.deployment_success_rate = Gauge(
            f"{namespace}_gitops_deployment_success_rate",
            "Deployment success rate (0-1)",
            ["environment"],
        )

        self.freeze_windows_active = Gauge(
            f"{namespace}_gitops_freeze_windows_active",
            "Number of active freeze windows",
        )

        self.time_since_last_deployment = Gauge(
            f"{namespace}_gitops_time_since_last_deployment_seconds",
            "Seconds since last deployment",
            ["environment", "cluster"],
        )

    def start_deployment(
        self,
        name: str,
        namespace: str,
        cluster: str,
        environment: str,
        tool: DeploymentTool,
        strategy: DeploymentStrategy = DeploymentStrategy.ROLLING,
        source_revision: str = "",
        target_revision: str = "",
        source_repo: str = "",
        initiated_by: str = "",
        message: str = "",
        labels: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Deployment]:
        """Start tracking a new deployment.

        Args:
            name: Deployment name
            namespace: Kubernetes namespace
            cluster: Target cluster
            environment: Environment (dev, staging, production)
            tool: GitOps tool
            strategy: Deployment strategy
            source_revision: Current revision
            target_revision: Target revision
            source_repo: Source repository
            initiated_by: User/system initiating
            message: Deployment message
            labels: Custom labels
            metadata: Custom metadata

        Returns:
            Deployment or None if blocked
        """
        # Check freeze windows
        if self._is_deployment_blocked(environment, cluster):
            logger.warning(
                f"Deployment blocked: {name} - freeze window active for {environment}/{cluster}"
            )
            return None

        deployment_id = f"deploy-{uuid.uuid4().hex[:12]}"

        deployment = Deployment(
            deployment_id=deployment_id,
            name=name,
            namespace=namespace,
            cluster=cluster,
            environment=environment,
            tool=tool,
            phase=DeploymentPhase.PENDING,
            strategy=strategy,
            source_revision=source_revision,
            target_revision=target_revision,
            source_repo=source_repo,
            started_at=datetime.now(),
            initiated_by=initiated_by,
            message=message,
            labels=labels or {},
            metadata=metadata or {},
        )

        with self._lock:
            self._active_deployments[deployment_id] = deployment

        # Update metrics
        self.deployments_in_progress.labels(
            environment=environment,
            cluster=cluster,
            tool=tool.value,
        ).inc()

        # Notify callbacks
        for callback in self._start_callbacks:
            try:
                callback(deployment)
            except Exception as e:
                logger.error(f"Start callback error: {e}")

        logger.info(
            f"Started deployment: {deployment_id} - {name} ({environment}/{cluster})"
        )

        return deployment

    def _is_deployment_blocked(self, environment: str, cluster: str) -> bool:
        """Check if deployment is blocked by freeze window."""
        with self._lock:
            for window in self._deployment_windows.values():
                if not window.is_active:
                    continue

                # Check environment match
                if window.environments and environment not in window.environments:
                    continue

                # Check cluster match
                if window.clusters and cluster not in window.clusters:
                    continue

                return True

        return False

    def update_phase(
        self,
        deployment_id: str,
        phase: DeploymentPhase,
        message: str = "",
    ) -> Optional[Deployment]:
        """Update deployment phase.

        Args:
            deployment_id: Deployment ID
            phase: New phase
            message: Phase message

        Returns:
            Updated deployment or None
        """
        with self._lock:
            deployment = self._active_deployments.get(deployment_id)
            if not deployment:
                return None

            old_phase = deployment.phase
            deployment.phase = phase
            if message:
                deployment.message = message

        # Notify callbacks
        for callback in self._phase_callbacks:
            try:
                callback(deployment, old_phase)
            except Exception as e:
                logger.error(f"Phase callback error: {e}")

        logger.debug(f"Deployment phase update: {deployment_id} -> {phase.value}")

        return deployment

    def add_resource(
        self,
        deployment_id: str,
        kind: str,
        name: str,
        namespace: str,
        action: str,
        before_image: str = "",
        after_image: str = "",
    ) -> bool:
        """Add a resource to deployment tracking.

        Args:
            deployment_id: Deployment ID
            kind: Resource kind
            name: Resource name
            namespace: Resource namespace
            action: Action (create, update, delete)
            before_image: Image before change
            after_image: Image after change

        Returns:
            True if added
        """
        resource = DeploymentResource(
            kind=kind,
            name=name,
            namespace=namespace,
            action=action,
            before_image=before_image,
            after_image=after_image,
        )

        with self._lock:
            deployment = self._active_deployments.get(deployment_id)
            if deployment:
                deployment.resources.append(resource)
                return True

        return False

    def add_change(
        self,
        deployment_id: str,
        change_type: str,
        resource: str,
        field_name: str,
        old_value: str,
        new_value: str,
    ) -> bool:
        """Record a deployment change.

        Args:
            deployment_id: Deployment ID
            change_type: Type of change
            resource: Resource name
            field_name: Changed field
            old_value: Previous value
            new_value: New value

        Returns:
            True if recorded
        """
        change = DeploymentChange(
            change_type=change_type,
            resource=resource,
            field=field_name,
            old_value=old_value,
            new_value=new_value,
        )

        with self._lock:
            deployment = self._active_deployments.get(deployment_id)
            if deployment:
                deployment.changes.append(change)

                # Update metrics
                self.deployment_changes.labels(
                    environment=deployment.environment,
                    change_type=change_type,
                ).inc()

                return True

        return False

    def complete_deployment(
        self,
        deployment_id: str,
        health_status: str = "Healthy",
        message: str = "",
    ) -> Optional[Deployment]:
        """Mark deployment as completed successfully.

        Args:
            deployment_id: Deployment ID
            health_status: Final health status
            message: Completion message

        Returns:
            Completed deployment or None
        """
        return self._finish_deployment(
            deployment_id,
            DeploymentPhase.COMPLETED,
            health_status,
            message,
        )

    def fail_deployment(
        self,
        deployment_id: str,
        error_message: str,
    ) -> Optional[Deployment]:
        """Mark deployment as failed.

        Args:
            deployment_id: Deployment ID
            error_message: Error message

        Returns:
            Failed deployment or None
        """
        return self._finish_deployment(
            deployment_id,
            DeploymentPhase.FAILED,
            "Unhealthy",
            error_message,
        )

    def rollback_deployment(
        self,
        deployment_id: str,
        message: str = "",
    ) -> Optional[Deployment]:
        """Mark deployment as rolled back.

        Args:
            deployment_id: Deployment ID
            message: Rollback message

        Returns:
            Rolled back deployment or None
        """
        return self._finish_deployment(
            deployment_id,
            DeploymentPhase.ROLLED_BACK,
            "RolledBack",
            message,
        )

    def _finish_deployment(
        self,
        deployment_id: str,
        phase: DeploymentPhase,
        health_status: str,
        message: str,
    ) -> Optional[Deployment]:
        """Finish a deployment with given status."""
        with self._lock:
            deployment = self._active_deployments.pop(deployment_id, None)
            if not deployment:
                return None

            deployment.phase = phase
            deployment.completed_at = datetime.now()
            deployment.health_status = health_status
            deployment.message = message

            if phase == DeploymentPhase.FAILED:
                deployment.error_message = message

            # Add to history
            key = f"{deployment.environment}/{deployment.cluster}/{deployment.name}"
            if key not in self._deployment_history:
                self._deployment_history[key] = []
            self._deployment_history[key].append(deployment)

            # Trim history
            if len(self._deployment_history[key]) > self._max_history:
                self._deployment_history[key] = self._deployment_history[key][-self._max_history:]

        # Update metrics
        result = "success" if deployment.is_successful else "failure"
        self.deployments_total.labels(
            environment=deployment.environment,
            cluster=deployment.cluster,
            tool=deployment.tool.value,
            result=result,
        ).inc()

        self.deployments_in_progress.labels(
            environment=deployment.environment,
            cluster=deployment.cluster,
            tool=deployment.tool.value,
        ).dec()

        self.deployment_duration.labels(
            environment=deployment.environment,
            tool=deployment.tool.value,
            strategy=deployment.strategy.value,
        ).observe(deployment.duration_seconds)

        self.deployment_resources.labels(
            environment=deployment.environment,
            tool=deployment.tool.value,
        ).observe(deployment.resource_count)

        self._update_success_rate(deployment.environment)

        # Notify callbacks
        callbacks = self._complete_callbacks if deployment.is_successful else self._failed_callbacks
        for callback in callbacks:
            try:
                callback(deployment)
            except Exception as e:
                logger.error(f"Finish callback error: {e}")

        logger.info(
            f"Deployment {phase.value}: {deployment_id} - {deployment.name} "
            f"({deployment.duration_seconds:.1f}s)"
        )

        return deployment

    def _update_success_rate(self, environment: str):
        """Update success rate metric for environment."""
        with self._lock:
            deployments = []
            for key, history in self._deployment_history.items():
                if key.startswith(f"{environment}/"):
                    deployments.extend(history)

        if not deployments:
            return

        # Last 100 deployments
        recent = sorted(deployments, key=lambda d: d.started_at or datetime.min)[-100:]
        successful = sum(1 for d in recent if d.is_successful)
        rate = successful / len(recent)

        self.deployment_success_rate.labels(environment=environment).set(rate)

    def get_deployment(self, deployment_id: str) -> Optional[Deployment]:
        """Get a deployment by ID.

        Args:
            deployment_id: Deployment ID

        Returns:
            Deployment or None
        """
        with self._lock:
            return self._active_deployments.get(deployment_id)

    def get_active_deployments(
        self,
        environment: Optional[str] = None,
        cluster: Optional[str] = None,
    ) -> List[Deployment]:
        """Get active deployments.

        Args:
            environment: Filter by environment
            cluster: Filter by cluster

        Returns:
            List of active deployments
        """
        with self._lock:
            deployments = list(self._active_deployments.values())

        if environment:
            deployments = [d for d in deployments if d.environment == environment]

        if cluster:
            deployments = [d for d in deployments if d.cluster == cluster]

        return deployments

    def get_deployment_history(
        self,
        name: str,
        environment: Optional[str] = None,
        cluster: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 50,
    ) -> List[Deployment]:
        """Get deployment history for an application.

        Args:
            name: Application/deployment name
            environment: Filter by environment
            cluster: Filter by cluster
            since: Only deployments after this time
            limit: Maximum results

        Returns:
            List of deployments
        """
        with self._lock:
            all_deployments = []
            for key, history in self._deployment_history.items():
                # Check if name matches
                if name in key:
                    all_deployments.extend(history)

        # Filter by environment
        if environment:
            all_deployments = [d for d in all_deployments if d.environment == environment]

        # Filter by cluster
        if cluster:
            all_deployments = [d for d in all_deployments if d.cluster == cluster]

        # Filter by time
        if since:
            all_deployments = [
                d for d in all_deployments
                if d.started_at and d.started_at >= since
            ]

        # Sort by time descending
        all_deployments.sort(key=lambda d: d.started_at or datetime.min, reverse=True)

        return all_deployments[:limit]

    def get_deployment_stats(
        self,
        environment: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> DeploymentStats:
        """Get deployment statistics.

        Args:
            environment: Filter by environment
            since: Only consider deployments after this time

        Returns:
            DeploymentStats
        """
        with self._lock:
            all_deployments = []
            for history in self._deployment_history.values():
                all_deployments.extend(history)

        if environment:
            all_deployments = [d for d in all_deployments if d.environment == environment]

        if since:
            all_deployments = [
                d for d in all_deployments
                if d.started_at and d.started_at >= since
            ]

        if not all_deployments:
            return DeploymentStats()

        total = len(all_deployments)
        successful = sum(1 for d in all_deployments if d.is_successful)
        failed = sum(1 for d in all_deployments if d.phase == DeploymentPhase.FAILED)
        rolled_back = sum(1 for d in all_deployments if d.phase == DeploymentPhase.ROLLED_BACK)

        # Average duration
        completed = [d for d in all_deployments if d.completed_at]
        avg_duration = (
            sum(d.duration_seconds for d in completed) / len(completed)
            if completed else 0.0
        )

        # Frequency
        if len(all_deployments) >= 2:
            sorted_deps = sorted(all_deployments, key=lambda d: d.started_at or datetime.min)
            first = sorted_deps[0].started_at
            last = sorted_deps[-1].started_at
            if first and last:
                days = (last - first).total_seconds() / 86400
                frequency = total / days if days > 0 else 0
            else:
                frequency = 0
        else:
            frequency = 0

        # Success rate
        success_rate = successful / total if total > 0 else 0.0

        # MTTR (Mean Time to Recovery) - average time of failed deployments
        failed_deployments = [d for d in all_deployments if d.is_failed]
        mttr = (
            sum(d.duration_seconds for d in failed_deployments) / len(failed_deployments)
            if failed_deployments else 0.0
        )

        return DeploymentStats(
            total_deployments=total,
            successful_deployments=successful,
            failed_deployments=failed,
            rolled_back_deployments=rolled_back,
            average_duration_seconds=avg_duration,
            deployment_frequency_per_day=frequency,
            success_rate=success_rate,
            mttr_seconds=mttr,
        )

    def add_freeze_window(
        self,
        name: str,
        start_time: datetime,
        end_time: datetime,
        environments: Optional[List[str]] = None,
        clusters: Optional[List[str]] = None,
        allow_emergency: bool = True,
        reason: str = "",
        created_by: str = "",
    ) -> DeploymentWindow:
        """Add a deployment freeze window.

        Args:
            name: Window name
            start_time: Window start
            end_time: Window end
            environments: Affected environments
            clusters: Affected clusters
            allow_emergency: Allow emergency deployments
            reason: Reason for freeze
            created_by: Creator

        Returns:
            DeploymentWindow
        """
        window_id = f"freeze-{uuid.uuid4().hex[:8]}"

        window = DeploymentWindow(
            window_id=window_id,
            name=name,
            start_time=start_time,
            end_time=end_time,
            environments=environments or [],
            clusters=clusters or [],
            allow_emergency=allow_emergency,
            reason=reason,
            created_by=created_by,
        )

        with self._lock:
            self._deployment_windows[window_id] = window

        self._update_freeze_window_metric()

        logger.info(f"Added freeze window: {name} ({start_time} - {end_time})")

        return window

    def remove_freeze_window(self, window_id: str) -> bool:
        """Remove a freeze window.

        Args:
            window_id: Window ID

        Returns:
            True if removed
        """
        with self._lock:
            if window_id in self._deployment_windows:
                del self._deployment_windows[window_id]
                self._update_freeze_window_metric()
                return True

        return False

    def _update_freeze_window_metric(self):
        """Update active freeze windows metric."""
        with self._lock:
            active = sum(1 for w in self._deployment_windows.values() if w.is_active)

        self.freeze_windows_active.set(active)

    def get_active_freeze_windows(self) -> List[DeploymentWindow]:
        """Get currently active freeze windows.

        Returns:
            List of active freeze windows
        """
        with self._lock:
            return [w for w in self._deployment_windows.values() if w.is_active]

    def on_deployment_started(self, callback: Callable[[Deployment], None]):
        """Register callback for deployment start.

        Args:
            callback: Function to call on deployment start
        """
        self._start_callbacks.append(callback)

    def on_deployment_completed(self, callback: Callable[[Deployment], None]):
        """Register callback for successful deployment.

        Args:
            callback: Function to call on successful deployment
        """
        self._complete_callbacks.append(callback)

    def on_deployment_failed(self, callback: Callable[[Deployment], None]):
        """Register callback for failed deployment.

        Args:
            callback: Function to call on failed deployment
        """
        self._failed_callbacks.append(callback)

    def on_phase_change(self, callback: Callable[[Deployment, DeploymentPhase], None]):
        """Register callback for deployment phase changes.

        Args:
            callback: Function to call on phase change
        """
        self._phase_callbacks.append(callback)

    def get_summary(self) -> Dict[str, Any]:
        """Get tracker summary.

        Returns:
            Summary dictionary
        """
        with self._lock:
            active = list(self._active_deployments.values())
            all_deployments = []
            for history in self._deployment_history.values():
                all_deployments.extend(history)
            freeze_windows = list(self._deployment_windows.values())

        # Last 24 hours stats
        cutoff = datetime.now() - timedelta(hours=24)
        recent = [d for d in all_deployments if d.started_at and d.started_at >= cutoff]

        # By tool
        by_tool: Dict[str, int] = {}
        for d in recent:
            by_tool[d.tool.value] = by_tool.get(d.tool.value, 0) + 1

        # By environment
        by_env: Dict[str, int] = {}
        for d in recent:
            by_env[d.environment] = by_env.get(d.environment, 0) + 1

        successful = sum(1 for d in recent if d.is_successful)
        failed = sum(1 for d in recent if d.is_failed)

        return {
            "active_deployments": len(active),
            "deployments_24h": len(recent),
            "successful_24h": successful,
            "failed_24h": failed,
            "success_rate_24h": successful / len(recent) if recent else 1.0,
            "by_tool_24h": by_tool,
            "by_environment_24h": by_env,
            "active_freeze_windows": sum(1 for w in freeze_windows if w.is_active),
            "total_freeze_windows": len(freeze_windows),
        }
