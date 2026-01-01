"""ArgoCD Connector

Connects to ArgoCD API for application and sync status monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
from enum import Enum
import threading
import logging
import json

from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)


class ApplicationHealthStatus(str, Enum):
    """ArgoCD application health status."""
    HEALTHY = "Healthy"
    PROGRESSING = "Progressing"
    DEGRADED = "Degraded"
    SUSPENDED = "Suspended"
    MISSING = "Missing"
    UNKNOWN = "Unknown"


class ApplicationSyncStatus(str, Enum):
    """ArgoCD application sync status."""
    SYNCED = "Synced"
    OUT_OF_SYNC = "OutOfSync"
    UNKNOWN = "Unknown"


class SyncOperationPhase(str, Enum):
    """ArgoCD sync operation phase."""
    RUNNING = "Running"
    SUCCEEDED = "Succeeded"
    FAILED = "Failed"
    ERROR = "Error"
    TERMINATED = "Terminated"


@dataclass
class ArgoCDConfig:
    """ArgoCD connection configuration.

    Usage:
        config = ArgoCDConfig(
            server_url="https://argocd.example.com",
            token="my-api-token",
            insecure=False,
        )
    """
    server_url: str
    token: str = ""
    username: str = ""
    password: str = ""
    insecure: bool = False
    grpc_web: bool = True
    timeout_seconds: int = 30
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class ApplicationResource:
    """Kubernetes resource managed by ArgoCD application."""
    group: str
    kind: str
    namespace: str
    name: str
    version: str
    sync_status: str = ""
    health_status: str = ""
    hook: bool = False
    requires_pruning: bool = False


@dataclass
class ApplicationSource:
    """ArgoCD application source configuration."""
    repo_url: str
    path: str = ""
    target_revision: str = "HEAD"
    chart: str = ""
    helm_values: Dict[str, Any] = field(default_factory=dict)
    kustomize_images: List[str] = field(default_factory=list)


@dataclass
class SyncOperation:
    """ArgoCD sync operation details."""
    sync_id: str
    application: str
    phase: SyncOperationPhase
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    message: str = ""
    revision: str = ""
    initiated_by: str = ""
    retry_count: int = 0
    resources_synced: int = 0
    resources_failed: int = 0

    @property
    def duration_seconds(self) -> float:
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sync_id": self.sync_id,
            "application": self.application,
            "phase": self.phase.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "duration_seconds": self.duration_seconds,
            "message": self.message,
            "revision": self.revision,
            "initiated_by": self.initiated_by,
            "retry_count": self.retry_count,
            "resources_synced": self.resources_synced,
            "resources_failed": self.resources_failed,
        }


@dataclass
class Application:
    """ArgoCD Application representation."""
    name: str
    namespace: str
    project: str
    source: ApplicationSource
    destination_server: str
    destination_namespace: str
    health_status: ApplicationHealthStatus = ApplicationHealthStatus.UNKNOWN
    sync_status: ApplicationSyncStatus = ApplicationSyncStatus.UNKNOWN
    sync_revision: str = ""
    target_revision: str = ""
    created_at: Optional[datetime] = None
    synced_at: Optional[datetime] = None
    resources: List[ApplicationResource] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    conditions: List[Dict[str, str]] = field(default_factory=list)
    operation_state: Optional[SyncOperation] = None

    @property
    def is_healthy(self) -> bool:
        return self.health_status == ApplicationHealthStatus.HEALTHY

    @property
    def is_synced(self) -> bool:
        return self.sync_status == ApplicationSyncStatus.SYNCED

    @property
    def resource_count(self) -> int:
        return len(self.resources)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "namespace": self.namespace,
            "project": self.project,
            "destination_server": self.destination_server,
            "destination_namespace": self.destination_namespace,
            "health_status": self.health_status.value,
            "sync_status": self.sync_status.value,
            "sync_revision": self.sync_revision,
            "target_revision": self.target_revision,
            "is_healthy": self.is_healthy,
            "is_synced": self.is_synced,
            "resource_count": self.resource_count,
            "labels": self.labels,
        }


class ArgoCDConnector:
    """Connector for ArgoCD API with observability.

    Usage:
        connector = ArgoCDConnector(config)

        # List applications
        apps = connector.list_applications()

        # Get application status
        app = connector.get_application("my-app")

        # Monitor sync operations
        connector.on_sync_event(handle_sync)

        # Get sync history
        history = connector.get_sync_history("my-app")
    """

    def __init__(
        self,
        config: ArgoCDConfig,
        namespace: str = "consciousness",
    ):
        self.config = config
        self.namespace = namespace
        self._lock = threading.Lock()

        # Application cache
        self._applications: Dict[str, Application] = {}
        self._sync_history: Dict[str, List[SyncOperation]] = {}
        self._callbacks: List[Callable[[SyncOperation], None]] = []
        self._connected = False
        self._max_history = 100

        # Prometheus metrics
        self.applications_total = Gauge(
            f"{namespace}_argocd_applications_total",
            "Total ArgoCD applications",
            ["project"],
        )

        self.application_health = Gauge(
            f"{namespace}_argocd_application_health",
            "Application health status (1=healthy, 0=unhealthy)",
            ["application", "project", "health_status"],
        )

        self.application_sync = Gauge(
            f"{namespace}_argocd_application_sync",
            "Application sync status (1=synced, 0=out_of_sync)",
            ["application", "project", "sync_status"],
        )

        self.sync_operations = Counter(
            f"{namespace}_argocd_sync_operations_total",
            "Total sync operations",
            ["application", "phase"],
        )

        self.sync_duration = Histogram(
            f"{namespace}_argocd_sync_duration_seconds",
            "Sync operation duration",
            ["application"],
            buckets=[1, 5, 10, 30, 60, 120, 300, 600],
        )

        self.api_requests = Counter(
            f"{namespace}_argocd_api_requests_total",
            "ArgoCD API requests",
            ["endpoint", "status"],
        )

        self.connection_status = Gauge(
            f"{namespace}_argocd_connection_status",
            "ArgoCD connection status (1=connected, 0=disconnected)",
        )

    def connect(self) -> bool:
        """Connect to ArgoCD server.

        Returns:
            True if connected successfully
        """
        try:
            # Simulate connection - in production, use actual ArgoCD client
            logger.info(f"Connecting to ArgoCD at {self.config.server_url}")

            # Validate configuration
            if not self.config.server_url:
                raise ValueError("Server URL is required")

            if not self.config.token and not (self.config.username and self.config.password):
                raise ValueError("Token or username/password is required")

            self._connected = True
            self.connection_status.set(1)

            self.api_requests.labels(endpoint="connect", status="success").inc()
            logger.info("Connected to ArgoCD")

            return True

        except Exception as e:
            self._connected = False
            self.connection_status.set(0)
            self.api_requests.labels(endpoint="connect", status="error").inc()
            logger.error(f"Failed to connect to ArgoCD: {e}")
            return False

    def disconnect(self):
        """Disconnect from ArgoCD server."""
        self._connected = False
        self.connection_status.set(0)
        logger.info("Disconnected from ArgoCD")

    @property
    def is_connected(self) -> bool:
        return self._connected

    def list_applications(
        self,
        project: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> List[Application]:
        """List all ArgoCD applications.

        Args:
            project: Filter by project name
            labels: Filter by labels

        Returns:
            List of applications
        """
        if not self._connected:
            logger.warning("Not connected to ArgoCD")
            return []

        try:
            with self._lock:
                apps = list(self._applications.values())

            # Filter by project
            if project:
                apps = [a for a in apps if a.project == project]

            # Filter by labels
            if labels:
                apps = [
                    a for a in apps
                    if all(a.labels.get(k) == v for k, v in labels.items())
                ]

            self.api_requests.labels(endpoint="list_applications", status="success").inc()
            return apps

        except Exception as e:
            self.api_requests.labels(endpoint="list_applications", status="error").inc()
            logger.error(f"Failed to list applications: {e}")
            return []

    def get_application(self, name: str) -> Optional[Application]:
        """Get application by name.

        Args:
            name: Application name

        Returns:
            Application or None
        """
        if not self._connected:
            logger.warning("Not connected to ArgoCD")
            return None

        try:
            with self._lock:
                app = self._applications.get(name)

            self.api_requests.labels(endpoint="get_application", status="success").inc()
            return app

        except Exception as e:
            self.api_requests.labels(endpoint="get_application", status="error").inc()
            logger.error(f"Failed to get application {name}: {e}")
            return None

    def refresh_application(self, name: str) -> Optional[Application]:
        """Refresh application state from ArgoCD.

        Args:
            name: Application name

        Returns:
            Updated application or None
        """
        if not self._connected:
            return None

        try:
            # In production, fetch from ArgoCD API
            app = self.get_application(name)
            if app:
                self._update_application_metrics(app)

            self.api_requests.labels(endpoint="refresh_application", status="success").inc()
            return app

        except Exception as e:
            self.api_requests.labels(endpoint="refresh_application", status="error").inc()
            logger.error(f"Failed to refresh application {name}: {e}")
            return None

    def sync_application(
        self,
        name: str,
        revision: Optional[str] = None,
        prune: bool = False,
        dry_run: bool = False,
    ) -> Optional[SyncOperation]:
        """Trigger application sync.

        Args:
            name: Application name
            revision: Target revision (optional)
            prune: Enable pruning
            dry_run: Dry run mode

        Returns:
            SyncOperation or None
        """
        if not self._connected:
            return None

        try:
            app = self.get_application(name)
            if not app:
                logger.warning(f"Application not found: {name}")
                return None

            # Create sync operation
            sync_op = SyncOperation(
                sync_id=f"sync-{name}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                application=name,
                phase=SyncOperationPhase.RUNNING,
                started_at=datetime.now(),
                revision=revision or app.target_revision,
            )

            # Store sync operation
            with self._lock:
                if name not in self._sync_history:
                    self._sync_history[name] = []
                self._sync_history[name].append(sync_op)

                # Trim history
                if len(self._sync_history[name]) > self._max_history:
                    self._sync_history[name] = self._sync_history[name][-self._max_history:]

            # Update metrics
            self.sync_operations.labels(
                application=name,
                phase=sync_op.phase.value,
            ).inc()

            self.api_requests.labels(endpoint="sync_application", status="success").inc()
            logger.info(f"Triggered sync for {name} (revision: {sync_op.revision})")

            # Notify callbacks
            for callback in self._callbacks:
                try:
                    callback(sync_op)
                except Exception as e:
                    logger.error(f"Sync callback error: {e}")

            return sync_op

        except Exception as e:
            self.api_requests.labels(endpoint="sync_application", status="error").inc()
            logger.error(f"Failed to sync application {name}: {e}")
            return None

    def get_sync_history(
        self,
        name: str,
        limit: int = 10,
    ) -> List[SyncOperation]:
        """Get sync operation history for an application.

        Args:
            name: Application name
            limit: Maximum number of operations to return

        Returns:
            List of sync operations
        """
        with self._lock:
            history = self._sync_history.get(name, [])
            return history[-limit:]

    def update_application(self, app: Application):
        """Update application in cache (for testing/simulation).

        Args:
            app: Application to update
        """
        with self._lock:
            self._applications[app.name] = app

        self._update_application_metrics(app)

    def _update_application_metrics(self, app: Application):
        """Update Prometheus metrics for an application."""
        # Health metric
        self.application_health.labels(
            application=app.name,
            project=app.project,
            health_status=app.health_status.value,
        ).set(1 if app.is_healthy else 0)

        # Sync metric
        self.application_sync.labels(
            application=app.name,
            project=app.project,
            sync_status=app.sync_status.value,
        ).set(1 if app.is_synced else 0)

    def record_sync_completion(
        self,
        sync_id: str,
        phase: SyncOperationPhase,
        message: str = "",
        resources_synced: int = 0,
        resources_failed: int = 0,
    ):
        """Record sync operation completion.

        Args:
            sync_id: Sync operation ID
            phase: Final phase
            message: Completion message
            resources_synced: Number of resources synced
            resources_failed: Number of resources failed
        """
        with self._lock:
            for app_history in self._sync_history.values():
                for sync_op in app_history:
                    if sync_op.sync_id == sync_id:
                        sync_op.phase = phase
                        sync_op.finished_at = datetime.now()
                        sync_op.message = message
                        sync_op.resources_synced = resources_synced
                        sync_op.resources_failed = resources_failed

                        # Update metrics
                        self.sync_operations.labels(
                            application=sync_op.application,
                            phase=phase.value,
                        ).inc()

                        if sync_op.duration_seconds > 0:
                            self.sync_duration.labels(
                                application=sync_op.application,
                            ).observe(sync_op.duration_seconds)

                        # Notify callbacks
                        for callback in self._callbacks:
                            try:
                                callback(sync_op)
                            except Exception as e:
                                logger.error(f"Sync callback error: {e}")

                        return

    def on_sync_event(self, callback: Callable[[SyncOperation], None]):
        """Register callback for sync events.

        Args:
            callback: Function to call on sync events
        """
        self._callbacks.append(callback)

    def get_projects(self) -> List[str]:
        """Get list of unique project names.

        Returns:
            List of project names
        """
        with self._lock:
            return list(set(app.project for app in self._applications.values()))

    def get_application_count_by_status(self) -> Dict[str, int]:
        """Get application count by health status.

        Returns:
            Dictionary of status -> count
        """
        with self._lock:
            counts: Dict[str, int] = {}
            for app in self._applications.values():
                status = app.health_status.value
                counts[status] = counts.get(status, 0) + 1
            return counts

    def get_summary(self) -> Dict[str, Any]:
        """Get connector summary.

        Returns:
            Summary dictionary
        """
        with self._lock:
            apps = list(self._applications.values())
            projects = set(a.project for a in apps)

        healthy = sum(1 for a in apps if a.is_healthy)
        synced = sum(1 for a in apps if a.is_synced)

        return {
            "connected": self._connected,
            "server_url": self.config.server_url,
            "total_applications": len(apps),
            "healthy_applications": healthy,
            "synced_applications": synced,
            "out_of_sync_applications": len(apps) - synced,
            "projects": len(projects),
            "status_breakdown": self.get_application_count_by_status(),
        }
