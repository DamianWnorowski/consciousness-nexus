"""ArgoCD Observability Module

ArgoCD integration for GitOps observability:
- Application sync monitoring
- Rollback detection and tracking
- Drift detection
- Deployment metrics
"""

from .connector import (
    ArgoCDConnector,
    ArgoCDConfig,
    Application,
    ApplicationSource,
    ApplicationResource,
    ApplicationHealthStatus,
    ApplicationSyncStatus,
    SyncOperation,
    SyncOperationPhase,
)
from .sync_tracker import (
    SyncTracker,
    SyncStatus,
    SyncDrift,
    SyncDriftType,
    SyncFailure,
    SyncFailureReason,
)
from .rollback_monitor import (
    RollbackMonitor,
    RollbackEvent,
    RollbackType,
    RollbackReason,
    RollbackStatus,
    RollbackPolicy,
    RollbackAnalysis,
    RevisionInfo,
)

__all__ = [
    # Connector
    "ArgoCDConnector",
    "ArgoCDConfig",
    "Application",
    "ApplicationSource",
    "ApplicationResource",
    "ApplicationHealthStatus",
    "ApplicationSyncStatus",
    "SyncOperation",
    "SyncOperationPhase",
    # Sync Tracker
    "SyncTracker",
    "SyncStatus",
    "SyncDrift",
    "SyncDriftType",
    "SyncFailure",
    "SyncFailureReason",
    # Rollback Monitor
    "RollbackMonitor",
    "RollbackEvent",
    "RollbackType",
    "RollbackReason",
    "RollbackStatus",
    "RollbackPolicy",
    "RollbackAnalysis",
    "RevisionInfo",
]
