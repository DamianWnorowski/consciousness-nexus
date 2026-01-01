"""Model Registry and Version Tracking

Provides model lifecycle management:
- Model version registration
- Deployment stage tracking
- Model metadata management
- Rollback support
"""

from __future__ import annotations

import time
import json
import hashlib
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from prometheus_client import Counter, Gauge, Info

logger = logging.getLogger(__name__)


class ModelStage(str, Enum):
    """Model deployment stages."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"


class DeploymentStatus(str, Enum):
    """Model deployment status."""
    PENDING = "pending"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class ModelType(str, Enum):
    """Types of models."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    RECOMMENDATION = "recommendation"
    NLP = "nlp"
    COMPUTER_VISION = "computer_vision"
    TIME_SERIES = "time_series"
    EMBEDDING = "embedding"
    CUSTOM = "custom"


@dataclass
class ModelMetadata:
    """Metadata for a model."""
    model_name: str
    model_type: ModelType
    framework: str
    description: str = ""
    owner: str = ""
    team: str = ""
    tags: List[str] = field(default_factory=list)
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    target_name: Optional[str] = None
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    training_data_info: Dict[str, Any] = field(default_factory=dict)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelMetrics:
    """Performance metrics for a model version."""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_roc: Optional[float] = None
    mse: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None
    latency_p50_ms: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    latency_p99_ms: Optional[float] = None
    throughput_rps: Optional[float] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ModelVersion:
    """A specific version of a model."""
    model_name: str
    version: str
    stage: ModelStage
    status: DeploymentStatus
    artifact_path: str
    created_at: datetime
    updated_at: datetime
    metadata: ModelMetadata
    metrics: ModelMetrics = field(default_factory=ModelMetrics)
    model_hash: Optional[str] = None
    parent_version: Optional[str] = None
    run_id: Optional[str] = None
    experiment_id: Optional[str] = None
    deployment_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StageTransition:
    """Record of a stage transition."""
    model_name: str
    version: str
    from_stage: ModelStage
    to_stage: ModelStage
    timestamp: datetime
    user: str
    reason: str = ""
    approved_by: Optional[str] = None


class ModelRegistry:
    """Registry for model versions and lifecycle management.

    Usage:
        registry = ModelRegistry(namespace="consciousness")

        # Register a new model version
        version = registry.register_model(
            model_name="fraud-detector",
            version="1.0.0",
            artifact_path="/models/fraud-detector/1.0.0",
            metadata=ModelMetadata(
                model_name="fraud-detector",
                model_type=ModelType.CLASSIFICATION,
                framework="sklearn",
            ),
        )

        # Transition to production
        registry.transition_stage(
            model_name="fraud-detector",
            version="1.0.0",
            to_stage=ModelStage.PRODUCTION,
            user="ml-engineer",
        )

        # Get production model
        prod_model = registry.get_production_model("fraud-detector")

        # Register callback for transitions
        registry.on_transition(handle_stage_change)
    """

    def __init__(self, namespace: str = "consciousness"):
        self.namespace = namespace

        self._models: Dict[str, Dict[str, ModelVersion]] = {}
        self._transitions: List[StageTransition] = []
        self._callbacks: List[Callable[[StageTransition], None]] = []
        self._lock = threading.Lock()
        self._max_transitions = 10000

        # Prometheus metrics
        self.models_registered = Gauge(
            f"{namespace}_model_registry_models_total",
            "Total registered models",
        )

        self.versions_registered = Gauge(
            f"{namespace}_model_registry_versions_total",
            "Total registered model versions",
            ["model_name"],
        )

        self.production_versions = Gauge(
            f"{namespace}_model_registry_production_versions",
            "Number of production model versions",
        )

        self.stage_transitions = Counter(
            f"{namespace}_model_registry_transitions_total",
            "Total stage transitions",
            ["model_name", "from_stage", "to_stage"],
        )

        self.deployment_status_gauge = Gauge(
            f"{namespace}_model_registry_deployment_status",
            "Deployment status (0=pending, 1=deploying, 2=deployed, 3=failed, 4=rolled_back)",
            ["model_name", "version"],
        )

        self.model_metrics_gauge = Gauge(
            f"{namespace}_model_registry_model_metric",
            "Model performance metrics",
            ["model_name", "version", "metric_name"],
        )

    def register_model(
        self,
        model_name: str,
        version: str,
        artifact_path: str,
        metadata: ModelMetadata,
        metrics: Optional[ModelMetrics] = None,
        stage: ModelStage = ModelStage.DEVELOPMENT,
        parent_version: Optional[str] = None,
        run_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
    ) -> ModelVersion:
        """Register a new model version.

        Args:
            model_name: Model name
            version: Version string
            artifact_path: Path to model artifacts
            metadata: Model metadata
            metrics: Performance metrics
            stage: Initial stage
            parent_version: Parent version if derived
            run_id: MLflow/experiment run ID
            experiment_id: Experiment ID

        Returns:
            Registered ModelVersion
        """
        now = datetime.now()

        # Generate model hash
        model_hash = self._generate_hash(model_name, version, artifact_path)

        model_version = ModelVersion(
            model_name=model_name,
            version=version,
            stage=stage,
            status=DeploymentStatus.PENDING,
            artifact_path=artifact_path,
            created_at=now,
            updated_at=now,
            metadata=metadata,
            metrics=metrics or ModelMetrics(),
            model_hash=model_hash,
            parent_version=parent_version,
            run_id=run_id,
            experiment_id=experiment_id,
        )

        with self._lock:
            if model_name not in self._models:
                self._models[model_name] = {}

            self._models[model_name][version] = model_version

        # Update metrics
        self._update_metrics()
        self._update_model_metrics(model_version)

        logger.info(f"Registered model: {model_name}:{version}")

        return model_version

    def _generate_hash(
        self,
        model_name: str,
        version: str,
        artifact_path: str,
    ) -> str:
        """Generate a hash for a model version."""
        content = f"{model_name}:{version}:{artifact_path}:{time.time()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get_model_version(
        self,
        model_name: str,
        version: str,
    ) -> Optional[ModelVersion]:
        """Get a specific model version.

        Args:
            model_name: Model name
            version: Version string

        Returns:
            ModelVersion or None
        """
        with self._lock:
            return self._models.get(model_name, {}).get(version)

    def get_latest_version(
        self,
        model_name: str,
        stage: Optional[ModelStage] = None,
    ) -> Optional[ModelVersion]:
        """Get the latest version of a model.

        Args:
            model_name: Model name
            stage: Filter by stage

        Returns:
            Latest ModelVersion or None
        """
        with self._lock:
            versions = self._models.get(model_name, {})

        if not versions:
            return None

        if stage:
            versions = {k: v for k, v in versions.items() if v.stage == stage}

        if not versions:
            return None

        # Sort by created_at descending
        sorted_versions = sorted(
            versions.values(),
            key=lambda v: v.created_at,
            reverse=True,
        )

        return sorted_versions[0] if sorted_versions else None

    def get_production_model(self, model_name: str) -> Optional[ModelVersion]:
        """Get the production version of a model.

        Args:
            model_name: Model name

        Returns:
            Production ModelVersion or None
        """
        return self.get_latest_version(model_name, ModelStage.PRODUCTION)

    def get_all_versions(
        self,
        model_name: str,
        stage: Optional[ModelStage] = None,
    ) -> List[ModelVersion]:
        """Get all versions of a model.

        Args:
            model_name: Model name
            stage: Filter by stage

        Returns:
            List of ModelVersion
        """
        with self._lock:
            versions = list(self._models.get(model_name, {}).values())

        if stage:
            versions = [v for v in versions if v.stage == stage]

        return sorted(versions, key=lambda v: v.created_at, reverse=True)

    def list_models(self) -> List[str]:
        """List all registered model names.

        Returns:
            List of model names
        """
        with self._lock:
            return list(self._models.keys())

    def transition_stage(
        self,
        model_name: str,
        version: str,
        to_stage: ModelStage,
        user: str,
        reason: str = "",
        approved_by: Optional[str] = None,
        archive_existing_production: bool = True,
    ) -> StageTransition:
        """Transition a model version to a new stage.

        Args:
            model_name: Model name
            version: Version string
            to_stage: Target stage
            user: User making the transition
            reason: Reason for transition
            approved_by: Approver (for production transitions)
            archive_existing_production: Archive existing production version

        Returns:
            StageTransition record
        """
        with self._lock:
            model_version = self._models.get(model_name, {}).get(version)

            if not model_version:
                raise ValueError(f"Model version not found: {model_name}:{version}")

            from_stage = model_version.stage

            # Archive existing production if transitioning to production
            if to_stage == ModelStage.PRODUCTION and archive_existing_production:
                for v in self._models.get(model_name, {}).values():
                    if v.stage == ModelStage.PRODUCTION and v.version != version:
                        v.stage = ModelStage.ARCHIVED
                        v.updated_at = datetime.now()
                        logger.info(f"Archived {model_name}:{v.version}")

            # Update stage
            model_version.stage = to_stage
            model_version.updated_at = datetime.now()

            transition = StageTransition(
                model_name=model_name,
                version=version,
                from_stage=from_stage,
                to_stage=to_stage,
                timestamp=datetime.now(),
                user=user,
                reason=reason,
                approved_by=approved_by,
            )

            self._transitions.append(transition)
            if len(self._transitions) > self._max_transitions:
                self._transitions = self._transitions[-self._max_transitions // 2:]

        # Update metrics
        self.stage_transitions.labels(
            model_name=model_name,
            from_stage=from_stage.value,
            to_stage=to_stage.value,
        ).inc()

        self._update_metrics()

        # Trigger callbacks
        for callback in self._callbacks:
            try:
                callback(transition)
            except Exception as e:
                logger.error(f"Transition callback error: {e}")

        logger.info(f"Transitioned {model_name}:{version} from {from_stage} to {to_stage}")

        return transition

    def update_deployment_status(
        self,
        model_name: str,
        version: str,
        status: DeploymentStatus,
        deployment_info: Optional[Dict[str, Any]] = None,
    ):
        """Update deployment status for a model version.

        Args:
            model_name: Model name
            version: Version string
            status: New deployment status
            deployment_info: Additional deployment information
        """
        with self._lock:
            model_version = self._models.get(model_name, {}).get(version)

            if not model_version:
                raise ValueError(f"Model version not found: {model_name}:{version}")

            model_version.status = status
            model_version.updated_at = datetime.now()

            if deployment_info:
                model_version.deployment_info.update(deployment_info)

        # Update metric
        status_map = {
            DeploymentStatus.PENDING: 0,
            DeploymentStatus.DEPLOYING: 1,
            DeploymentStatus.DEPLOYED: 2,
            DeploymentStatus.FAILED: 3,
            DeploymentStatus.ROLLED_BACK: 4,
        }

        self.deployment_status_gauge.labels(
            model_name=model_name,
            version=version,
        ).set(status_map.get(status, -1))

        logger.info(f"Updated deployment status: {model_name}:{version} -> {status}")

    def update_metrics(
        self,
        model_name: str,
        version: str,
        metrics: ModelMetrics,
    ):
        """Update performance metrics for a model version.

        Args:
            model_name: Model name
            version: Version string
            metrics: New metrics
        """
        with self._lock:
            model_version = self._models.get(model_name, {}).get(version)

            if not model_version:
                raise ValueError(f"Model version not found: {model_name}:{version}")

            model_version.metrics = metrics
            model_version.updated_at = datetime.now()

        self._update_model_metrics(model_version)

    def _update_model_metrics(self, model_version: ModelVersion):
        """Update Prometheus metrics for a model version."""
        metrics = model_version.metrics
        labels = {
            "model_name": model_version.model_name,
            "version": model_version.version,
        }

        metric_values = {
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1_score": metrics.f1_score,
            "auc_roc": metrics.auc_roc,
            "mse": metrics.mse,
            "rmse": metrics.rmse,
            "mae": metrics.mae,
            "r2_score": metrics.r2_score,
            "latency_p50_ms": metrics.latency_p50_ms,
            "latency_p95_ms": metrics.latency_p95_ms,
            "latency_p99_ms": metrics.latency_p99_ms,
            "throughput_rps": metrics.throughput_rps,
        }

        for metric_name, value in metric_values.items():
            if value is not None:
                self.model_metrics_gauge.labels(
                    **labels,
                    metric_name=metric_name,
                ).set(value)

        for metric_name, value in metrics.custom_metrics.items():
            self.model_metrics_gauge.labels(
                **labels,
                metric_name=metric_name,
            ).set(value)

    def rollback(
        self,
        model_name: str,
        user: str,
        reason: str = "Rollback",
    ) -> Optional[StageTransition]:
        """Rollback to previous production version.

        Args:
            model_name: Model name
            user: User performing rollback
            reason: Reason for rollback

        Returns:
            StageTransition or None if no previous version
        """
        # Find current production
        current_prod = self.get_production_model(model_name)
        if not current_prod:
            logger.warning(f"No production model to rollback: {model_name}")
            return None

        # Find previous production (most recent archived)
        with self._lock:
            archived = [
                v for v in self._models.get(model_name, {}).values()
                if v.stage == ModelStage.ARCHIVED and v.version != current_prod.version
            ]

        if not archived:
            logger.warning(f"No archived version to rollback to: {model_name}")
            return None

        # Get most recently archived
        previous = max(archived, key=lambda v: v.updated_at)

        # Transition current to failed/rolled_back
        self.update_deployment_status(
            model_name,
            current_prod.version,
            DeploymentStatus.ROLLED_BACK,
        )

        # Archive current production
        self.transition_stage(
            model_name,
            current_prod.version,
            ModelStage.ARCHIVED,
            user,
            reason=f"Rolled back: {reason}",
            archive_existing_production=False,
        )

        # Promote previous to production
        transition = self.transition_stage(
            model_name,
            previous.version,
            ModelStage.PRODUCTION,
            user,
            reason=f"Rollback from {current_prod.version}",
            archive_existing_production=False,
        )

        logger.info(f"Rolled back {model_name} from {current_prod.version} to {previous.version}")

        return transition

    def compare_versions(
        self,
        model_name: str,
        version_a: str,
        version_b: str,
    ) -> Dict[str, Any]:
        """Compare two model versions.

        Args:
            model_name: Model name
            version_a: First version
            version_b: Second version

        Returns:
            Comparison dictionary
        """
        a = self.get_model_version(model_name, version_a)
        b = self.get_model_version(model_name, version_b)

        if not a or not b:
            return {"error": "Version not found"}

        metric_diffs = {}

        for metric_name in ["accuracy", "precision", "recall", "f1_score", "auc_roc",
                           "mse", "rmse", "mae", "r2_score"]:
            val_a = getattr(a.metrics, metric_name, None)
            val_b = getattr(b.metrics, metric_name, None)

            if val_a is not None and val_b is not None:
                metric_diffs[metric_name] = {
                    "version_a": val_a,
                    "version_b": val_b,
                    "diff": val_b - val_a,
                    "pct_change": ((val_b - val_a) / val_a * 100) if val_a != 0 else 0,
                }

        return {
            "model_name": model_name,
            "version_a": version_a,
            "version_b": version_b,
            "created_diff_hours": (b.created_at - a.created_at).total_seconds() / 3600,
            "stage_a": a.stage.value,
            "stage_b": b.stage.value,
            "metric_comparison": metric_diffs,
            "hyperparameter_diff": self._compare_dicts(
                a.metadata.hyperparameters,
                b.metadata.hyperparameters,
            ),
        }

    def _compare_dicts(
        self,
        dict_a: Dict[str, Any],
        dict_b: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compare two dictionaries."""
        all_keys = set(dict_a.keys()) | set(dict_b.keys())

        diff = {}
        for key in all_keys:
            val_a = dict_a.get(key)
            val_b = dict_b.get(key)

            if val_a != val_b:
                diff[key] = {"a": val_a, "b": val_b}

        return diff

    def get_transition_history(
        self,
        model_name: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[StageTransition]:
        """Get transition history.

        Args:
            model_name: Filter by model
            since: Only transitions after this time
            limit: Maximum results

        Returns:
            List of transitions
        """
        with self._lock:
            transitions = list(self._transitions)

        if model_name:
            transitions = [t for t in transitions if t.model_name == model_name]

        if since:
            transitions = [t for t in transitions if t.timestamp >= since]

        return transitions[-limit:]

    def on_transition(self, callback: Callable[[StageTransition], None]):
        """Register callback for stage transitions.

        Args:
            callback: Function to call on transitions
        """
        self._callbacks.append(callback)

    def delete_version(
        self,
        model_name: str,
        version: str,
    ):
        """Delete a model version (only if not in production).

        Args:
            model_name: Model name
            version: Version to delete
        """
        with self._lock:
            model_version = self._models.get(model_name, {}).get(version)

            if not model_version:
                return

            if model_version.stage == ModelStage.PRODUCTION:
                raise ValueError("Cannot delete production model version")

            del self._models[model_name][version]

            if not self._models[model_name]:
                del self._models[model_name]

        self._update_metrics()
        logger.info(f"Deleted model version: {model_name}:{version}")

    def _update_metrics(self):
        """Update registry-level Prometheus metrics."""
        with self._lock:
            model_count = len(self._models)
            prod_count = 0

            for model_name, versions in self._models.items():
                self.versions_registered.labels(model_name=model_name).set(len(versions))

                for v in versions.values():
                    if v.stage == ModelStage.PRODUCTION:
                        prod_count += 1

        self.models_registered.set(model_count)
        self.production_versions.set(prod_count)

    def get_summary(self) -> Dict[str, Any]:
        """Get registry summary.

        Returns:
            Summary dictionary
        """
        with self._lock:
            model_count = len(self._models)
            version_count = sum(len(v) for v in self._models.values())

            stage_counts = {stage: 0 for stage in ModelStage}
            status_counts = {status: 0 for status in DeploymentStatus}

            for versions in self._models.values():
                for v in versions.values():
                    stage_counts[v.stage] += 1
                    status_counts[v.status] += 1

        recent_transitions = self.get_transition_history(since=datetime.now() - timedelta(hours=24))

        return {
            "total_models": model_count,
            "total_versions": version_count,
            "by_stage": {s.value: c for s, c in stage_counts.items()},
            "by_status": {s.value: c for s, c in status_counts.items()},
            "transitions_24h": len(recent_transitions),
            "callbacks_registered": len(self._callbacks),
        }
