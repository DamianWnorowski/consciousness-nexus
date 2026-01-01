"""ML Model Monitoring Module

Comprehensive ML observability with:
- Data and concept drift detection (KS test, PSI)
- Evidently AI integration for ML monitoring
- Feature store monitoring
- Model registry and version tracking
- Prediction logging and analysis
"""

from .drift_detector import (
    DriftDetector,
    DriftResult,
    DriftType,
    DriftSeverity,
    FeatureDrift,
    DatasetDrift,
)
from .evidently_client import (
    EvidentlyClient,
    EvidentlyReport,
    ReportType,
    MetricResult,
)
from .feature_store import (
    FeatureStoreMonitor,
    FeatureMetadata,
    FeatureHealth,
    FeatureFreshness,
)
from .model_registry import (
    ModelRegistry,
    ModelVersion,
    ModelStage,
    ModelMetadata,
    DeploymentStatus,
)
from .prediction_logger import (
    PredictionLogger,
    PredictionRecord,
    PredictionAnalytics,
    PredictionDistribution,
)

__all__ = [
    # Drift Detection
    "DriftDetector",
    "DriftResult",
    "DriftType",
    "DriftSeverity",
    "FeatureDrift",
    "DatasetDrift",
    # Evidently
    "EvidentlyClient",
    "EvidentlyReport",
    "ReportType",
    "MetricResult",
    # Feature Store
    "FeatureStoreMonitor",
    "FeatureMetadata",
    "FeatureHealth",
    "FeatureFreshness",
    # Model Registry
    "ModelRegistry",
    "ModelVersion",
    "ModelStage",
    "ModelMetadata",
    "DeploymentStatus",
    # Prediction Logging
    "PredictionLogger",
    "PredictionRecord",
    "PredictionAnalytics",
    "PredictionDistribution",
]
