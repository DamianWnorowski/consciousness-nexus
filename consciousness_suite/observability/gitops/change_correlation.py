"""Deployment-Incident Correlation

Correlates deployments with incidents for root cause analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Tuple
from datetime import datetime, timedelta
from enum import Enum
import threading
import logging
import statistics

from prometheus_client import Counter, Gauge, Histogram

from .deployment_tracker import Deployment, DeploymentPhase

logger = logging.getLogger(__name__)


class IncidentSeverity(str, Enum):
    """Incident severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class IncidentStatus(str, Enum):
    """Incident status."""
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    INVESTIGATING = "investigating"
    MITIGATING = "mitigating"
    RESOLVED = "resolved"
    CLOSED = "closed"


class CorrelationConfidence(str, Enum):
    """Correlation confidence level."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class CorrelationType(str, Enum):
    """Type of correlation between deployment and incident."""
    TEMPORAL = "temporal"  # Time-based correlation
    RESOURCE = "resource"  # Same resources affected
    ERROR_PATTERN = "error_pattern"  # Error messages match
    METRIC_ANOMALY = "metric_anomaly"  # Metrics changed after deploy
    ROLLBACK_TRIGGER = "rollback_trigger"  # Deploy caused rollback
    MANUAL = "manual"  # Manually correlated


@dataclass
class Incident:
    """Incident representation."""
    incident_id: str
    title: str
    description: str
    severity: IncidentSeverity
    status: IncidentStatus
    service: str
    environment: str
    cluster: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    detected_by: str = ""  # alert name, user, etc.
    affected_resources: List[str] = field(default_factory=list)
    error_messages: List[str] = field(default_factory=list)
    metrics_affected: List[str] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        end = self.resolved_at or datetime.now()
        return (end - self.created_at).total_seconds()

    @property
    def time_to_acknowledge_seconds(self) -> Optional[float]:
        if self.acknowledged_at:
            return (self.acknowledged_at - self.created_at).total_seconds()
        return None

    @property
    def time_to_resolve_seconds(self) -> Optional[float]:
        if self.resolved_at:
            return (self.resolved_at - self.created_at).total_seconds()
        return None

    @property
    def is_resolved(self) -> bool:
        return self.status in (IncidentStatus.RESOLVED, IncidentStatus.CLOSED)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "title": self.title,
            "severity": self.severity.value,
            "status": self.status.value,
            "service": self.service,
            "environment": self.environment,
            "cluster": self.cluster,
            "created_at": self.created_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "duration_seconds": self.duration_seconds,
            "is_resolved": self.is_resolved,
        }


@dataclass
class DeploymentIncidentCorrelation:
    """Correlation between a deployment and an incident."""
    correlation_id: str
    deployment: Deployment
    incident: Incident
    correlation_type: CorrelationType
    confidence: CorrelationConfidence
    confidence_score: float  # 0.0 to 1.0
    time_delta_seconds: float  # Time between deployment and incident
    matching_resources: List[str] = field(default_factory=list)
    matching_errors: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    verified: Optional[bool] = None  # Manual verification
    verified_by: str = ""
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "correlation_id": self.correlation_id,
            "deployment_id": self.deployment.deployment_id,
            "deployment_name": self.deployment.name,
            "incident_id": self.incident.incident_id,
            "incident_title": self.incident.title,
            "correlation_type": self.correlation_type.value,
            "confidence": self.confidence.value,
            "confidence_score": self.confidence_score,
            "time_delta_seconds": self.time_delta_seconds,
            "matching_resources": self.matching_resources,
            "evidence": self.evidence,
            "verified": self.verified,
        }


@dataclass
class CorrelationRule:
    """Rule for correlating deployments with incidents."""
    rule_id: str
    name: str
    enabled: bool = True
    max_time_window_seconds: int = 3600  # 1 hour default
    min_confidence_score: float = 0.3
    weight: float = 1.0
    environments: List[str] = field(default_factory=list)  # Empty = all
    services: List[str] = field(default_factory=list)  # Empty = all
    severity_filter: List[IncidentSeverity] = field(default_factory=list)


@dataclass
class ChangeFailureMetrics:
    """Change failure rate metrics."""
    total_deployments: int = 0
    deployments_with_incidents: int = 0
    change_failure_rate: float = 0.0
    average_time_to_incident_seconds: float = 0.0
    incidents_by_severity: Dict[str, int] = field(default_factory=dict)


class ChangeCorrelator:
    """Correlates deployments with incidents for root cause analysis.

    Usage:
        correlator = ChangeCorrelator()

        # Register incidents
        correlator.register_incident(incident)

        # Correlate with deployment
        correlations = correlator.correlate_deployment(deployment)

        # Get correlation history
        history = correlator.get_correlations(deployment_id="deploy-123")

        # Get change failure metrics
        metrics = correlator.get_change_failure_metrics("production")
    """

    def __init__(
        self,
        namespace: str = "consciousness",
        default_time_window_seconds: int = 3600,
    ):
        self.namespace = namespace
        self.default_time_window = timedelta(seconds=default_time_window_seconds)
        self._lock = threading.Lock()

        # State
        self._incidents: Dict[str, Incident] = {}
        self._correlations: Dict[str, List[DeploymentIncidentCorrelation]] = {}
        self._rules: Dict[str, CorrelationRule] = {}

        # Callbacks
        self._correlation_callbacks: List[Callable[[DeploymentIncidentCorrelation], None]] = []

        self._max_incidents = 10000
        self._max_correlations = 5000

        # Create default rules
        self._create_default_rules()

        # Prometheus metrics
        self.correlations_total = Counter(
            f"{namespace}_gitops_correlations_total",
            "Total correlations found",
            ["environment", "type", "confidence"],
        )

        self.change_failure_rate = Gauge(
            f"{namespace}_gitops_change_failure_rate",
            "Change failure rate (0-1)",
            ["environment"],
        )

        self.time_to_incident = Histogram(
            f"{namespace}_gitops_time_to_incident_seconds",
            "Time from deployment to incident",
            ["environment", "severity"],
            buckets=[60, 300, 600, 1800, 3600, 7200, 14400],
        )

        self.incidents_per_deployment = Histogram(
            f"{namespace}_gitops_incidents_per_deployment",
            "Incidents correlated per deployment",
            ["environment"],
            buckets=[0, 1, 2, 3, 5, 10, 20],
        )

        self.correlation_confidence = Histogram(
            f"{namespace}_gitops_correlation_confidence",
            "Correlation confidence scores",
            ["type"],
            buckets=[0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0],
        )

    def _create_default_rules(self):
        """Create default correlation rules."""
        self._rules["temporal"] = CorrelationRule(
            rule_id="temporal",
            name="Temporal Correlation",
            max_time_window_seconds=3600,
            weight=0.5,
        )

        self._rules["resource"] = CorrelationRule(
            rule_id="resource",
            name="Resource Correlation",
            max_time_window_seconds=7200,
            weight=0.8,
        )

        self._rules["high_severity"] = CorrelationRule(
            rule_id="high_severity",
            name="High Severity Incidents",
            max_time_window_seconds=1800,
            severity_filter=[IncidentSeverity.CRITICAL, IncidentSeverity.HIGH],
            weight=1.0,
        )

    def register_incident(self, incident: Incident):
        """Register an incident for correlation.

        Args:
            incident: Incident to register
        """
        with self._lock:
            self._incidents[incident.incident_id] = incident

            # Trim old resolved incidents
            if len(self._incidents) > self._max_incidents:
                self._trim_old_incidents()

        logger.debug(f"Registered incident: {incident.incident_id}")

    def _trim_old_incidents(self):
        """Remove old resolved incidents."""
        resolved = [
            (iid, i) for iid, i in self._incidents.items()
            if i.is_resolved
        ]

        # Sort by resolution time
        resolved.sort(key=lambda x: x[1].resolved_at or datetime.min)

        # Remove oldest half
        to_remove = len(resolved) // 2
        for iid, _ in resolved[:to_remove]:
            del self._incidents[iid]

    def correlate_deployment(
        self,
        deployment: Deployment,
        time_window: Optional[timedelta] = None,
    ) -> List[DeploymentIncidentCorrelation]:
        """Find incidents correlated with a deployment.

        Args:
            deployment: Deployment to correlate
            time_window: Time window to search (default: 1 hour after deployment)

        Returns:
            List of correlations
        """
        window = time_window or self.default_time_window
        correlations = []

        # Get incidents in time window
        with self._lock:
            incidents = list(self._incidents.values())

        deploy_time = deployment.started_at or datetime.now()
        deploy_end = deployment.completed_at or datetime.now()

        # Find incidents that started after deployment began
        for incident in incidents:
            # Check time window
            if incident.created_at < deploy_time:
                continue  # Incident before deployment

            time_delta = (incident.created_at - deploy_time).total_seconds()
            if time_delta > window.total_seconds():
                continue  # Outside time window

            # Check environment match
            if incident.environment != deployment.environment:
                continue

            # Calculate correlation
            correlation = self._calculate_correlation(deployment, incident, time_delta)

            if correlation:
                correlations.append(correlation)

        # Store correlations
        if correlations:
            with self._lock:
                if deployment.deployment_id not in self._correlations:
                    self._correlations[deployment.deployment_id] = []
                self._correlations[deployment.deployment_id].extend(correlations)

            # Update metrics
            for corr in correlations:
                self.correlations_total.labels(
                    environment=deployment.environment,
                    type=corr.correlation_type.value,
                    confidence=corr.confidence.value,
                ).inc()

                self.correlation_confidence.labels(
                    type=corr.correlation_type.value,
                ).observe(corr.confidence_score)

                self.time_to_incident.labels(
                    environment=deployment.environment,
                    severity=corr.incident.severity.value,
                ).observe(corr.time_delta_seconds)

            self.incidents_per_deployment.labels(
                environment=deployment.environment,
            ).observe(len(correlations))

            # Notify callbacks
            for corr in correlations:
                for callback in self._correlation_callbacks:
                    try:
                        callback(corr)
                    except Exception as e:
                        logger.error(f"Correlation callback error: {e}")

        return correlations

    def _calculate_correlation(
        self,
        deployment: Deployment,
        incident: Incident,
        time_delta: float,
    ) -> Optional[DeploymentIncidentCorrelation]:
        """Calculate correlation between deployment and incident."""
        evidence = []
        confidence_factors = []

        # Temporal correlation - closer = higher confidence
        if time_delta < 300:  # 5 minutes
            temporal_score = 0.9
            evidence.append(f"Incident occurred {time_delta:.0f}s after deployment")
        elif time_delta < 900:  # 15 minutes
            temporal_score = 0.7
            evidence.append(f"Incident occurred {time_delta/60:.1f}min after deployment")
        elif time_delta < 1800:  # 30 minutes
            temporal_score = 0.5
            evidence.append(f"Incident occurred {time_delta/60:.1f}min after deployment")
        else:
            temporal_score = 0.3
            evidence.append(f"Incident occurred {time_delta/3600:.1f}h after deployment")

        confidence_factors.append(temporal_score)

        # Resource correlation
        matching_resources = []
        deploy_resources = set(r.name for r in deployment.resources)

        for affected in incident.affected_resources:
            if affected in deploy_resources:
                matching_resources.append(affected)

        if matching_resources:
            resource_score = min(1.0, len(matching_resources) / max(len(deploy_resources), 1))
            confidence_factors.append(resource_score)
            evidence.append(f"Matching resources: {', '.join(matching_resources[:5])}")
        else:
            confidence_factors.append(0.2)

        # Cluster correlation
        if incident.cluster and deployment.cluster:
            if incident.cluster == deployment.cluster:
                confidence_factors.append(0.8)
                evidence.append(f"Same cluster: {deployment.cluster}")
            else:
                confidence_factors.append(0.3)

        # Service correlation (check if deployment name matches service)
        if incident.service.lower() in deployment.name.lower():
            confidence_factors.append(0.9)
            evidence.append(f"Service name match: {incident.service}")
        elif deployment.name.lower() in incident.service.lower():
            confidence_factors.append(0.7)
            evidence.append(f"Partial service match")

        # Severity weight
        severity_weights = {
            IncidentSeverity.CRITICAL: 1.0,
            IncidentSeverity.HIGH: 0.9,
            IncidentSeverity.MEDIUM: 0.7,
            IncidentSeverity.LOW: 0.5,
            IncidentSeverity.INFO: 0.3,
        }
        severity_weight = severity_weights.get(incident.severity, 0.5)

        # Calculate final confidence score
        if confidence_factors:
            base_score = statistics.mean(confidence_factors)
            final_score = base_score * severity_weight
        else:
            final_score = 0.1

        # Determine confidence level
        if final_score >= 0.7:
            confidence = CorrelationConfidence.HIGH
        elif final_score >= 0.4:
            confidence = CorrelationConfidence.MEDIUM
        elif final_score >= 0.2:
            confidence = CorrelationConfidence.LOW
        else:
            return None  # Too low to correlate

        # Determine correlation type
        if matching_resources:
            correlation_type = CorrelationType.RESOURCE
        else:
            correlation_type = CorrelationType.TEMPORAL

        correlation_id = (
            f"corr-{deployment.deployment_id[:8]}-{incident.incident_id[:8]}"
        )

        return DeploymentIncidentCorrelation(
            correlation_id=correlation_id,
            deployment=deployment,
            incident=incident,
            correlation_type=correlation_type,
            confidence=confidence,
            confidence_score=final_score,
            time_delta_seconds=time_delta,
            matching_resources=matching_resources,
            evidence=evidence,
        )

    def get_correlations(
        self,
        deployment_id: Optional[str] = None,
        incident_id: Optional[str] = None,
        confidence: Optional[CorrelationConfidence] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[DeploymentIncidentCorrelation]:
        """Get correlation history.

        Args:
            deployment_id: Filter by deployment
            incident_id: Filter by incident
            confidence: Filter by confidence level
            since: Only correlations after this time
            limit: Maximum results

        Returns:
            List of correlations
        """
        with self._lock:
            if deployment_id:
                all_correlations = list(self._correlations.get(deployment_id, []))
            else:
                all_correlations = []
                for corrs in self._correlations.values():
                    all_correlations.extend(corrs)

        # Filter by incident
        if incident_id:
            all_correlations = [
                c for c in all_correlations
                if c.incident.incident_id == incident_id
            ]

        # Filter by confidence
        if confidence:
            all_correlations = [
                c for c in all_correlations
                if c.confidence == confidence
            ]

        # Filter by time
        if since:
            all_correlations = [
                c for c in all_correlations
                if c.created_at >= since
            ]

        # Sort by confidence score descending
        all_correlations.sort(key=lambda c: c.confidence_score, reverse=True)

        return all_correlations[:limit]

    def verify_correlation(
        self,
        correlation_id: str,
        verified: bool,
        verified_by: str,
        notes: str = "",
    ) -> bool:
        """Manually verify a correlation.

        Args:
            correlation_id: Correlation ID
            verified: Whether correlation is verified
            verified_by: User verifying
            notes: Verification notes

        Returns:
            True if updated
        """
        with self._lock:
            for corrs in self._correlations.values():
                for corr in corrs:
                    if corr.correlation_id == correlation_id:
                        corr.verified = verified
                        corr.verified_by = verified_by
                        corr.notes = notes

                        logger.info(
                            f"Correlation {correlation_id} verified: {verified} by {verified_by}"
                        )

                        return True

        return False

    def add_manual_correlation(
        self,
        deployment: Deployment,
        incident: Incident,
        notes: str = "",
        created_by: str = "",
    ) -> DeploymentIncidentCorrelation:
        """Add a manual correlation.

        Args:
            deployment: Deployment
            incident: Incident
            notes: Correlation notes
            created_by: User creating correlation

        Returns:
            DeploymentIncidentCorrelation
        """
        time_delta = (
            incident.created_at - (deployment.started_at or datetime.now())
        ).total_seconds()

        correlation_id = (
            f"manual-{deployment.deployment_id[:8]}-{incident.incident_id[:8]}"
        )

        correlation = DeploymentIncidentCorrelation(
            correlation_id=correlation_id,
            deployment=deployment,
            incident=incident,
            correlation_type=CorrelationType.MANUAL,
            confidence=CorrelationConfidence.HIGH,
            confidence_score=1.0,
            time_delta_seconds=time_delta,
            evidence=[f"Manually correlated by {created_by}"],
            verified=True,
            verified_by=created_by,
            notes=notes,
        )

        with self._lock:
            if deployment.deployment_id not in self._correlations:
                self._correlations[deployment.deployment_id] = []
            self._correlations[deployment.deployment_id].append(correlation)

        logger.info(
            f"Manual correlation added: {deployment.deployment_id} <-> {incident.incident_id}"
        )

        return correlation

    def get_change_failure_metrics(
        self,
        environment: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> ChangeFailureMetrics:
        """Calculate change failure rate metrics.

        Args:
            environment: Filter by environment
            since: Only consider deployments after this time

        Returns:
            ChangeFailureMetrics
        """
        with self._lock:
            all_correlations = []
            for corrs in self._correlations.values():
                all_correlations.extend(corrs)

        if environment:
            all_correlations = [
                c for c in all_correlations
                if c.deployment.environment == environment
            ]

        if since:
            all_correlations = [
                c for c in all_correlations
                if c.created_at >= since
            ]

        # Get unique deployments with incidents
        deployments_with_incidents = set(
            c.deployment.deployment_id for c in all_correlations
            if c.confidence_score >= 0.5  # Only count medium+ confidence
        )

        # We need total deployments - this would come from DeploymentTracker
        # For now, we'll estimate from correlations
        all_deployment_ids = set(
            c.deployment.deployment_id for c in all_correlations
        )

        total = len(all_deployment_ids) if all_deployment_ids else 1
        with_incidents = len(deployments_with_incidents)

        # Calculate average time to incident
        time_deltas = [c.time_delta_seconds for c in all_correlations]
        avg_time = statistics.mean(time_deltas) if time_deltas else 0.0

        # Count by severity
        by_severity: Dict[str, int] = {}
        for c in all_correlations:
            sev = c.incident.severity.value
            by_severity[sev] = by_severity.get(sev, 0) + 1

        failure_rate = with_incidents / total if total > 0 else 0.0

        # Update metric
        if environment:
            self.change_failure_rate.labels(environment=environment).set(failure_rate)

        return ChangeFailureMetrics(
            total_deployments=total,
            deployments_with_incidents=with_incidents,
            change_failure_rate=failure_rate,
            average_time_to_incident_seconds=avg_time,
            incidents_by_severity=by_severity,
        )

    def get_deployment_risk_score(
        self,
        deployment: Deployment,
    ) -> Tuple[float, List[str]]:
        """Calculate risk score for a deployment based on historical correlations.

        Args:
            deployment: Deployment to assess

        Returns:
            Tuple of (risk_score 0-1, risk_factors)
        """
        risk_factors = []
        scores = []

        # Check historical correlations for same service
        with self._lock:
            all_correlations = []
            for corrs in self._correlations.values():
                all_correlations.extend(corrs)

        # Filter to same service/namespace
        relevant = [
            c for c in all_correlations
            if c.deployment.name == deployment.name
            or c.deployment.namespace == deployment.namespace
        ]

        if not relevant:
            return 0.2, ["No historical data for risk assessment"]

        # Recent failure rate
        recent = [c for c in relevant if c.deployment.is_failed]
        if recent:
            failure_score = min(1.0, len(recent) / len(relevant))
            scores.append(failure_score)
            risk_factors.append(f"Historical failure rate: {failure_score:.0%}")

        # High severity incident history
        high_sev = [
            c for c in relevant
            if c.incident.severity in (IncidentSeverity.CRITICAL, IncidentSeverity.HIGH)
        ]
        if high_sev:
            sev_score = min(1.0, len(high_sev) / 10)
            scores.append(sev_score)
            risk_factors.append(f"{len(high_sev)} high severity incidents in history")

        # Time since last incident
        if relevant:
            latest = max(relevant, key=lambda c: c.created_at)
            days_since = (datetime.now() - latest.created_at).days

            if days_since < 7:
                time_score = 0.8
                risk_factors.append(f"Recent incident {days_since} days ago")
            elif days_since < 30:
                time_score = 0.5
            else:
                time_score = 0.2

            scores.append(time_score)

        final_score = statistics.mean(scores) if scores else 0.2

        return final_score, risk_factors

    def on_correlation_found(
        self,
        callback: Callable[[DeploymentIncidentCorrelation], None],
    ):
        """Register callback for new correlations.

        Args:
            callback: Function to call when correlation is found
        """
        self._correlation_callbacks.append(callback)

    def add_rule(self, rule: CorrelationRule):
        """Add a correlation rule.

        Args:
            rule: Correlation rule
        """
        with self._lock:
            self._rules[rule.rule_id] = rule

        logger.info(f"Added correlation rule: {rule.rule_id}")

    def get_summary(self) -> Dict[str, Any]:
        """Get correlator summary.

        Returns:
            Summary dictionary
        """
        with self._lock:
            all_correlations = []
            for corrs in self._correlations.values():
                all_correlations.extend(corrs)
            incidents = list(self._incidents.values())

        # By confidence
        by_confidence: Dict[str, int] = {}
        for c in all_correlations:
            by_confidence[c.confidence.value] = by_confidence.get(c.confidence.value, 0) + 1

        # By type
        by_type: Dict[str, int] = {}
        for c in all_correlations:
            by_type[c.correlation_type.value] = by_type.get(c.correlation_type.value, 0) + 1

        # Verified stats
        verified = sum(1 for c in all_correlations if c.verified is True)
        rejected = sum(1 for c in all_correlations if c.verified is False)

        # Active incidents
        active_incidents = sum(1 for i in incidents if not i.is_resolved)

        return {
            "total_correlations": len(all_correlations),
            "by_confidence": by_confidence,
            "by_type": by_type,
            "verified_correlations": verified,
            "rejected_correlations": rejected,
            "pending_verification": len(all_correlations) - verified - rejected,
            "total_incidents": len(incidents),
            "active_incidents": active_incidents,
            "rules_count": len(self._rules),
        }
