"""Chaos Engineering Observability Hooks

Integration between chaos experiments and observability systems.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import threading
import logging
import hashlib

from prometheus_client import Counter, Gauge, Histogram, Info

logger = logging.getLogger(__name__)


class ExperimentPhase(str, Enum):
    """Phases of a chaos experiment."""
    PENDING = "pending"
    INITIALIZING = "initializing"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"
    ROLLING_BACK = "rolling_back"


@dataclass
class ExperimentEvent:
    """Event from a chaos experiment."""
    experiment_id: str
    experiment_name: str
    phase: ExperimentPhase
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    affected_services: List[str] = field(default_factory=list)
    metrics_snapshot: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IncidentCorrelation:
    """Correlation between chaos experiment and incident."""
    experiment_id: str
    incident_id: str
    correlation_score: float  # 0.0 to 1.0
    correlation_type: str  # "caused", "coincidental", "unrelated"
    evidence: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


class ExperimentMetrics:
    """Metrics collection for chaos experiments.

    Usage:
        metrics = ExperimentMetrics()

        # Start tracking an experiment
        metrics.start_experiment("exp-001", "latency-test", ["api-gateway"])

        # Record events during experiment
        metrics.record_event(ExperimentEvent(...))

        # End experiment
        metrics.end_experiment("exp-001", success=True)
    """

    def __init__(self, namespace: str = "consciousness"):
        self.namespace = namespace
        self._experiments: Dict[str, Dict[str, Any]] = {}
        self._events: List[ExperimentEvent] = []
        self._lock = threading.Lock()
        self._max_events = 10000

        # Prometheus metrics
        self.experiments_total = Counter(
            f"{namespace}_chaos_experiments_total",
            "Total chaos experiments run",
            ["status"],
        )

        self.experiment_duration = Histogram(
            f"{namespace}_chaos_experiment_duration_seconds",
            "Duration of chaos experiments",
            ["experiment_name"],
            buckets=[10, 30, 60, 120, 300, 600, 1800, 3600],
        )

        self.active_experiments = Gauge(
            f"{namespace}_chaos_active_experiments",
            "Number of currently running experiments",
        )

        self.affected_services = Gauge(
            f"{namespace}_chaos_affected_services",
            "Number of services affected by active experiments",
        )

        self.experiment_impact = Gauge(
            f"{namespace}_chaos_experiment_impact",
            "Impact score of current experiment",
            ["experiment_id", "metric"],
        )

        self.blast_radius = Gauge(
            f"{namespace}_chaos_blast_radius",
            "Blast radius (affected services count)",
            ["experiment_id"],
        )

    def start_experiment(
        self,
        experiment_id: str,
        name: str,
        affected_services: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Record the start of a chaos experiment.

        Args:
            experiment_id: Unique experiment identifier
            name: Experiment name
            affected_services: List of services affected
            metadata: Additional context
        """
        with self._lock:
            self._experiments[experiment_id] = {
                "name": name,
                "affected_services": affected_services,
                "start_time": datetime.now(),
                "end_time": None,
                "phase": ExperimentPhase.RUNNING,
                "success": None,
                "metadata": metadata or {},
                "baseline_metrics": {},
                "impact_metrics": {},
            }

        # Update metrics
        self.active_experiments.inc()
        self.blast_radius.labels(experiment_id=experiment_id).set(len(affected_services))

        # Record start event
        self.record_event(ExperimentEvent(
            experiment_id=experiment_id,
            experiment_name=name,
            phase=ExperimentPhase.RUNNING,
            message=f"Experiment started: {name}",
            affected_services=affected_services,
            metadata=metadata or {},
        ))

        logger.info(
            f"Started chaos experiment {experiment_id}: {name} "
            f"(affecting {len(affected_services)} services)"
        )

    def end_experiment(
        self,
        experiment_id: str,
        success: bool,
        summary: Optional[str] = None,
        impact_metrics: Optional[Dict[str, float]] = None,
    ):
        """Record the end of a chaos experiment.

        Args:
            experiment_id: Experiment identifier
            success: Whether experiment succeeded
            summary: Summary message
            impact_metrics: Metrics showing impact
        """
        with self._lock:
            if experiment_id not in self._experiments:
                logger.warning(f"Unknown experiment {experiment_id}")
                return

            exp = self._experiments[experiment_id]
            exp["end_time"] = datetime.now()
            exp["success"] = success
            exp["phase"] = ExperimentPhase.COMPLETED if success else ExperimentPhase.FAILED
            exp["impact_metrics"] = impact_metrics or {}

            duration = (exp["end_time"] - exp["start_time"]).total_seconds()

        # Update metrics
        self.active_experiments.dec()
        self.experiments_total.labels(
            status="success" if success else "failure"
        ).inc()
        self.experiment_duration.labels(
            experiment_name=exp["name"]
        ).observe(duration)

        # Clear impact metrics
        self.blast_radius.labels(experiment_id=experiment_id).set(0)

        # Record end event
        self.record_event(ExperimentEvent(
            experiment_id=experiment_id,
            experiment_name=exp["name"],
            phase=exp["phase"],
            message=summary or f"Experiment completed: {'success' if success else 'failure'}",
            affected_services=exp["affected_services"],
            metrics_snapshot=impact_metrics or {},
        ))

        logger.info(
            f"Ended chaos experiment {experiment_id}: "
            f"{'success' if success else 'failure'} ({duration:.1f}s)"
        )

    def abort_experiment(
        self,
        experiment_id: str,
        reason: str,
    ):
        """Abort a running experiment.

        Args:
            experiment_id: Experiment identifier
            reason: Abort reason
        """
        with self._lock:
            if experiment_id in self._experiments:
                exp = self._experiments[experiment_id]
                exp["phase"] = ExperimentPhase.ABORTED
                exp["end_time"] = datetime.now()
                exp["abort_reason"] = reason

        self.active_experiments.dec()
        self.experiments_total.labels(status="aborted").inc()

        logger.warning(f"Aborted chaos experiment {experiment_id}: {reason}")

    def record_event(self, event: ExperimentEvent):
        """Record an experiment event.

        Args:
            event: Experiment event
        """
        with self._lock:
            self._events.append(event)
            if len(self._events) > self._max_events:
                self._events = self._events[-self._max_events // 2:]

    def set_baseline_metrics(
        self,
        experiment_id: str,
        metrics: Dict[str, float],
    ):
        """Set baseline metrics before experiment.

        Args:
            experiment_id: Experiment identifier
            metrics: Baseline metric values
        """
        with self._lock:
            if experiment_id in self._experiments:
                self._experiments[experiment_id]["baseline_metrics"] = metrics

    def record_impact(
        self,
        experiment_id: str,
        metric_name: str,
        baseline_value: float,
        current_value: float,
    ):
        """Record impact of experiment on a metric.

        Args:
            experiment_id: Experiment identifier
            metric_name: Name of the metric
            baseline_value: Value before experiment
            current_value: Current value
        """
        impact = abs(current_value - baseline_value) / baseline_value if baseline_value else 0

        self.experiment_impact.labels(
            experiment_id=experiment_id,
            metric=metric_name,
        ).set(impact)

        with self._lock:
            if experiment_id in self._experiments:
                self._experiments[experiment_id]["impact_metrics"][metric_name] = {
                    "baseline": baseline_value,
                    "current": current_value,
                    "impact": impact,
                }

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment details.

        Args:
            experiment_id: Experiment identifier

        Returns:
            Experiment data or None
        """
        with self._lock:
            return self._experiments.get(experiment_id, {}).copy()

    def get_active_experiments(self) -> List[Dict[str, Any]]:
        """Get all active experiments.

        Returns:
            List of active experiments
        """
        with self._lock:
            return [
                {**exp, "id": eid}
                for eid, exp in self._experiments.items()
                if exp["phase"] == ExperimentPhase.RUNNING
            ]

    def get_recent_events(
        self,
        count: int = 50,
        experiment_id: Optional[str] = None,
    ) -> List[ExperimentEvent]:
        """Get recent experiment events.

        Args:
            count: Number of events to return
            experiment_id: Filter by experiment

        Returns:
            List of events
        """
        with self._lock:
            events = self._events.copy()

        if experiment_id:
            events = [e for e in events if e.experiment_id == experiment_id]

        return list(reversed(events[-count:]))


class IncidentCorrelator:
    """Correlates chaos experiments with incidents.

    Usage:
        correlator = IncidentCorrelator(experiment_metrics)

        # Report an incident during experiment
        correlation = correlator.check_correlation(
            incident_id="inc-123",
            incident_time=datetime.now(),
            affected_services=["api-gateway"],
            symptoms=["high_latency", "error_spike"]
        )
    """

    def __init__(
        self,
        metrics: ExperimentMetrics,
        correlation_window_seconds: float = 300,
    ):
        self.metrics = metrics
        self.correlation_window = timedelta(seconds=correlation_window_seconds)
        self._correlations: List[IncidentCorrelation] = []
        self._lock = threading.Lock()

    def check_correlation(
        self,
        incident_id: str,
        incident_time: datetime,
        affected_services: List[str],
        symptoms: List[str],
    ) -> Optional[IncidentCorrelation]:
        """Check if an incident correlates with active experiments.

        Args:
            incident_id: Incident identifier
            incident_time: When incident occurred
            affected_services: Services affected by incident
            symptoms: Observed symptoms

        Returns:
            IncidentCorrelation if correlation found
        """
        active = self.metrics.get_active_experiments()

        for exp in active:
            # Check service overlap
            exp_services = set(exp.get("affected_services", []))
            inc_services = set(affected_services)
            service_overlap = exp_services & inc_services

            if not service_overlap:
                continue

            # Check time window
            exp_start = exp.get("start_time")
            if exp_start:
                if incident_time < exp_start - self.correlation_window:
                    continue
                if incident_time > exp_start + timedelta(hours=1):
                    continue

            # Calculate correlation score
            service_score = len(service_overlap) / max(len(inc_services), 1)
            time_score = 1.0 if exp_start and incident_time >= exp_start else 0.5

            # Check symptom match
            symptom_keywords = ["latency", "error", "timeout", "failure", "unavailable"]
            symptom_score = sum(
                1 for s in symptoms
                if any(k in s.lower() for k in symptom_keywords)
            ) / max(len(symptoms), 1)

            correlation_score = (service_score * 0.5 + time_score * 0.3 + symptom_score * 0.2)

            # Determine correlation type
            if correlation_score > 0.8:
                correlation_type = "caused"
            elif correlation_score > 0.5:
                correlation_type = "coincidental"
            else:
                correlation_type = "unrelated"

            evidence = [
                f"Service overlap: {service_overlap}",
                f"Time proximity: experiment started at {exp_start}",
                f"Symptoms: {symptoms}",
            ]

            correlation = IncidentCorrelation(
                experiment_id=exp.get("id", "unknown"),
                incident_id=incident_id,
                correlation_score=correlation_score,
                correlation_type=correlation_type,
                evidence=evidence,
            )

            with self._lock:
                self._correlations.append(correlation)

            if correlation_type == "caused":
                logger.warning(
                    f"Incident {incident_id} likely CAUSED by experiment {exp.get('id')}"
                )
            elif correlation_type == "coincidental":
                logger.info(
                    f"Incident {incident_id} may be related to experiment {exp.get('id')}"
                )

            return correlation

        return None

    def get_correlations(
        self,
        experiment_id: Optional[str] = None,
        incident_id: Optional[str] = None,
    ) -> List[IncidentCorrelation]:
        """Get correlation records.

        Args:
            experiment_id: Filter by experiment
            incident_id: Filter by incident

        Returns:
            List of correlations
        """
        with self._lock:
            correlations = self._correlations.copy()

        if experiment_id:
            correlations = [c for c in correlations if c.experiment_id == experiment_id]
        if incident_id:
            correlations = [c for c in correlations if c.incident_id == incident_id]

        return correlations


class ChaosObservabilityHook:
    """Hook for integrating chaos with observability systems.

    Usage:
        hook = ChaosObservabilityHook(experiment_metrics)

        # Register callbacks
        hook.on_experiment_start(notify_team)
        hook.on_impact_threshold(auto_abort)

        # Start experiment through hook
        await hook.run_experiment(
            name="latency-test",
            affected_services=["api-gateway"],
            faults=[latency_fault],
            duration_seconds=60,
        )
    """

    def __init__(
        self,
        metrics: ExperimentMetrics,
        auto_abort_threshold: float = 0.5,
    ):
        self.metrics = metrics
        self.auto_abort_threshold = auto_abort_threshold
        self._on_start: List[Callable] = []
        self._on_end: List[Callable] = []
        self._on_impact: List[Callable] = []
        self._on_abort: List[Callable] = []

    def on_experiment_start(self, callback: Callable[[str, str, List[str]], None]):
        """Register callback for experiment start.

        Callback receives: experiment_id, name, affected_services
        """
        self._on_start.append(callback)

    def on_experiment_end(self, callback: Callable[[str, bool, Dict[str, float]], None]):
        """Register callback for experiment end.

        Callback receives: experiment_id, success, impact_metrics
        """
        self._on_end.append(callback)

    def on_impact_threshold(self, callback: Callable[[str, str, float], None]):
        """Register callback for impact threshold breach.

        Callback receives: experiment_id, metric_name, impact_value
        """
        self._on_impact.append(callback)

    def on_abort(self, callback: Callable[[str, str], None]):
        """Register callback for experiment abort.

        Callback receives: experiment_id, reason
        """
        self._on_abort.append(callback)

    def _trigger_callbacks(self, callbacks: List[Callable], *args):
        """Trigger callbacks safely."""
        for cb in callbacks:
            try:
                cb(*args)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    async def run_experiment(
        self,
        name: str,
        affected_services: List[str],
        faults: List[Any],  # InjectedFault list
        duration_seconds: float = 60,
        baseline_collector: Optional[Callable[[], Dict[str, float]]] = None,
        impact_collector: Optional[Callable[[], Dict[str, float]]] = None,
        abort_on_impact: bool = True,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Run a chaos experiment with full observability.

        Args:
            name: Experiment name
            affected_services: Services being tested
            faults: List of faults to inject
            duration_seconds: How long to run
            baseline_collector: Function to collect baseline metrics
            impact_collector: Function to collect impact metrics
            abort_on_impact: Whether to auto-abort on high impact

        Returns:
            Tuple of (success, results_dict)
        """
        import asyncio

        experiment_id = hashlib.sha256(
            f"{name}:{time.time_ns()}".encode()
        ).hexdigest()[:12]

        # Collect baseline
        baseline = {}
        if baseline_collector:
            try:
                baseline = baseline_collector()
                self.metrics.set_baseline_metrics(experiment_id, baseline)
            except Exception as e:
                logger.error(f"Failed to collect baseline: {e}")

        # Start experiment
        self.metrics.start_experiment(
            experiment_id=experiment_id,
            name=name,
            affected_services=affected_services,
            metadata={"faults": len(faults), "duration": duration_seconds},
        )

        self._trigger_callbacks(self._on_start, experiment_id, name, affected_services)

        success = True
        impact_metrics = {}
        aborted = False

        try:
            # Monitor during experiment
            check_interval = min(10, duration_seconds / 6)
            elapsed = 0

            while elapsed < duration_seconds:
                await asyncio.sleep(check_interval)
                elapsed += check_interval

                # Check impact
                if impact_collector and abort_on_impact:
                    try:
                        current = impact_collector()
                        for metric, value in current.items():
                            baseline_value = baseline.get(metric, value)
                            if baseline_value:
                                impact = abs(value - baseline_value) / baseline_value
                                self.metrics.record_impact(
                                    experiment_id, metric, baseline_value, value
                                )

                                if impact > self.auto_abort_threshold:
                                    self._trigger_callbacks(
                                        self._on_impact, experiment_id, metric, impact
                                    )

                                    # Auto-abort
                                    reason = f"Impact threshold exceeded: {metric}={impact:.2%}"
                                    self.metrics.abort_experiment(experiment_id, reason)
                                    self._trigger_callbacks(self._on_abort, experiment_id, reason)
                                    aborted = True
                                    success = False
                                    break

                    except Exception as e:
                        logger.error(f"Impact check failed: {e}")

                if aborted:
                    break

            # Final impact collection
            if impact_collector and not aborted:
                try:
                    impact_metrics = impact_collector()
                except Exception as e:
                    logger.error(f"Final impact collection failed: {e}")

        except asyncio.CancelledError:
            self.metrics.abort_experiment(experiment_id, "Cancelled")
            success = False
        except Exception as e:
            self.metrics.abort_experiment(experiment_id, str(e))
            success = False
            logger.error(f"Experiment failed: {e}")

        if not aborted:
            self.metrics.end_experiment(
                experiment_id=experiment_id,
                success=success,
                impact_metrics=impact_metrics,
            )

        self._trigger_callbacks(self._on_end, experiment_id, success, impact_metrics)

        return success, {
            "experiment_id": experiment_id,
            "name": name,
            "success": success,
            "aborted": aborted,
            "duration_seconds": elapsed if aborted else duration_seconds,
            "baseline": baseline,
            "impact": impact_metrics,
        }
