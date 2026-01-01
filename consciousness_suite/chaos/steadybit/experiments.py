"""Steadybit Experiment Definitions

Defines and manages Steadybit chaos experiments.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
from enum import Enum
import logging
import json

logger = logging.getLogger(__name__)


class ExperimentStatus(str, Enum):
    """Status of a Steadybit experiment."""
    CREATED = "created"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"
    PAUSED = "paused"


class TargetType(str, Enum):
    """Types of attack targets."""
    CONTAINER = "container"
    HOST = "host"
    KUBERNETES_POD = "kubernetes-pod"
    KUBERNETES_DEPLOYMENT = "kubernetes-deployment"
    KUBERNETES_NODE = "kubernetes-node"
    AWS_EC2 = "aws-ec2"
    AWS_ECS = "aws-ecs"
    AWS_LAMBDA = "aws-lambda"
    AZURE_VM = "azure-vm"
    GCP_VM = "gcp-vm"
    APPLICATION = "application"
    DATABASE = "database"


@dataclass
class ExperimentTarget:
    """Target for a Steadybit experiment."""
    target_type: TargetType
    attributes: Dict[str, str]
    selection_mode: str = "all"  # all, random, percentage
    percentage: float = 100.0
    count: Optional[int] = None


@dataclass
class ExperimentStep:
    """A step in a Steadybit experiment."""
    action_id: str
    parameters: Dict[str, Any]
    targets: List[ExperimentTarget]
    duration_seconds: float
    radius: str = "all"  # all, one, percentage
    radius_percentage: float = 100.0


@dataclass
class ExperimentGuardrail:
    """Guardrail/abort condition for experiment."""
    check_type: str  # http, prometheus, datadog, custom
    config: Dict[str, Any]
    abort_on_failure: bool = True


@dataclass
class SteadybitExperiment:
    """A Steadybit chaos experiment definition.

    Usage:
        experiment = SteadybitExperiment(
            name="Payment Service Latency Test",
            description="Test payment service resilience to latency",
            team="platform",
            environment="staging",
            steps=[
                ExperimentStep(
                    action_id="container-network-delay",
                    parameters={"delay": 500, "jitter": 100},
                    targets=[payment_target],
                    duration_seconds=60,
                ),
            ],
            guardrails=[error_rate_guardrail],
        )
    """
    name: str
    description: str
    team: str
    environment: str
    steps: List[ExperimentStep]
    guardrails: List[ExperimentGuardrail] = field(default_factory=list)
    hypothesis: Optional[str] = None
    expected_outcome: Optional[str] = None
    status: ExperimentStatus = ExperimentStatus.CREATED
    id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_api_format(self) -> Dict[str, Any]:
        """Convert to Steadybit API format."""
        return {
            "name": self.name,
            "description": self.description,
            "team": self.team,
            "environment": self.environment,
            "hypothesis": self.hypothesis,
            "expectedOutcome": self.expected_outcome,
            "lanes": [
                {
                    "steps": [
                        {
                            "type": "action",
                            "actionId": step.action_id,
                            "parameters": step.parameters,
                            "duration": f"{int(step.duration_seconds)}s",
                            "radius": {
                                "mode": step.radius,
                                "percentage": step.radius_percentage,
                            },
                            "targets": [
                                {
                                    "type": t.target_type.value,
                                    "attributes": t.attributes,
                                    "selection": {
                                        "mode": t.selection_mode,
                                        "percentage": t.percentage,
                                    },
                                }
                                for t in step.targets
                            ],
                        }
                        for step in self.steps
                    ]
                }
            ],
            "guardrails": [
                {
                    "type": g.check_type,
                    "config": g.config,
                    "abortOnFailure": g.abort_on_failure,
                }
                for g in self.guardrails
            ],
            "tags": self.tags,
        }


class ExperimentBuilder:
    """Fluent builder for Steadybit experiments.

    Usage:
        experiment = (
            ExperimentBuilder("payment-latency-test")
            .description("Test payment service latency handling")
            .team("platform")
            .environment("staging")
            .target_kubernetes_deployment("payment-service", namespace="prod")
            .add_latency_attack(delay_ms=500, duration_seconds=60)
            .add_guardrail_prometheus("error_rate > 0.05")
            .hypothesis("Payment service handles 500ms latency without errors")
            .build()
        )
    """

    def __init__(self, name: str):
        self._name = name
        self._description = ""
        self._team = "default"
        self._environment = "staging"
        self._hypothesis: Optional[str] = None
        self._expected_outcome: Optional[str] = None
        self._steps: List[ExperimentStep] = []
        self._guardrails: List[ExperimentGuardrail] = []
        self._current_targets: List[ExperimentTarget] = []
        self._tags: List[str] = []
        self._metadata: Dict[str, Any] = {}

    def description(self, desc: str) -> ExperimentBuilder:
        """Set experiment description."""
        self._description = desc
        return self

    def team(self, team: str) -> ExperimentBuilder:
        """Set owning team."""
        self._team = team
        return self

    def environment(self, env: str) -> ExperimentBuilder:
        """Set target environment."""
        self._environment = env
        return self

    def hypothesis(self, hypothesis: str) -> ExperimentBuilder:
        """Set experiment hypothesis."""
        self._hypothesis = hypothesis
        return self

    def expected_outcome(self, outcome: str) -> ExperimentBuilder:
        """Set expected outcome."""
        self._expected_outcome = outcome
        return self

    def tag(self, *tags: str) -> ExperimentBuilder:
        """Add tags."""
        self._tags.extend(tags)
        return self

    # Target methods
    def target_container(
        self,
        name_pattern: str,
        image_pattern: Optional[str] = None,
    ) -> ExperimentBuilder:
        """Target containers by name/image pattern."""
        attrs = {"container.name": name_pattern}
        if image_pattern:
            attrs["container.image"] = image_pattern

        self._current_targets.append(ExperimentTarget(
            target_type=TargetType.CONTAINER,
            attributes=attrs,
        ))
        return self

    def target_kubernetes_deployment(
        self,
        deployment: str,
        namespace: str = "default",
    ) -> ExperimentBuilder:
        """Target Kubernetes deployment."""
        self._current_targets.append(ExperimentTarget(
            target_type=TargetType.KUBERNETES_DEPLOYMENT,
            attributes={
                "k8s.deployment.name": deployment,
                "k8s.namespace": namespace,
            },
        ))
        return self

    def target_kubernetes_pod(
        self,
        label_selector: str,
        namespace: str = "default",
    ) -> ExperimentBuilder:
        """Target Kubernetes pods by label selector."""
        self._current_targets.append(ExperimentTarget(
            target_type=TargetType.KUBERNETES_POD,
            attributes={
                "k8s.label": label_selector,
                "k8s.namespace": namespace,
            },
        ))
        return self

    def target_host(self, hostname_pattern: str) -> ExperimentBuilder:
        """Target hosts by hostname pattern."""
        self._current_targets.append(ExperimentTarget(
            target_type=TargetType.HOST,
            attributes={"host.name": hostname_pattern},
        ))
        return self

    def with_selection(
        self,
        mode: str = "all",
        percentage: float = 100.0,
        count: Optional[int] = None,
    ) -> ExperimentBuilder:
        """Set selection mode for current targets."""
        if self._current_targets:
            target = self._current_targets[-1]
            target.selection_mode = mode
            target.percentage = percentage
            target.count = count
        return self

    # Attack methods
    def add_latency_attack(
        self,
        delay_ms: int,
        duration_seconds: float,
        jitter_ms: int = 0,
        port: Optional[int] = None,
    ) -> ExperimentBuilder:
        """Add network latency attack."""
        params = {
            "delay": delay_ms,
            "jitter": jitter_ms,
        }
        if port:
            params["port"] = port

        self._steps.append(ExperimentStep(
            action_id="container-network-delay",
            parameters=params,
            targets=self._current_targets.copy(),
            duration_seconds=duration_seconds,
        ))
        return self

    def add_packet_loss_attack(
        self,
        loss_percentage: float,
        duration_seconds: float,
        port: Optional[int] = None,
    ) -> ExperimentBuilder:
        """Add packet loss attack."""
        params = {"loss": loss_percentage}
        if port:
            params["port"] = port

        self._steps.append(ExperimentStep(
            action_id="container-network-loss",
            parameters=params,
            targets=self._current_targets.copy(),
            duration_seconds=duration_seconds,
        ))
        return self

    def add_cpu_stress_attack(
        self,
        cpu_load_percentage: int,
        duration_seconds: float,
        core_count: Optional[int] = None,
    ) -> ExperimentBuilder:
        """Add CPU stress attack."""
        params = {"cpuLoad": cpu_load_percentage}
        if core_count:
            params["cores"] = core_count

        self._steps.append(ExperimentStep(
            action_id="container-stress-cpu",
            parameters=params,
            targets=self._current_targets.copy(),
            duration_seconds=duration_seconds,
        ))
        return self

    def add_memory_stress_attack(
        self,
        memory_percentage: int,
        duration_seconds: float,
    ) -> ExperimentBuilder:
        """Add memory stress attack."""
        self._steps.append(ExperimentStep(
            action_id="container-stress-memory",
            parameters={"memoryLoad": memory_percentage},
            targets=self._current_targets.copy(),
            duration_seconds=duration_seconds,
        ))
        return self

    def add_kill_container_attack(
        self,
        duration_seconds: float,
        graceful: bool = True,
    ) -> ExperimentBuilder:
        """Add container kill attack."""
        self._steps.append(ExperimentStep(
            action_id="container-stop",
            parameters={"graceful": graceful},
            targets=self._current_targets.copy(),
            duration_seconds=duration_seconds,
        ))
        return self

    def add_pod_delete_attack(
        self,
        duration_seconds: float,
        grace_period_seconds: int = 30,
    ) -> ExperimentBuilder:
        """Add Kubernetes pod delete attack."""
        self._steps.append(ExperimentStep(
            action_id="kubernetes-pod-delete",
            parameters={"gracePeriodSeconds": grace_period_seconds},
            targets=self._current_targets.copy(),
            duration_seconds=duration_seconds,
        ))
        return self

    def add_dns_failure_attack(
        self,
        duration_seconds: float,
        hostnames: Optional[List[str]] = None,
    ) -> ExperimentBuilder:
        """Add DNS failure attack."""
        params = {}
        if hostnames:
            params["hostnames"] = hostnames

        self._steps.append(ExperimentStep(
            action_id="container-network-dns",
            parameters=params,
            targets=self._current_targets.copy(),
            duration_seconds=duration_seconds,
        ))
        return self

    # Guardrail methods
    def add_guardrail_http(
        self,
        url: str,
        expected_status: int = 200,
        timeout_seconds: float = 10,
        abort_on_failure: bool = True,
    ) -> ExperimentBuilder:
        """Add HTTP health check guardrail."""
        self._guardrails.append(ExperimentGuardrail(
            check_type="http",
            config={
                "url": url,
                "expectedStatus": expected_status,
                "timeout": f"{timeout_seconds}s",
            },
            abort_on_failure=abort_on_failure,
        ))
        return self

    def add_guardrail_prometheus(
        self,
        query: str,
        abort_on_failure: bool = True,
    ) -> ExperimentBuilder:
        """Add Prometheus query guardrail."""
        self._guardrails.append(ExperimentGuardrail(
            check_type="prometheus",
            config={"query": query},
            abort_on_failure=abort_on_failure,
        ))
        return self

    def add_guardrail_datadog(
        self,
        monitor_id: str,
        abort_on_failure: bool = True,
    ) -> ExperimentBuilder:
        """Add Datadog monitor guardrail."""
        self._guardrails.append(ExperimentGuardrail(
            check_type="datadog",
            config={"monitorId": monitor_id},
            abort_on_failure=abort_on_failure,
        ))
        return self

    def build(self) -> SteadybitExperiment:
        """Build the experiment."""
        return SteadybitExperiment(
            name=self._name,
            description=self._description,
            team=self._team,
            environment=self._environment,
            steps=self._steps,
            guardrails=self._guardrails,
            hypothesis=self._hypothesis,
            expected_outcome=self._expected_outcome,
            tags=self._tags,
            metadata=self._metadata,
        )


class SteadybitClient:
    """Client for Steadybit API.

    Usage:
        client = SteadybitClient(
            base_url="https://steadybit.example.com",
            api_token="your-token"
        )

        # Create and run experiment
        experiment = builder.build()
        run_id = await client.run_experiment(experiment)
        result = await client.wait_for_completion(run_id)
    """

    def __init__(
        self,
        base_url: str,
        api_token: str,
        timeout_seconds: float = 30,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_token = api_token
        self.timeout = timeout_seconds
        self._client = None

    async def _get_client(self):
        """Get HTTP client (lazy initialization)."""
        if self._client is None:
            try:
                import httpx
                self._client = httpx.AsyncClient(
                    base_url=self.base_url,
                    headers={
                        "Authorization": f"Bearer {self.api_token}",
                        "Content-Type": "application/json",
                    },
                    timeout=self.timeout,
                )
            except ImportError:
                logger.warning("httpx not installed, using mock client")
                self._client = MockSteadybitClient()
        return self._client

    async def create_experiment(
        self,
        experiment: SteadybitExperiment,
    ) -> str:
        """Create an experiment definition.

        Args:
            experiment: Experiment to create

        Returns:
            Experiment ID
        """
        client = await self._get_client()

        if isinstance(client, MockSteadybitClient):
            return client.create_experiment(experiment)

        response = await client.post(
            "/api/experiments",
            json=experiment.to_api_format(),
        )
        response.raise_for_status()
        data = response.json()
        return data["id"]

    async def run_experiment(
        self,
        experiment: Union[str, SteadybitExperiment],
    ) -> str:
        """Run an experiment.

        Args:
            experiment: Experiment ID or SteadybitExperiment

        Returns:
            Execution/run ID
        """
        client = await self._get_client()

        if isinstance(experiment, SteadybitExperiment):
            experiment_id = await self.create_experiment(experiment)
        else:
            experiment_id = experiment

        if isinstance(client, MockSteadybitClient):
            return client.run_experiment(experiment_id)

        response = await client.post(
            f"/api/experiments/{experiment_id}/runs",
        )
        response.raise_for_status()
        data = response.json()
        return data["id"]

    async def get_run_status(self, run_id: str) -> Dict[str, Any]:
        """Get status of an experiment run.

        Args:
            run_id: Run/execution ID

        Returns:
            Run status details
        """
        client = await self._get_client()

        if isinstance(client, MockSteadybitClient):
            return client.get_run_status(run_id)

        response = await client.get(f"/api/runs/{run_id}")
        response.raise_for_status()
        return response.json()

    async def wait_for_completion(
        self,
        run_id: str,
        poll_interval_seconds: float = 5,
        timeout_seconds: float = 3600,
    ) -> Dict[str, Any]:
        """Wait for experiment run to complete.

        Args:
            run_id: Run ID
            poll_interval_seconds: How often to check
            timeout_seconds: Maximum wait time

        Returns:
            Final run status
        """
        start_time = time.time()

        while True:
            status = await self.get_run_status(run_id)

            if status.get("status") in ("completed", "failed", "aborted"):
                return status

            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                raise TimeoutError(f"Experiment {run_id} did not complete in time")

            await asyncio.sleep(poll_interval_seconds)

    async def abort_run(self, run_id: str):
        """Abort a running experiment.

        Args:
            run_id: Run ID to abort
        """
        client = await self._get_client()

        if isinstance(client, MockSteadybitClient):
            return client.abort_run(run_id)

        response = await client.post(f"/api/runs/{run_id}/abort")
        response.raise_for_status()

    async def list_experiments(
        self,
        team: Optional[str] = None,
        environment: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List experiments.

        Args:
            team: Filter by team
            environment: Filter by environment

        Returns:
            List of experiments
        """
        client = await self._get_client()

        params = {}
        if team:
            params["team"] = team
        if environment:
            params["environment"] = environment

        if isinstance(client, MockSteadybitClient):
            return []

        response = await client.get("/api/experiments", params=params)
        response.raise_for_status()
        return response.json().get("experiments", [])

    async def close(self):
        """Close the client."""
        if self._client and not isinstance(self._client, MockSteadybitClient):
            await self._client.aclose()


class MockSteadybitClient:
    """Mock client for testing without Steadybit server."""

    def __init__(self):
        self._experiments: Dict[str, Dict] = {}
        self._runs: Dict[str, Dict] = {}
        self._counter = 0

    def create_experiment(self, experiment: SteadybitExperiment) -> str:
        self._counter += 1
        exp_id = f"exp-mock-{self._counter}"
        self._experiments[exp_id] = experiment.to_api_format()
        logger.info(f"Mock: Created experiment {exp_id}")
        return exp_id

    def run_experiment(self, experiment_id: str) -> str:
        self._counter += 1
        run_id = f"run-mock-{self._counter}"
        self._runs[run_id] = {
            "id": run_id,
            "experimentId": experiment_id,
            "status": "completed",
            "startedAt": datetime.now().isoformat(),
            "completedAt": datetime.now().isoformat(),
            "success": True,
        }
        logger.info(f"Mock: Started run {run_id} for {experiment_id}")
        return run_id

    def get_run_status(self, run_id: str) -> Dict[str, Any]:
        return self._runs.get(run_id, {"status": "unknown"})

    def abort_run(self, run_id: str):
        if run_id in self._runs:
            self._runs[run_id]["status"] = "aborted"
        logger.info(f"Mock: Aborted run {run_id}")
