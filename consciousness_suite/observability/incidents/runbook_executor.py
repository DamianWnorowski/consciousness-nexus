"""Runbook Executor

Automated runbook execution for incident response.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import threading
import logging

from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)


class StepType(str, Enum):
    """Runbook step types."""
    SHELL = "shell"
    HTTP = "http"
    KUBERNETES = "kubernetes"
    DATABASE = "database"
    NOTIFICATION = "notification"
    APPROVAL = "approval"
    WAIT = "wait"
    CONDITION = "condition"
    PARALLEL = "parallel"
    SCRIPT = "script"


class ExecutionStatus(str, Enum):
    """Execution status."""
    PENDING = "pending"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    SUCCESS = "success"
    FAILURE = "failure"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class RunbookStep:
    """A step in a runbook."""
    step_id: str
    name: str
    step_type: StepType
    config: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 300.0
    retry_count: int = 0
    retry_delay_seconds: float = 5.0
    continue_on_failure: bool = False
    condition: Optional[str] = None  # Expression to evaluate
    requires_approval: bool = False
    approval_timeout_minutes: int = 30
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "name": self.name,
            "step_type": self.step_type.value,
            "timeout_seconds": self.timeout_seconds,
            "requires_approval": self.requires_approval,
        }


@dataclass
class StepResult:
    """Result of a runbook step execution."""
    step_id: str
    step_name: str
    step_type: StepType
    status: ExecutionStatus
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0
    output: Optional[str] = None
    error_message: Optional[str] = None
    retry_attempts: int = 0
    approved_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "step_name": self.step_name,
            "step_type": self.step_type.value,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "output": self.output[:500] if self.output else None,
            "error_message": self.error_message,
            "retry_attempts": self.retry_attempts,
        }


@dataclass
class Runbook:
    """A runbook definition."""
    runbook_id: str
    name: str
    description: str = ""
    steps: List[RunbookStep] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    max_execution_time_minutes: int = 60
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "runbook_id": self.runbook_id,
            "name": self.name,
            "description": self.description,
            "steps": [s.to_dict() for s in self.steps],
            "enabled": self.enabled,
            "tags": self.tags,
        }


@dataclass
class ExecutionResult:
    """Result of a runbook execution."""
    execution_id: str
    runbook_id: str
    runbook_name: str
    incident_id: Optional[str] = None
    status: ExecutionStatus = ExecutionStatus.PENDING
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0
    step_results: List[StepResult] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    triggered_by: str = "system"
    error_message: Optional[str] = None

    @property
    def steps_passed(self) -> int:
        return sum(1 for r in self.step_results if r.status == ExecutionStatus.SUCCESS)

    @property
    def steps_failed(self) -> int:
        return sum(1 for r in self.step_results if r.status == ExecutionStatus.FAILURE)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "runbook_id": self.runbook_id,
            "runbook_name": self.runbook_name,
            "incident_id": self.incident_id,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "steps_passed": self.steps_passed,
            "steps_failed": self.steps_failed,
            "step_results": [r.to_dict() for r in self.step_results],
            "triggered_by": self.triggered_by,
        }


class RunbookExecutor:
    """Executes automated runbooks.

    Usage:
        executor = RunbookExecutor()

        # Register step handler
        executor.register_handler(StepType.SHELL, shell_handler)

        # Create runbook
        runbook = Runbook(
            runbook_id="restart-service",
            name="Restart Service",
            steps=[
                RunbookStep(
                    step_id="stop",
                    name="Stop Service",
                    step_type=StepType.SHELL,
                    config={"command": "systemctl stop myservice"},
                ),
                RunbookStep(
                    step_id="wait",
                    name="Wait",
                    step_type=StepType.WAIT,
                    config={"seconds": 5},
                ),
                RunbookStep(
                    step_id="start",
                    name="Start Service",
                    step_type=StepType.SHELL,
                    config={"command": "systemctl start myservice"},
                ),
            ],
        )
        executor.add_runbook(runbook)

        # Execute
        result = await executor.execute("restart-service", incident_id="inc-123")
    """

    def __init__(self, namespace: str = "consciousness"):
        self.namespace = namespace

        self._runbooks: Dict[str, Runbook] = {}
        self._executions: Dict[str, ExecutionResult] = {}
        self._handlers: Dict[StepType, Callable] = {}
        self._pending_approvals: Dict[str, asyncio.Event] = {}
        self._lock = threading.Lock()

        # Register default handlers
        self._register_default_handlers()

        # Callbacks
        self._on_start: List[Callable[[ExecutionResult], None]] = []
        self._on_complete: List[Callable[[ExecutionResult], None]] = []
        self._on_step_complete: List[Callable[[StepResult], None]] = []

        # Metrics
        self.executions_total = Counter(
            f"{namespace}_runbook_executions_total",
            "Total runbook executions",
            ["runbook", "status"],
        )

        self.execution_duration = Histogram(
            f"{namespace}_runbook_execution_duration_seconds",
            "Runbook execution duration",
            ["runbook"],
            buckets=[10, 30, 60, 120, 300, 600, 1800, 3600],
        )

        self.step_duration = Histogram(
            f"{namespace}_runbook_step_duration_seconds",
            "Step execution duration",
            ["runbook", "step_type"],
            buckets=[1, 5, 10, 30, 60, 120, 300],
        )

        self.active_executions = Gauge(
            f"{namespace}_runbook_active_executions",
            "Active runbook executions",
        )

        self.pending_approvals = Gauge(
            f"{namespace}_runbook_pending_approvals",
            "Pending approval requests",
        )

    def _register_default_handlers(self):
        """Register default step handlers."""
        self._handlers[StepType.WAIT] = self._handle_wait
        self._handlers[StepType.CONDITION] = self._handle_condition
        self._handlers[StepType.NOTIFICATION] = self._handle_notification

    def register_handler(
        self,
        step_type: StepType,
        handler: Callable[[RunbookStep, Dict[str, Any]], StepResult],
    ):
        """Register a step handler.

        Args:
            step_type: Step type to handle
            handler: Async function(step, variables) -> StepResult
        """
        self._handlers[step_type] = handler

    def add_runbook(self, runbook: Runbook):
        """Add a runbook.

        Args:
            runbook: Runbook definition
        """
        with self._lock:
            self._runbooks[runbook.runbook_id] = runbook

        logger.info(f"Added runbook: {runbook.runbook_id}")

    def remove_runbook(self, runbook_id: str):
        """Remove a runbook.

        Args:
            runbook_id: Runbook ID
        """
        with self._lock:
            self._runbooks.pop(runbook_id, None)

    def get_runbook(self, runbook_id: str) -> Optional[Runbook]:
        """Get a runbook.

        Args:
            runbook_id: Runbook ID

        Returns:
            Runbook or None
        """
        with self._lock:
            return self._runbooks.get(runbook_id)

    async def execute(
        self,
        runbook_id: str,
        incident_id: Optional[str] = None,
        variables: Optional[Dict[str, Any]] = None,
        triggered_by: str = "system",
    ) -> ExecutionResult:
        """Execute a runbook.

        Args:
            runbook_id: Runbook ID
            incident_id: Associated incident ID
            variables: Runtime variables
            triggered_by: User or system that triggered execution

        Returns:
            ExecutionResult
        """
        import uuid
        import time

        runbook = self.get_runbook(runbook_id)
        if not runbook or not runbook.enabled:
            return ExecutionResult(
                execution_id=str(uuid.uuid4()),
                runbook_id=runbook_id,
                runbook_name="Unknown",
                status=ExecutionStatus.FAILURE,
                error_message="Runbook not found or disabled",
            )

        execution_id = str(uuid.uuid4())
        start_time = time.time()

        # Merge variables
        run_variables = dict(runbook.variables)
        if variables:
            run_variables.update(variables)

        result = ExecutionResult(
            execution_id=execution_id,
            runbook_id=runbook_id,
            runbook_name=runbook.name,
            incident_id=incident_id,
            status=ExecutionStatus.RUNNING,
            variables=run_variables,
            triggered_by=triggered_by,
        )

        with self._lock:
            self._executions[execution_id] = result

        self._update_active_count()

        # Trigger start callbacks
        for callback in self._on_start:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Start callback error: {e}")

        # Execute steps
        try:
            for step in runbook.steps:
                # Check condition
                if step.condition:
                    if not self._evaluate_condition(step.condition, run_variables):
                        step_result = StepResult(
                            step_id=step.step_id,
                            step_name=step.name,
                            step_type=step.step_type,
                            status=ExecutionStatus.SKIPPED,
                        )
                        result.step_results.append(step_result)
                        continue

                # Execute step
                step_result = await self._execute_step(step, run_variables, execution_id)
                result.step_results.append(step_result)

                # Update variables from step output
                if step_result.metadata.get("output_variables"):
                    run_variables.update(step_result.metadata["output_variables"])

                # Trigger step callbacks
                for callback in self._on_step_complete:
                    try:
                        callback(step_result)
                    except Exception as e:
                        logger.error(f"Step callback error: {e}")

                # Check if we should continue
                if step_result.status == ExecutionStatus.FAILURE:
                    if not step.continue_on_failure:
                        result.status = ExecutionStatus.FAILURE
                        break

            # Determine final status
            if result.status == ExecutionStatus.RUNNING:
                if all(r.status in [ExecutionStatus.SUCCESS, ExecutionStatus.SKIPPED]
                       for r in result.step_results):
                    result.status = ExecutionStatus.SUCCESS
                else:
                    result.status = ExecutionStatus.FAILURE

        except asyncio.CancelledError:
            result.status = ExecutionStatus.CANCELLED
        except Exception as e:
            result.status = ExecutionStatus.FAILURE
            result.error_message = str(e)
            logger.error(f"Runbook execution error: {e}")

        # Finalize
        result.completed_at = datetime.now()
        result.duration_seconds = time.time() - start_time
        result.variables = run_variables

        # Update metrics
        self.executions_total.labels(
            runbook=runbook_id,
            status=result.status.value,
        ).inc()

        self.execution_duration.labels(runbook=runbook_id).observe(result.duration_seconds)

        self._update_active_count()

        # Trigger complete callbacks
        for callback in self._on_complete:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Complete callback error: {e}")

        return result

    async def _execute_step(
        self,
        step: RunbookStep,
        variables: Dict[str, Any],
        execution_id: str,
    ) -> StepResult:
        """Execute a single step.

        Args:
            step: Step to execute
            variables: Current variables
            execution_id: Execution ID

        Returns:
            StepResult
        """
        import time

        start_time = time.time()
        attempts = 0

        # Check for approval
        if step.requires_approval:
            result = await self._wait_for_approval(step, execution_id)
            if result:
                return result

        while attempts <= step.retry_count:
            attempts += 1

            try:
                handler = self._handlers.get(step.step_type)
                if not handler:
                    return StepResult(
                        step_id=step.step_id,
                        step_name=step.name,
                        step_type=step.step_type,
                        status=ExecutionStatus.FAILURE,
                        error_message=f"No handler for step type: {step.step_type}",
                    )

                # Execute with timeout
                if asyncio.iscoroutinefunction(handler):
                    result = await asyncio.wait_for(
                        handler(step, variables),
                        timeout=step.timeout_seconds,
                    )
                else:
                    loop = asyncio.get_event_loop()
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, handler, step, variables),
                        timeout=step.timeout_seconds,
                    )

                result.duration_seconds = time.time() - start_time
                result.retry_attempts = attempts - 1

                # Update step metrics
                self.step_duration.labels(
                    runbook=step.metadata.get("runbook_id", "unknown"),
                    step_type=step.step_type.value,
                ).observe(result.duration_seconds)

                if result.status == ExecutionStatus.SUCCESS:
                    return result

                # Retry on failure
                if attempts <= step.retry_count:
                    await asyncio.sleep(step.retry_delay_seconds)

            except asyncio.TimeoutError:
                return StepResult(
                    step_id=step.step_id,
                    step_name=step.name,
                    step_type=step.step_type,
                    status=ExecutionStatus.TIMEOUT,
                    duration_seconds=time.time() - start_time,
                    error_message="Step execution timeout",
                    retry_attempts=attempts - 1,
                )

            except Exception as e:
                if attempts <= step.retry_count:
                    await asyncio.sleep(step.retry_delay_seconds)
                else:
                    return StepResult(
                        step_id=step.step_id,
                        step_name=step.name,
                        step_type=step.step_type,
                        status=ExecutionStatus.FAILURE,
                        duration_seconds=time.time() - start_time,
                        error_message=str(e),
                        retry_attempts=attempts - 1,
                    )

        return result

    async def _wait_for_approval(
        self,
        step: RunbookStep,
        execution_id: str,
    ) -> Optional[StepResult]:
        """Wait for step approval.

        Args:
            step: Step requiring approval
            execution_id: Execution ID

        Returns:
            StepResult if approval times out or is rejected
        """
        approval_key = f"{execution_id}:{step.step_id}"
        event = asyncio.Event()

        with self._lock:
            self._pending_approvals[approval_key] = event

        self.pending_approvals.inc()

        try:
            await asyncio.wait_for(
                event.wait(),
                timeout=step.approval_timeout_minutes * 60,
            )
            return None  # Approved, continue execution

        except asyncio.TimeoutError:
            return StepResult(
                step_id=step.step_id,
                step_name=step.name,
                step_type=step.step_type,
                status=ExecutionStatus.TIMEOUT,
                error_message="Approval timeout",
            )

        finally:
            with self._lock:
                self._pending_approvals.pop(approval_key, None)
            self.pending_approvals.dec()

    async def approve_step(
        self,
        execution_id: str,
        step_id: str,
        approved_by: str,
    ) -> bool:
        """Approve a pending step.

        Args:
            execution_id: Execution ID
            step_id: Step ID
            approved_by: User approving

        Returns:
            True if approved
        """
        approval_key = f"{execution_id}:{step_id}"

        with self._lock:
            event = self._pending_approvals.get(approval_key)

        if event:
            event.set()
            logger.info(f"Step {step_id} approved by {approved_by}")
            return True

        return False

    def _evaluate_condition(
        self,
        condition: str,
        variables: Dict[str, Any],
    ) -> bool:
        """Evaluate a condition expression.

        Args:
            condition: Condition expression
            variables: Variables for evaluation

        Returns:
            True if condition is met
        """
        try:
            # Simple expression evaluation
            # In production, use a proper expression parser
            return eval(condition, {"__builtins__": {}}, variables)
        except Exception as e:
            logger.warning(f"Condition evaluation error: {e}")
            return False

    async def _handle_wait(
        self,
        step: RunbookStep,
        variables: Dict[str, Any],
    ) -> StepResult:
        """Handle wait step."""
        seconds = step.config.get("seconds", 1)
        await asyncio.sleep(seconds)

        return StepResult(
            step_id=step.step_id,
            step_name=step.name,
            step_type=step.step_type,
            status=ExecutionStatus.SUCCESS,
            output=f"Waited {seconds} seconds",
        )

    async def _handle_condition(
        self,
        step: RunbookStep,
        variables: Dict[str, Any],
    ) -> StepResult:
        """Handle condition step."""
        expression = step.config.get("expression", "True")
        result = self._evaluate_condition(expression, variables)

        return StepResult(
            step_id=step.step_id,
            step_name=step.name,
            step_type=step.step_type,
            status=ExecutionStatus.SUCCESS if result else ExecutionStatus.FAILURE,
            output=f"Condition evaluated to {result}",
        )

    async def _handle_notification(
        self,
        step: RunbookStep,
        variables: Dict[str, Any],
    ) -> StepResult:
        """Handle notification step."""
        message = step.config.get("message", "Notification from runbook")
        channel = step.config.get("channel", "default")

        logger.info(f"Notification [{channel}]: {message}")

        return StepResult(
            step_id=step.step_id,
            step_name=step.name,
            step_type=step.step_type,
            status=ExecutionStatus.SUCCESS,
            output=f"Sent notification to {channel}",
        )

    def _update_active_count(self):
        """Update active executions gauge."""
        with self._lock:
            active = sum(
                1 for e in self._executions.values()
                if e.status == ExecutionStatus.RUNNING
            )
        self.active_executions.set(active)

    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running execution.

        Args:
            execution_id: Execution ID

        Returns:
            True if cancelled
        """
        with self._lock:
            execution = self._executions.get(execution_id)

        if not execution or execution.status != ExecutionStatus.RUNNING:
            return False

        execution.status = ExecutionStatus.CANCELLED
        execution.completed_at = datetime.now()

        self._update_active_count()

        return True

    def on_start(self, callback: Callable[[ExecutionResult], None]):
        """Register execution start callback.

        Args:
            callback: Function to call on execution start
        """
        self._on_start.append(callback)

    def on_complete(self, callback: Callable[[ExecutionResult], None]):
        """Register execution complete callback.

        Args:
            callback: Function to call on execution complete
        """
        self._on_complete.append(callback)

    def on_step_complete(self, callback: Callable[[StepResult], None]):
        """Register step complete callback.

        Args:
            callback: Function to call on step complete
        """
        self._on_step_complete.append(callback)

    def get_execution(self, execution_id: str) -> Optional[ExecutionResult]:
        """Get execution result.

        Args:
            execution_id: Execution ID

        Returns:
            ExecutionResult or None
        """
        with self._lock:
            return self._executions.get(execution_id)

    def get_recent_executions(
        self,
        runbook_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[ExecutionResult]:
        """Get recent executions.

        Args:
            runbook_id: Filter by runbook
            limit: Maximum results

        Returns:
            List of executions
        """
        with self._lock:
            executions = list(self._executions.values())

        if runbook_id:
            executions = [e for e in executions if e.runbook_id == runbook_id]

        executions.sort(key=lambda e: e.started_at, reverse=True)

        return executions[:limit]

    def get_summary(self) -> Dict[str, Any]:
        """Get executor summary.

        Returns:
            Summary dictionary
        """
        with self._lock:
            runbooks = list(self._runbooks.values())
            executions = list(self._executions.values())

        # Count by status
        by_status: Dict[str, int] = {}
        for e in executions:
            s = e.status.value
            by_status[s] = by_status.get(s, 0) + 1

        return {
            "total_runbooks": len(runbooks),
            "enabled_runbooks": sum(1 for r in runbooks if r.enabled),
            "total_executions": len(executions),
            "by_status": by_status,
            "active_executions": by_status.get("running", 0),
            "pending_approvals": len(self._pending_approvals),
        }
