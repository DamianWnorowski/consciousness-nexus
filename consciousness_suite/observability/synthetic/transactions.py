"""Transaction Probes

Multi-step transaction testing for synthetic monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
from enum import Enum
import logging
import re

from .probes import SyntheticProbe, ProbeConfig, ProbeResult, ProbeType, ProbeStatus

logger = logging.getLogger(__name__)


class StepType(str, Enum):
    """Transaction step types."""
    HTTP = "http"
    WAIT = "wait"
    EXTRACT = "extract"
    ASSERT = "assert"
    SCRIPT = "script"


@dataclass
class TransactionStep:
    """A step in a transaction."""
    step_id: str
    name: str
    step_type: StepType
    config: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 30.0
    continue_on_failure: bool = False
    extract_to: Dict[str, str] = field(default_factory=dict)  # var_name -> extraction_path
    assertions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class StepResult:
    """Result of a transaction step."""
    step_id: str
    step_name: str
    step_type: StepType
    status: ProbeStatus
    latency_ms: float
    timestamp: datetime = field(default_factory=datetime.now)
    response_code: Optional[int] = None
    response_body: Optional[str] = None
    extracted_values: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    assertions_passed: int = 0
    assertions_failed: int = 0
    assertion_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "step_name": self.step_name,
            "step_type": self.step_type.value,
            "status": self.status.value,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp.isoformat(),
            "response_code": self.response_code,
            "extracted_values": self.extracted_values,
            "error_message": self.error_message,
            "assertions_passed": self.assertions_passed,
            "assertions_failed": self.assertions_failed,
        }


@dataclass
class TransactionResult(ProbeResult):
    """Result of a transaction execution."""
    steps: List[StepResult] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    steps_passed: int = 0
    steps_failed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "steps_passed": self.steps_passed,
            "steps_failed": self.steps_failed,
            "steps": [s.to_dict() for s in self.steps],
        })
        return base


class TransactionProbe(SyntheticProbe):
    """Multi-step transaction probe.

    Usage:
        probe = TransactionProbe()

        steps = [
            TransactionStep(
                step_id="login",
                name="Login",
                step_type=StepType.HTTP,
                config={
                    "url": "https://api.example.com/login",
                    "method": "POST",
                    "body": {"username": "test", "password": "test"},
                },
                extract_to={"token": "$.data.token"},
            ),
            TransactionStep(
                step_id="get-profile",
                name="Get Profile",
                step_type=StepType.HTTP,
                config={
                    "url": "https://api.example.com/profile",
                    "headers": {"Authorization": "Bearer ${token}"},
                },
            ),
        ]

        config = ProbeConfig(...)
        result = await probe.execute_transaction(config, steps)
    """

    def __init__(self, namespace: str = "consciousness"):
        super().__init__(namespace)
        self._variables: Dict[str, Any] = {}

    async def execute(self, config: ProbeConfig) -> ProbeResult:
        """Execute probe (delegates to execute_transaction).

        For transactions, use execute_transaction directly.
        """
        return ProbeResult(
            probe_id=config.probe_id,
            probe_name=config.name,
            probe_type=config.probe_type,
            status=ProbeStatus.ERROR,
            target=config.target,
            error_message="Use execute_transaction for transaction probes",
        )

    async def execute_transaction(
        self,
        config: ProbeConfig,
        steps: List[TransactionStep],
        initial_variables: Optional[Dict[str, Any]] = None,
    ) -> TransactionResult:
        """Execute a multi-step transaction.

        Args:
            config: Probe configuration
            steps: Transaction steps
            initial_variables: Initial variable values

        Returns:
            TransactionResult
        """
        import time

        start_time = time.time()
        self._variables = initial_variables.copy() if initial_variables else {}

        step_results: List[StepResult] = []
        steps_passed = 0
        steps_failed = 0
        overall_status = ProbeStatus.SUCCESS

        for step in steps:
            step_result = await self._execute_step(step)
            step_results.append(step_result)

            # Update variables with extracted values
            self._variables.update(step_result.extracted_values)

            if step_result.status == ProbeStatus.SUCCESS:
                steps_passed += 1
            else:
                steps_failed += 1
                if not step.continue_on_failure:
                    overall_status = ProbeStatus.FAILURE
                    break
                elif overall_status == ProbeStatus.SUCCESS:
                    overall_status = ProbeStatus.DEGRADED

        total_latency = (time.time() - start_time) * 1000

        return TransactionResult(
            probe_id=config.probe_id,
            probe_name=config.name,
            probe_type=ProbeType.TRANSACTION,
            status=overall_status,
            target=config.target,
            latency_ms=total_latency,
            steps=step_results,
            variables=self._variables.copy(),
            steps_passed=steps_passed,
            steps_failed=steps_failed,
        )

    async def _execute_step(self, step: TransactionStep) -> StepResult:
        """Execute a single transaction step.

        Args:
            step: Transaction step

        Returns:
            StepResult
        """
        import time

        start_time = time.time()

        try:
            if step.step_type == StepType.HTTP:
                return await self._execute_http_step(step, start_time)
            elif step.step_type == StepType.WAIT:
                return await self._execute_wait_step(step, start_time)
            elif step.step_type == StepType.EXTRACT:
                return self._execute_extract_step(step, start_time)
            elif step.step_type == StepType.ASSERT:
                return self._execute_assert_step(step, start_time)
            elif step.step_type == StepType.SCRIPT:
                return await self._execute_script_step(step, start_time)
            else:
                return StepResult(
                    step_id=step.step_id,
                    step_name=step.name,
                    step_type=step.step_type,
                    status=ProbeStatus.ERROR,
                    latency_ms=(time.time() - start_time) * 1000,
                    error_message=f"Unknown step type: {step.step_type}",
                )

        except Exception as e:
            return StepResult(
                step_id=step.step_id,
                step_name=step.name,
                step_type=step.step_type,
                status=ProbeStatus.ERROR,
                latency_ms=(time.time() - start_time) * 1000,
                error_message=str(e),
            )

    async def _execute_http_step(
        self,
        step: TransactionStep,
        start_time: float,
    ) -> StepResult:
        """Execute HTTP step."""
        import time
        import httpx
        import json

        config = step.config
        url = self._substitute_variables(config.get("url", ""))
        method = config.get("method", "GET").upper()
        headers = config.get("headers", {})
        body = config.get("body")

        # Substitute variables in headers
        headers = {
            k: self._substitute_variables(str(v))
            for k, v in headers.items()
        }

        # Substitute variables in body
        if body and isinstance(body, str):
            body = self._substitute_variables(body)
        elif body and isinstance(body, dict):
            body = json.dumps(body)

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(step.timeout_seconds),
            ) as client:
                if method == "GET":
                    response = await client.get(url, headers=headers)
                elif method == "POST":
                    response = await client.post(url, headers=headers, content=body)
                elif method == "PUT":
                    response = await client.put(url, headers=headers, content=body)
                elif method == "DELETE":
                    response = await client.delete(url, headers=headers)
                else:
                    response = await client.request(method, url, headers=headers, content=body)

                latency = (time.time() - start_time) * 1000

                # Extract values
                extracted = {}
                for var_name, path in step.extract_to.items():
                    try:
                        value = self._extract_value(response.text, path)
                        if value is not None:
                            extracted[var_name] = value
                    except Exception as e:
                        logger.warning(f"Extraction failed for {var_name}: {e}")

                # Check expected status
                expected_status = config.get("expected_status", [200, 201])
                if not isinstance(expected_status, list):
                    expected_status = [expected_status]

                status = ProbeStatus.SUCCESS if response.status_code in expected_status else ProbeStatus.FAILURE

                # Run assertions
                result = StepResult(
                    step_id=step.step_id,
                    step_name=step.name,
                    step_type=step.step_type,
                    status=status,
                    latency_ms=latency,
                    response_code=response.status_code,
                    response_body=response.text[:5000],
                    extracted_values=extracted,
                )

                self._run_step_assertions(result, step.assertions)

                return result

        except httpx.TimeoutException:
            return StepResult(
                step_id=step.step_id,
                step_name=step.name,
                step_type=step.step_type,
                status=ProbeStatus.TIMEOUT,
                latency_ms=(time.time() - start_time) * 1000,
                error_message="Request timeout",
            )

    async def _execute_wait_step(
        self,
        step: TransactionStep,
        start_time: float,
    ) -> StepResult:
        """Execute wait step."""
        import time
        import asyncio

        duration = step.config.get("duration_seconds", 1.0)
        await asyncio.sleep(duration)

        return StepResult(
            step_id=step.step_id,
            step_name=step.name,
            step_type=step.step_type,
            status=ProbeStatus.SUCCESS,
            latency_ms=(time.time() - start_time) * 1000,
        )

    def _execute_extract_step(
        self,
        step: TransactionStep,
        start_time: float,
    ) -> StepResult:
        """Execute extraction step."""
        import time

        extracted = {}
        source = step.config.get("source", "")
        source = self._substitute_variables(source)

        for var_name, path in step.extract_to.items():
            try:
                value = self._extract_value(source, path)
                if value is not None:
                    extracted[var_name] = value
            except Exception as e:
                logger.warning(f"Extraction failed for {var_name}: {e}")

        return StepResult(
            step_id=step.step_id,
            step_name=step.name,
            step_type=step.step_type,
            status=ProbeStatus.SUCCESS if extracted else ProbeStatus.FAILURE,
            latency_ms=(time.time() - start_time) * 1000,
            extracted_values=extracted,
        )

    def _execute_assert_step(
        self,
        step: TransactionStep,
        start_time: float,
    ) -> StepResult:
        """Execute assertion step."""
        import time

        result = StepResult(
            step_id=step.step_id,
            step_name=step.name,
            step_type=step.step_type,
            status=ProbeStatus.SUCCESS,
            latency_ms=(time.time() - start_time) * 1000,
        )

        self._run_step_assertions(result, step.assertions)

        if result.assertions_failed > 0:
            result.status = ProbeStatus.FAILURE

        return result

    async def _execute_script_step(
        self,
        step: TransactionStep,
        start_time: float,
    ) -> StepResult:
        """Execute custom script step."""
        import time

        # Get script function from config
        script_fn = step.config.get("function")

        if not script_fn or not callable(script_fn):
            return StepResult(
                step_id=step.step_id,
                step_name=step.name,
                step_type=step.step_type,
                status=ProbeStatus.ERROR,
                latency_ms=(time.time() - start_time) * 1000,
                error_message="Script function not provided or not callable",
            )

        try:
            import asyncio

            if asyncio.iscoroutinefunction(script_fn):
                result_data = await script_fn(self._variables)
            else:
                result_data = script_fn(self._variables)

            extracted = {}
            if isinstance(result_data, dict):
                extracted = result_data

            return StepResult(
                step_id=step.step_id,
                step_name=step.name,
                step_type=step.step_type,
                status=ProbeStatus.SUCCESS,
                latency_ms=(time.time() - start_time) * 1000,
                extracted_values=extracted,
            )

        except Exception as e:
            return StepResult(
                step_id=step.step_id,
                step_name=step.name,
                step_type=step.step_type,
                status=ProbeStatus.ERROR,
                latency_ms=(time.time() - start_time) * 1000,
                error_message=str(e),
            )

    def _substitute_variables(self, text: str) -> str:
        """Substitute ${variable} placeholders in text."""
        pattern = r'\$\{(\w+)\}'

        def replace(match):
            var_name = match.group(1)
            return str(self._variables.get(var_name, match.group(0)))

        return re.sub(pattern, replace, text)

    def _extract_value(self, source: str, path: str) -> Any:
        """Extract value from source using path.

        Supports:
        - JSONPath: $.data.token
        - Regex: regex:token=([a-z0-9]+)
        - Header: header:Content-Type
        """
        if path.startswith("$."):
            return self._json_path_extract(source, path)
        elif path.startswith("regex:"):
            pattern = path[6:]
            match = re.search(pattern, source)
            if match:
                return match.group(1) if match.groups() else match.group(0)
            return None
        else:
            return self._json_path_extract(source, path)

    def _json_path_extract(self, source: str, path: str) -> Any:
        """Extract value using JSONPath."""
        import json

        try:
            data = json.loads(source)
        except json.JSONDecodeError:
            return None

        # Simple JSONPath implementation
        parts = path.strip("$.").split(".")
        current = data

        for part in parts:
            # Handle array index
            if "[" in part:
                name, idx_str = part.split("[")
                idx = int(idx_str.rstrip("]"))
                if name and isinstance(current, dict):
                    current = current.get(name, [])
                if isinstance(current, list) and len(current) > idx:
                    current = current[idx]
                else:
                    return None
            elif isinstance(current, dict):
                current = current.get(part)
            else:
                return None

            if current is None:
                return None

        return current

    def _run_step_assertions(
        self,
        result: StepResult,
        assertions: List[Dict[str, Any]],
    ):
        """Run assertions for a step."""
        for assertion in assertions:
            try:
                passed = self._check_step_assertion(result, assertion)
                if passed:
                    result.assertions_passed += 1
                else:
                    result.assertions_failed += 1
                    result.assertion_errors.append(f"Failed: {assertion}")
            except Exception as e:
                result.assertions_failed += 1
                result.assertion_errors.append(f"Error: {e}")

    def _check_step_assertion(
        self,
        result: StepResult,
        assertion: Dict[str, Any],
    ) -> bool:
        """Check a single step assertion."""
        assertion_type = assertion.get("type", "status_code")

        if assertion_type == "status_code":
            expected = assertion.get("expected", 200)
            return result.response_code == expected

        elif assertion_type == "variable_equals":
            var_name = assertion.get("variable")
            expected = assertion.get("expected")
            return self._variables.get(var_name) == expected

        elif assertion_type == "variable_exists":
            var_name = assertion.get("variable")
            return var_name in self._variables

        elif assertion_type == "variable_matches":
            var_name = assertion.get("variable")
            pattern = assertion.get("pattern", "")
            value = str(self._variables.get(var_name, ""))
            return bool(re.match(pattern, value))

        elif assertion_type == "body_contains":
            text = assertion.get("text", "")
            return text in (result.response_body or "")

        elif assertion_type == "latency":
            max_ms = assertion.get("max_ms", 1000)
            return result.latency_ms <= max_ms

        return True
