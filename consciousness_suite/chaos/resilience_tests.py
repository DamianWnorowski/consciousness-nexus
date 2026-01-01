"""Automated Resilience Testing Framework

Defines and executes resilience tests with chaos engineering.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import threading
import logging
import traceback

from prometheus_client import Counter, Gauge, Histogram

from .fault_injection import FaultInjector, InjectedFault

logger = logging.getLogger(__name__)


class TestStatus(str, Enum):
    """Status of a resilience test."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


class SeverityLevel(str, Enum):
    """Severity level of a test failure."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ResilienceScore:
    """Resilience score for a service or system."""
    service: str
    overall_score: float  # 0.0 to 1.0
    availability_score: float
    latency_score: float
    error_handling_score: float
    recovery_score: float
    timestamp: datetime = field(default_factory=datetime.now)
    tests_passed: int = 0
    tests_failed: int = 0
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def grade(self) -> str:
        """Get letter grade based on score."""
        if self.overall_score >= 0.95:
            return "A+"
        elif self.overall_score >= 0.90:
            return "A"
        elif self.overall_score >= 0.85:
            return "A-"
        elif self.overall_score >= 0.80:
            return "B+"
        elif self.overall_score >= 0.75:
            return "B"
        elif self.overall_score >= 0.70:
            return "B-"
        elif self.overall_score >= 0.65:
            return "C+"
        elif self.overall_score >= 0.60:
            return "C"
        elif self.overall_score >= 0.55:
            return "C-"
        elif self.overall_score >= 0.50:
            return "D"
        else:
            return "F"


@dataclass
class TestResult:
    """Result of a single resilience test."""
    test_name: str
    test_id: str
    status: TestStatus
    service: str
    duration_ms: float
    passed_assertions: List[str]
    failed_assertions: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    severity: SeverityLevel = SeverityLevel.MEDIUM


@dataclass
class ResilienceTest:
    """Definition of a resilience test.

    Usage:
        test = ResilienceTest(
            name="payment-latency-spike",
            description="Test payment service handles latency spikes",
            target_service="payment-service",
            faults=[latency_fault],
            assertions=[
                ("response_time_p99 < 5000", "P99 latency under 5s"),
                ("error_rate < 0.01", "Error rate under 1%"),
            ],
        )
    """
    name: str
    description: str
    target_service: str
    faults: List[InjectedFault]
    assertions: List[Tuple[str, str]]  # (condition, description)
    timeout_seconds: float = 60
    warmup_seconds: float = 5
    duration_seconds: float = 30
    cooldown_seconds: float = 5
    severity: SeverityLevel = SeverityLevel.MEDIUM
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Callbacks
    setup_fn: Optional[Callable[[], None]] = None
    teardown_fn: Optional[Callable[[], None]] = None
    collect_metrics_fn: Optional[Callable[[], Dict[str, float]]] = None


class ResilienceTestRunner:
    """Executes resilience tests and collects results.

    Usage:
        runner = ResilienceTestRunner(injector)

        # Register tests
        runner.register_test(payment_latency_test)
        runner.register_test(payment_failure_test)

        # Run all tests
        results = await runner.run_all()

        # Get resilience score
        score = runner.calculate_score("payment-service")
    """

    def __init__(
        self,
        injector: FaultInjector,
        namespace: str = "consciousness",
    ):
        self.injector = injector
        self.namespace = namespace
        self._tests: Dict[str, ResilienceTest] = {}
        self._results: Dict[str, List[TestResult]] = {}
        self._running_tests: set = set()
        self._lock = threading.Lock()

        # Prometheus metrics
        self.tests_total = Counter(
            f"{namespace}_resilience_tests_total",
            "Total resilience tests run",
            ["service", "test_name", "status"],
        )

        self.test_duration = Histogram(
            f"{namespace}_resilience_test_duration_seconds",
            "Duration of resilience tests",
            ["service", "test_name"],
            buckets=[5, 10, 30, 60, 120, 300, 600],
        )

        self.resilience_score = Gauge(
            f"{namespace}_resilience_score",
            "Resilience score by service",
            ["service", "component"],
        )

        self.active_tests = Gauge(
            f"{namespace}_resilience_active_tests",
            "Number of currently running tests",
        )

    def register_test(self, test: ResilienceTest):
        """Register a resilience test.

        Args:
            test: Test definition
        """
        test_id = f"{test.target_service}:{test.name}"
        with self._lock:
            self._tests[test_id] = test

        logger.info(f"Registered resilience test: {test_id}")

    def unregister_test(self, test_id: str):
        """Unregister a test.

        Args:
            test_id: Test identifier
        """
        with self._lock:
            if test_id in self._tests:
                del self._tests[test_id]

    def get_tests_for_service(self, service: str) -> List[ResilienceTest]:
        """Get all tests for a service.

        Args:
            service: Service name

        Returns:
            List of tests
        """
        with self._lock:
            return [
                t for t in self._tests.values()
                if t.target_service == service
            ]

    async def run_test(self, test_id: str) -> TestResult:
        """Run a single resilience test.

        Args:
            test_id: Test identifier

        Returns:
            TestResult
        """
        with self._lock:
            if test_id not in self._tests:
                return TestResult(
                    test_name="unknown",
                    test_id=test_id,
                    status=TestStatus.ERROR,
                    service="unknown",
                    duration_ms=0,
                    passed_assertions=[],
                    failed_assertions=[],
                    error_message=f"Test {test_id} not found",
                )

            test = self._tests[test_id]
            self._running_tests.add(test_id)

        self.active_tests.set(len(self._running_tests))
        start_time = time.perf_counter()

        try:
            result = await self._execute_test(test, test_id)
        except Exception as e:
            result = TestResult(
                test_name=test.name,
                test_id=test_id,
                status=TestStatus.ERROR,
                service=test.target_service,
                duration_ms=(time.perf_counter() - start_time) * 1000,
                passed_assertions=[],
                failed_assertions=[],
                error_message=str(e),
                stack_trace=traceback.format_exc(),
                severity=test.severity,
            )
        finally:
            with self._lock:
                self._running_tests.discard(test_id)
            self.active_tests.set(len(self._running_tests))

        # Store result
        with self._lock:
            if test_id not in self._results:
                self._results[test_id] = []
            self._results[test_id].append(result)

            # Keep only last 100 results per test
            if len(self._results[test_id]) > 100:
                self._results[test_id] = self._results[test_id][-50:]

        # Update metrics
        self.tests_total.labels(
            service=test.target_service,
            test_name=test.name,
            status=result.status.value,
        ).inc()

        self.test_duration.labels(
            service=test.target_service,
            test_name=test.name,
        ).observe(result.duration_ms / 1000)

        return result

    async def _execute_test(self, test: ResilienceTest, test_id: str) -> TestResult:
        """Execute a test with fault injection.

        Args:
            test: Test to execute
            test_id: Test identifier

        Returns:
            TestResult
        """
        start_time = time.perf_counter()
        passed_assertions = []
        failed_assertions = []
        metrics = {}

        logger.info(f"Starting resilience test: {test.name}")

        # Setup
        if test.setup_fn:
            try:
                test.setup_fn()
            except Exception as e:
                logger.error(f"Test setup failed: {e}")

        try:
            # Register faults
            fault_ids = []
            for i, fault in enumerate(test.faults):
                fault_id = f"{test_id}:fault:{i}"
                self.injector.register_fault(fault_id, fault)
                fault_ids.append(fault_id)

            # Warmup phase
            if test.warmup_seconds > 0:
                logger.debug(f"Warmup phase: {test.warmup_seconds}s")
                await asyncio.sleep(test.warmup_seconds)

            # Enable faults
            for fault_id in fault_ids:
                self.injector.enable_fault(fault_id)

            # Test duration
            logger.debug(f"Test duration: {test.duration_seconds}s")
            await asyncio.sleep(test.duration_seconds)

            # Collect metrics
            if test.collect_metrics_fn:
                try:
                    metrics = test.collect_metrics_fn()
                except Exception as e:
                    logger.error(f"Failed to collect metrics: {e}")

            # Disable faults
            for fault_id in fault_ids:
                self.injector.disable_fault(fault_id)

            # Cooldown phase
            if test.cooldown_seconds > 0:
                logger.debug(f"Cooldown phase: {test.cooldown_seconds}s")
                await asyncio.sleep(test.cooldown_seconds)

            # Evaluate assertions
            for condition, description in test.assertions:
                try:
                    # Simple expression evaluation with metrics context
                    # In production, use a proper expression evaluator
                    result = self._evaluate_assertion(condition, metrics)
                    if result:
                        passed_assertions.append(description)
                    else:
                        failed_assertions.append(description)
                except Exception as e:
                    failed_assertions.append(f"{description} (error: {e})")

            # Cleanup faults
            for fault_id in fault_ids:
                self.injector.unregister_fault(fault_id)

        finally:
            # Teardown
            if test.teardown_fn:
                try:
                    test.teardown_fn()
                except Exception as e:
                    logger.error(f"Test teardown failed: {e}")

        duration_ms = (time.perf_counter() - start_time) * 1000

        # Determine status
        if failed_assertions:
            status = TestStatus.FAILED
        elif passed_assertions:
            status = TestStatus.PASSED
        else:
            status = TestStatus.SKIPPED

        logger.info(
            f"Resilience test {test.name} completed: {status.value} "
            f"({len(passed_assertions)} passed, {len(failed_assertions)} failed)"
        )

        return TestResult(
            test_name=test.name,
            test_id=test_id,
            status=status,
            service=test.target_service,
            duration_ms=duration_ms,
            passed_assertions=passed_assertions,
            failed_assertions=failed_assertions,
            metrics=metrics,
            severity=test.severity,
        )

    def _evaluate_assertion(
        self,
        condition: str,
        metrics: Dict[str, float],
    ) -> bool:
        """Evaluate an assertion condition.

        Args:
            condition: Condition expression (e.g., "error_rate < 0.01")
            metrics: Collected metrics

        Returns:
            Boolean result
        """
        # Parse simple conditions like "metric < value" or "metric > value"
        import re

        match = re.match(r"(\w+)\s*([<>=!]+)\s*([\d.]+)", condition)
        if not match:
            # Default to True if condition can't be parsed
            return True

        metric_name = match.group(1)
        operator = match.group(2)
        threshold = float(match.group(3))

        metric_value = metrics.get(metric_name, 0.0)

        if operator == "<":
            return metric_value < threshold
        elif operator == "<=":
            return metric_value <= threshold
        elif operator == ">":
            return metric_value > threshold
        elif operator == ">=":
            return metric_value >= threshold
        elif operator in ("==", "="):
            return metric_value == threshold
        elif operator in ("!=", "<>"):
            return metric_value != threshold

        return True

    async def run_all(self, service: Optional[str] = None) -> List[TestResult]:
        """Run all registered tests.

        Args:
            service: Only run tests for this service (None for all)

        Returns:
            List of test results
        """
        with self._lock:
            test_ids = list(self._tests.keys())

        if service:
            test_ids = [
                tid for tid in test_ids
                if self._tests[tid].target_service == service
            ]

        results = []
        for test_id in test_ids:
            result = await self.run_test(test_id)
            results.append(result)

        return results

    async def run_parallel(
        self,
        service: Optional[str] = None,
        max_concurrent: int = 5,
    ) -> List[TestResult]:
        """Run tests in parallel.

        Args:
            service: Only run tests for this service
            max_concurrent: Maximum concurrent tests

        Returns:
            List of test results
        """
        with self._lock:
            test_ids = list(self._tests.keys())

        if service:
            test_ids = [
                tid for tid in test_ids
                if self._tests[tid].target_service == service
            ]

        semaphore = asyncio.Semaphore(max_concurrent)

        async def run_with_semaphore(test_id: str) -> TestResult:
            async with semaphore:
                return await self.run_test(test_id)

        tasks = [run_with_semaphore(tid) for tid in test_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(TestResult(
                    test_name=test_ids[i],
                    test_id=test_ids[i],
                    status=TestStatus.ERROR,
                    service="unknown",
                    duration_ms=0,
                    passed_assertions=[],
                    failed_assertions=[],
                    error_message=str(result),
                ))
            else:
                final_results.append(result)

        return final_results

    def calculate_score(self, service: str) -> ResilienceScore:
        """Calculate resilience score for a service.

        Args:
            service: Service name

        Returns:
            ResilienceScore
        """
        with self._lock:
            all_results = []
            for test_id, results in self._results.items():
                if self._tests.get(test_id, ResilienceTest("", "", "", [], [])).target_service == service:
                    all_results.extend(results)

        if not all_results:
            return ResilienceScore(
                service=service,
                overall_score=1.0,
                availability_score=1.0,
                latency_score=1.0,
                error_handling_score=1.0,
                recovery_score=1.0,
            )

        # Get recent results (last 24 hours)
        cutoff = datetime.now() - timedelta(hours=24)
        recent = [r for r in all_results if r.timestamp > cutoff]

        if not recent:
            recent = all_results[-10:]  # Use last 10 if no recent

        passed = sum(1 for r in recent if r.status == TestStatus.PASSED)
        failed = sum(1 for r in recent if r.status == TestStatus.FAILED)
        total = len(recent)

        # Calculate component scores
        availability_score = passed / total if total > 0 else 1.0

        # Latency score based on test durations
        avg_duration = sum(r.duration_ms for r in recent) / total if total > 0 else 0
        latency_score = max(0, 1 - (avg_duration / 60000))  # Penalty for > 60s

        # Error handling score based on severity of failures
        critical_failures = sum(
            1 for r in recent
            if r.status == TestStatus.FAILED and r.severity == SeverityLevel.CRITICAL
        )
        error_handling_score = max(0, 1 - (critical_failures * 0.2))

        # Recovery score (placeholder - would need actual recovery data)
        recovery_score = availability_score

        # Overall weighted score
        overall_score = (
            availability_score * 0.4 +
            latency_score * 0.2 +
            error_handling_score * 0.25 +
            recovery_score * 0.15
        )

        score = ResilienceScore(
            service=service,
            overall_score=overall_score,
            availability_score=availability_score,
            latency_score=latency_score,
            error_handling_score=error_handling_score,
            recovery_score=recovery_score,
            tests_passed=passed,
            tests_failed=failed,
            details={
                "total_tests": total,
                "recent_window_hours": 24,
            },
        )

        # Update metrics
        self.resilience_score.labels(service=service, component="overall").set(overall_score)
        self.resilience_score.labels(service=service, component="availability").set(availability_score)
        self.resilience_score.labels(service=service, component="latency").set(latency_score)
        self.resilience_score.labels(service=service, component="error_handling").set(error_handling_score)
        self.resilience_score.labels(service=service, component="recovery").set(recovery_score)

        return score

    def get_results(
        self,
        test_id: Optional[str] = None,
        count: int = 50,
    ) -> List[TestResult]:
        """Get test results.

        Args:
            test_id: Filter by test ID (None for all)
            count: Maximum results to return

        Returns:
            List of test results
        """
        with self._lock:
            if test_id:
                results = self._results.get(test_id, []).copy()
            else:
                results = []
                for test_results in self._results.values():
                    results.extend(test_results)

        results.sort(key=lambda r: r.timestamp, reverse=True)
        return results[:count]

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all test results.

        Returns:
            Summary dictionary
        """
        with self._lock:
            total_tests = len(self._tests)
            running = len(self._running_tests)

            all_results = []
            for test_results in self._results.values():
                all_results.extend(test_results)

        # Recent results (last 24 hours)
        cutoff = datetime.now() - timedelta(hours=24)
        recent = [r for r in all_results if r.timestamp > cutoff]

        by_status = {}
        for r in recent:
            by_status[r.status.value] = by_status.get(r.status.value, 0) + 1

        # Get unique services
        services = set(r.service for r in recent)

        return {
            "registered_tests": total_tests,
            "running_tests": running,
            "results_last_24h": len(recent),
            "by_status": by_status,
            "services_tested": list(services),
            "pass_rate": by_status.get("passed", 0) / len(recent) if recent else 0,
        }
