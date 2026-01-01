"""Synthetic Probes

Base probe definitions for synthetic monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import threading
import logging

from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)


class ProbeType(str, Enum):
    """Types of synthetic probes."""
    HTTP = "http"
    HTTPS = "https"
    TCP = "tcp"
    UDP = "udp"
    DNS = "dns"
    SSL = "ssl"
    GRPC = "grpc"
    WEBSOCKET = "websocket"
    TRANSACTION = "transaction"
    CUSTOM = "custom"


class ProbeStatus(str, Enum):
    """Probe execution status."""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    ERROR = "error"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class ProbeConfig:
    """Configuration for a synthetic probe."""
    probe_id: str
    name: str
    probe_type: ProbeType
    target: str  # URL, hostname, or address
    interval_seconds: int = 60
    timeout_seconds: float = 30.0
    retry_count: int = 1
    retry_delay_seconds: float = 5.0
    locations: List[str] = field(default_factory=list)  # Probe locations
    assertions: List[Dict[str, Any]] = field(default_factory=list)
    headers: Dict[str, str] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    alert_threshold: int = 2  # Failures before alerting
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProbeResult:
    """Result of a probe execution."""
    probe_id: str
    probe_name: str
    probe_type: ProbeType
    status: ProbeStatus
    target: str
    location: str = "default"
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    response_code: Optional[int] = None
    response_body: Optional[str] = None
    response_headers: Dict[str, str] = field(default_factory=dict)
    error_message: Optional[str] = None
    assertions_passed: int = 0
    assertions_failed: int = 0
    assertion_errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return self.status == ProbeStatus.SUCCESS

    @property
    def is_healthy(self) -> bool:
        return self.status in [ProbeStatus.SUCCESS, ProbeStatus.DEGRADED]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "probe_name": self.probe_name,
            "probe_type": self.probe_type.value,
            "status": self.status.value,
            "target": self.target,
            "location": self.location,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp.isoformat(),
            "response_code": self.response_code,
            "error_message": self.error_message,
            "assertions_passed": self.assertions_passed,
            "assertions_failed": self.assertions_failed,
            "assertion_errors": self.assertion_errors,
        }


class SyntheticProbe(ABC):
    """Base class for synthetic probes.

    Usage:
        class CustomProbe(SyntheticProbe):
            async def execute(self, config: ProbeConfig) -> ProbeResult:
                # Implementation
                pass

        probe = CustomProbe(namespace="consciousness")
        result = await probe.run(config)
    """

    def __init__(self, namespace: str = "consciousness"):
        self.namespace = namespace
        self._lock = threading.Lock()

        # Result history
        self._history: Dict[str, List[ProbeResult]] = {}
        self._max_history = 100

        # Callbacks
        self._callbacks: List[Callable[[ProbeResult], None]] = []

        # Metrics
        self.probe_executions = Counter(
            f"{namespace}_synthetic_probe_executions_total",
            "Total probe executions",
            ["probe_id", "probe_type", "status", "location"],
        )

        self.probe_latency = Histogram(
            f"{namespace}_synthetic_probe_latency_seconds",
            "Probe latency",
            ["probe_id", "probe_type", "location"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        )

        self.probe_success_rate = Gauge(
            f"{namespace}_synthetic_probe_success_rate",
            "Probe success rate",
            ["probe_id"],
        )

        self.active_probes = Gauge(
            f"{namespace}_synthetic_active_probes",
            "Active probes",
        )

    @abstractmethod
    async def execute(self, config: ProbeConfig) -> ProbeResult:
        """Execute the probe.

        Args:
            config: Probe configuration

        Returns:
            ProbeResult
        """
        pass

    async def run(
        self,
        config: ProbeConfig,
        location: str = "default",
    ) -> ProbeResult:
        """Run the probe with retries and result recording.

        Args:
            config: Probe configuration
            location: Probe location

        Returns:
            ProbeResult
        """
        import time

        attempts = 0
        last_result = None

        while attempts <= config.retry_count:
            attempts += 1
            start_time = time.time()

            try:
                result = await self.execute(config)
                result.location = location
                result.latency_ms = (time.time() - start_time) * 1000

                # Run assertions
                self._run_assertions(result, config.assertions)

                if result.is_success:
                    last_result = result
                    break

                last_result = result

            except Exception as e:
                logger.error(f"Probe {config.probe_id} error: {e}")
                last_result = ProbeResult(
                    probe_id=config.probe_id,
                    probe_name=config.name,
                    probe_type=config.probe_type,
                    status=ProbeStatus.ERROR,
                    target=config.target,
                    location=location,
                    latency_ms=(time.time() - start_time) * 1000,
                    error_message=str(e),
                )

            if attempts <= config.retry_count:
                import asyncio
                await asyncio.sleep(config.retry_delay_seconds)

        if last_result:
            self._record_result(last_result)

        return last_result

    def _run_assertions(
        self,
        result: ProbeResult,
        assertions: List[Dict[str, Any]],
    ):
        """Run assertions on probe result."""
        for assertion in assertions:
            try:
                if self._check_assertion(result, assertion):
                    result.assertions_passed += 1
                else:
                    result.assertions_failed += 1
                    result.assertion_errors.append(
                        f"Assertion failed: {assertion}"
                    )
                    if result.status == ProbeStatus.SUCCESS:
                        result.status = ProbeStatus.DEGRADED
            except Exception as e:
                result.assertions_failed += 1
                result.assertion_errors.append(f"Assertion error: {e}")

        if result.assertions_failed > 0 and result.assertions_passed == 0:
            result.status = ProbeStatus.FAILURE

    def _check_assertion(
        self,
        result: ProbeResult,
        assertion: Dict[str, Any],
    ) -> bool:
        """Check a single assertion."""
        assertion_type = assertion.get("type", "status_code")

        if assertion_type == "status_code":
            expected = assertion.get("expected", 200)
            return result.response_code == expected

        elif assertion_type == "status_code_range":
            min_code = assertion.get("min", 200)
            max_code = assertion.get("max", 299)
            return min_code <= (result.response_code or 0) <= max_code

        elif assertion_type == "latency":
            max_latency = assertion.get("max_ms", 1000)
            return result.latency_ms <= max_latency

        elif assertion_type == "body_contains":
            text = assertion.get("text", "")
            return text in (result.response_body or "")

        elif assertion_type == "body_not_contains":
            text = assertion.get("text", "")
            return text not in (result.response_body or "")

        elif assertion_type == "header_exists":
            header = assertion.get("header", "")
            return header.lower() in {k.lower() for k in result.response_headers}

        elif assertion_type == "header_value":
            header = assertion.get("header", "")
            expected = assertion.get("value", "")
            for k, v in result.response_headers.items():
                if k.lower() == header.lower():
                    return v == expected
            return False

        elif assertion_type == "json_path":
            import json
            path = assertion.get("path", "")
            expected = assertion.get("expected")
            try:
                data = json.loads(result.response_body or "{}")
                value = self._json_path_extract(data, path)
                return value == expected
            except Exception:
                return False

        return True

    def _json_path_extract(self, data: Any, path: str) -> Any:
        """Extract value from JSON using simple path."""
        parts = path.strip("$.").split(".")
        current = data
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, list):
                try:
                    current = current[int(part)]
                except (ValueError, IndexError):
                    return None
            else:
                return None
        return current

    def _record_result(self, result: ProbeResult):
        """Record probe result."""
        with self._lock:
            if result.probe_id not in self._history:
                self._history[result.probe_id] = []

            history = self._history[result.probe_id]
            history.append(result)

            # Trim history
            if len(history) > self._max_history:
                self._history[result.probe_id] = history[-self._max_history:]

        # Update metrics
        self.probe_executions.labels(
            probe_id=result.probe_id,
            probe_type=result.probe_type.value,
            status=result.status.value,
            location=result.location,
        ).inc()

        self.probe_latency.labels(
            probe_id=result.probe_id,
            probe_type=result.probe_type.value,
            location=result.location,
        ).observe(result.latency_ms / 1000)

        # Calculate success rate
        with self._lock:
            history = self._history.get(result.probe_id, [])
            if history:
                success_count = sum(1 for r in history if r.is_success)
                success_rate = success_count / len(history)
                self.probe_success_rate.labels(probe_id=result.probe_id).set(success_rate)

        # Trigger callbacks
        for callback in self._callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Probe callback error: {e}")

    def on_result(self, callback: Callable[[ProbeResult], None]):
        """Register result callback.

        Args:
            callback: Function to call with ProbeResult
        """
        self._callbacks.append(callback)

    def get_history(
        self,
        probe_id: str,
        limit: int = 50,
    ) -> List[ProbeResult]:
        """Get probe result history.

        Args:
            probe_id: Probe ID
            limit: Maximum results

        Returns:
            List of results
        """
        with self._lock:
            history = self._history.get(probe_id, [])
            return list(reversed(history[-limit:]))

    def get_statistics(self, probe_id: str) -> Dict[str, Any]:
        """Get probe statistics.

        Args:
            probe_id: Probe ID

        Returns:
            Statistics dictionary
        """
        with self._lock:
            history = self._history.get(probe_id, [])

        if not history:
            return {"probe_id": probe_id, "total_executions": 0}

        success_count = sum(1 for r in history if r.is_success)
        latencies = [r.latency_ms for r in history]

        return {
            "probe_id": probe_id,
            "total_executions": len(history),
            "success_count": success_count,
            "failure_count": len(history) - success_count,
            "success_rate": success_count / len(history),
            "avg_latency_ms": sum(latencies) / len(latencies),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "p50_latency_ms": sorted(latencies)[len(latencies) // 2],
            "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) >= 20 else max(latencies),
            "last_execution": history[-1].timestamp.isoformat(),
            "last_status": history[-1].status.value,
        }
