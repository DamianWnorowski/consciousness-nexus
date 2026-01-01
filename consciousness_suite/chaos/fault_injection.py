"""Programmatic Fault Injection

Injects controlled faults for chaos engineering experiments.
"""

from __future__ import annotations

import asyncio
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar
from datetime import datetime, timedelta
from enum import Enum
from contextlib import contextmanager, asynccontextmanager
import threading
import logging

from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)

T = TypeVar("T")


class FaultType(str, Enum):
    """Types of faults that can be injected."""
    LATENCY = "latency"
    ERROR = "error"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    NETWORK_PARTITION = "network_partition"
    NETWORK_LOSS = "network_loss"
    CPU_STRESS = "cpu_stress"
    MEMORY_PRESSURE = "memory_pressure"
    DISK_FULL = "disk_full"
    DNS_FAILURE = "dns_failure"
    CONNECTION_RESET = "connection_reset"
    TIMEOUT = "timeout"


class FaultScope(str, Enum):
    """Scope of fault application."""
    ALL = "all"              # Affect all requests
    PERCENTAGE = "percentage"  # Affect N% of requests
    TARGETED = "targeted"    # Affect specific requests (by header, etc.)
    TIME_WINDOW = "time_window"  # Affect requests in time window


@dataclass
class InjectedFault:
    """Base class for injected faults."""
    fault_type: FaultType
    target_service: str
    enabled: bool = True
    probability: float = 1.0  # 0.0 to 1.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def should_inject(self) -> bool:
        """Check if fault should be injected now."""
        if not self.enabled:
            return False

        now = datetime.now()
        if self.start_time and now < self.start_time:
            return False
        if self.end_time and now > self.end_time:
            return False

        return random.random() < self.probability


@dataclass
class LatencyFault(InjectedFault):
    """Latency injection fault."""
    min_delay_ms: float = 100
    max_delay_ms: float = 500
    fault_type: FaultType = field(default=FaultType.LATENCY, init=False)

    def get_delay(self) -> float:
        """Get random delay in seconds."""
        delay_ms = random.uniform(self.min_delay_ms, self.max_delay_ms)
        return delay_ms / 1000

    async def apply_async(self):
        """Apply latency asynchronously."""
        await asyncio.sleep(self.get_delay())

    def apply_sync(self):
        """Apply latency synchronously."""
        time.sleep(self.get_delay())


@dataclass
class ErrorFault(InjectedFault):
    """Error injection fault."""
    error_type: Type[Exception] = Exception
    error_message: str = "Injected fault error"
    http_status_code: int = 500
    fault_type: FaultType = field(default=FaultType.ERROR, init=False)

    def raise_error(self):
        """Raise the configured error."""
        raise self.error_type(self.error_message)


@dataclass
class ResourceFault(InjectedFault):
    """Resource exhaustion fault."""
    resource_type: str = "memory"  # memory, cpu, disk
    intensity: float = 0.8  # 0.0 to 1.0
    duration_seconds: float = 30
    fault_type: FaultType = field(default=FaultType.RESOURCE_EXHAUSTION, init=False)


@dataclass
class NetworkFault(InjectedFault):
    """Network-related fault."""
    packet_loss_percent: float = 0.0
    partition_targets: List[str] = field(default_factory=list)
    bandwidth_limit_kbps: Optional[int] = None
    fault_type: FaultType = field(default=FaultType.NETWORK_PARTITION, init=False)


class FaultInjector:
    """Central fault injection manager.

    Usage:
        injector = FaultInjector()

        # Register a latency fault
        fault = LatencyFault(
            target_service="payment-service",
            min_delay_ms=200,
            max_delay_ms=1000,
            probability=0.3
        )
        injector.register_fault("slow-payments", fault)

        # Use in code
        async with injector.maybe_inject_async("payment-service"):
            result = await payment_client.process()

        # Or decorator style
        @injector.chaos_wrapper("payment-service")
        async def process_payment():
            ...
    """

    def __init__(self, namespace: str = "consciousness", enabled: bool = True):
        self.namespace = namespace
        self.enabled = enabled
        self._faults: Dict[str, InjectedFault] = {}
        self._service_faults: Dict[str, List[str]] = {}
        self._lock = threading.Lock()
        self._injection_history: List[Dict[str, Any]] = []
        self._max_history = 1000

        # Prometheus metrics
        self.injections_total = Counter(
            f"{namespace}_chaos_injections_total",
            "Total fault injections",
            ["service", "fault_type", "fault_id"],
        )

        self.active_faults = Gauge(
            f"{namespace}_chaos_active_faults",
            "Number of active faults",
            ["service"],
        )

        self.injection_latency = Histogram(
            f"{namespace}_chaos_injection_latency_seconds",
            "Latency added by injections",
            ["service"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        )

        self.affected_requests = Counter(
            f"{namespace}_chaos_affected_requests_total",
            "Requests affected by chaos",
            ["service", "fault_type"],
        )

    def register_fault(self, fault_id: str, fault: InjectedFault):
        """Register a fault configuration.

        Args:
            fault_id: Unique identifier for this fault
            fault: Fault configuration
        """
        with self._lock:
            self._faults[fault_id] = fault

            # Index by service
            service = fault.target_service
            if service not in self._service_faults:
                self._service_faults[service] = []
            if fault_id not in self._service_faults[service]:
                self._service_faults[service].append(fault_id)

        # Update metrics
        self._update_active_faults_metric(service)

        logger.info(f"Registered fault {fault_id}: {fault.fault_type.value} for {service}")

    def unregister_fault(self, fault_id: str):
        """Unregister a fault.

        Args:
            fault_id: Fault identifier to remove
        """
        with self._lock:
            if fault_id in self._faults:
                fault = self._faults[fault_id]
                service = fault.target_service

                del self._faults[fault_id]

                if service in self._service_faults:
                    self._service_faults[service] = [
                        f for f in self._service_faults[service] if f != fault_id
                    ]

                self._update_active_faults_metric(service)
                logger.info(f"Unregistered fault {fault_id}")

    def enable_fault(self, fault_id: str):
        """Enable a registered fault."""
        with self._lock:
            if fault_id in self._faults:
                self._faults[fault_id].enabled = True
                self._update_active_faults_metric(self._faults[fault_id].target_service)

    def disable_fault(self, fault_id: str):
        """Disable a registered fault."""
        with self._lock:
            if fault_id in self._faults:
                self._faults[fault_id].enabled = False
                self._update_active_faults_metric(self._faults[fault_id].target_service)

    def _update_active_faults_metric(self, service: str):
        """Update the active faults metric for a service."""
        with self._lock:
            active = sum(
                1 for fid in self._service_faults.get(service, [])
                if self._faults.get(fid, InjectedFault(FaultType.ERROR, "")).enabled
            )
        self.active_faults.labels(service=service).set(active)

    def get_faults_for_service(self, service: str) -> List[InjectedFault]:
        """Get all faults targeting a service.

        Args:
            service: Service name

        Returns:
            List of applicable faults
        """
        with self._lock:
            fault_ids = self._service_faults.get(service, [])
            return [self._faults[fid] for fid in fault_ids if fid in self._faults]

    def _record_injection(
        self,
        fault_id: str,
        fault: InjectedFault,
        applied: bool,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Record an injection event."""
        record = {
            "fault_id": fault_id,
            "fault_type": fault.fault_type.value,
            "service": fault.target_service,
            "applied": applied,
            "timestamp": datetime.now().isoformat(),
            "details": details or {},
        }

        with self._lock:
            self._injection_history.append(record)
            if len(self._injection_history) > self._max_history:
                self._injection_history = self._injection_history[-self._max_history // 2:]

        if applied:
            self.injections_total.labels(
                service=fault.target_service,
                fault_type=fault.fault_type.value,
                fault_id=fault_id,
            ).inc()

            self.affected_requests.labels(
                service=fault.target_service,
                fault_type=fault.fault_type.value,
            ).inc()

    @contextmanager
    def maybe_inject_sync(self, service: str):
        """Context manager for synchronous fault injection.

        Args:
            service: Target service name
        """
        if not self.enabled:
            yield
            return

        faults = self.get_faults_for_service(service)

        for fault_id, fault in [(fid, self._faults.get(fid)) for fid in self._service_faults.get(service, [])]:
            if fault and fault.should_inject():
                if isinstance(fault, LatencyFault):
                    delay = fault.get_delay()
                    self._record_injection(fault_id, fault, True, {"delay_ms": delay * 1000})
                    self.injection_latency.labels(service=service).observe(delay)
                    fault.apply_sync()

                elif isinstance(fault, ErrorFault):
                    self._record_injection(fault_id, fault, True, {
                        "error": fault.error_message,
                        "status_code": fault.http_status_code,
                    })
                    fault.raise_error()

        yield

    @asynccontextmanager
    async def maybe_inject_async(self, service: str):
        """Async context manager for fault injection.

        Args:
            service: Target service name
        """
        if not self.enabled:
            yield
            return

        for fault_id in self._service_faults.get(service, []):
            fault = self._faults.get(fault_id)
            if fault and fault.should_inject():
                if isinstance(fault, LatencyFault):
                    delay = fault.get_delay()
                    self._record_injection(fault_id, fault, True, {"delay_ms": delay * 1000})
                    self.injection_latency.labels(service=service).observe(delay)
                    await fault.apply_async()

                elif isinstance(fault, ErrorFault):
                    self._record_injection(fault_id, fault, True, {
                        "error": fault.error_message,
                        "status_code": fault.http_status_code,
                    })
                    fault.raise_error()

        yield

    def chaos_wrapper(
        self,
        service: str,
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator for chaos injection.

        Args:
            service: Target service name

        Returns:
            Decorated function with chaos injection
        """
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            if asyncio.iscoroutinefunction(func):
                async def async_wrapper(*args, **kwargs):
                    async with self.maybe_inject_async(service):
                        return await func(*args, **kwargs)
                return async_wrapper
            else:
                def sync_wrapper(*args, **kwargs):
                    with self.maybe_inject_sync(service):
                        return func(*args, **kwargs)
                return sync_wrapper
        return decorator

    def get_injection_history(
        self,
        count: int = 100,
        service: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get recent injection history.

        Args:
            count: Number of records to return
            service: Filter by service

        Returns:
            List of injection records
        """
        with self._lock:
            history = self._injection_history.copy()

        if service:
            history = [h for h in history if h["service"] == service]

        return list(reversed(history[-count:]))

    def get_status(self) -> Dict[str, Any]:
        """Get current injector status.

        Returns:
            Status dictionary
        """
        with self._lock:
            total_faults = len(self._faults)
            enabled_faults = sum(1 for f in self._faults.values() if f.enabled)
            services = list(self._service_faults.keys())

        return {
            "enabled": self.enabled,
            "total_faults": total_faults,
            "enabled_faults": enabled_faults,
            "services_with_faults": services,
            "recent_injections": len(self._injection_history),
        }

    def clear_all(self):
        """Clear all registered faults."""
        with self._lock:
            self._faults.clear()
            self._service_faults.clear()

        logger.info("Cleared all faults")


# Predefined fault templates
def create_latency_spike(
    service: str,
    min_delay_ms: float = 500,
    max_delay_ms: float = 2000,
    probability: float = 0.3,
    duration_minutes: float = 5,
) -> LatencyFault:
    """Create a latency spike fault.

    Args:
        service: Target service
        min_delay_ms: Minimum latency to add
        max_delay_ms: Maximum latency to add
        probability: Probability of injection (0-1)
        duration_minutes: How long the fault should be active

    Returns:
        Configured LatencyFault
    """
    now = datetime.now()
    return LatencyFault(
        target_service=service,
        min_delay_ms=min_delay_ms,
        max_delay_ms=max_delay_ms,
        probability=probability,
        start_time=now,
        end_time=now + timedelta(minutes=duration_minutes),
    )


def create_error_burst(
    service: str,
    error_rate: float = 0.5,
    status_code: int = 500,
    duration_minutes: float = 5,
) -> ErrorFault:
    """Create an error burst fault.

    Args:
        service: Target service
        error_rate: Rate of errors (0-1)
        status_code: HTTP status code to return
        duration_minutes: How long the fault should be active

    Returns:
        Configured ErrorFault
    """
    now = datetime.now()
    return ErrorFault(
        target_service=service,
        probability=error_rate,
        http_status_code=status_code,
        error_message=f"Chaos injection: Service {service} unavailable",
        start_time=now,
        end_time=now + timedelta(minutes=duration_minutes),
    )


def create_total_failure(service: str, duration_minutes: float = 5) -> ErrorFault:
    """Create a total service failure fault.

    Args:
        service: Target service
        duration_minutes: How long the fault should be active

    Returns:
        Configured ErrorFault with 100% failure rate
    """
    return create_error_burst(
        service,
        error_rate=1.0,
        status_code=503,
        duration_minutes=duration_minutes,
    )
