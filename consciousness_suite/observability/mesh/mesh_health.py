"""Service Mesh Health Monitor

Comprehensive health monitoring for service mesh deployments.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set
from datetime import datetime, timedelta
from enum import Enum
import threading
import logging

from prometheus_client import Counter, Gauge, Histogram, Info

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Service health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ProbeType(str, Enum):
    """Types of health probes."""
    LIVENESS = "liveness"
    READINESS = "readiness"
    STARTUP = "startup"
    CUSTOM = "custom"


@dataclass
class ServiceHealth:
    """Health status of a service."""
    service: str
    status: HealthStatus
    healthy_instances: int
    total_instances: int
    avg_latency_ms: float
    error_rate: float
    last_check: datetime
    uptime: timedelta
    probes: Dict[ProbeType, bool] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthCheck:
    """A health check result."""
    service: str
    probe_type: ProbeType
    success: bool
    latency_ms: float
    timestamp: datetime = field(default_factory=datetime.now)
    message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MeshZone:
    """A mesh zone (region/datacenter)."""
    name: str
    services: Set[str] = field(default_factory=set)
    healthy_services: int = 0
    total_services: int = 0
    latency_to_other_zones: Dict[str, float] = field(default_factory=dict)


class MeshHealthMonitor:
    """Monitors overall health of the service mesh.

    Usage:
        monitor = MeshHealthMonitor()

        # Register services
        monitor.register_service("api-gateway", zone="us-east-1")
        monitor.register_service("user-service", zone="us-east-1")

        # Record health checks
        monitor.record_health_check(HealthCheck(
            service="api-gateway",
            probe_type=ProbeType.LIVENESS,
            success=True,
            latency_ms=5.0
        ))

        # Get mesh health
        health = monitor.get_mesh_health()
    """

    def __init__(
        self,
        namespace: str = "consciousness",
        check_interval_seconds: float = 30.0,
    ):
        self.namespace = namespace
        self.check_interval = check_interval_seconds

        self._services: Dict[str, Dict[str, Any]] = {}
        self._health_checks: Dict[str, List[HealthCheck]] = {}
        self._service_health: Dict[str, ServiceHealth] = {}
        self._zones: Dict[str, MeshZone] = {}
        self._lock = threading.Lock()
        self._max_checks = 1000

        # Callbacks for health changes
        self._on_health_change: List[Callable[[str, HealthStatus, HealthStatus], None]] = []

        # Prometheus metrics
        self.service_health_status = Gauge(
            f"{namespace}_mesh_service_health",
            "Service health status (0=unknown, 1=unhealthy, 2=degraded, 3=healthy)",
            ["service", "zone"],
        )

        self.service_instances = Gauge(
            f"{namespace}_mesh_service_instances",
            "Number of service instances",
            ["service", "zone", "state"],
        )

        self.probe_results = Counter(
            f"{namespace}_mesh_probe_results_total",
            "Health probe results",
            ["service", "probe_type", "result"],
        )

        self.probe_latency = Histogram(
            f"{namespace}_mesh_probe_latency_seconds",
            "Health probe latency",
            ["service", "probe_type"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
        )

        self.mesh_health_score = Gauge(
            f"{namespace}_mesh_overall_health_score",
            "Overall mesh health score (0-1)",
        )

        self.zone_health = Gauge(
            f"{namespace}_mesh_zone_health",
            "Zone health score (0-1)",
            ["zone"],
        )

        self.zone_latency = Gauge(
            f"{namespace}_mesh_zone_latency_seconds",
            "Latency between zones",
            ["source_zone", "target_zone"],
        )

    def register_service(
        self,
        service: str,
        zone: str = "default",
        instances: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Register a service with the mesh monitor.

        Args:
            service: Service name
            zone: Zone/region name
            instances: Number of instances
            metadata: Additional service metadata
        """
        with self._lock:
            self._services[service] = {
                "zone": zone,
                "instances": instances,
                "healthy_instances": instances,
                "registered_at": datetime.now(),
                "metadata": metadata or {},
            }

            # Initialize health state
            if service not in self._service_health:
                self._service_health[service] = ServiceHealth(
                    service=service,
                    status=HealthStatus.UNKNOWN,
                    healthy_instances=instances,
                    total_instances=instances,
                    avg_latency_ms=0,
                    error_rate=0,
                    last_check=datetime.now(),
                    uptime=timedelta(),
                )

            # Update zone
            if zone not in self._zones:
                self._zones[zone] = MeshZone(name=zone)
            self._zones[zone].services.add(service)
            self._zones[zone].total_services = len(self._zones[zone].services)

        # Update metrics
        self.service_instances.labels(
            service=service, zone=zone, state="healthy"
        ).set(instances)
        self.service_instances.labels(
            service=service, zone=zone, state="total"
        ).set(instances)

        logger.info(f"Registered service {service} in zone {zone} ({instances} instances)")

    def unregister_service(self, service: str):
        """Unregister a service from monitoring.

        Args:
            service: Service name to unregister
        """
        with self._lock:
            if service in self._services:
                zone = self._services[service]["zone"]
                del self._services[service]

                if service in self._service_health:
                    del self._service_health[service]

                if service in self._health_checks:
                    del self._health_checks[service]

                if zone in self._zones:
                    self._zones[zone].services.discard(service)
                    self._zones[zone].total_services = len(self._zones[zone].services)

        logger.info(f"Unregistered service {service}")

    def record_health_check(self, check: HealthCheck):
        """Record a health check result.

        Args:
            check: Health check result
        """
        service = check.service

        with self._lock:
            if service not in self._health_checks:
                self._health_checks[service] = []

            self._health_checks[service].append(check)

            # Trim old checks
            if len(self._health_checks[service]) > self._max_checks:
                self._health_checks[service] = self._health_checks[service][-self._max_checks // 2:]

            # Update service health
            self._update_service_health(service)

        # Update metrics
        self.probe_results.labels(
            service=service,
            probe_type=check.probe_type.value,
            result="success" if check.success else "failure",
        ).inc()

        self.probe_latency.labels(
            service=service,
            probe_type=check.probe_type.value,
        ).observe(check.latency_ms / 1000)

        if not check.success:
            logger.warning(
                f"Health check failed for {service} ({check.probe_type.value}): "
                f"{check.message}"
            )

    def _update_service_health(self, service: str):
        """Update calculated health for a service.

        Must be called with lock held.
        """
        if service not in self._health_checks:
            return

        checks = self._health_checks[service]
        if not checks:
            return

        # Get recent checks (last 5 minutes)
        cutoff = datetime.now() - timedelta(minutes=5)
        recent = [c for c in checks if c.timestamp > cutoff]

        if not recent:
            return

        # Calculate metrics
        success_count = sum(1 for c in recent if c.success)
        total = len(recent)
        success_rate = success_count / total
        error_rate = 1 - success_rate
        avg_latency = sum(c.latency_ms for c in recent) / total

        # Determine health status
        old_status = (
            self._service_health[service].status
            if service in self._service_health
            else HealthStatus.UNKNOWN
        )

        if success_rate >= 0.99:
            status = HealthStatus.HEALTHY
        elif success_rate >= 0.95:
            status = HealthStatus.DEGRADED
        elif success_rate > 0:
            status = HealthStatus.UNHEALTHY
        else:
            status = HealthStatus.UNHEALTHY

        # Get probe status
        probe_status = {}
        for probe_type in ProbeType:
            probe_checks = [c for c in recent if c.probe_type == probe_type]
            if probe_checks:
                probe_status[probe_type] = probe_checks[-1].success

        # Calculate uptime
        service_info = self._services.get(service, {})
        registered_at = service_info.get("registered_at", datetime.now())
        uptime = datetime.now() - registered_at

        # Get instance counts
        total_instances = service_info.get("instances", 1)
        healthy_instances = int(total_instances * success_rate)

        # Update health record
        self._service_health[service] = ServiceHealth(
            service=service,
            status=status,
            healthy_instances=healthy_instances,
            total_instances=total_instances,
            avg_latency_ms=avg_latency,
            error_rate=error_rate,
            last_check=recent[-1].timestamp,
            uptime=uptime,
            probes=probe_status,
        )

        # Update instances in services dict
        if service in self._services:
            self._services[service]["healthy_instances"] = healthy_instances

        # Trigger callbacks if status changed
        if status != old_status:
            for callback in self._on_health_change:
                try:
                    callback(service, old_status, status)
                except Exception as e:
                    logger.error(f"Health change callback error: {e}")

        # Update metrics
        zone = service_info.get("zone", "default")
        status_value = {
            HealthStatus.UNKNOWN: 0,
            HealthStatus.UNHEALTHY: 1,
            HealthStatus.DEGRADED: 2,
            HealthStatus.HEALTHY: 3,
        }.get(status, 0)

        self.service_health_status.labels(service=service, zone=zone).set(status_value)
        self.service_instances.labels(
            service=service, zone=zone, state="healthy"
        ).set(healthy_instances)

        # Update zone health
        self._update_zone_health(zone)

    def _update_zone_health(self, zone: str):
        """Update zone health based on service health.

        Must be called with lock held.
        """
        if zone not in self._zones:
            return

        mesh_zone = self._zones[zone]
        healthy = 0

        for service in mesh_zone.services:
            if service in self._service_health:
                if self._service_health[service].status == HealthStatus.HEALTHY:
                    healthy += 1

        mesh_zone.healthy_services = healthy

        # Calculate zone health score
        if mesh_zone.total_services > 0:
            score = healthy / mesh_zone.total_services
            self.zone_health.labels(zone=zone).set(score)

    def on_health_change(
        self,
        callback: Callable[[str, HealthStatus, HealthStatus], None],
    ):
        """Register a callback for health status changes.

        Args:
            callback: Function(service, old_status, new_status)
        """
        self._on_health_change.append(callback)

    def get_service_health(self, service: str) -> Optional[ServiceHealth]:
        """Get health status of a service.

        Args:
            service: Service name

        Returns:
            ServiceHealth or None if not found
        """
        with self._lock:
            return self._service_health.get(service)

    def get_all_service_health(self) -> Dict[str, ServiceHealth]:
        """Get health status of all services.

        Returns:
            Dictionary of service name to health
        """
        with self._lock:
            return self._service_health.copy()

    def get_unhealthy_services(self) -> List[ServiceHealth]:
        """Get all unhealthy or degraded services.

        Returns:
            List of unhealthy service health records
        """
        with self._lock:
            return [
                h for h in self._service_health.values()
                if h.status in (HealthStatus.UNHEALTHY, HealthStatus.DEGRADED)
            ]

    def get_zone_health(self, zone: str) -> Optional[MeshZone]:
        """Get health of a zone.

        Args:
            zone: Zone name

        Returns:
            MeshZone or None
        """
        with self._lock:
            return self._zones.get(zone)

    def get_all_zones(self) -> Dict[str, MeshZone]:
        """Get all zone health records.

        Returns:
            Dictionary of zone name to MeshZone
        """
        with self._lock:
            return self._zones.copy()

    def record_zone_latency(
        self,
        source_zone: str,
        target_zone: str,
        latency_ms: float,
    ):
        """Record latency between zones.

        Args:
            source_zone: Source zone name
            target_zone: Target zone name
            latency_ms: Measured latency in milliseconds
        """
        with self._lock:
            if source_zone in self._zones:
                self._zones[source_zone].latency_to_other_zones[target_zone] = latency_ms

        self.zone_latency.labels(
            source_zone=source_zone,
            target_zone=target_zone,
        ).set(latency_ms / 1000)

    def get_mesh_health(self) -> Dict[str, Any]:
        """Get overall mesh health summary.

        Returns:
            Dictionary with mesh health metrics
        """
        with self._lock:
            total_services = len(self._services)
            healthy_services = sum(
                1 for h in self._service_health.values()
                if h.status == HealthStatus.HEALTHY
            )
            degraded_services = sum(
                1 for h in self._service_health.values()
                if h.status == HealthStatus.DEGRADED
            )
            unhealthy_services = sum(
                1 for h in self._service_health.values()
                if h.status == HealthStatus.UNHEALTHY
            )

            # Calculate overall score
            if total_services > 0:
                score = (
                    healthy_services * 1.0 +
                    degraded_services * 0.5 +
                    unhealthy_services * 0.0
                ) / total_services
            else:
                score = 1.0

            # Get zone summary
            zones = {
                name: {
                    "healthy_services": z.healthy_services,
                    "total_services": z.total_services,
                    "health_score": (
                        z.healthy_services / z.total_services
                        if z.total_services > 0 else 1.0
                    ),
                }
                for name, z in self._zones.items()
            }

            # Get total instances
            total_instances = sum(s.get("instances", 0) for s in self._services.values())
            healthy_instances = sum(s.get("healthy_instances", 0) for s in self._services.values())

        # Update overall health metric
        self.mesh_health_score.set(score)

        return {
            "overall_score": score,
            "total_services": total_services,
            "healthy_services": healthy_services,
            "degraded_services": degraded_services,
            "unhealthy_services": unhealthy_services,
            "total_instances": total_instances,
            "healthy_instances": healthy_instances,
            "zones": zones,
            "timestamp": datetime.now().isoformat(),
        }

    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """Get service dependency graph based on health checks.

        Returns:
            Dictionary mapping service to its health check sources
        """
        # This would be populated by actual dependency data
        # For now, return based on registered services
        with self._lock:
            return {
                service: list(self._services.keys())
                for service in self._services
            }

    def get_recent_checks(
        self,
        service: Optional[str] = None,
        count: int = 50,
    ) -> List[HealthCheck]:
        """Get recent health checks.

        Args:
            service: Filter by service (None for all)
            count: Number of checks to return

        Returns:
            List of recent health checks
        """
        with self._lock:
            if service:
                checks = self._health_checks.get(service, [])
            else:
                checks = []
                for svc_checks in self._health_checks.values():
                    checks.extend(svc_checks)

        checks.sort(key=lambda c: c.timestamp, reverse=True)
        return checks[:count]


class HealthProber:
    """Performs health probes against services.

    Usage:
        prober = HealthProber(monitor)

        # Add probe target
        prober.add_target("api-gateway", "http://api:8080/health", ProbeType.LIVENESS)

        # Start probing
        await prober.start()
    """

    def __init__(
        self,
        monitor: MeshHealthMonitor,
        default_timeout_ms: float = 5000,
        default_interval_seconds: float = 30,
    ):
        self.monitor = monitor
        self.default_timeout = default_timeout_ms
        self.default_interval = default_interval_seconds
        self._targets: List[Dict[str, Any]] = []
        self._running = False
        self._task: Optional[asyncio.Task] = None

    def add_target(
        self,
        service: str,
        url: str,
        probe_type: ProbeType,
        timeout_ms: Optional[float] = None,
        interval_seconds: Optional[float] = None,
    ):
        """Add a probe target.

        Args:
            service: Service name
            url: Health endpoint URL
            probe_type: Type of probe
            timeout_ms: Probe timeout
            interval_seconds: Probe interval
        """
        self._targets.append({
            "service": service,
            "url": url,
            "probe_type": probe_type,
            "timeout_ms": timeout_ms or self.default_timeout,
            "interval_seconds": interval_seconds or self.default_interval,
            "last_probe": None,
        })

    async def start(self):
        """Start the probing loop."""
        self._running = True
        self._task = asyncio.create_task(self._probe_loop())

    async def stop(self):
        """Stop the probing loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _probe_loop(self):
        """Main probing loop."""
        try:
            import httpx
        except ImportError:
            logger.warning("httpx not installed, using mock probing")
            httpx = None

        while self._running:
            for target in self._targets:
                now = datetime.now()
                interval = timedelta(seconds=target["interval_seconds"])

                # Check if it's time to probe
                if target["last_probe"] and (now - target["last_probe"]) < interval:
                    continue

                target["last_probe"] = now

                # Perform probe
                start_time = time.perf_counter()
                success = False
                message = None
                details = {}

                if httpx:
                    try:
                        async with httpx.AsyncClient() as client:
                            response = await client.get(
                                target["url"],
                                timeout=target["timeout_ms"] / 1000,
                            )
                            success = response.status_code == 200
                            details["status_code"] = response.status_code
                            if not success:
                                message = f"Status {response.status_code}"
                    except httpx.TimeoutException:
                        message = "Timeout"
                    except Exception as e:
                        message = str(e)
                else:
                    # Mock probe for testing
                    import random
                    success = random.random() > 0.1
                    if not success:
                        message = "Mock failure"

                latency_ms = (time.perf_counter() - start_time) * 1000

                # Record the check
                self.monitor.record_health_check(HealthCheck(
                    service=target["service"],
                    probe_type=target["probe_type"],
                    success=success,
                    latency_ms=latency_ms,
                    message=message,
                    details=details,
                ))

            # Short sleep between probe cycles
            await asyncio.sleep(1)
