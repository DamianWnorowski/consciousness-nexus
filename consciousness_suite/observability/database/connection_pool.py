"""Connection Pool Monitor

Monitor database connection pool metrics and health.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import threading
import logging

from prometheus_client import Gauge, Counter, Histogram, Summary

logger = logging.getLogger(__name__)


class PoolState(str, Enum):
    """Connection pool states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    SATURATED = "saturated"
    EXHAUSTED = "exhausted"


class ConnectionState(str, Enum):
    """Individual connection states."""
    IDLE = "idle"
    ACTIVE = "active"
    WAITING = "waiting"
    CLOSED = "closed"


@dataclass
class ConnectionStats:
    """Statistics for a single connection."""
    connection_id: str
    state: ConnectionState
    created_at: datetime
    last_used: datetime
    queries_executed: int = 0
    total_time_ms: float = 0
    current_query: Optional[str] = None
    client_info: Optional[str] = None
    backend_pid: Optional[int] = None

    @property
    def idle_time_seconds(self) -> float:
        """Time since last use."""
        if self.state == ConnectionState.ACTIVE:
            return 0
        return (datetime.now() - self.last_used).total_seconds()

    @property
    def age_seconds(self) -> float:
        """Connection age."""
        return (datetime.now() - self.created_at).total_seconds()


@dataclass
class PoolMetrics:
    """Connection pool metrics snapshot."""
    timestamp: datetime = field(default_factory=datetime.now)
    pool_name: str = "default"

    # Connection counts
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    waiting_requests: int = 0

    # Pool configuration
    min_connections: int = 0
    max_connections: int = 0

    # Performance
    avg_wait_time_ms: float = 0
    avg_query_time_ms: float = 0
    queries_per_second: float = 0

    # Health
    connection_errors: int = 0
    timeouts: int = 0
    pool_state: PoolState = PoolState.HEALTHY

    @property
    def utilization(self) -> float:
        """Pool utilization percentage."""
        if self.max_connections == 0:
            return 0
        return self.active_connections / self.max_connections * 100

    @property
    def available_connections(self) -> int:
        """Available connections."""
        return self.max_connections - self.active_connections


@dataclass
class PoolHealth:
    """Pool health assessment."""
    pool_name: str
    state: PoolState
    score: float  # 0-100
    utilization: float
    wait_queue_depth: int
    connection_churn: float
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class ConnectionPoolMonitor:
    """Monitors database connection pools.

    Usage:
        monitor = ConnectionPoolMonitor()

        # Register a pool
        monitor.register_pool(
            pool_name="primary",
            min_connections=5,
            max_connections=20,
        )

        # Update metrics
        monitor.update_pool_metrics(
            pool_name="primary",
            active=10,
            idle=5,
            waiting=0,
        )

        # Record connection events
        monitor.record_connection_acquired("primary", wait_time_ms=5)
        monitor.record_connection_released("primary", query_time_ms=100)

        # Get health assessment
        health = monitor.get_pool_health("primary")
    """

    def __init__(self, namespace: str = "consciousness"):
        self.namespace = namespace

        self._pools: Dict[str, Dict[str, Any]] = {}
        self._connections: Dict[str, Dict[str, ConnectionStats]] = {}
        self._metrics_history: Dict[str, List[PoolMetrics]] = {}
        self._lock = threading.Lock()

        self._health_callbacks: List[Callable[[PoolHealth], None]] = []

        # Prometheus metrics
        self.pool_connections = Gauge(
            f"{namespace}_db_pool_connections",
            "Connection pool connections",
            ["pool", "state"],
        )

        self.pool_utilization = Gauge(
            f"{namespace}_db_pool_utilization_percent",
            "Connection pool utilization percentage",
            ["pool"],
        )

        self.pool_waiting = Gauge(
            f"{namespace}_db_pool_waiting_requests",
            "Requests waiting for connection",
            ["pool"],
        )

        self.connection_acquired = Counter(
            f"{namespace}_db_connections_acquired_total",
            "Total connections acquired",
            ["pool"],
        )

        self.connection_released = Counter(
            f"{namespace}_db_connections_released_total",
            "Total connections released",
            ["pool"],
        )

        self.connection_errors = Counter(
            f"{namespace}_db_connection_errors_total",
            "Connection errors",
            ["pool", "error_type"],
        )

        self.connection_wait_time = Histogram(
            f"{namespace}_db_connection_wait_seconds",
            "Time waiting for connection",
            ["pool"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
        )

        self.query_duration = Histogram(
            f"{namespace}_db_pool_query_duration_seconds",
            "Query duration per pool",
            ["pool"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        )

        self.pool_health_score = Gauge(
            f"{namespace}_db_pool_health_score",
            "Pool health score (0-100)",
            ["pool"],
        )

    def register_pool(
        self,
        pool_name: str,
        min_connections: int,
        max_connections: int,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Register a connection pool.

        Args:
            pool_name: Pool identifier
            min_connections: Minimum connections
            max_connections: Maximum connections
            metadata: Additional metadata
        """
        with self._lock:
            self._pools[pool_name] = {
                "name": pool_name,
                "min_connections": min_connections,
                "max_connections": max_connections,
                "metadata": metadata or {},
                "created_at": datetime.now(),
            }
            self._connections[pool_name] = {}
            self._metrics_history[pool_name] = []

        logger.info(f"Registered pool: {pool_name} (min={min_connections}, max={max_connections})")

    def update_pool_metrics(
        self,
        pool_name: str,
        active: int,
        idle: int,
        waiting: int = 0,
        avg_wait_time_ms: float = 0,
        avg_query_time_ms: float = 0,
        queries_per_second: float = 0,
    ):
        """Update pool metrics.

        Args:
            pool_name: Pool identifier
            active: Active connections
            idle: Idle connections
            waiting: Waiting requests
            avg_wait_time_ms: Average wait time
            avg_query_time_ms: Average query time
            queries_per_second: Query rate
        """
        with self._lock:
            pool_config = self._pools.get(pool_name)
            if not pool_config:
                logger.warning(f"Unknown pool: {pool_name}")
                return

            total = active + idle
            state = self._determine_pool_state(active, idle, waiting, pool_config["max_connections"])

            metrics = PoolMetrics(
                pool_name=pool_name,
                total_connections=total,
                active_connections=active,
                idle_connections=idle,
                waiting_requests=waiting,
                min_connections=pool_config["min_connections"],
                max_connections=pool_config["max_connections"],
                avg_wait_time_ms=avg_wait_time_ms,
                avg_query_time_ms=avg_query_time_ms,
                queries_per_second=queries_per_second,
                pool_state=state,
            )

            self._metrics_history[pool_name].append(metrics)

            # Keep only recent history
            if len(self._metrics_history[pool_name]) > 1000:
                self._metrics_history[pool_name] = self._metrics_history[pool_name][-500:]

        # Update Prometheus metrics
        self.pool_connections.labels(pool=pool_name, state="active").set(active)
        self.pool_connections.labels(pool=pool_name, state="idle").set(idle)
        self.pool_connections.labels(pool=pool_name, state="total").set(total)
        self.pool_utilization.labels(pool=pool_name).set(metrics.utilization)
        self.pool_waiting.labels(pool=pool_name).set(waiting)

        # Check health and trigger callbacks
        if state != PoolState.HEALTHY:
            health = self.get_pool_health(pool_name)
            for callback in self._health_callbacks:
                try:
                    callback(health)
                except Exception as e:
                    logger.error(f"Health callback error: {e}")

    def _determine_pool_state(
        self,
        active: int,
        idle: int,
        waiting: int,
        max_connections: int,
    ) -> PoolState:
        """Determine pool state based on metrics."""
        if max_connections == 0:
            return PoolState.HEALTHY

        utilization = active / max_connections

        if waiting > 0:
            if idle == 0:
                return PoolState.EXHAUSTED
            return PoolState.SATURATED

        if utilization > 0.9:
            return PoolState.SATURATED
        elif utilization > 0.7:
            return PoolState.DEGRADED

        return PoolState.HEALTHY

    def record_connection_acquired(
        self,
        pool_name: str,
        wait_time_ms: float,
        connection_id: Optional[str] = None,
    ):
        """Record connection acquisition.

        Args:
            pool_name: Pool identifier
            wait_time_ms: Time waited for connection
            connection_id: Optional connection ID
        """
        self.connection_acquired.labels(pool=pool_name).inc()
        self.connection_wait_time.labels(pool=pool_name).observe(wait_time_ms / 1000)

        if connection_id:
            with self._lock:
                if pool_name in self._connections:
                    if connection_id not in self._connections[pool_name]:
                        self._connections[pool_name][connection_id] = ConnectionStats(
                            connection_id=connection_id,
                            state=ConnectionState.ACTIVE,
                            created_at=datetime.now(),
                            last_used=datetime.now(),
                        )
                    else:
                        self._connections[pool_name][connection_id].state = ConnectionState.ACTIVE
                        self._connections[pool_name][connection_id].last_used = datetime.now()

    def record_connection_released(
        self,
        pool_name: str,
        query_time_ms: float,
        connection_id: Optional[str] = None,
    ):
        """Record connection release.

        Args:
            pool_name: Pool identifier
            query_time_ms: Time spent on query
            connection_id: Optional connection ID
        """
        self.connection_released.labels(pool=pool_name).inc()
        self.query_duration.labels(pool=pool_name).observe(query_time_ms / 1000)

        if connection_id:
            with self._lock:
                if pool_name in self._connections:
                    conn = self._connections[pool_name].get(connection_id)
                    if conn:
                        conn.state = ConnectionState.IDLE
                        conn.last_used = datetime.now()
                        conn.queries_executed += 1
                        conn.total_time_ms += query_time_ms

    def record_connection_error(
        self,
        pool_name: str,
        error_type: str,
        error_message: Optional[str] = None,
    ):
        """Record connection error.

        Args:
            pool_name: Pool identifier
            error_type: Type of error
            error_message: Error details
        """
        self.connection_errors.labels(
            pool=pool_name,
            error_type=error_type,
        ).inc()

        logger.warning(f"Connection error in {pool_name}: {error_type} - {error_message}")

    def record_connection_created(
        self,
        pool_name: str,
        connection_id: str,
        client_info: Optional[str] = None,
        backend_pid: Optional[int] = None,
    ):
        """Record new connection creation.

        Args:
            pool_name: Pool identifier
            connection_id: Connection ID
            client_info: Client information
            backend_pid: Backend process ID
        """
        with self._lock:
            if pool_name in self._connections:
                self._connections[pool_name][connection_id] = ConnectionStats(
                    connection_id=connection_id,
                    state=ConnectionState.IDLE,
                    created_at=datetime.now(),
                    last_used=datetime.now(),
                    client_info=client_info,
                    backend_pid=backend_pid,
                )

    def record_connection_closed(
        self,
        pool_name: str,
        connection_id: str,
    ):
        """Record connection closure.

        Args:
            pool_name: Pool identifier
            connection_id: Connection ID
        """
        with self._lock:
            if pool_name in self._connections:
                conn = self._connections[pool_name].get(connection_id)
                if conn:
                    conn.state = ConnectionState.CLOSED
                    # Don't delete immediately - keep for analysis
                    logger.debug(f"Connection {connection_id} closed in {pool_name}")

    def on_health_change(self, callback: Callable[[PoolHealth], None]):
        """Register callback for health changes.

        Args:
            callback: Function to call with PoolHealth
        """
        self._health_callbacks.append(callback)

    def get_pool_health(self, pool_name: str) -> Optional[PoolHealth]:
        """Get health assessment for a pool.

        Args:
            pool_name: Pool identifier

        Returns:
            PoolHealth assessment or None
        """
        with self._lock:
            pool_config = self._pools.get(pool_name)
            history = self._metrics_history.get(pool_name, [])

        if not pool_config:
            return None

        if not history:
            return PoolHealth(
                pool_name=pool_name,
                state=PoolState.HEALTHY,
                score=100.0,
                utilization=0,
                wait_queue_depth=0,
                connection_churn=0,
            )

        # Analyze recent metrics
        recent = history[-10:] if len(history) >= 10 else history
        latest = recent[-1]

        # Calculate health score
        score = 100.0
        issues = []
        recommendations = []

        # Utilization penalty
        if latest.utilization > 90:
            score -= 30
            issues.append("Pool nearly exhausted (>90% utilization)")
            recommendations.append("Increase max_connections or optimize queries")
        elif latest.utilization > 75:
            score -= 15
            issues.append("High utilization (>75%)")

        # Wait queue penalty
        if latest.waiting_requests > 0:
            score -= min(20, latest.waiting_requests * 2)
            issues.append(f"Requests waiting for connections: {latest.waiting_requests}")
            recommendations.append("Consider connection pooler like PgBouncer")

        # Wait time penalty
        avg_wait = sum(m.avg_wait_time_ms for m in recent) / len(recent)
        if avg_wait > 100:
            score -= 20
            issues.append(f"High average wait time: {avg_wait:.1f}ms")
        elif avg_wait > 50:
            score -= 10
            issues.append(f"Elevated wait time: {avg_wait:.1f}ms")

        # Connection churn (connections created/closed rapidly)
        connection_churn = self._calculate_churn(pool_name)
        if connection_churn > 0.5:
            score -= 15
            issues.append(f"High connection churn: {connection_churn:.2f}/s")
            recommendations.append("Consider connection warming or keep-alive")

        # Determine state from score
        if score >= 80:
            state = PoolState.HEALTHY
        elif score >= 60:
            state = PoolState.DEGRADED
        elif score >= 40:
            state = PoolState.SATURATED
        else:
            state = PoolState.EXHAUSTED

        score = max(0, min(100, score))

        # Update metric
        self.pool_health_score.labels(pool=pool_name).set(score)

        return PoolHealth(
            pool_name=pool_name,
            state=state,
            score=score,
            utilization=latest.utilization,
            wait_queue_depth=latest.waiting_requests,
            connection_churn=connection_churn,
            issues=issues,
            recommendations=recommendations,
        )

    def _calculate_churn(self, pool_name: str) -> float:
        """Calculate connection churn rate."""
        with self._lock:
            connections = self._connections.get(pool_name, {})

        if not connections:
            return 0.0

        now = datetime.now()
        recent_window = timedelta(minutes=5)

        recent_created = sum(
            1 for c in connections.values()
            if now - c.created_at < recent_window
        )

        return recent_created / 300  # per second

    def get_pool_metrics(self, pool_name: str) -> Optional[PoolMetrics]:
        """Get latest pool metrics.

        Args:
            pool_name: Pool identifier

        Returns:
            Latest PoolMetrics or None
        """
        with self._lock:
            history = self._metrics_history.get(pool_name, [])

        return history[-1] if history else None

    def get_connections(self, pool_name: str) -> List[ConnectionStats]:
        """Get connection details for a pool.

        Args:
            pool_name: Pool identifier

        Returns:
            List of connection statistics
        """
        with self._lock:
            connections = self._connections.get(pool_name, {})
            return list(connections.values())

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all pools.

        Returns:
            Summary dictionary
        """
        summary = {
            "pools": {},
            "total_connections": 0,
            "total_active": 0,
            "unhealthy_pools": [],
        }

        for pool_name in self._pools:
            metrics = self.get_pool_metrics(pool_name)
            health = self.get_pool_health(pool_name)

            if metrics and health:
                summary["pools"][pool_name] = {
                    "connections": metrics.total_connections,
                    "active": metrics.active_connections,
                    "utilization": metrics.utilization,
                    "health_score": health.score,
                    "state": health.state.value,
                }

                summary["total_connections"] += metrics.total_connections
                summary["total_active"] += metrics.active_connections

                if health.state != PoolState.HEALTHY:
                    summary["unhealthy_pools"].append(pool_name)

        return summary
