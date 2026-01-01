"""Routing Tracer for Service Mesh

Traces and visualizes request routing decisions through the service mesh.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from datetime import datetime, timedelta
from enum import Enum
import threading
import logging
import hashlib

from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)


class RouteType(str, Enum):
    """Types of routing decisions."""
    DIRECT = "direct"           # Direct service-to-service
    LOAD_BALANCED = "load_balanced"  # Load balancer selection
    CANARY = "canary"           # Canary deployment
    BLUE_GREEN = "blue_green"   # Blue-green deployment
    SHADOW = "shadow"           # Traffic shadowing
    RETRY = "retry"             # Retry to different instance
    FAILOVER = "failover"       # Failover to backup
    RATE_LIMITED = "rate_limited"  # Rate limit redirect


class RoutingStrategy(str, Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    RANDOM = "random"
    WEIGHTED = "weighted"
    CONSISTENT_HASH = "consistent_hash"
    ZONE_AFFINITY = "zone_affinity"


@dataclass
class RouteDecision:
    """A single routing decision."""
    request_id: str
    source_service: str
    target_service: str
    route_type: RouteType
    strategy: RoutingStrategy
    selected_endpoint: str
    available_endpoints: List[str]
    decision_time_ms: float
    timestamp: datetime = field(default_factory=datetime.now)
    version: Optional[str] = None
    zone: Optional[str] = None
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RouteTrace:
    """Complete trace of a request through the mesh."""
    trace_id: str
    request_id: str
    source_service: str
    final_service: str
    hops: List[RouteDecision]
    total_time_ms: float
    success: bool
    timestamp: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None


@dataclass
class ServiceEndpoint:
    """A service endpoint in the mesh."""
    address: str
    port: int
    service: str
    version: str = "v1"
    zone: str = "default"
    weight: float = 1.0
    healthy: bool = True
    active_connections: int = 0
    last_response_time_ms: float = 0.0


class RoutingTracer:
    """Traces and records routing decisions in a service mesh.

    Usage:
        tracer = RoutingTracer()

        # Record a routing decision
        decision = tracer.record_decision(
            request_id="req-123",
            source_service="api-gateway",
            target_service="user-service",
            route_type=RouteType.LOAD_BALANCED,
            strategy=RoutingStrategy.LEAST_CONNECTIONS,
            selected_endpoint="10.0.1.5:8080",
            available_endpoints=["10.0.1.5:8080", "10.0.1.6:8080"],
            decision_time_ms=0.5
        )

        # Start a trace
        trace_id = tracer.start_trace("req-123", "api-gateway")
        tracer.add_hop(trace_id, decision)
        tracer.complete_trace(trace_id, success=True)
    """

    def __init__(self, namespace: str = "consciousness"):
        self.namespace = namespace
        self._decisions: List[RouteDecision] = []
        self._traces: Dict[str, RouteTrace] = {}
        self._active_traces: Dict[str, List[RouteDecision]] = {}
        self._endpoints: Dict[str, List[ServiceEndpoint]] = {}
        self._lock = threading.Lock()
        self._max_decisions = 10000
        self._max_traces = 1000

        # Prometheus metrics
        self.routing_decisions = Counter(
            f"{namespace}_mesh_routing_decisions_total",
            "Total routing decisions made",
            ["source", "target", "route_type", "strategy"],
        )

        self.routing_latency = Histogram(
            f"{namespace}_mesh_routing_decision_latency_seconds",
            "Time to make routing decision",
            ["source", "target"],
            buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1],
        )

        self.endpoint_count = Gauge(
            f"{namespace}_mesh_endpoints_available",
            "Number of available endpoints",
            ["service"],
        )

        self.trace_hops = Histogram(
            f"{namespace}_mesh_trace_hops",
            "Number of hops per trace",
            ["source", "final"],
            buckets=[1, 2, 3, 4, 5, 7, 10, 15, 20],
        )

        self.canary_traffic = Gauge(
            f"{namespace}_mesh_canary_traffic_percentage",
            "Percentage of traffic to canary",
            ["service", "version"],
        )

    def record_decision(
        self,
        request_id: str,
        source_service: str,
        target_service: str,
        route_type: RouteType,
        strategy: RoutingStrategy,
        selected_endpoint: str,
        available_endpoints: List[str],
        decision_time_ms: float,
        version: Optional[str] = None,
        zone: Optional[str] = None,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RouteDecision:
        """Record a routing decision.

        Args:
            request_id: Unique request identifier
            source_service: Source service name
            target_service: Target service name
            route_type: Type of routing decision
            strategy: Load balancing strategy used
            selected_endpoint: Chosen endpoint
            available_endpoints: All available endpoints
            decision_time_ms: Time taken for decision in ms
            version: Target version (for canary/blue-green)
            zone: Target zone
            weight: Traffic weight
            metadata: Additional context

        Returns:
            RouteDecision object
        """
        decision = RouteDecision(
            request_id=request_id,
            source_service=source_service,
            target_service=target_service,
            route_type=route_type,
            strategy=strategy,
            selected_endpoint=selected_endpoint,
            available_endpoints=available_endpoints,
            decision_time_ms=decision_time_ms,
            version=version,
            zone=zone,
            weight=weight,
            metadata=metadata or {},
        )

        with self._lock:
            self._decisions.append(decision)
            if len(self._decisions) > self._max_decisions:
                self._decisions = self._decisions[-self._max_decisions // 2:]

        # Update metrics
        self.routing_decisions.labels(
            source=source_service,
            target=target_service,
            route_type=route_type.value,
            strategy=strategy.value,
        ).inc()

        self.routing_latency.labels(
            source=source_service,
            target=target_service,
        ).observe(decision_time_ms / 1000)

        logger.debug(
            f"Route decision: {source_service} -> {target_service} "
            f"via {selected_endpoint} ({route_type.value})"
        )

        return decision

    def start_trace(
        self,
        request_id: str,
        source_service: str,
    ) -> str:
        """Start a new routing trace.

        Args:
            request_id: Request identifier
            source_service: Originating service

        Returns:
            Trace ID
        """
        trace_id = hashlib.sha256(
            f"{request_id}:{time.time_ns()}".encode()
        ).hexdigest()[:16]

        with self._lock:
            self._active_traces[trace_id] = []

        logger.debug(f"Started trace {trace_id} for request {request_id}")
        return trace_id

    def add_hop(self, trace_id: str, decision: RouteDecision):
        """Add a routing hop to an active trace.

        Args:
            trace_id: Trace identifier
            decision: Routing decision for this hop
        """
        with self._lock:
            if trace_id in self._active_traces:
                self._active_traces[trace_id].append(decision)

    def complete_trace(
        self,
        trace_id: str,
        success: bool,
        error: Optional[str] = None,
    ) -> Optional[RouteTrace]:
        """Complete a routing trace.

        Args:
            trace_id: Trace identifier
            success: Whether the request succeeded
            error: Error message if failed

        Returns:
            Completed RouteTrace or None
        """
        with self._lock:
            if trace_id not in self._active_traces:
                return None

            hops = self._active_traces.pop(trace_id)

            if not hops:
                return None

            # Calculate total time
            total_time = sum(h.decision_time_ms for h in hops)

            trace = RouteTrace(
                trace_id=trace_id,
                request_id=hops[0].request_id,
                source_service=hops[0].source_service,
                final_service=hops[-1].target_service,
                hops=hops,
                total_time_ms=total_time,
                success=success,
                error=error,
            )

            self._traces[trace_id] = trace
            if len(self._traces) > self._max_traces:
                # Remove oldest traces
                oldest = list(self._traces.keys())[:len(self._traces) - self._max_traces // 2]
                for key in oldest:
                    del self._traces[key]

        # Update metrics
        self.trace_hops.labels(
            source=trace.source_service,
            final=trace.final_service,
        ).observe(len(hops))

        logger.debug(
            f"Completed trace {trace_id}: {trace.source_service} -> "
            f"{trace.final_service} ({len(hops)} hops, {success=})"
        )

        return trace

    def register_endpoint(
        self,
        service: str,
        endpoint: ServiceEndpoint,
    ):
        """Register a service endpoint.

        Args:
            service: Service name
            endpoint: Endpoint details
        """
        with self._lock:
            if service not in self._endpoints:
                self._endpoints[service] = []

            # Update or add endpoint
            existing = next(
                (e for e in self._endpoints[service]
                 if e.address == endpoint.address and e.port == endpoint.port),
                None
            )
            if existing:
                self._endpoints[service].remove(existing)
            self._endpoints[service].append(endpoint)

        # Update metric
        healthy_count = sum(
            1 for e in self._endpoints.get(service, []) if e.healthy
        )
        self.endpoint_count.labels(service=service).set(healthy_count)

    def unregister_endpoint(
        self,
        service: str,
        address: str,
        port: int,
    ):
        """Unregister a service endpoint.

        Args:
            service: Service name
            address: Endpoint address
            port: Endpoint port
        """
        with self._lock:
            if service in self._endpoints:
                self._endpoints[service] = [
                    e for e in self._endpoints[service]
                    if not (e.address == address and e.port == port)
                ]

        # Update metric
        healthy_count = sum(
            1 for e in self._endpoints.get(service, []) if e.healthy
        )
        self.endpoint_count.labels(service=service).set(healthy_count)

    def get_endpoints(
        self,
        service: str,
        healthy_only: bool = True,
    ) -> List[ServiceEndpoint]:
        """Get endpoints for a service.

        Args:
            service: Service name
            healthy_only: Only return healthy endpoints

        Returns:
            List of endpoints
        """
        with self._lock:
            endpoints = self._endpoints.get(service, [])
            if healthy_only:
                return [e for e in endpoints if e.healthy]
            return endpoints.copy()

    def select_endpoint(
        self,
        service: str,
        strategy: RoutingStrategy,
        hash_key: Optional[str] = None,
        zone_preference: Optional[str] = None,
    ) -> Optional[ServiceEndpoint]:
        """Select an endpoint using the specified strategy.

        Args:
            service: Target service
            strategy: Load balancing strategy
            hash_key: Key for consistent hashing
            zone_preference: Preferred zone

        Returns:
            Selected endpoint or None
        """
        endpoints = self.get_endpoints(service, healthy_only=True)

        if not endpoints:
            return None

        if strategy == RoutingStrategy.ROUND_ROBIN:
            # Simple round-robin (would need state for true round-robin)
            import random
            return random.choice(endpoints)

        elif strategy == RoutingStrategy.LEAST_CONNECTIONS:
            return min(endpoints, key=lambda e: e.active_connections)

        elif strategy == RoutingStrategy.RANDOM:
            import random
            return random.choice(endpoints)

        elif strategy == RoutingStrategy.WEIGHTED:
            import random
            total_weight = sum(e.weight for e in endpoints)
            r = random.uniform(0, total_weight)
            cumulative = 0
            for endpoint in endpoints:
                cumulative += endpoint.weight
                if r <= cumulative:
                    return endpoint
            return endpoints[-1]

        elif strategy == RoutingStrategy.CONSISTENT_HASH:
            if not hash_key:
                import random
                return random.choice(endpoints)

            # Simple consistent hash
            h = int(hashlib.md5(hash_key.encode()).hexdigest(), 16)
            return endpoints[h % len(endpoints)]

        elif strategy == RoutingStrategy.ZONE_AFFINITY:
            if zone_preference:
                zone_endpoints = [e for e in endpoints if e.zone == zone_preference]
                if zone_endpoints:
                    import random
                    return random.choice(zone_endpoints)
            import random
            return random.choice(endpoints)

        return endpoints[0]

    def get_recent_decisions(
        self,
        count: int = 50,
        source: Optional[str] = None,
        target: Optional[str] = None,
    ) -> List[RouteDecision]:
        """Get recent routing decisions.

        Args:
            count: Number of decisions to return
            source: Filter by source service
            target: Filter by target service

        Returns:
            List of routing decisions
        """
        with self._lock:
            decisions = self._decisions.copy()

        if source:
            decisions = [d for d in decisions if d.source_service == source]
        if target:
            decisions = [d for d in decisions if d.target_service == target]

        return list(reversed(decisions[-count:]))

    def get_trace(self, trace_id: str) -> Optional[RouteTrace]:
        """Get a completed trace by ID."""
        with self._lock:
            return self._traces.get(trace_id)

    def get_recent_traces(
        self,
        count: int = 20,
        success_only: bool = False,
    ) -> List[RouteTrace]:
        """Get recent completed traces.

        Args:
            count: Number of traces to return
            success_only: Only return successful traces

        Returns:
            List of traces
        """
        with self._lock:
            traces = list(self._traces.values())

        if success_only:
            traces = [t for t in traces if t.success]

        # Sort by timestamp descending
        traces.sort(key=lambda t: t.timestamp, reverse=True)
        return traces[:count]

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics.

        Returns:
            Dictionary with routing stats
        """
        with self._lock:
            decisions = self._decisions.copy()
            traces = list(self._traces.values())

        if not decisions:
            return {
                "total_decisions": 0,
                "by_route_type": {},
                "by_strategy": {},
                "avg_decision_time_ms": 0,
            }

        by_route_type: Dict[str, int] = {}
        by_strategy: Dict[str, int] = {}

        for d in decisions:
            by_route_type[d.route_type.value] = by_route_type.get(
                d.route_type.value, 0
            ) + 1
            by_strategy[d.strategy.value] = by_strategy.get(
                d.strategy.value, 0
            ) + 1

        success_rate = (
            sum(1 for t in traces if t.success) / len(traces)
            if traces else 0
        )

        return {
            "total_decisions": len(decisions),
            "by_route_type": by_route_type,
            "by_strategy": by_strategy,
            "avg_decision_time_ms": sum(d.decision_time_ms for d in decisions) / len(decisions),
            "total_traces": len(traces),
            "trace_success_rate": success_rate,
            "avg_hops_per_trace": (
                sum(len(t.hops) for t in traces) / len(traces)
                if traces else 0
            ),
        }

    def get_service_topology(self) -> Dict[str, Set[str]]:
        """Build service topology from routing decisions.

        Returns:
            Dictionary mapping services to their downstream dependencies
        """
        topology: Dict[str, Set[str]] = {}

        with self._lock:
            for decision in self._decisions:
                if decision.source_service not in topology:
                    topology[decision.source_service] = set()
                topology[decision.source_service].add(decision.target_service)

        return topology

    def track_canary(
        self,
        service: str,
        stable_version: str,
        canary_version: str,
        canary_percentage: float,
    ):
        """Track canary deployment traffic split.

        Args:
            service: Service name
            stable_version: Stable version
            canary_version: Canary version
            canary_percentage: Percentage of traffic to canary
        """
        self.canary_traffic.labels(
            service=service,
            version=stable_version,
        ).set(100 - canary_percentage)

        self.canary_traffic.labels(
            service=service,
            version=canary_version,
        ).set(canary_percentage)
