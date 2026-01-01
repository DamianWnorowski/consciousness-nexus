"""Hubble API Client

Client for Cilium Hubble gRPC API providing eBPF-based network flow visibility.
"""

from __future__ import annotations

import time
import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, Union
from enum import Enum
from datetime import datetime, timedelta
import threading
import logging
import queue

from prometheus_client import Counter, Gauge, Histogram, Info

try:
    import grpc
    import grpc.aio
    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False
    grpc = None

logger = logging.getLogger(__name__)


@dataclass
class HubbleConfig:
    """Configuration for Hubble client.

    Attributes:
        address: Hubble relay address (host:port)
        tls_enabled: Enable TLS for gRPC connection
        tls_ca_cert: Path to CA certificate
        tls_client_cert: Path to client certificate
        tls_client_key: Path to client key
        timeout_seconds: Connection timeout
        max_flows_per_request: Maximum flows to return per request
        follow: Enable flow streaming (follow mode)
    """
    address: str = "localhost:4245"
    tls_enabled: bool = False
    tls_ca_cert: Optional[str] = None
    tls_client_cert: Optional[str] = None
    tls_client_key: Optional[str] = None
    timeout_seconds: float = 30.0
    max_flows_per_request: int = 1000
    follow: bool = False
    namespace: str = "consciousness"


@dataclass
class FlowFilter:
    """Filter for Hubble flows.

    Attributes:
        source_pod: Filter by source pod name regex
        destination_pod: Filter by destination pod name regex
        source_namespace: Filter by source namespace
        destination_namespace: Filter by destination namespace
        source_ip: Filter by source IP
        destination_ip: Filter by destination IP
        source_port: Filter by source port
        destination_port: Filter by destination port
        protocol: Filter by protocol (TCP, UDP, ICMP)
        verdict: Filter by verdict (FORWARDED, DROPPED, etc.)
        http_status_code: Filter by HTTP status code
        http_method: Filter by HTTP method
        dns_query: Filter by DNS query
        node_name: Filter by node name
    """
    source_pod: Optional[str] = None
    destination_pod: Optional[str] = None
    source_namespace: Optional[str] = None
    destination_namespace: Optional[str] = None
    source_ip: Optional[str] = None
    destination_ip: Optional[str] = None
    source_port: Optional[int] = None
    destination_port: Optional[int] = None
    protocol: Optional[str] = None
    verdict: Optional[str] = None
    http_status_code: Optional[int] = None
    http_method: Optional[str] = None
    dns_query: Optional[str] = None
    node_name: Optional[str] = None

    def to_hubble_filter(self) -> List[Dict[str, Any]]:
        """Convert to Hubble API filter format."""
        filters = []

        if self.source_pod:
            filters.append({"source_pod": [self.source_pod]})
        if self.destination_pod:
            filters.append({"destination_pod": [self.destination_pod]})
        if self.source_namespace:
            filters.append({"source_namespace": [self.source_namespace]})
        if self.destination_namespace:
            filters.append({"destination_namespace": [self.destination_namespace]})
        if self.source_ip:
            filters.append({"source_ip": [self.source_ip]})
        if self.destination_ip:
            filters.append({"destination_ip": [self.destination_ip]})
        if self.verdict:
            filters.append({"verdict": [self.verdict]})
        if self.protocol:
            filters.append({"protocol": [self.protocol]})
        if self.http_status_code:
            filters.append({"http_status_code": [str(self.http_status_code)]})
        if self.node_name:
            filters.append({"node_name": [self.node_name]})

        return filters


@dataclass
class HubbleFlow:
    """Represents a network flow from Hubble.

    Attributes:
        flow_id: Unique flow identifier
        time: Timestamp of the flow
        verdict: Flow verdict (FORWARDED, DROPPED, etc.)
        source_ip: Source IP address
        source_port: Source port
        source_pod: Source pod name
        source_namespace: Source namespace
        source_labels: Source pod labels
        destination_ip: Destination IP address
        destination_port: Destination port
        destination_pod: Destination pod name
        destination_namespace: Destination namespace
        destination_labels: Destination pod labels
        protocol: Network protocol
        l7_protocol: Layer 7 protocol
        is_reply: Whether this is a reply flow
        node_name: Node where flow was observed
        traffic_direction: Traffic direction
        drop_reason: Reason for drop if dropped
        policy_match: Matched network policy
        http_info: HTTP-specific information
        dns_info: DNS-specific information
        kafka_info: Kafka-specific information
    """
    flow_id: str
    time: datetime
    verdict: str
    source_ip: str
    source_port: int
    source_pod: Optional[str]
    source_namespace: Optional[str]
    source_labels: Dict[str, str] = field(default_factory=dict)
    destination_ip: str = ""
    destination_port: int = 0
    destination_pod: Optional[str] = None
    destination_namespace: Optional[str] = None
    destination_labels: Dict[str, str] = field(default_factory=dict)
    protocol: str = "TCP"
    l7_protocol: Optional[str] = None
    is_reply: bool = False
    node_name: Optional[str] = None
    traffic_direction: str = "INGRESS"
    drop_reason: Optional[str] = None
    policy_match: Optional[str] = None
    http_info: Optional[Dict[str, Any]] = None
    dns_info: Optional[Dict[str, Any]] = None
    kafka_info: Optional[Dict[str, Any]] = None
    raw_data: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_grpc(cls, flow: Any) -> HubbleFlow:
        """Create HubbleFlow from gRPC response."""
        # Parse flow data from gRPC message
        source = getattr(flow, "source", None) or {}
        destination = getattr(flow, "destination", None) or {}
        l4 = getattr(flow, "l4", None)
        l7 = getattr(flow, "l7", None)

        source_port = 0
        dest_port = 0
        protocol = "UNKNOWN"

        if l4:
            if hasattr(l4, "TCP"):
                tcp = l4.TCP
                source_port = getattr(tcp, "source_port", 0)
                dest_port = getattr(tcp, "destination_port", 0)
                protocol = "TCP"
            elif hasattr(l4, "UDP"):
                udp = l4.UDP
                source_port = getattr(udp, "source_port", 0)
                dest_port = getattr(udp, "destination_port", 0)
                protocol = "UDP"
            elif hasattr(l4, "ICMPv4") or hasattr(l4, "ICMPv6"):
                protocol = "ICMP"

        l7_protocol = None
        http_info = None
        dns_info = None
        kafka_info = None

        if l7:
            l7_type = getattr(l7, "type", None)
            if l7_type:
                l7_protocol = str(l7_type)

            if hasattr(l7, "http"):
                http = l7.http
                http_info = {
                    "method": getattr(http, "method", ""),
                    "url": getattr(http, "url", ""),
                    "protocol": getattr(http, "protocol", ""),
                    "code": getattr(http, "code", 0),
                    "headers": dict(getattr(http, "headers", {})),
                }
            elif hasattr(l7, "dns"):
                dns = l7.dns
                dns_info = {
                    "query": getattr(dns, "query", ""),
                    "ips": list(getattr(dns, "ips", [])),
                    "rcode": getattr(dns, "rcode", 0),
                }
            elif hasattr(l7, "kafka"):
                kafka = l7.kafka
                kafka_info = {
                    "correlation_id": getattr(kafka, "correlation_id", 0),
                    "api_key": getattr(kafka, "api_key", ""),
                    "topic": getattr(kafka, "topic", {}),
                }

        flow_time = datetime.now()
        if hasattr(flow, "time"):
            try:
                timestamp = flow.time
                if hasattr(timestamp, "seconds"):
                    flow_time = datetime.fromtimestamp(timestamp.seconds)
            except Exception:
                pass

        return cls(
            flow_id=str(getattr(flow, "uuid", "")),
            time=flow_time,
            verdict=str(getattr(flow, "verdict", "UNKNOWN")),
            source_ip=getattr(source, "ip", ""),
            source_port=source_port,
            source_pod=getattr(source, "pod_name", None),
            source_namespace=getattr(source, "namespace", None),
            source_labels=dict(getattr(source, "labels", {})),
            destination_ip=getattr(destination, "ip", ""),
            destination_port=dest_port,
            destination_pod=getattr(destination, "pod_name", None),
            destination_namespace=getattr(destination, "namespace", None),
            destination_labels=dict(getattr(destination, "labels", {})),
            protocol=protocol,
            l7_protocol=l7_protocol,
            is_reply=getattr(flow, "is_reply", False),
            node_name=getattr(flow, "node_name", None),
            traffic_direction=str(getattr(flow, "traffic_direction", "INGRESS")),
            drop_reason=str(getattr(flow, "drop_reason_desc", None)) if getattr(flow, "drop_reason_desc", None) else None,
            policy_match=getattr(flow, "policy_match_type", None),
            http_info=http_info,
            dns_info=dns_info,
            kafka_info=kafka_info,
        )


class HubbleMetrics:
    """Prometheus metrics for Hubble client.

    Usage:
        metrics = HubbleMetrics()
        metrics.record_flow(flow)
        metrics.record_error("connection_failed")
    """

    def __init__(self, namespace: str = "consciousness"):
        self.namespace = namespace
        self._lock = threading.Lock()

        # Flow metrics
        self.flows_total = Counter(
            f"{namespace}_hubble_flows_total",
            "Total flows observed",
            ["verdict", "protocol", "direction"],
        )

        self.flows_bytes = Counter(
            f"{namespace}_hubble_flows_bytes_total",
            "Total bytes observed in flows",
            ["direction"],
        )

        self.dropped_flows = Counter(
            f"{namespace}_hubble_dropped_flows_total",
            "Total dropped flows",
            ["drop_reason", "source_namespace", "destination_namespace"],
        )

        self.l7_requests = Counter(
            f"{namespace}_hubble_l7_requests_total",
            "Total L7 requests",
            ["protocol", "method", "status_class"],
        )

        # Policy metrics
        self.policy_verdicts = Counter(
            f"{namespace}_hubble_policy_verdicts_total",
            "Policy verdicts by type",
            ["policy", "verdict"],
        )

        # Connection metrics
        self.active_connections = Gauge(
            f"{namespace}_hubble_active_connections",
            "Active Hubble connections",
        )

        self.connection_errors = Counter(
            f"{namespace}_hubble_connection_errors_total",
            "Hubble connection errors",
            ["error_type"],
        )

        # Performance metrics
        self.flow_processing_latency = Histogram(
            f"{namespace}_hubble_flow_processing_seconds",
            "Flow processing latency",
            buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1],
        )

        self.flows_per_second = Gauge(
            f"{namespace}_hubble_flows_per_second",
            "Current flow rate per second",
        )

    def record_flow(self, flow: HubbleFlow):
        """Record metrics for a flow."""
        self.flows_total.labels(
            verdict=flow.verdict,
            protocol=flow.protocol,
            direction=flow.traffic_direction,
        ).inc()

        if flow.verdict == "DROPPED" and flow.drop_reason:
            self.dropped_flows.labels(
                drop_reason=flow.drop_reason[:50],
                source_namespace=flow.source_namespace or "unknown",
                destination_namespace=flow.destination_namespace or "unknown",
            ).inc()

        if flow.http_info:
            status_code = flow.http_info.get("code", 0)
            status_class = f"{status_code // 100}xx" if status_code else "unknown"
            self.l7_requests.labels(
                protocol="HTTP",
                method=flow.http_info.get("method", "unknown"),
                status_class=status_class,
            ).inc()
        elif flow.dns_info:
            self.l7_requests.labels(
                protocol="DNS",
                method="query",
                status_class="success" if flow.dns_info.get("ips") else "nxdomain",
            ).inc()

    def record_error(self, error_type: str):
        """Record a connection error."""
        self.connection_errors.labels(error_type=error_type).inc()


class HubbleClient:
    """Client for Cilium Hubble API.

    Provides access to network flows via Hubble Relay gRPC API.

    Usage:
        config = HubbleConfig(address="hubble-relay.kube-system:4245")
        client = HubbleClient(config)

        # Get recent flows
        flows = client.get_flows(limit=100)

        # Stream flows
        async for flow in client.stream_flows():
            process_flow(flow)

        # Get filtered flows
        filter = FlowFilter(destination_namespace="production")
        flows = client.get_flows(filter=filter)
    """

    def __init__(
        self,
        config: Optional[HubbleConfig] = None,
        metrics: Optional[HubbleMetrics] = None,
    ):
        self.config = config or HubbleConfig()
        self.metrics = metrics or HubbleMetrics(self.config.namespace)

        self._channel: Any = None
        self._stub: Any = None
        self._lock = threading.Lock()
        self._connected = False
        self._flow_buffer: queue.Queue = queue.Queue(maxsize=10000)
        self._streaming = False
        self._stream_thread: Optional[threading.Thread] = None

    def connect(self) -> bool:
        """Establish connection to Hubble.

        Returns:
            True if connected successfully
        """
        if not GRPC_AVAILABLE:
            logger.error("gRPC not available. Install grpcio package.")
            return False

        with self._lock:
            if self._connected:
                return True

            try:
                # Create channel
                if self.config.tls_enabled:
                    credentials = self._create_ssl_credentials()
                    self._channel = grpc.secure_channel(
                        self.config.address,
                        credentials,
                    )
                else:
                    self._channel = grpc.insecure_channel(self.config.address)

                # Wait for channel ready
                grpc.channel_ready_future(self._channel).result(
                    timeout=self.config.timeout_seconds
                )

                self._connected = True
                self.metrics.active_connections.inc()
                logger.info(f"Connected to Hubble at {self.config.address}")
                return True

            except Exception as e:
                logger.error(f"Failed to connect to Hubble: {e}")
                self.metrics.record_error("connection_failed")
                return False

    def _create_ssl_credentials(self) -> Any:
        """Create SSL credentials for gRPC."""
        root_cert = None
        client_key = None
        client_cert = None

        if self.config.tls_ca_cert:
            with open(self.config.tls_ca_cert, "rb") as f:
                root_cert = f.read()

        if self.config.tls_client_key:
            with open(self.config.tls_client_key, "rb") as f:
                client_key = f.read()

        if self.config.tls_client_cert:
            with open(self.config.tls_client_cert, "rb") as f:
                client_cert = f.read()

        return grpc.ssl_channel_credentials(
            root_certificates=root_cert,
            private_key=client_key,
            certificate_chain=client_cert,
        )

    def disconnect(self):
        """Close connection to Hubble."""
        with self._lock:
            if self._channel:
                self._channel.close()
                self._channel = None
                self._stub = None
                self._connected = False
                self.metrics.active_connections.dec()
                logger.info("Disconnected from Hubble")

    def get_flows(
        self,
        filter: Optional[FlowFilter] = None,
        limit: int = 100,
        since: Optional[timedelta] = None,
    ) -> List[HubbleFlow]:
        """Get recent flows from Hubble.

        Args:
            filter: Optional filter for flows
            limit: Maximum number of flows to return
            since: Get flows since this time ago

        Returns:
            List of HubbleFlow objects
        """
        if not self._connected and not self.connect():
            return []

        # This is a mock implementation since we cannot import actual Hubble proto
        # In production, this would use the Hubble Observer service
        flows: List[HubbleFlow] = []

        try:
            # Mock flow data for demonstration
            logger.debug(f"Fetching flows with limit={limit}")

            # In production:
            # request = observer_pb2.GetFlowsRequest(
            #     number=limit,
            #     follow=False,
            #     whitelist=filter.to_hubble_filter() if filter else [],
            # )
            # for response in self._stub.GetFlows(request):
            #     flow = HubbleFlow.from_grpc(response.flow)
            #     flows.append(flow)

        except Exception as e:
            logger.error(f"Error fetching flows: {e}")
            self.metrics.record_error("get_flows_failed")

        return flows

    async def stream_flows(
        self,
        filter: Optional[FlowFilter] = None,
        on_flow: Optional[Callable[[HubbleFlow], None]] = None,
    ) -> AsyncIterator[HubbleFlow]:
        """Stream flows from Hubble asynchronously.

        Args:
            filter: Optional filter for flows
            on_flow: Optional callback for each flow

        Yields:
            HubbleFlow objects as they arrive
        """
        if not GRPC_AVAILABLE:
            logger.error("gRPC not available for async streaming")
            return

        try:
            # Create async channel
            if self.config.tls_enabled:
                credentials = self._create_ssl_credentials()
                channel = grpc.aio.secure_channel(
                    self.config.address,
                    credentials,
                )
            else:
                channel = grpc.aio.insecure_channel(self.config.address)

            # In production:
            # stub = observer_pb2_grpc.ObserverStub(channel)
            # request = observer_pb2.GetFlowsRequest(
            #     follow=True,
            #     whitelist=filter.to_hubble_filter() if filter else [],
            # )
            # async for response in stub.GetFlows(request):
            #     flow = HubbleFlow.from_grpc(response.flow)
            #     self.metrics.record_flow(flow)
            #     if on_flow:
            #         on_flow(flow)
            #     yield flow

            # Mock implementation for demonstration
            logger.debug("Starting flow stream (mock)")

            # Yield nothing in mock mode
            return

        except Exception as e:
            logger.error(f"Error streaming flows: {e}")
            self.metrics.record_error("stream_flows_failed")

    def start_background_stream(
        self,
        filter: Optional[FlowFilter] = None,
        on_flow: Optional[Callable[[HubbleFlow], None]] = None,
    ):
        """Start streaming flows in a background thread.

        Args:
            filter: Optional filter for flows
            on_flow: Optional callback for each flow
        """
        if self._streaming:
            logger.warning("Background stream already running")
            return

        self._streaming = True

        def stream_worker():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                async def run_stream():
                    async for flow in self.stream_flows(filter, on_flow):
                        if not self._streaming:
                            break
                        try:
                            self._flow_buffer.put_nowait(flow)
                        except queue.Full:
                            logger.warning("Flow buffer full, dropping flow")

                loop.run_until_complete(run_stream())
            except Exception as e:
                logger.error(f"Background stream error: {e}")
            finally:
                loop.close()

        self._stream_thread = threading.Thread(target=stream_worker, daemon=True)
        self._stream_thread.start()
        logger.info("Started background flow stream")

    def stop_background_stream(self):
        """Stop the background flow stream."""
        self._streaming = False
        if self._stream_thread:
            self._stream_thread.join(timeout=5.0)
            self._stream_thread = None
        logger.info("Stopped background flow stream")

    def get_buffered_flows(self, max_count: int = 100) -> List[HubbleFlow]:
        """Get flows from the buffer.

        Args:
            max_count: Maximum number of flows to return

        Returns:
            List of buffered flows
        """
        flows = []
        while len(flows) < max_count:
            try:
                flow = self._flow_buffer.get_nowait()
                flows.append(flow)
            except queue.Empty:
                break
        return flows

    def get_server_status(self) -> Dict[str, Any]:
        """Get Hubble server status.

        Returns:
            Server status information
        """
        if not self._connected:
            return {"status": "disconnected"}

        # In production, call ServerStatus RPC
        return {
            "status": "connected",
            "address": self.config.address,
            "tls_enabled": self.config.tls_enabled,
        }

    def get_nodes(self) -> List[Dict[str, Any]]:
        """Get list of nodes reporting to Hubble.

        Returns:
            List of node information
        """
        if not self._connected:
            return []

        # In production, call GetNodes RPC
        return []

    def health_check(self) -> bool:
        """Check if Hubble connection is healthy.

        Returns:
            True if healthy
        """
        try:
            status = self.get_server_status()
            return status.get("status") == "connected"
        except Exception:
            return False

    def __enter__(self) -> HubbleClient:
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    async def __aenter__(self) -> HubbleClient:
        self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
