"""MCP Server Implementation

Model Context Protocol server for Claude Code integration.
Provides observability tools and data resources via MCP.

Usage:
    from consciousness_suite.observability.ai_platform.mcp_server import (
        MCPServer,
        MCPServerConfig,
    )

    config = MCPServerConfig(
        name="consciousness-observability",
        host="localhost",
        port=8765,
    )
    server = MCPServer(config)
    await server.start()
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union

from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)


class MCPConnectionState(str, Enum):
    """MCP connection states."""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"


class MCPMessageType(str, Enum):
    """MCP message types."""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"


@dataclass
class MCPServerConfig:
    """Configuration for MCP server."""
    name: str = "consciousness-observability"
    version: str = "1.0.0"
    host: str = "localhost"
    port: int = 8765
    max_connections: int = 100
    connection_timeout_seconds: float = 30.0
    heartbeat_interval_seconds: float = 30.0
    enable_authentication: bool = True
    api_keys: List[str] = field(default_factory=list)
    enable_tls: bool = False
    tls_cert_path: Optional[str] = None
    tls_key_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPMessage:
    """MCP protocol message."""
    id: str
    type: MCPMessageType
    method: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        msg = {
            "jsonrpc": "2.0",
            "id": self.id,
        }
        if self.type == MCPMessageType.REQUEST:
            msg["method"] = self.method
            if self.params:
                msg["params"] = self.params
        elif self.type == MCPMessageType.RESPONSE:
            if self.result is not None:
                msg["result"] = self.result
            if self.error:
                msg["error"] = self.error
        elif self.type == MCPMessageType.NOTIFICATION:
            msg["method"] = self.method
            if self.params:
                msg["params"] = self.params
        elif self.type == MCPMessageType.ERROR:
            msg["error"] = self.error
        return msg

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPMessage":
        """Create from dictionary."""
        msg_id = data.get("id", str(uuid.uuid4()))

        if "method" in data and "id" in data:
            return cls(
                id=msg_id,
                type=MCPMessageType.REQUEST,
                method=data.get("method"),
                params=data.get("params"),
            )
        elif "result" in data or ("error" in data and "id" in data):
            return cls(
                id=msg_id,
                type=MCPMessageType.RESPONSE,
                result=data.get("result"),
                error=data.get("error"),
            )
        elif "method" in data and "id" not in data:
            return cls(
                id=msg_id,
                type=MCPMessageType.NOTIFICATION,
                method=data.get("method"),
                params=data.get("params"),
            )
        else:
            return cls(
                id=msg_id,
                type=MCPMessageType.ERROR,
                error={"code": -32600, "message": "Invalid Request"},
            )


@dataclass
class MCPConnection:
    """Represents an MCP client connection."""
    connection_id: str
    state: MCPConnectionState
    client_info: Dict[str, Any] = field(default_factory=dict)
    connected_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    authenticated: bool = False
    capabilities: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = datetime.now()

    def is_stale(self, timeout_seconds: float) -> bool:
        """Check if connection is stale."""
        elapsed = (datetime.now() - self.last_activity).total_seconds()
        return elapsed > timeout_seconds


class MCPHandler(ABC):
    """Abstract base class for MCP method handlers."""

    @abstractmethod
    async def handle(
        self,
        connection: MCPConnection,
        params: Optional[Dict[str, Any]],
    ) -> Any:
        """Handle an MCP method call.

        Args:
            connection: The client connection
            params: Method parameters

        Returns:
            Method result
        """
        pass


class MCPServer:
    """MCP Server for Claude Code integration.

    Provides observability tools and resources via the
    Model Context Protocol for AI assistant integration.

    Usage:
        server = MCPServer(config)

        # Register tools
        server.register_tool("metrics_query", MetricsQueryTool())

        # Register resources
        server.register_resource("metrics", MetricsResource())

        # Start server
        await server.start()
    """

    def __init__(
        self,
        config: MCPServerConfig,
        namespace: str = "consciousness",
    ):
        self.config = config
        self.namespace = namespace
        self._lock = threading.Lock()

        # Connection management
        self._connections: Dict[str, MCPConnection] = {}
        self._handlers: Dict[str, MCPHandler] = {}

        # Tool and resource registries
        self._tools: Dict[str, Any] = {}
        self._resources: Dict[str, Any] = {}

        # Server state
        self._server = None
        self._running = False
        self._heartbeat_task = None

        # Initialize metrics
        self._init_metrics()

        # Register built-in handlers
        self._register_builtin_handlers()

        logger.info(f"MCPServer initialized: {config.name}@{config.host}:{config.port}")

    def _init_metrics(self):
        """Initialize Prometheus metrics."""
        self.connections_total = Counter(
            f"{self.namespace}_mcp_connections_total",
            "Total MCP connections",
            ["state"],
        )

        self.active_connections = Gauge(
            f"{self.namespace}_mcp_active_connections",
            "Active MCP connections",
        )

        self.requests_total = Counter(
            f"{self.namespace}_mcp_requests_total",
            "Total MCP requests",
            ["method", "status"],
        )

        self.request_latency = Histogram(
            f"{self.namespace}_mcp_request_latency_seconds",
            "MCP request latency",
            ["method"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
        )

        self.messages_sent = Counter(
            f"{self.namespace}_mcp_messages_sent_total",
            "Total MCP messages sent",
            ["type"],
        )

        self.messages_received = Counter(
            f"{self.namespace}_mcp_messages_received_total",
            "Total MCP messages received",
            ["type"],
        )

    def _register_builtin_handlers(self):
        """Register built-in MCP method handlers."""

        class InitializeHandler(MCPHandler):
            def __init__(self, server: MCPServer):
                self.server = server

            async def handle(
                self,
                connection: MCPConnection,
                params: Optional[Dict[str, Any]],
            ) -> Dict[str, Any]:
                client_info = params or {}
                connection.client_info = client_info

                return {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {"listChanged": True},
                        "resources": {"subscribe": True, "listChanged": True},
                        "prompts": {"listChanged": True},
                        "logging": {},
                    },
                    "serverInfo": {
                        "name": self.server.config.name,
                        "version": self.server.config.version,
                    },
                }

        class ListToolsHandler(MCPHandler):
            def __init__(self, server: MCPServer):
                self.server = server

            async def handle(
                self,
                connection: MCPConnection,
                params: Optional[Dict[str, Any]],
            ) -> Dict[str, Any]:
                tools = []
                for name, tool in self.server._tools.items():
                    tool_def = {
                        "name": name,
                        "description": getattr(tool, "description", ""),
                        "inputSchema": getattr(tool, "input_schema", {}),
                    }
                    tools.append(tool_def)
                return {"tools": tools}

        class CallToolHandler(MCPHandler):
            def __init__(self, server: MCPServer):
                self.server = server

            async def handle(
                self,
                connection: MCPConnection,
                params: Optional[Dict[str, Any]],
            ) -> Dict[str, Any]:
                if not params:
                    raise ValueError("Missing tool call parameters")

                tool_name = params.get("name")
                arguments = params.get("arguments", {})

                if tool_name not in self.server._tools:
                    raise ValueError(f"Unknown tool: {tool_name}")

                tool = self.server._tools[tool_name]
                result = await tool.execute(arguments)

                return {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result.to_dict())
                            if hasattr(result, "to_dict")
                            else str(result),
                        }
                    ],
                    "isError": False,
                }

        class ListResourcesHandler(MCPHandler):
            def __init__(self, server: MCPServer):
                self.server = server

            async def handle(
                self,
                connection: MCPConnection,
                params: Optional[Dict[str, Any]],
            ) -> Dict[str, Any]:
                resources = []
                for name, resource in self.server._resources.items():
                    resource_def = {
                        "uri": f"consciousness://{name}",
                        "name": name,
                        "description": getattr(resource, "description", ""),
                        "mimeType": getattr(resource, "mime_type", "application/json"),
                    }
                    resources.append(resource_def)
                return {"resources": resources}

        class ReadResourceHandler(MCPHandler):
            def __init__(self, server: MCPServer):
                self.server = server

            async def handle(
                self,
                connection: MCPConnection,
                params: Optional[Dict[str, Any]],
            ) -> Dict[str, Any]:
                if not params:
                    raise ValueError("Missing resource parameters")

                uri = params.get("uri", "")
                # Parse consciousness://resource_name
                if uri.startswith("consciousness://"):
                    resource_name = uri.replace("consciousness://", "")
                else:
                    resource_name = uri

                if resource_name not in self.server._resources:
                    raise ValueError(f"Unknown resource: {resource_name}")

                resource = self.server._resources[resource_name]
                content = await resource.read(params.get("arguments", {}))

                return {
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": getattr(
                                resource, "mime_type", "application/json"
                            ),
                            "text": json.dumps(content)
                            if isinstance(content, (dict, list))
                            else str(content),
                        }
                    ]
                }

        class PingHandler(MCPHandler):
            async def handle(
                self,
                connection: MCPConnection,
                params: Optional[Dict[str, Any]],
            ) -> Dict[str, Any]:
                return {}

        # Register handlers
        self._handlers["initialize"] = InitializeHandler(self)
        self._handlers["tools/list"] = ListToolsHandler(self)
        self._handlers["tools/call"] = CallToolHandler(self)
        self._handlers["resources/list"] = ListResourcesHandler(self)
        self._handlers["resources/read"] = ReadResourceHandler(self)
        self._handlers["ping"] = PingHandler()

    def register_tool(self, name: str, tool: Any):
        """Register an observability tool.

        Args:
            name: Tool name
            tool: Tool instance with execute() method
        """
        with self._lock:
            self._tools[name] = tool
            logger.info(f"Registered tool: {name}")

    def register_resource(self, name: str, resource: Any):
        """Register a data resource.

        Args:
            name: Resource name
            resource: Resource instance with read() method
        """
        with self._lock:
            self._resources[name] = resource
            logger.info(f"Registered resource: {name}")

    def register_handler(self, method: str, handler: MCPHandler):
        """Register a custom method handler.

        Args:
            method: Method name
            handler: Handler instance
        """
        with self._lock:
            self._handlers[method] = handler
            logger.info(f"Registered handler: {method}")

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ):
        """Handle a new client connection."""
        connection_id = str(uuid.uuid4())
        addr = writer.get_extra_info("peername")

        connection = MCPConnection(
            connection_id=connection_id,
            state=MCPConnectionState.CONNECTING,
        )

        with self._lock:
            self._connections[connection_id] = connection
            self.active_connections.set(len(self._connections))

        self.connections_total.labels(state="connecting").inc()
        logger.info(f"New connection from {addr}: {connection_id}")

        try:
            connection.state = MCPConnectionState.CONNECTED
            self.connections_total.labels(state="connected").inc()

            # Read messages
            buffer = b""
            while self._running:
                try:
                    data = await asyncio.wait_for(
                        reader.read(4096),
                        timeout=self.config.connection_timeout_seconds,
                    )
                    if not data:
                        break

                    buffer += data
                    connection.update_activity()

                    # Process complete messages (newline-delimited JSON)
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        if line.strip():
                            await self._process_message(connection, line, writer)

                except asyncio.TimeoutError:
                    # Check for stale connection
                    if connection.is_stale(self.config.heartbeat_interval_seconds * 2):
                        logger.warning(f"Connection {connection_id} timed out")
                        break
                except Exception as e:
                    logger.error(f"Error reading from {connection_id}: {e}")
                    break

        except Exception as e:
            logger.error(f"Connection error for {connection_id}: {e}")
            connection.state = MCPConnectionState.ERROR

        finally:
            connection.state = MCPConnectionState.DISCONNECTED
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

            with self._lock:
                self._connections.pop(connection_id, None)
                self.active_connections.set(len(self._connections))

            self.connections_total.labels(state="disconnected").inc()
            logger.info(f"Connection closed: {connection_id}")

    async def _process_message(
        self,
        connection: MCPConnection,
        raw_message: bytes,
        writer: asyncio.StreamWriter,
    ):
        """Process an incoming MCP message."""
        try:
            data = json.loads(raw_message.decode("utf-8"))
            message = MCPMessage.from_dict(data)

            self.messages_received.labels(type=message.type.value).inc()

            if message.type == MCPMessageType.REQUEST:
                await self._handle_request(connection, message, writer)
            elif message.type == MCPMessageType.NOTIFICATION:
                await self._handle_notification(connection, message)

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {e}")
            error_response = MCPMessage(
                id="unknown",
                type=MCPMessageType.ERROR,
                error={"code": -32700, "message": "Parse error"},
            )
            await self._send_message(writer, error_response)

    async def _handle_request(
        self,
        connection: MCPConnection,
        message: MCPMessage,
        writer: asyncio.StreamWriter,
    ):
        """Handle an MCP request."""
        start_time = time.perf_counter()
        method = message.method or ""
        status = "success"

        try:
            handler = self._handlers.get(method)
            if not handler:
                raise ValueError(f"Unknown method: {method}")

            result = await handler.handle(connection, message.params)

            response = MCPMessage(
                id=message.id,
                type=MCPMessageType.RESPONSE,
                result=result,
            )

        except Exception as e:
            status = "error"
            logger.error(f"Error handling {method}: {e}")
            response = MCPMessage(
                id=message.id,
                type=MCPMessageType.RESPONSE,
                error={"code": -32603, "message": str(e)},
            )

        finally:
            latency = time.perf_counter() - start_time
            self.requests_total.labels(method=method, status=status).inc()
            self.request_latency.labels(method=method).observe(latency)

        await self._send_message(writer, response)

    async def _handle_notification(
        self,
        connection: MCPConnection,
        message: MCPMessage,
    ):
        """Handle an MCP notification."""
        method = message.method or ""

        if method == "notifications/cancelled":
            # Handle cancellation
            logger.debug(f"Request cancelled: {message.params}")
        elif method == "notifications/progress":
            # Handle progress update
            logger.debug(f"Progress: {message.params}")

    async def _send_message(
        self,
        writer: asyncio.StreamWriter,
        message: MCPMessage,
    ):
        """Send an MCP message to client."""
        try:
            data = json.dumps(message.to_dict()) + "\n"
            writer.write(data.encode("utf-8"))
            await writer.drain()
            self.messages_sent.labels(type=message.type.value).inc()
        except Exception as e:
            logger.error(f"Error sending message: {e}")

    async def _heartbeat_loop(self):
        """Periodic heartbeat to check connection health."""
        while self._running:
            try:
                await asyncio.sleep(self.config.heartbeat_interval_seconds)

                with self._lock:
                    stale_connections = [
                        conn_id
                        for conn_id, conn in self._connections.items()
                        if conn.is_stale(self.config.heartbeat_interval_seconds * 3)
                    ]

                for conn_id in stale_connections:
                    logger.warning(f"Removing stale connection: {conn_id}")
                    with self._lock:
                        self._connections.pop(conn_id, None)
                        self.active_connections.set(len(self._connections))

            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

    async def start(self):
        """Start the MCP server."""
        if self._running:
            logger.warning("Server already running")
            return

        self._running = True

        # Start TCP server
        if self.config.enable_tls and self.config.tls_cert_path:
            import ssl

            ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ssl_context.load_cert_chain(
                self.config.tls_cert_path,
                self.config.tls_key_path,
            )
            self._server = await asyncio.start_server(
                self._handle_connection,
                self.config.host,
                self.config.port,
                ssl=ssl_context,
            )
        else:
            self._server = await asyncio.start_server(
                self._handle_connection,
                self.config.host,
                self.config.port,
            )

        # Start heartbeat task
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        logger.info(
            f"MCP Server started on {self.config.host}:{self.config.port}"
        )

        async with self._server:
            await self._server.serve_forever()

    async def stop(self):
        """Stop the MCP server."""
        self._running = False

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        if self._server:
            self._server.close()
            await self._server.wait_closed()

        # Close all connections
        with self._lock:
            self._connections.clear()
            self.active_connections.set(0)

        logger.info("MCP Server stopped")

    def get_connection(self, connection_id: str) -> Optional[MCPConnection]:
        """Get connection by ID.

        Args:
            connection_id: Connection ID

        Returns:
            MCPConnection or None
        """
        with self._lock:
            return self._connections.get(connection_id)

    def get_connections(self) -> List[MCPConnection]:
        """Get all active connections.

        Returns:
            List of connections
        """
        with self._lock:
            return list(self._connections.values())

    def get_statistics(self) -> Dict[str, Any]:
        """Get server statistics.

        Returns:
            Statistics dictionary
        """
        with self._lock:
            connections = list(self._connections.values())

        return {
            "server_name": self.config.name,
            "version": self.config.version,
            "host": self.config.host,
            "port": self.config.port,
            "running": self._running,
            "active_connections": len(connections),
            "max_connections": self.config.max_connections,
            "registered_tools": list(self._tools.keys()),
            "registered_resources": list(self._resources.keys()),
            "registered_handlers": list(self._handlers.keys()),
        }
