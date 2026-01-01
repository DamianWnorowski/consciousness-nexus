"""AI Platform Integration Module

Comprehensive AI platform observability with:
- MCP (Model Context Protocol) Server for Claude Code integration
- Coralogix SDK client for log analytics
- Autonomous observability agent (Olly) integration
- Metrics query, trace lookup, and alert status tools
"""

from .mcp_server import (
    MCPServer,
    MCPServerConfig,
    MCPConnection,
    MCPConnectionState,
)
from .mcp_server.tools import (
    ObservabilityTool,
    MetricsQueryTool,
    TraceLookupTool,
    AlertStatusTool,
    ToolRegistry,
    ToolResult,
)
from .mcp_server.resources import (
    MCPResource,
    MetricsResource,
    TracesResource,
    AlertsResource,
    ResourceRegistry,
)
from .coralogix import (
    CoralogixClient,
    CoralogixConfig,
    LogEntry,
    LogSeverity,
)
from .coralogix.olly_agent import (
    OllyAgent,
    OllyAgentConfig,
    OllyTask,
    OllyTaskResult,
    OllyTaskType,
)

__all__ = [
    # MCP Server
    "MCPServer",
    "MCPServerConfig",
    "MCPConnection",
    "MCPConnectionState",
    # MCP Tools
    "ObservabilityTool",
    "MetricsQueryTool",
    "TraceLookupTool",
    "AlertStatusTool",
    "ToolRegistry",
    "ToolResult",
    # MCP Resources
    "MCPResource",
    "MetricsResource",
    "TracesResource",
    "AlertsResource",
    "ResourceRegistry",
    # Coralogix
    "CoralogixClient",
    "CoralogixConfig",
    "LogEntry",
    "LogSeverity",
    # Olly Agent
    "OllyAgent",
    "OllyAgentConfig",
    "OllyTask",
    "OllyTaskResult",
    "OllyTaskType",
]
