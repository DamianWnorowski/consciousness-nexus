"""Cilium Hubble Integration

eBPF-based network observability via Hubble:
- Real-time flow visibility
- Network policy enforcement
- Service dependency mapping
"""

from .hubble_client import (
    HubbleClient,
    HubbleConfig,
    HubbleFlow,
    FlowFilter,
    HubbleMetrics,
)
from .flows import (
    FlowAnalyzer,
    FlowRecord,
    FlowDirection,
    FlowVerdict,
    FlowType,
    FlowAggregation,
    L7Protocol,
    FlowStats,
)
from .policies import (
    PolicyMonitor,
    NetworkPolicy,
    PolicyVerdict,
    PolicyMatch,
    PolicyRule,
    PolicyEndpoint,
    PolicyStats,
)

__all__ = [
    # Hubble Client
    "HubbleClient",
    "HubbleConfig",
    "HubbleFlow",
    "FlowFilter",
    "HubbleMetrics",
    # Flow Analysis
    "FlowAnalyzer",
    "FlowRecord",
    "FlowDirection",
    "FlowVerdict",
    "FlowType",
    "FlowAggregation",
    "L7Protocol",
    "FlowStats",
    # Network Policy
    "PolicyMonitor",
    "NetworkPolicy",
    "PolicyVerdict",
    "PolicyMatch",
    "PolicyRule",
    "PolicyEndpoint",
    "PolicyStats",
]
