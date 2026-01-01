"""Coralogix Integration Module

SDK client and autonomous agent integration for Coralogix:
- Log ingestion with automatic batching
- Metrics export
- Span/trace export
- Autonomous observability agent (Olly)
"""

from .client import (
    CoralogixClient,
    CoralogixConfig,
    LogEntry,
    LogSeverity,
    LogBatch,
)

__all__ = [
    "CoralogixClient",
    "CoralogixConfig",
    "LogEntry",
    "LogSeverity",
    "LogBatch",
]
