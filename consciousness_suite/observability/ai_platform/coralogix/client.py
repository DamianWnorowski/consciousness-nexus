"""Coralogix API Client

Integration with Coralogix observability platform.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class LogSeverity(int, Enum):
    DEBUG = 1
    VERBOSE = 2
    INFO = 3
    WARNING = 4
    ERROR = 5
    CRITICAL = 6

@dataclass
class LogEntry:
    text: str
    severity: LogSeverity
    metadata: Dict[str, Any]

@dataclass
class LogBatch:
    entries: List[LogEntry]

class CoralogixConfig:
    def __init__(self, api_key: str, domain: str = "coralogix.com"):
        self.api_key = api_key
        self.domain = domain

class CoralogixClient:
    def __init__(self, config: CoralogixConfig):
        self.config = config

    async def send_logs(self, logs: List[Dict[str, Any]]):
        pass

    async def send_metrics(self, metrics: List[Dict[str, Any]]):
        pass
