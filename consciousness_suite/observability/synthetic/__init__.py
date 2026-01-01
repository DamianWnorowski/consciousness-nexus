"""Synthetic Monitoring Module

Proactive monitoring with synthetic probes:
- HTTP/HTTPS endpoint probes
- Multi-step transaction tests
- SSL certificate monitoring
- DNS health monitoring
- Scheduled probe execution
"""

from .probes import (
    SyntheticProbe,
    ProbeResult,
    ProbeConfig,
    ProbeType,
    ProbeStatus,
)
from .scheduler import (
    ProbeScheduler,
    ScheduledProbe,
    ScheduleConfig,
)
from .endpoints import (
    EndpointProbe,
    HTTPProbeConfig,
    HTTPProbeResult,
    HealthCheckResult,
)
from .transactions import (
    TransactionProbe,
    TransactionStep,
    TransactionResult,
    StepResult,
)
from .ssl_monitor import (
    SSLMonitor,
    CertificateInfo,
    SSLCheckResult,
    CertificateStatus,
)
from .dns_monitor import (
    DNSMonitor,
    DNSRecord,
    DNSCheckResult,
    DNSRecordType,
)

__all__ = [
    # Probes
    "SyntheticProbe",
    "ProbeResult",
    "ProbeConfig",
    "ProbeType",
    "ProbeStatus",
    # Scheduler
    "ProbeScheduler",
    "ScheduledProbe",
    "ScheduleConfig",
    # Endpoints
    "EndpointProbe",
    "HTTPProbeConfig",
    "HTTPProbeResult",
    "HealthCheckResult",
    # Transactions
    "TransactionProbe",
    "TransactionStep",
    "TransactionResult",
    "StepResult",
    # SSL
    "SSLMonitor",
    "CertificateInfo",
    "SSLCheckResult",
    "CertificateStatus",
    # DNS
    "DNSMonitor",
    "DNSRecord",
    "DNSCheckResult",
    "DNSRecordType",
]
