"""DNS Monitor

DNS health and resolution monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import threading
import logging
import socket

from prometheus_client import Gauge, Counter, Histogram

logger = logging.getLogger(__name__)


class DNSRecordType(str, Enum):
    """DNS record types."""
    A = "A"
    AAAA = "AAAA"
    CNAME = "CNAME"
    MX = "MX"
    NS = "NS"
    TXT = "TXT"
    SOA = "SOA"
    PTR = "PTR"
    SRV = "SRV"
    CAA = "CAA"


class DNSStatus(str, Enum):
    """DNS check status."""
    RESOLVED = "resolved"
    NXDOMAIN = "nxdomain"
    TIMEOUT = "timeout"
    SERVFAIL = "servfail"
    REFUSED = "refused"
    ERROR = "error"


@dataclass
class DNSRecord:
    """A DNS record."""
    name: str
    record_type: DNSRecordType
    value: str
    ttl: int = 0
    priority: Optional[int] = None  # For MX records
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.record_type.value,
            "value": self.value,
            "ttl": self.ttl,
            "priority": self.priority,
        }


@dataclass
class DNSCheckResult:
    """Result of DNS check."""
    hostname: str
    record_type: DNSRecordType
    status: DNSStatus
    records: List[DNSRecord] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    latency_ms: float = 0.0
    nameserver: str = ""
    error_message: Optional[str] = None
    expected_values: List[str] = field(default_factory=list)
    values_match: bool = True

    @property
    def is_resolved(self) -> bool:
        return self.status == DNSStatus.RESOLVED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hostname": self.hostname,
            "record_type": self.record_type.value,
            "status": self.status.value,
            "records": [r.to_dict() for r in self.records],
            "timestamp": self.timestamp.isoformat(),
            "latency_ms": self.latency_ms,
            "nameserver": self.nameserver,
            "error_message": self.error_message,
            "values_match": self.values_match,
        }


class DNSMonitor:
    """DNS health and resolution monitor.

    Usage:
        monitor = DNSMonitor()

        # Check single hostname
        result = await monitor.check_dns("api.example.com")

        # Add hostname to monitor
        monitor.add_hostname(
            "api.example.com",
            record_type=DNSRecordType.A,
            expected_values=["192.168.1.1"],
        )

        # Check all
        results = await monitor.check_all()

        # Get propagation status
        prop = await monitor.check_propagation("example.com", DNSRecordType.A)
    """

    def __init__(
        self,
        namespace: str = "consciousness",
        default_nameservers: Optional[List[str]] = None,
    ):
        self.namespace = namespace
        self.default_nameservers = default_nameservers or [
            "8.8.8.8",      # Google
            "8.8.4.4",      # Google
            "1.1.1.1",      # Cloudflare
            "1.0.0.1",      # Cloudflare
            "9.9.9.9",      # Quad9
        ]

        self._hostnames: Dict[str, Dict[str, Any]] = {}
        self._results: Dict[str, DNSCheckResult] = {}
        self._lock = threading.Lock()

        # Alert callbacks
        self._alert_callbacks: List[Callable[[DNSCheckResult], None]] = []

        # Metrics
        self.dns_resolution_latency = Histogram(
            f"{namespace}_dns_resolution_latency_seconds",
            "DNS resolution latency",
            ["hostname", "record_type"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
        )

        self.dns_resolution_status = Gauge(
            f"{namespace}_dns_resolution_status",
            "DNS resolution status (1=resolved, 0=failed)",
            ["hostname", "record_type"],
        )

        self.dns_checks_total = Counter(
            f"{namespace}_dns_checks_total",
            "Total DNS checks",
            ["hostname", "record_type", "status"],
        )

        self.dns_ttl = Gauge(
            f"{namespace}_dns_ttl_seconds",
            "DNS record TTL",
            ["hostname", "record_type"],
        )

        self.dns_record_count = Gauge(
            f"{namespace}_dns_record_count",
            "Number of DNS records",
            ["hostname", "record_type"],
        )

    def add_hostname(
        self,
        hostname: str,
        record_type: DNSRecordType = DNSRecordType.A,
        expected_values: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Add hostname to monitor.

        Args:
            hostname: Hostname to monitor
            record_type: DNS record type
            expected_values: Expected record values
            metadata: Additional metadata
        """
        key = f"{hostname}:{record_type.value}"
        with self._lock:
            self._hostnames[key] = {
                "hostname": hostname,
                "record_type": record_type,
                "expected_values": expected_values or [],
                "metadata": metadata or {},
            }

        logger.info(f"Added DNS monitor: {key}")

    def remove_hostname(
        self,
        hostname: str,
        record_type: DNSRecordType = DNSRecordType.A,
    ):
        """Remove hostname from monitoring.

        Args:
            hostname: Hostname
            record_type: Record type
        """
        key = f"{hostname}:{record_type.value}"
        with self._lock:
            self._hostnames.pop(key, None)
            self._results.pop(key, None)

    async def check_dns(
        self,
        hostname: str,
        record_type: DNSRecordType = DNSRecordType.A,
        nameserver: Optional[str] = None,
        timeout_seconds: float = 5.0,
        expected_values: Optional[List[str]] = None,
    ) -> DNSCheckResult:
        """Check DNS resolution for hostname.

        Args:
            hostname: Hostname to resolve
            record_type: DNS record type
            nameserver: Specific nameserver to use
            timeout_seconds: Query timeout
            expected_values: Expected record values

        Returns:
            DNSCheckResult
        """
        import time
        import asyncio

        start_time = time.time()

        try:
            # Try using dnspython if available, otherwise fall back to socket
            try:
                result = await self._check_dns_resolver(
                    hostname,
                    record_type,
                    nameserver,
                    timeout_seconds,
                )
            except ImportError:
                result = await self._check_dns_socket(
                    hostname,
                    record_type,
                    timeout_seconds,
                )

            result.latency_ms = (time.time() - start_time) * 1000

            # Check expected values
            if expected_values:
                result.expected_values = expected_values
                actual_values = [r.value for r in result.records]
                result.values_match = all(v in actual_values for v in expected_values)

            # Record result
            self._record_result(result)

            return result

        except Exception as e:
            result = DNSCheckResult(
                hostname=hostname,
                record_type=record_type,
                status=DNSStatus.ERROR,
                latency_ms=(time.time() - start_time) * 1000,
                error_message=str(e),
            )

            self._record_result(result)
            return result

    async def _check_dns_resolver(
        self,
        hostname: str,
        record_type: DNSRecordType,
        nameserver: Optional[str],
        timeout: float,
    ) -> DNSCheckResult:
        """Check DNS using dnspython resolver."""
        import dns.resolver
        import asyncio

        resolver = dns.resolver.Resolver()

        if nameserver:
            resolver.nameservers = [nameserver]

        resolver.timeout = timeout
        resolver.lifetime = timeout

        try:
            loop = asyncio.get_event_loop()
            answer = await loop.run_in_executor(
                None,
                lambda: resolver.resolve(hostname, record_type.value),
            )

            records = []
            for rdata in answer:
                if record_type == DNSRecordType.MX:
                    records.append(DNSRecord(
                        name=hostname,
                        record_type=record_type,
                        value=str(rdata.exchange).rstrip("."),
                        ttl=answer.ttl,
                        priority=rdata.preference,
                    ))
                else:
                    records.append(DNSRecord(
                        name=hostname,
                        record_type=record_type,
                        value=str(rdata).rstrip("."),
                        ttl=answer.ttl,
                    ))

            return DNSCheckResult(
                hostname=hostname,
                record_type=record_type,
                status=DNSStatus.RESOLVED,
                records=records,
                nameserver=nameserver or "system",
            )

        except dns.resolver.NXDOMAIN:
            return DNSCheckResult(
                hostname=hostname,
                record_type=record_type,
                status=DNSStatus.NXDOMAIN,
                nameserver=nameserver or "system",
                error_message="Domain does not exist",
            )

        except dns.resolver.NoAnswer:
            return DNSCheckResult(
                hostname=hostname,
                record_type=record_type,
                status=DNSStatus.RESOLVED,
                records=[],
                nameserver=nameserver or "system",
            )

        except dns.resolver.Timeout:
            return DNSCheckResult(
                hostname=hostname,
                record_type=record_type,
                status=DNSStatus.TIMEOUT,
                nameserver=nameserver or "system",
                error_message="DNS query timeout",
            )

        except dns.resolver.NoNameservers:
            return DNSCheckResult(
                hostname=hostname,
                record_type=record_type,
                status=DNSStatus.SERVFAIL,
                nameserver=nameserver or "system",
                error_message="No nameservers available",
            )

    async def _check_dns_socket(
        self,
        hostname: str,
        record_type: DNSRecordType,
        timeout: float,
    ) -> DNSCheckResult:
        """Check DNS using socket (fallback for A/AAAA only)."""
        import asyncio

        if record_type not in [DNSRecordType.A, DNSRecordType.AAAA]:
            return DNSCheckResult(
                hostname=hostname,
                record_type=record_type,
                status=DNSStatus.ERROR,
                error_message="Socket fallback only supports A/AAAA records. Install dnspython for full support.",
            )

        family = socket.AF_INET if record_type == DNSRecordType.A else socket.AF_INET6

        try:
            loop = asyncio.get_event_loop()

            socket.setdefaulttimeout(timeout)
            infos = await loop.run_in_executor(
                None,
                lambda: socket.getaddrinfo(hostname, None, family),
            )

            records = []
            seen = set()

            for info in infos:
                addr = info[4][0]
                if addr not in seen:
                    seen.add(addr)
                    records.append(DNSRecord(
                        name=hostname,
                        record_type=record_type,
                        value=addr,
                        ttl=0,
                    ))

            return DNSCheckResult(
                hostname=hostname,
                record_type=record_type,
                status=DNSStatus.RESOLVED,
                records=records,
                nameserver="system",
            )

        except socket.gaierror as e:
            if "Name or service not known" in str(e) or "NXDOMAIN" in str(e):
                return DNSCheckResult(
                    hostname=hostname,
                    record_type=record_type,
                    status=DNSStatus.NXDOMAIN,
                    nameserver="system",
                    error_message=str(e),
                )
            return DNSCheckResult(
                hostname=hostname,
                record_type=record_type,
                status=DNSStatus.ERROR,
                nameserver="system",
                error_message=str(e),
            )

        except socket.timeout:
            return DNSCheckResult(
                hostname=hostname,
                record_type=record_type,
                status=DNSStatus.TIMEOUT,
                nameserver="system",
                error_message="DNS resolution timeout",
            )

    def _record_result(self, result: DNSCheckResult):
        """Record DNS check result."""
        key = f"{result.hostname}:{result.record_type.value}"

        with self._lock:
            self._results[key] = result

        # Update metrics
        self.dns_checks_total.labels(
            hostname=result.hostname,
            record_type=result.record_type.value,
            status=result.status.value,
        ).inc()

        self.dns_resolution_latency.labels(
            hostname=result.hostname,
            record_type=result.record_type.value,
        ).observe(result.latency_ms / 1000)

        status_value = 1 if result.is_resolved else 0
        self.dns_resolution_status.labels(
            hostname=result.hostname,
            record_type=result.record_type.value,
        ).set(status_value)

        if result.records:
            self.dns_record_count.labels(
                hostname=result.hostname,
                record_type=result.record_type.value,
            ).set(len(result.records))

            avg_ttl = sum(r.ttl for r in result.records) / len(result.records)
            self.dns_ttl.labels(
                hostname=result.hostname,
                record_type=result.record_type.value,
            ).set(avg_ttl)

        # Check for alerts
        if not result.is_resolved or not result.values_match:
            self._trigger_alert(result)

    def _trigger_alert(self, result: DNSCheckResult):
        """Trigger alert for DNS issue."""
        logger.warning(f"DNS alert for {result.hostname}: {result.status.value}")

        for callback in self._alert_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

    def on_alert(self, callback: Callable[[DNSCheckResult], None]):
        """Register alert callback.

        Args:
            callback: Function to call on DNS alerts
        """
        self._alert_callbacks.append(callback)

    async def check_all(self) -> List[DNSCheckResult]:
        """Check all registered hostnames.

        Returns:
            List of check results
        """
        import asyncio

        with self._lock:
            hostnames = list(self._hostnames.values())

        tasks = [
            self.check_dns(
                h["hostname"],
                h["record_type"],
                expected_values=h.get("expected_values"),
            )
            for h in hostnames
        ]

        return await asyncio.gather(*tasks)

    async def check_propagation(
        self,
        hostname: str,
        record_type: DNSRecordType = DNSRecordType.A,
        expected_value: Optional[str] = None,
        nameservers: Optional[List[str]] = None,
    ) -> Dict[str, DNSCheckResult]:
        """Check DNS propagation across nameservers.

        Args:
            hostname: Hostname to check
            record_type: Record type
            expected_value: Expected record value
            nameservers: Nameservers to check

        Returns:
            Dict of nameserver -> result
        """
        import asyncio

        servers = nameservers or self.default_nameservers
        expected = [expected_value] if expected_value else None

        tasks = {
            ns: self.check_dns(
                hostname,
                record_type,
                nameserver=ns,
                expected_values=expected,
            )
            for ns in servers
        }

        results = {}
        for ns, task in tasks.items():
            results[ns] = await task

        return results

    async def check_consistency(
        self,
        hostname: str,
        record_type: DNSRecordType = DNSRecordType.A,
    ) -> Dict[str, Any]:
        """Check DNS consistency across nameservers.

        Args:
            hostname: Hostname to check
            record_type: Record type

        Returns:
            Consistency report
        """
        results = await self.check_propagation(hostname, record_type)

        # Collect all unique values
        all_values: Dict[str, List[str]] = {}
        for ns, result in results.items():
            values = tuple(sorted(r.value for r in result.records))
            key = str(values)
            if key not in all_values:
                all_values[key] = []
            all_values[key].append(ns)

        consistent = len(all_values) == 1
        resolved_count = sum(1 for r in results.values() if r.is_resolved)

        return {
            "hostname": hostname,
            "record_type": record_type.value,
            "consistent": consistent,
            "nameservers_checked": len(results),
            "resolved_count": resolved_count,
            "unique_value_sets": len(all_values),
            "values_by_nameserver": {
                ns: [r.value for r in result.records]
                for ns, result in results.items()
            },
            "nameservers_by_value": all_values,
        }

    def get_result(
        self,
        hostname: str,
        record_type: DNSRecordType = DNSRecordType.A,
    ) -> Optional[DNSCheckResult]:
        """Get latest result for hostname.

        Args:
            hostname: Hostname
            record_type: Record type

        Returns:
            DNSCheckResult or None
        """
        key = f"{hostname}:{record_type.value}"
        with self._lock:
            return self._results.get(key)

    def get_failed_resolutions(self) -> List[DNSCheckResult]:
        """Get all failed DNS resolutions.

        Returns:
            List of failed results
        """
        with self._lock:
            results = list(self._results.values())

        return [r for r in results if not r.is_resolved]

    def get_summary(self) -> Dict[str, Any]:
        """Get DNS monitoring summary.

        Returns:
            Summary dictionary
        """
        with self._lock:
            results = list(self._results.values())

        resolved = sum(1 for r in results if r.is_resolved)
        failed = len(results) - resolved
        mismatched = sum(1 for r in results if not r.values_match)

        # Average latency
        latencies = [r.latency_ms for r in results]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        return {
            "total_hostnames": len(self._hostnames),
            "total_checks": len(results),
            "resolved_count": resolved,
            "failed_count": failed,
            "mismatched_count": mismatched,
            "average_latency_ms": avg_latency,
            "nameservers": self.default_nameservers,
        }
