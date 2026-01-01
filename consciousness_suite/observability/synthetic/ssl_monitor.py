"""SSL Certificate Monitor

SSL/TLS certificate monitoring and expiration alerts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import threading
import logging
import ssl
import socket

from prometheus_client import Gauge, Counter

logger = logging.getLogger(__name__)


class CertificateStatus(str, Enum):
    """Certificate status."""
    VALID = "valid"
    EXPIRING_SOON = "expiring_soon"
    EXPIRED = "expired"
    INVALID = "invalid"
    SELF_SIGNED = "self_signed"
    REVOKED = "revoked"
    UNKNOWN = "unknown"


@dataclass
class CertificateInfo:
    """SSL certificate information."""
    hostname: str
    port: int = 443
    subject: Dict[str, str] = field(default_factory=dict)
    issuer: Dict[str, str] = field(default_factory=dict)
    serial_number: str = ""
    version: int = 0
    not_before: Optional[datetime] = None
    not_after: Optional[datetime] = None
    signature_algorithm: str = ""
    san: List[str] = field(default_factory=list)  # Subject Alternative Names
    is_self_signed: bool = False
    chain_length: int = 0
    ocsp_stapling: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def days_until_expiry(self) -> int:
        """Days until certificate expires."""
        if self.not_after:
            delta = self.not_after - datetime.now()
            return max(0, delta.days)
        return -1

    @property
    def is_expired(self) -> bool:
        """Check if certificate is expired."""
        if self.not_after:
            return datetime.now() > self.not_after
        return False

    @property
    def is_valid(self) -> bool:
        """Check if certificate is currently valid."""
        now = datetime.now()
        if self.not_before and self.not_after:
            return self.not_before <= now <= self.not_after
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hostname": self.hostname,
            "port": self.port,
            "subject": self.subject,
            "issuer": self.issuer,
            "serial_number": self.serial_number,
            "version": self.version,
            "not_before": self.not_before.isoformat() if self.not_before else None,
            "not_after": self.not_after.isoformat() if self.not_after else None,
            "days_until_expiry": self.days_until_expiry,
            "is_expired": self.is_expired,
            "is_valid": self.is_valid,
            "is_self_signed": self.is_self_signed,
            "signature_algorithm": self.signature_algorithm,
            "san": self.san,
            "chain_length": self.chain_length,
        }


@dataclass
class SSLCheckResult:
    """Result of SSL check."""
    hostname: str
    port: int
    status: CertificateStatus
    certificate: Optional[CertificateInfo] = None
    timestamp: datetime = field(default_factory=datetime.now)
    latency_ms: float = 0.0
    protocol_version: str = ""
    cipher_suite: str = ""
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hostname": self.hostname,
            "port": self.port,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "latency_ms": self.latency_ms,
            "protocol_version": self.protocol_version,
            "cipher_suite": self.cipher_suite,
            "error_message": self.error_message,
            "warnings": self.warnings,
            "certificate": self.certificate.to_dict() if self.certificate else None,
        }


class SSLMonitor:
    """SSL/TLS certificate monitor.

    Usage:
        monitor = SSLMonitor()

        # Check single host
        result = await monitor.check_ssl("api.example.com")

        # Monitor multiple hosts
        monitor.add_host("api.example.com")
        monitor.add_host("web.example.com", port=8443)

        # Get expiring certificates
        expiring = monitor.get_expiring_certificates(days=30)
    """

    def __init__(
        self,
        namespace: str = "consciousness",
        warning_days: int = 30,
        critical_days: int = 7,
    ):
        self.namespace = namespace
        self.warning_days = warning_days
        self.critical_days = critical_days

        self._hosts: Dict[str, Dict[str, Any]] = {}
        self._results: Dict[str, SSLCheckResult] = {}
        self._lock = threading.Lock()

        # Alert callbacks
        self._alert_callbacks: List[Callable[[SSLCheckResult], None]] = []

        # Metrics
        self.certificate_days_remaining = Gauge(
            f"{namespace}_ssl_certificate_days_remaining",
            "Days until certificate expires",
            ["hostname", "port"],
        )

        self.ssl_check_status = Gauge(
            f"{namespace}_ssl_check_status",
            "SSL check status (1=valid, 0=invalid/expired)",
            ["hostname", "port"],
        )

        self.ssl_check_latency = Gauge(
            f"{namespace}_ssl_check_latency_seconds",
            "SSL check latency",
            ["hostname", "port"],
        )

        self.certificates_expiring = Gauge(
            f"{namespace}_ssl_certificates_expiring",
            "Number of certificates expiring within threshold",
            ["threshold_days"],
        )

        self.ssl_checks_total = Counter(
            f"{namespace}_ssl_checks_total",
            "Total SSL checks",
            ["hostname", "status"],
        )

    def add_host(
        self,
        hostname: str,
        port: int = 443,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Add host to monitor.

        Args:
            hostname: Hostname to check
            port: Port number
            metadata: Additional metadata
        """
        key = f"{hostname}:{port}"
        with self._lock:
            self._hosts[key] = {
                "hostname": hostname,
                "port": port,
                "metadata": metadata or {},
            }

        logger.info(f"Added SSL monitor: {key}")

    def remove_host(self, hostname: str, port: int = 443):
        """Remove host from monitoring.

        Args:
            hostname: Hostname
            port: Port number
        """
        key = f"{hostname}:{port}"
        with self._lock:
            self._hosts.pop(key, None)
            self._results.pop(key, None)

    async def check_ssl(
        self,
        hostname: str,
        port: int = 443,
        timeout_seconds: float = 10.0,
    ) -> SSLCheckResult:
        """Check SSL certificate for a host.

        Args:
            hostname: Hostname to check
            port: Port number
            timeout_seconds: Connection timeout

        Returns:
            SSLCheckResult
        """
        import time
        import asyncio

        start_time = time.time()

        try:
            # Run in thread pool for sync socket operations
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._check_ssl_sync,
                hostname,
                port,
                timeout_seconds,
            )

            result.latency_ms = (time.time() - start_time) * 1000

            # Record result
            self._record_result(result)

            return result

        except Exception as e:
            result = SSLCheckResult(
                hostname=hostname,
                port=port,
                status=CertificateStatus.UNKNOWN,
                latency_ms=(time.time() - start_time) * 1000,
                error_message=str(e),
            )

            self._record_result(result)
            return result

    def _check_ssl_sync(
        self,
        hostname: str,
        port: int,
        timeout: float,
    ) -> SSLCheckResult:
        """Synchronous SSL check."""
        warnings = []

        try:
            # Create SSL context
            context = ssl.create_default_context()
            context.check_hostname = True
            context.verify_mode = ssl.CERT_REQUIRED

            # Connect
            with socket.create_connection((hostname, port), timeout=timeout) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    # Get certificate
                    cert = ssock.getpeercert()
                    cert_binary = ssock.getpeercert(binary_form=True)

                    # Parse certificate
                    cert_info = self._parse_certificate(cert, hostname, port)

                    # Get connection info
                    protocol_version = ssock.version()
                    cipher = ssock.cipher()
                    cipher_suite = f"{cipher[0]}:{cipher[1]}" if cipher else ""

                    # Check certificate chain
                    try:
                        chain = ssock.get_channel_binding()
                        cert_info.chain_length = 1  # Simplified
                    except Exception:
                        pass

                    # Determine status
                    status = CertificateStatus.VALID

                    if cert_info.is_expired:
                        status = CertificateStatus.EXPIRED
                    elif cert_info.is_self_signed:
                        status = CertificateStatus.SELF_SIGNED
                        warnings.append("Certificate is self-signed")
                    elif cert_info.days_until_expiry <= self.critical_days:
                        status = CertificateStatus.EXPIRING_SOON
                        warnings.append(f"Certificate expires in {cert_info.days_until_expiry} days (critical)")
                    elif cert_info.days_until_expiry <= self.warning_days:
                        status = CertificateStatus.EXPIRING_SOON
                        warnings.append(f"Certificate expires in {cert_info.days_until_expiry} days")

                    # Check for weak algorithms
                    if "sha1" in cert_info.signature_algorithm.lower():
                        warnings.append("Uses weak SHA-1 signature algorithm")

                    return SSLCheckResult(
                        hostname=hostname,
                        port=port,
                        status=status,
                        certificate=cert_info,
                        protocol_version=protocol_version,
                        cipher_suite=cipher_suite,
                        warnings=warnings,
                    )

        except ssl.SSLCertVerificationError as e:
            return SSLCheckResult(
                hostname=hostname,
                port=port,
                status=CertificateStatus.INVALID,
                error_message=f"Certificate verification failed: {e}",
            )

        except ssl.SSLError as e:
            return SSLCheckResult(
                hostname=hostname,
                port=port,
                status=CertificateStatus.INVALID,
                error_message=f"SSL error: {e}",
            )

        except socket.timeout:
            return SSLCheckResult(
                hostname=hostname,
                port=port,
                status=CertificateStatus.UNKNOWN,
                error_message="Connection timeout",
            )

        except socket.gaierror as e:
            return SSLCheckResult(
                hostname=hostname,
                port=port,
                status=CertificateStatus.UNKNOWN,
                error_message=f"DNS resolution failed: {e}",
            )

        except Exception as e:
            return SSLCheckResult(
                hostname=hostname,
                port=port,
                status=CertificateStatus.UNKNOWN,
                error_message=str(e),
            )

    def _parse_certificate(
        self,
        cert: Dict[str, Any],
        hostname: str,
        port: int,
    ) -> CertificateInfo:
        """Parse certificate dictionary."""
        # Parse subject
        subject = {}
        for item in cert.get("subject", ()):
            for key, value in item:
                subject[key] = value

        # Parse issuer
        issuer = {}
        for item in cert.get("issuer", ()):
            for key, value in item:
                issuer[key] = value

        # Parse dates
        not_before = None
        not_after = None

        if "notBefore" in cert:
            try:
                not_before = datetime.strptime(
                    cert["notBefore"],
                    "%b %d %H:%M:%S %Y %Z"
                )
            except ValueError:
                pass

        if "notAfter" in cert:
            try:
                not_after = datetime.strptime(
                    cert["notAfter"],
                    "%b %d %H:%M:%S %Y %Z"
                )
            except ValueError:
                pass

        # Parse SANs
        san = []
        for item in cert.get("subjectAltName", ()):
            if item[0] == "DNS":
                san.append(item[1])

        # Check if self-signed
        is_self_signed = subject == issuer

        return CertificateInfo(
            hostname=hostname,
            port=port,
            subject=subject,
            issuer=issuer,
            serial_number=str(cert.get("serialNumber", "")),
            version=cert.get("version", 0),
            not_before=not_before,
            not_after=not_after,
            san=san,
            is_self_signed=is_self_signed,
        )

    def _record_result(self, result: SSLCheckResult):
        """Record check result."""
        key = f"{result.hostname}:{result.port}"

        with self._lock:
            self._results[key] = result

        # Update metrics
        self.ssl_checks_total.labels(
            hostname=result.hostname,
            status=result.status.value,
        ).inc()

        self.ssl_check_latency.labels(
            hostname=result.hostname,
            port=str(result.port),
        ).set(result.latency_ms / 1000)

        if result.certificate:
            self.certificate_days_remaining.labels(
                hostname=result.hostname,
                port=str(result.port),
            ).set(result.certificate.days_until_expiry)

        status_value = 1 if result.status == CertificateStatus.VALID else 0
        self.ssl_check_status.labels(
            hostname=result.hostname,
            port=str(result.port),
        ).set(status_value)

        # Check for alerts
        if result.status in [CertificateStatus.EXPIRED, CertificateStatus.INVALID]:
            self._trigger_alert(result)
        elif result.status == CertificateStatus.EXPIRING_SOON:
            if result.certificate and result.certificate.days_until_expiry <= self.critical_days:
                self._trigger_alert(result)

        # Update expiring count
        self._update_expiring_count()

    def _trigger_alert(self, result: SSLCheckResult):
        """Trigger alert for certificate issue."""
        logger.warning(f"SSL alert for {result.hostname}:{result.port}: {result.status.value}")

        for callback in self._alert_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

    def _update_expiring_count(self):
        """Update count of expiring certificates."""
        with self._lock:
            results = list(self._results.values())

        for threshold in [7, 14, 30, 60, 90]:
            count = sum(
                1 for r in results
                if r.certificate and r.certificate.days_until_expiry <= threshold
            )
            self.certificates_expiring.labels(threshold_days=str(threshold)).set(count)

    def on_alert(self, callback: Callable[[SSLCheckResult], None]):
        """Register alert callback.

        Args:
            callback: Function to call on certificate alerts
        """
        self._alert_callbacks.append(callback)

    async def check_all(self) -> List[SSLCheckResult]:
        """Check all registered hosts.

        Returns:
            List of check results
        """
        import asyncio

        with self._lock:
            hosts = list(self._hosts.values())

        tasks = [
            self.check_ssl(h["hostname"], h["port"])
            for h in hosts
        ]

        return await asyncio.gather(*tasks)

    def get_expiring_certificates(
        self,
        days: int = 30,
    ) -> List[SSLCheckResult]:
        """Get certificates expiring within days.

        Args:
            days: Threshold in days

        Returns:
            List of expiring certificates
        """
        with self._lock:
            results = list(self._results.values())

        return [
            r for r in results
            if r.certificate and r.certificate.days_until_expiry <= days
        ]

    def get_invalid_certificates(self) -> List[SSLCheckResult]:
        """Get invalid or expired certificates.

        Returns:
            List of invalid certificates
        """
        with self._lock:
            results = list(self._results.values())

        return [
            r for r in results
            if r.status in [
                CertificateStatus.EXPIRED,
                CertificateStatus.INVALID,
                CertificateStatus.REVOKED,
            ]
        ]

    def get_result(
        self,
        hostname: str,
        port: int = 443,
    ) -> Optional[SSLCheckResult]:
        """Get latest result for host.

        Args:
            hostname: Hostname
            port: Port number

        Returns:
            SSLCheckResult or None
        """
        key = f"{hostname}:{port}"
        with self._lock:
            return self._results.get(key)

    def get_summary(self) -> Dict[str, Any]:
        """Get SSL monitoring summary.

        Returns:
            Summary dictionary
        """
        with self._lock:
            results = list(self._results.values())

        # Count by status
        by_status: Dict[str, int] = {}
        for r in results:
            s = r.status.value
            by_status[s] = by_status.get(s, 0) + 1

        # Expiring counts
        expiring_7 = sum(1 for r in results if r.certificate and r.certificate.days_until_expiry <= 7)
        expiring_30 = sum(1 for r in results if r.certificate and r.certificate.days_until_expiry <= 30)

        return {
            "total_hosts": len(self._hosts),
            "total_checks": len(results),
            "by_status": by_status,
            "valid_count": by_status.get("valid", 0),
            "invalid_count": by_status.get("invalid", 0) + by_status.get("expired", 0),
            "expiring_7_days": expiring_7,
            "expiring_30_days": expiring_30,
            "warning_threshold_days": self.warning_days,
            "critical_threshold_days": self.critical_days,
        }
