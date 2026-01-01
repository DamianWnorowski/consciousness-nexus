"""Endpoint Probes

HTTP/HTTPS endpoint health check probes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum
import logging

from .probes import SyntheticProbe, ProbeConfig, ProbeResult, ProbeType, ProbeStatus

logger = logging.getLogger(__name__)


class HTTPMethod(str, Enum):
    """HTTP methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


@dataclass
class HTTPProbeConfig(ProbeConfig):
    """HTTP probe configuration."""
    method: HTTPMethod = HTTPMethod.GET
    body: Optional[str] = None
    follow_redirects: bool = True
    verify_ssl: bool = True
    expected_status_codes: List[int] = field(default_factory=lambda: [200])
    username: Optional[str] = None
    password: Optional[str] = None
    bearer_token: Optional[str] = None


@dataclass
class HTTPProbeResult(ProbeResult):
    """HTTP probe result."""
    ssl_info: Optional[Dict[str, Any]] = None
    redirect_chain: List[str] = field(default_factory=list)
    dns_lookup_ms: float = 0.0
    tcp_connect_ms: float = 0.0
    ssl_handshake_ms: float = 0.0
    time_to_first_byte_ms: float = 0.0
    content_transfer_ms: float = 0.0


@dataclass
class HealthCheckResult:
    """Health check aggregated result."""
    endpoint: str
    healthy: bool
    status: ProbeStatus
    latency_ms: float
    checks_passed: int
    checks_failed: int
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)


class EndpointProbe(SyntheticProbe):
    """HTTP/HTTPS endpoint probe.

    Usage:
        probe = EndpointProbe()

        config = HTTPProbeConfig(
            probe_id="api-health",
            name="API Health Check",
            probe_type=ProbeType.HTTPS,
            target="https://api.example.com/health",
            method=HTTPMethod.GET,
            expected_status_codes=[200],
        )

        result = await probe.run(config)
    """

    def __init__(self, namespace: str = "consciousness"):
        super().__init__(namespace)
        self._session = None

    async def execute(self, config: ProbeConfig) -> ProbeResult:
        """Execute HTTP probe.

        Args:
            config: Probe configuration

        Returns:
            HTTPProbeResult
        """
        import time

        # Handle both ProbeConfig and HTTPProbeConfig
        method = HTTPMethod.GET
        body = None
        follow_redirects = True
        verify_ssl = True
        expected_codes = [200]
        username = None
        password = None
        bearer_token = None

        if isinstance(config, HTTPProbeConfig):
            method = config.method
            body = config.body
            follow_redirects = config.follow_redirects
            verify_ssl = config.verify_ssl
            expected_codes = config.expected_status_codes
            username = config.username
            password = config.password
            bearer_token = config.bearer_token

        start_time = time.time()
        timing = {
            "dns_lookup_ms": 0.0,
            "tcp_connect_ms": 0.0,
            "ssl_handshake_ms": 0.0,
            "time_to_first_byte_ms": 0.0,
            "content_transfer_ms": 0.0,
        }

        try:
            import httpx

            headers = dict(config.headers)
            headers.setdefault("User-Agent", f"Consciousness-Synthetic-Probe/1.0")

            if bearer_token:
                headers["Authorization"] = f"Bearer {bearer_token}"

            auth = None
            if username and password:
                auth = httpx.BasicAuth(username, password)

            async with httpx.AsyncClient(
                verify=verify_ssl,
                follow_redirects=follow_redirects,
                timeout=httpx.Timeout(config.timeout_seconds),
                auth=auth,
            ) as client:
                # Make request
                request_start = time.time()

                if method == HTTPMethod.GET:
                    response = await client.get(config.target, headers=headers)
                elif method == HTTPMethod.POST:
                    response = await client.post(config.target, headers=headers, content=body)
                elif method == HTTPMethod.PUT:
                    response = await client.put(config.target, headers=headers, content=body)
                elif method == HTTPMethod.PATCH:
                    response = await client.patch(config.target, headers=headers, content=body)
                elif method == HTTPMethod.DELETE:
                    response = await client.delete(config.target, headers=headers)
                elif method == HTTPMethod.HEAD:
                    response = await client.head(config.target, headers=headers)
                elif method == HTTPMethod.OPTIONS:
                    response = await client.options(config.target, headers=headers)
                else:
                    response = await client.get(config.target, headers=headers)

                request_end = time.time()
                timing["time_to_first_byte_ms"] = (request_end - request_start) * 1000

                # Calculate status
                status = ProbeStatus.SUCCESS
                if response.status_code not in expected_codes:
                    status = ProbeStatus.FAILURE

                # Get redirect chain
                redirect_chain = []
                if hasattr(response, "history") and response.history:
                    redirect_chain = [str(r.url) for r in response.history]

                # Get SSL info
                ssl_info = None
                if config.target.startswith("https://"):
                    ssl_info = self._extract_ssl_info(response)

                return HTTPProbeResult(
                    probe_id=config.probe_id,
                    probe_name=config.name,
                    probe_type=config.probe_type,
                    status=status,
                    target=config.target,
                    latency_ms=(time.time() - start_time) * 1000,
                    response_code=response.status_code,
                    response_body=response.text[:10000] if response.text else None,
                    response_headers=dict(response.headers),
                    ssl_info=ssl_info,
                    redirect_chain=redirect_chain,
                    dns_lookup_ms=timing["dns_lookup_ms"],
                    tcp_connect_ms=timing["tcp_connect_ms"],
                    ssl_handshake_ms=timing["ssl_handshake_ms"],
                    time_to_first_byte_ms=timing["time_to_first_byte_ms"],
                    content_transfer_ms=timing["content_transfer_ms"],
                )

        except Exception as e:
            error_type = type(e).__name__

            if "timeout" in str(e).lower() or "Timeout" in error_type:
                status = ProbeStatus.TIMEOUT
            else:
                status = ProbeStatus.ERROR

            return HTTPProbeResult(
                probe_id=config.probe_id,
                probe_name=config.name,
                probe_type=config.probe_type,
                status=status,
                target=config.target,
                latency_ms=(time.time() - start_time) * 1000,
                error_message=str(e),
            )

    def _extract_ssl_info(self, response) -> Optional[Dict[str, Any]]:
        """Extract SSL certificate info from response."""
        try:
            # This is a simplified version
            # Full SSL extraction requires access to socket-level info
            return {
                "verified": True,
                "protocol": "TLS",
            }
        except Exception:
            return None

    async def health_check(
        self,
        endpoints: List[str],
        timeout_seconds: float = 10.0,
    ) -> Dict[str, HealthCheckResult]:
        """Run health checks on multiple endpoints.

        Args:
            endpoints: List of endpoint URLs
            timeout_seconds: Timeout per endpoint

        Returns:
            Dict of endpoint -> HealthCheckResult
        """
        import asyncio

        results: Dict[str, HealthCheckResult] = {}

        async def check_endpoint(url: str) -> tuple:
            config = HTTPProbeConfig(
                probe_id=f"health-{hash(url)}",
                name="Health Check",
                probe_type=ProbeType.HTTPS if url.startswith("https") else ProbeType.HTTP,
                target=url,
                timeout_seconds=timeout_seconds,
            )

            result = await self.run(config)

            return url, HealthCheckResult(
                endpoint=url,
                healthy=result.is_success,
                status=result.status,
                latency_ms=result.latency_ms,
                checks_passed=result.assertions_passed,
                checks_failed=result.assertions_failed,
                details={
                    "status_code": result.response_code,
                    "error": result.error_message,
                },
            )

        tasks = [check_endpoint(url) for url in endpoints]
        check_results = await asyncio.gather(*tasks, return_exceptions=True)

        for item in check_results:
            if isinstance(item, tuple):
                url, health_result = item
                results[url] = health_result
            elif isinstance(item, Exception):
                logger.error(f"Health check error: {item}")

        return results

    async def check_endpoint_sequence(
        self,
        endpoints: List[str],
        stop_on_failure: bool = False,
    ) -> List[HealthCheckResult]:
        """Check endpoints in sequence.

        Args:
            endpoints: List of endpoint URLs
            stop_on_failure: Stop on first failure

        Returns:
            List of health check results
        """
        results = []

        for url in endpoints:
            config = HTTPProbeConfig(
                probe_id=f"seq-{hash(url)}",
                name="Sequential Check",
                probe_type=ProbeType.HTTPS if url.startswith("https") else ProbeType.HTTP,
                target=url,
            )

            probe_result = await self.run(config)

            health_result = HealthCheckResult(
                endpoint=url,
                healthy=probe_result.is_success,
                status=probe_result.status,
                latency_ms=probe_result.latency_ms,
                checks_passed=probe_result.assertions_passed,
                checks_failed=probe_result.assertions_failed,
            )

            results.append(health_result)

            if stop_on_failure and not health_result.healthy:
                break

        return results
