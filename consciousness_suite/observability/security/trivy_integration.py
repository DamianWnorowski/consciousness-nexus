"""Trivy Integration

Integration with Trivy security scanner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum
import subprocess
import json
import logging
import asyncio

from prometheus_client import Gauge, Counter, Histogram

logger = logging.getLogger(__name__)


class TrivySeverity(str, Enum):
    """Trivy severity levels."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNKNOWN = "UNKNOWN"


class TrivyClass(str, Enum):
    """Trivy finding classes."""
    OS_PKG_VULN = "os-pkgvuln"
    LANG_PKG_VULN = "lang-pkgvuln"
    CONFIG = "config"
    SECRET = "secret"
    LICENSE = "license"


@dataclass
class TrivyVulnerability:
    """A vulnerability found by Trivy."""
    vuln_id: str
    pkg_name: str
    installed_version: str
    severity: TrivySeverity
    title: str = ""
    description: str = ""
    fixed_version: Optional[str] = None
    primary_url: Optional[str] = None
    data_source: Optional[str] = None
    cvss_score: Optional[float] = None
    cwe_ids: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)


@dataclass
class TrivyMisconfiguration:
    """A misconfiguration found by Trivy."""
    misconfig_id: str
    avd_id: str
    title: str
    description: str
    severity: TrivySeverity
    resolution: str = ""
    status: str = "FAIL"
    resource: str = ""
    provider: str = ""
    service: str = ""
    cause_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrivySecret:
    """A secret found by Trivy."""
    rule_id: str
    category: str
    severity: TrivySeverity
    title: str
    start_line: int = 0
    end_line: int = 0
    match: str = ""
    code: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrivyResult:
    """Result of a Trivy scan."""
    target: str
    target_type: str
    scan_time: datetime = field(default_factory=datetime.now)
    duration_seconds: float = 0
    vulnerabilities: List[TrivyVulnerability] = field(default_factory=list)
    misconfigurations: List[TrivyMisconfiguration] = field(default_factory=list)
    secrets: List[TrivySecret] = field(default_factory=list)
    artifact_name: str = ""
    artifact_type: str = ""
    os_family: str = ""
    os_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def is_success(self) -> bool:
        return self.error is None

    @property
    def total_findings(self) -> int:
        return len(self.vulnerabilities) + len(self.misconfigurations) + len(self.secrets)

    @property
    def critical_vulns(self) -> int:
        return sum(1 for v in self.vulnerabilities if v.severity == TrivySeverity.CRITICAL)

    @property
    def high_vulns(self) -> int:
        return sum(1 for v in self.vulnerabilities if v.severity == TrivySeverity.HIGH)

    @property
    def has_critical(self) -> bool:
        return self.critical_vulns > 0

    @property
    def has_secrets(self) -> bool:
        return len(self.secrets) > 0

    def get_vulns_by_severity(self, severity: TrivySeverity) -> List[TrivyVulnerability]:
        """Get vulnerabilities by severity."""
        return [v for v in self.vulnerabilities if v.severity == severity]

    def get_fixable_vulns(self) -> List[TrivyVulnerability]:
        """Get vulnerabilities with fixes available."""
        return [v for v in self.vulnerabilities if v.fixed_version]

    def to_summary(self) -> Dict[str, Any]:
        """Get result summary."""
        return {
            "target": self.target,
            "target_type": self.target_type,
            "scan_time": self.scan_time.isoformat(),
            "duration_seconds": self.duration_seconds,
            "vulnerabilities": {
                "total": len(self.vulnerabilities),
                "critical": self.critical_vulns,
                "high": self.high_vulns,
                "fixable": len(self.get_fixable_vulns()),
            },
            "misconfigurations": len(self.misconfigurations),
            "secrets": len(self.secrets),
            "os": f"{self.os_family} {self.os_name}",
            "success": self.is_success,
            "error": self.error,
        }


class TrivyScanner:
    """Trivy security scanner integration.

    Usage:
        scanner = TrivyScanner()

        # Scan an image
        result = await scanner.scan_image("nginx:latest")

        # Check results
        if result.has_critical:
            print(f"Found {result.critical_vulns} critical vulnerabilities!")

        # Scan filesystem
        result = await scanner.scan_filesystem("/app")

        # Scan config files
        result = await scanner.scan_config("./k8s")
    """

    def __init__(
        self,
        namespace: str = "consciousness",
        trivy_path: str = "trivy",
        timeout_seconds: int = 300,
        cache_dir: Optional[str] = None,
    ):
        self.namespace = namespace
        self.trivy_path = trivy_path
        self.timeout_seconds = timeout_seconds
        self.cache_dir = cache_dir

        self._scan_history: List[TrivyResult] = []

        # Prometheus metrics
        self.trivy_scans = Counter(
            f"{namespace}_trivy_scans_total",
            "Total Trivy scans",
            ["target_type", "result"],
        )

        self.trivy_vulns = Gauge(
            f"{namespace}_trivy_vulnerabilities",
            "Vulnerabilities found",
            ["target", "severity"],
        )

        self.trivy_misconfigs = Gauge(
            f"{namespace}_trivy_misconfigurations",
            "Misconfigurations found",
            ["target", "severity"],
        )

        self.trivy_secrets = Gauge(
            f"{namespace}_trivy_secrets",
            "Secrets found",
            ["target"],
        )

        self.trivy_duration = Histogram(
            f"{namespace}_trivy_scan_duration_seconds",
            "Scan duration",
            ["target_type"],
            buckets=[5, 10, 30, 60, 120, 300, 600],
        )

    def _build_command(
        self,
        target: str,
        target_type: str,
        extra_args: Optional[List[str]] = None,
    ) -> List[str]:
        """Build Trivy command."""
        cmd = [self.trivy_path]

        if target_type == "image":
            cmd.extend(["image", "--format", "json"])
        elif target_type == "filesystem":
            cmd.extend(["fs", "--format", "json"])
        elif target_type == "config":
            cmd.extend(["config", "--format", "json"])
        elif target_type == "sbom":
            cmd.extend(["sbom", "--format", "json"])
        elif target_type == "repo":
            cmd.extend(["repo", "--format", "json"])

        # Add security checks
        cmd.extend(["--scanners", "vuln,misconfig,secret"])

        # Add severity filter
        cmd.extend(["--severity", "CRITICAL,HIGH,MEDIUM,LOW"])

        if self.cache_dir:
            cmd.extend(["--cache-dir", self.cache_dir])

        if extra_args:
            cmd.extend(extra_args)

        cmd.append(target)

        return cmd

    async def _run_trivy(
        self,
        target: str,
        target_type: str,
        extra_args: Optional[List[str]] = None,
    ) -> TrivyResult:
        """Run Trivy and parse results."""
        start_time = datetime.now()

        cmd = self._build_command(target, target_type, extra_args)
        logger.info(f"Running: {' '.join(cmd)}")

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout_seconds,
            )

            duration = (datetime.now() - start_time).total_seconds()

            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                logger.error(f"Trivy failed: {error_msg}")
                return TrivyResult(
                    target=target,
                    target_type=target_type,
                    duration_seconds=duration,
                    error=error_msg,
                )

            # Parse JSON output
            result = self._parse_output(stdout.decode(), target, target_type)
            result.duration_seconds = duration

            return result

        except asyncio.TimeoutError:
            duration = (datetime.now() - start_time).total_seconds()
            return TrivyResult(
                target=target,
                target_type=target_type,
                duration_seconds=duration,
                error=f"Scan timed out after {self.timeout_seconds}s",
            )

        except FileNotFoundError:
            return TrivyResult(
                target=target,
                target_type=target_type,
                error=f"Trivy not found at {self.trivy_path}",
            )

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            return TrivyResult(
                target=target,
                target_type=target_type,
                duration_seconds=duration,
                error=str(e),
            )

    def _parse_output(
        self,
        output: str,
        target: str,
        target_type: str,
    ) -> TrivyResult:
        """Parse Trivy JSON output."""
        try:
            data = json.loads(output)
        except json.JSONDecodeError as e:
            return TrivyResult(
                target=target,
                target_type=target_type,
                error=f"Failed to parse output: {e}",
            )

        result = TrivyResult(
            target=target,
            target_type=target_type,
            artifact_name=data.get("ArtifactName", ""),
            artifact_type=data.get("ArtifactType", ""),
            metadata=data.get("Metadata", {}),
        )

        # Parse OS info
        if "Metadata" in data:
            os_info = data["Metadata"].get("OS", {})
            result.os_family = os_info.get("Family", "")
            result.os_name = os_info.get("Name", "")

        # Parse results
        for target_result in data.get("Results", []):
            # Vulnerabilities
            for vuln in target_result.get("Vulnerabilities", []):
                result.vulnerabilities.append(TrivyVulnerability(
                    vuln_id=vuln.get("VulnerabilityID", ""),
                    pkg_name=vuln.get("PkgName", ""),
                    installed_version=vuln.get("InstalledVersion", ""),
                    severity=TrivySeverity(vuln.get("Severity", "UNKNOWN")),
                    title=vuln.get("Title", ""),
                    description=vuln.get("Description", ""),
                    fixed_version=vuln.get("FixedVersion"),
                    primary_url=vuln.get("PrimaryURL"),
                    data_source=vuln.get("DataSource", {}).get("Name"),
                    cvss_score=self._extract_cvss_score(vuln),
                    cwe_ids=vuln.get("CweIDs", []),
                    references=vuln.get("References", []),
                ))

            # Misconfigurations
            for misconfig in target_result.get("Misconfigurations", []):
                result.misconfigurations.append(TrivyMisconfiguration(
                    misconfig_id=misconfig.get("ID", ""),
                    avd_id=misconfig.get("AVDID", ""),
                    title=misconfig.get("Title", ""),
                    description=misconfig.get("Description", ""),
                    severity=TrivySeverity(misconfig.get("Severity", "UNKNOWN")),
                    resolution=misconfig.get("Resolution", ""),
                    status=misconfig.get("Status", "FAIL"),
                    resource=misconfig.get("CauseMetadata", {}).get("Resource", ""),
                    provider=misconfig.get("CauseMetadata", {}).get("Provider", ""),
                    service=misconfig.get("CauseMetadata", {}).get("Service", ""),
                    cause_metadata=misconfig.get("CauseMetadata", {}),
                ))

            # Secrets
            for secret in target_result.get("Secrets", []):
                result.secrets.append(TrivySecret(
                    rule_id=secret.get("RuleID", ""),
                    category=secret.get("Category", ""),
                    severity=TrivySeverity(secret.get("Severity", "UNKNOWN")),
                    title=secret.get("Title", ""),
                    start_line=secret.get("StartLine", 0),
                    end_line=secret.get("EndLine", 0),
                    match=secret.get("Match", ""),
                    code=secret.get("Code", {}),
                ))

        return result

    def _extract_cvss_score(self, vuln: Dict[str, Any]) -> Optional[float]:
        """Extract CVSS score from vulnerability data."""
        cvss = vuln.get("CVSS", {})

        # Try CVSS v3 first
        for source in cvss.values():
            if isinstance(source, dict):
                if "V3Score" in source:
                    return source["V3Score"]
                if "V2Score" in source:
                    return source["V2Score"]

        return None

    async def scan_image(
        self,
        image: str,
        ignore_unfixed: bool = False,
        platform: Optional[str] = None,
    ) -> TrivyResult:
        """Scan a container image.

        Args:
            image: Image name (e.g., nginx:latest)
            ignore_unfixed: Ignore unfixed vulnerabilities
            platform: Target platform (e.g., linux/amd64)

        Returns:
            TrivyResult
        """
        extra_args = []

        if ignore_unfixed:
            extra_args.append("--ignore-unfixed")

        if platform:
            extra_args.extend(["--platform", platform])

        result = await self._run_trivy(image, "image", extra_args)
        self._record_result(result)

        return result

    async def scan_filesystem(
        self,
        path: str,
    ) -> TrivyResult:
        """Scan a filesystem path.

        Args:
            path: Filesystem path

        Returns:
            TrivyResult
        """
        result = await self._run_trivy(path, "filesystem")
        self._record_result(result)

        return result

    async def scan_config(
        self,
        path: str,
    ) -> TrivyResult:
        """Scan configuration files (IaC).

        Args:
            path: Path to config files

        Returns:
            TrivyResult
        """
        result = await self._run_trivy(path, "config")
        self._record_result(result)

        return result

    async def scan_sbom(
        self,
        sbom_path: str,
    ) -> TrivyResult:
        """Scan an SBOM file.

        Args:
            sbom_path: Path to SBOM

        Returns:
            TrivyResult
        """
        result = await self._run_trivy(sbom_path, "sbom")
        self._record_result(result)

        return result

    async def scan_repo(
        self,
        repo_url: str,
    ) -> TrivyResult:
        """Scan a git repository.

        Args:
            repo_url: Repository URL

        Returns:
            TrivyResult
        """
        result = await self._run_trivy(repo_url, "repo")
        self._record_result(result)

        return result

    def _record_result(self, result: TrivyResult):
        """Record scan result and update metrics."""
        self._scan_history.append(result)

        # Trim history
        if len(self._scan_history) > 1000:
            self._scan_history = self._scan_history[-500:]

        # Update metrics
        status = "success" if result.is_success else "error"
        self.trivy_scans.labels(
            target_type=result.target_type,
            result=status,
        ).inc()

        self.trivy_duration.labels(
            target_type=result.target_type,
        ).observe(result.duration_seconds)

        target_short = result.target[:50]

        for severity in TrivySeverity:
            vuln_count = len(result.get_vulns_by_severity(severity))
            self.trivy_vulns.labels(
                target=target_short,
                severity=severity.value,
            ).set(vuln_count)

            misconfig_count = sum(
                1 for m in result.misconfigurations
                if m.severity == severity
            )
            self.trivy_misconfigs.labels(
                target=target_short,
                severity=severity.value,
            ).set(misconfig_count)

        self.trivy_secrets.labels(target=target_short).set(len(result.secrets))

    def get_scan_history(
        self,
        limit: int = 100,
    ) -> List[TrivyResult]:
        """Get scan history.

        Args:
            limit: Maximum results

        Returns:
            List of results
        """
        results = list(self._scan_history)
        results.sort(key=lambda r: r.scan_time, reverse=True)
        return results[:limit]

    def get_summary(self) -> Dict[str, Any]:
        """Get scanner summary.

        Returns:
            Summary dictionary
        """
        total_scans = len(self._scan_history)
        successful = sum(1 for r in self._scan_history if r.is_success)

        all_vulns = []
        all_misconfigs = []
        all_secrets = []

        for result in self._scan_history:
            all_vulns.extend(result.vulnerabilities)
            all_misconfigs.extend(result.misconfigurations)
            all_secrets.extend(result.secrets)

        return {
            "total_scans": total_scans,
            "successful_scans": successful,
            "failed_scans": total_scans - successful,
            "total_vulnerabilities": len(all_vulns),
            "critical_vulnerabilities": sum(1 for v in all_vulns if v.severity == TrivySeverity.CRITICAL),
            "high_vulnerabilities": sum(1 for v in all_vulns if v.severity == TrivySeverity.HIGH),
            "total_misconfigurations": len(all_misconfigs),
            "total_secrets": len(all_secrets),
            "unique_cves": len(set(v.vuln_id for v in all_vulns)),
        }
