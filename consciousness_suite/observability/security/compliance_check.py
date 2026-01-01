"""Compliance Checker

Compliance verification against security frameworks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
from enum import Enum
import threading
import logging

from prometheus_client import Gauge, Counter

logger = logging.getLogger(__name__)


class ComplianceFramework(str, Enum):
    """Compliance frameworks."""
    CIS_DOCKER = "cis_docker"
    CIS_KUBERNETES = "cis_kubernetes"
    NIST_800_53 = "nist_800_53"
    PCI_DSS = "pci_dss"
    HIPAA = "hipaa"
    SOC2 = "soc2"
    GDPR = "gdpr"
    ISO_27001 = "iso_27001"
    CUSTOM = "custom"


class PolicySeverity(str, Enum):
    """Policy violation severity."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class PolicyStatus(str, Enum):
    """Policy check status."""
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"
    ERROR = "error"


@dataclass
class CompliancePolicy:
    """A compliance policy definition."""
    policy_id: str
    title: str
    description: str
    framework: ComplianceFramework
    severity: PolicySeverity
    category: str = ""
    subcategory: str = ""
    remediation: str = ""
    references: List[str] = field(default_factory=list)
    check_function: Optional[Callable[[Dict[str, Any]], PolicyStatus]] = None
    enabled: bool = True
    tags: List[str] = field(default_factory=list)


@dataclass
class PolicyViolation:
    """A policy violation finding."""
    policy: CompliancePolicy
    status: PolicyStatus
    resource: str
    resource_type: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    evidence: Dict[str, Any] = field(default_factory=dict)
    remediation_steps: List[str] = field(default_factory=list)


@dataclass
class ComplianceResult:
    """Result of a compliance check."""
    framework: ComplianceFramework
    target: str
    scan_time: datetime = field(default_factory=datetime.now)
    duration_seconds: float = 0
    violations: List[PolicyViolation] = field(default_factory=list)
    passed_policies: List[str] = field(default_factory=list)
    skipped_policies: List[str] = field(default_factory=list)
    total_policies: int = 0
    error: Optional[str] = None

    @property
    def is_compliant(self) -> bool:
        """Check if fully compliant (no fails or criticals)."""
        critical_fails = [
            v for v in self.violations
            if v.status == PolicyStatus.FAIL and v.policy.severity == PolicySeverity.CRITICAL
        ]
        return len(critical_fails) == 0

    @property
    def compliance_score(self) -> float:
        """Calculate compliance score (0-100)."""
        if self.total_policies == 0:
            return 100.0

        passed = len(self.passed_policies)
        failed = sum(1 for v in self.violations if v.status == PolicyStatus.FAIL)

        return (passed / (passed + failed)) * 100 if (passed + failed) > 0 else 100.0

    @property
    def critical_violations(self) -> List[PolicyViolation]:
        return [v for v in self.violations if v.policy.severity == PolicySeverity.CRITICAL]

    @property
    def high_violations(self) -> List[PolicyViolation]:
        return [v for v in self.violations if v.policy.severity == PolicySeverity.HIGH]

    def to_summary(self) -> Dict[str, Any]:
        """Get result summary."""
        by_severity: Dict[str, int] = {}
        for severity in PolicySeverity:
            count = sum(1 for v in self.violations if v.policy.severity == severity)
            by_severity[severity.value] = count

        return {
            "framework": self.framework.value,
            "target": self.target,
            "scan_time": self.scan_time.isoformat(),
            "is_compliant": self.is_compliant,
            "compliance_score": self.compliance_score,
            "total_policies": self.total_policies,
            "passed": len(self.passed_policies),
            "failed": len(self.violations),
            "skipped": len(self.skipped_policies),
            "by_severity": by_severity,
        }


class ComplianceChecker:
    """Checks compliance against security frameworks.

    Usage:
        checker = ComplianceChecker()

        # Add policies
        checker.add_policy(CompliancePolicy(
            policy_id="CIS-4.1",
            title="Ensure container images are updated",
            description="Container images should be regularly updated",
            framework=ComplianceFramework.CIS_DOCKER,
            severity=PolicySeverity.MEDIUM,
        ))

        # Check compliance
        result = await checker.check_compliance(
            framework=ComplianceFramework.CIS_DOCKER,
            target="production-cluster",
            context={"images": image_list},
        )

        # Get report
        report = checker.generate_report(result)
    """

    def __init__(self, namespace: str = "consciousness"):
        self.namespace = namespace

        self._policies: Dict[str, CompliancePolicy] = {}
        self._results_history: List[ComplianceResult] = []
        self._lock = threading.Lock()

        self._violation_callbacks: List[Callable[[PolicyViolation], None]] = []

        # Initialize with some default policies
        self._init_default_policies()

        # Prometheus metrics
        self.compliance_score = Gauge(
            f"{namespace}_compliance_score",
            "Compliance score (0-100)",
            ["framework", "target"],
        )

        self.compliance_violations = Gauge(
            f"{namespace}_compliance_violations",
            "Compliance violations",
            ["framework", "severity"],
        )

        self.compliance_checks = Counter(
            f"{namespace}_compliance_checks_total",
            "Total compliance checks",
            ["framework", "result"],
        )

        self.policy_status = Gauge(
            f"{namespace}_compliance_policy_status",
            "Policy status (1=pass, 0=fail)",
            ["policy_id", "framework"],
        )

    def _init_default_policies(self):
        """Initialize default compliance policies."""
        # CIS Docker Benchmark policies
        docker_policies = [
            CompliancePolicy(
                policy_id="CIS-DOCKER-4.1",
                title="Ensure container images are updated",
                description="Container images should use the latest base images",
                framework=ComplianceFramework.CIS_DOCKER,
                severity=PolicySeverity.MEDIUM,
                category="Container Images",
                remediation="Update container base images regularly",
            ),
            CompliancePolicy(
                policy_id="CIS-DOCKER-4.2",
                title="Ensure containers use only trusted base images",
                description="Only use official or verified images",
                framework=ComplianceFramework.CIS_DOCKER,
                severity=PolicySeverity.HIGH,
                category="Container Images",
                remediation="Use images from trusted registries only",
            ),
            CompliancePolicy(
                policy_id="CIS-DOCKER-5.1",
                title="Ensure container is not running as root",
                description="Containers should run as non-root user",
                framework=ComplianceFramework.CIS_DOCKER,
                severity=PolicySeverity.HIGH,
                category="Container Runtime",
                remediation="Add USER directive to Dockerfile",
            ),
            CompliancePolicy(
                policy_id="CIS-DOCKER-5.7",
                title="Ensure privileged containers are not used",
                description="Do not run containers in privileged mode",
                framework=ComplianceFramework.CIS_DOCKER,
                severity=PolicySeverity.CRITICAL,
                category="Container Runtime",
                remediation="Remove --privileged flag from container runs",
            ),
            CompliancePolicy(
                policy_id="CIS-DOCKER-5.12",
                title="Ensure container's root filesystem is mounted read-only",
                description="Container root filesystem should be read-only",
                framework=ComplianceFramework.CIS_DOCKER,
                severity=PolicySeverity.MEDIUM,
                category="Container Runtime",
                remediation="Use --read-only flag when running containers",
            ),
        ]

        # CIS Kubernetes policies
        k8s_policies = [
            CompliancePolicy(
                policy_id="CIS-K8S-1.1.1",
                title="Ensure API server pod specification file permissions",
                description="API server pod spec file should have restricted permissions",
                framework=ComplianceFramework.CIS_KUBERNETES,
                severity=PolicySeverity.HIGH,
                category="Control Plane",
                remediation="Set file permissions to 644 or more restrictive",
            ),
            CompliancePolicy(
                policy_id="CIS-K8S-5.1.1",
                title="Ensure cluster-admin role is only used where required",
                description="Minimize use of cluster-admin role",
                framework=ComplianceFramework.CIS_KUBERNETES,
                severity=PolicySeverity.HIGH,
                category="RBAC and Service Accounts",
                remediation="Review and minimize cluster-admin bindings",
            ),
            CompliancePolicy(
                policy_id="CIS-K8S-5.2.1",
                title="Minimize the admission of privileged containers",
                description="Pods should not use privileged containers",
                framework=ComplianceFramework.CIS_KUBERNETES,
                severity=PolicySeverity.CRITICAL,
                category="Pod Security Policies",
                remediation="Use Pod Security Policies to restrict privileged containers",
            ),
        ]

        # SOC2 policies
        soc2_policies = [
            CompliancePolicy(
                policy_id="SOC2-CC6.1",
                title="Logical access security",
                description="Restrict logical access to information assets",
                framework=ComplianceFramework.SOC2,
                severity=PolicySeverity.HIGH,
                category="Security",
                remediation="Implement RBAC and access controls",
            ),
            CompliancePolicy(
                policy_id="SOC2-CC7.1",
                title="Incident management",
                description="Monitor systems for security incidents",
                framework=ComplianceFramework.SOC2,
                severity=PolicySeverity.HIGH,
                category="Operations",
                remediation="Implement security monitoring and alerting",
            ),
            CompliancePolicy(
                policy_id="SOC2-CC8.1",
                title="Change management",
                description="Manage changes to systems properly",
                framework=ComplianceFramework.SOC2,
                severity=PolicySeverity.MEDIUM,
                category="Operations",
                remediation="Implement change management process",
            ),
        ]

        for policy in docker_policies + k8s_policies + soc2_policies:
            self._policies[policy.policy_id] = policy

    def add_policy(self, policy: CompliancePolicy):
        """Add or update a compliance policy.

        Args:
            policy: Policy to add
        """
        with self._lock:
            self._policies[policy.policy_id] = policy

        logger.info(f"Added policy: {policy.policy_id}")

    def remove_policy(self, policy_id: str):
        """Remove a policy.

        Args:
            policy_id: Policy ID to remove
        """
        with self._lock:
            self._policies.pop(policy_id, None)

    def get_policies(
        self,
        framework: Optional[ComplianceFramework] = None,
    ) -> List[CompliancePolicy]:
        """Get policies, optionally filtered by framework.

        Args:
            framework: Framework to filter by

        Returns:
            List of policies
        """
        with self._lock:
            policies = list(self._policies.values())

        if framework:
            policies = [p for p in policies if p.framework == framework]

        return policies

    def on_violation(self, callback: Callable[[PolicyViolation], None]):
        """Register callback for policy violations.

        Args:
            callback: Function to call with PolicyViolation
        """
        self._violation_callbacks.append(callback)

    async def check_compliance(
        self,
        framework: ComplianceFramework,
        target: str,
        context: Dict[str, Any],
    ) -> ComplianceResult:
        """Check compliance against a framework.

        Args:
            framework: Framework to check
            target: Target being checked
            context: Context data for checks

        Returns:
            ComplianceResult
        """
        start_time = datetime.now()

        policies = self.get_policies(framework)
        if not policies:
            return ComplianceResult(
                framework=framework,
                target=target,
                error=f"No policies defined for {framework.value}",
            )

        result = ComplianceResult(
            framework=framework,
            target=target,
            total_policies=len(policies),
        )

        for policy in policies:
            if not policy.enabled:
                result.skipped_policies.append(policy.policy_id)
                continue

            try:
                if policy.check_function:
                    status = policy.check_function(context)
                else:
                    # Default: simulate pass (real implementation would check)
                    status = PolicyStatus.PASS

                if status == PolicyStatus.PASS:
                    result.passed_policies.append(policy.policy_id)
                    self.policy_status.labels(
                        policy_id=policy.policy_id,
                        framework=framework.value,
                    ).set(1)
                elif status in [PolicyStatus.FAIL, PolicyStatus.WARN]:
                    violation = PolicyViolation(
                        policy=policy,
                        status=status,
                        resource=target,
                        resource_type="unknown",
                        message=f"Policy {policy.policy_id} failed",
                    )
                    result.violations.append(violation)

                    self.policy_status.labels(
                        policy_id=policy.policy_id,
                        framework=framework.value,
                    ).set(0)

                    # Trigger callbacks
                    for callback in self._violation_callbacks:
                        try:
                            callback(violation)
                        except Exception as e:
                            logger.error(f"Violation callback error: {e}")

                else:
                    result.skipped_policies.append(policy.policy_id)

            except Exception as e:
                logger.error(f"Policy check error for {policy.policy_id}: {e}")
                result.skipped_policies.append(policy.policy_id)

        result.duration_seconds = (datetime.now() - start_time).total_seconds()

        # Store result
        with self._lock:
            self._results_history.append(result)
            if len(self._results_history) > 1000:
                self._results_history = self._results_history[-500:]

        # Update metrics
        self.compliance_score.labels(
            framework=framework.value,
            target=target[:50],
        ).set(result.compliance_score)

        for severity in PolicySeverity:
            count = sum(1 for v in result.violations if v.policy.severity == severity)
            self.compliance_violations.labels(
                framework=framework.value,
                severity=severity.value,
            ).set(count)

        status = "compliant" if result.is_compliant else "non_compliant"
        self.compliance_checks.labels(
            framework=framework.value,
            result=status,
        ).inc()

        return result

    def generate_report(
        self,
        result: ComplianceResult,
        format: str = "summary",
    ) -> Dict[str, Any]:
        """Generate compliance report.

        Args:
            result: Compliance result
            format: Report format (summary, detailed, executive)

        Returns:
            Report dictionary
        """
        report = {
            "report_type": "compliance_report",
            "generated_at": datetime.now().isoformat(),
            "framework": result.framework.value,
            "target": result.target,
            "scan_time": result.scan_time.isoformat(),
            "is_compliant": result.is_compliant,
            "compliance_score": result.compliance_score,
            "summary": result.to_summary(),
        }

        if format in ["detailed", "executive"]:
            # Add violation details
            report["violations"] = [
                {
                    "policy_id": v.policy.policy_id,
                    "title": v.policy.title,
                    "severity": v.policy.severity.value,
                    "status": v.status.value,
                    "resource": v.resource,
                    "message": v.message,
                    "remediation": v.policy.remediation,
                }
                for v in result.violations
            ]

        if format == "detailed":
            # Add all policies
            report["all_policies"] = [
                {
                    "policy_id": p.policy_id,
                    "title": p.title,
                    "severity": p.severity.value,
                    "status": "passed" if p.policy_id in result.passed_policies
                             else "failed" if any(v.policy.policy_id == p.policy_id for v in result.violations)
                             else "skipped",
                }
                for p in self.get_policies(result.framework)
            ]

        if format == "executive":
            # Add executive summary
            report["executive_summary"] = {
                "overall_status": "Compliant" if result.is_compliant else "Non-Compliant",
                "score_grade": self._score_to_grade(result.compliance_score),
                "critical_issues": len(result.critical_violations),
                "high_priority_issues": len(result.high_violations),
                "key_findings": [
                    v.policy.title for v in result.critical_violations[:5]
                ],
                "recommendations": [
                    v.policy.remediation for v in result.violations[:5]
                    if v.policy.remediation
                ],
            }

        return report

    def _score_to_grade(self, score: float) -> str:
        """Convert score to letter grade."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        return "F"

    def get_history(
        self,
        framework: Optional[ComplianceFramework] = None,
        limit: int = 100,
    ) -> List[ComplianceResult]:
        """Get compliance check history.

        Args:
            framework: Filter by framework
            limit: Maximum results

        Returns:
            List of results
        """
        with self._lock:
            results = list(self._results_history)

        if framework:
            results = [r for r in results if r.framework == framework]

        results.sort(key=lambda r: r.scan_time, reverse=True)

        return results[:limit]

    def get_trend(
        self,
        framework: ComplianceFramework,
        days: int = 30,
    ) -> Dict[str, Any]:
        """Get compliance trend over time.

        Args:
            framework: Framework to analyze
            days: Number of days

        Returns:
            Trend data
        """
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(days=days)

        with self._lock:
            results = [
                r for r in self._results_history
                if r.framework == framework and r.scan_time >= cutoff
            ]

        if not results:
            return {
                "framework": framework.value,
                "period_days": days,
                "data_points": 0,
                "avg_score": 0,
                "trend": "unknown",
            }

        scores = [r.compliance_score for r in results]
        avg_score = sum(scores) / len(scores)

        # Calculate trend
        if len(scores) >= 2:
            first_half = sum(scores[:len(scores) // 2]) / (len(scores) // 2)
            second_half = sum(scores[len(scores) // 2:]) / (len(scores) - len(scores) // 2)

            if second_half > first_half + 5:
                trend = "improving"
            elif second_half < first_half - 5:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        return {
            "framework": framework.value,
            "period_days": days,
            "data_points": len(results),
            "avg_score": avg_score,
            "min_score": min(scores),
            "max_score": max(scores),
            "trend": trend,
            "latest_score": scores[-1] if scores else 0,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get overall compliance summary.

        Returns:
            Summary dictionary
        """
        with self._lock:
            results = list(self._results_history)
            policies = dict(self._policies)

        # By framework
        by_framework: Dict[str, Dict[str, Any]] = {}

        for framework in ComplianceFramework:
            framework_results = [r for r in results if r.framework == framework]
            if framework_results:
                latest = max(framework_results, key=lambda r: r.scan_time)
                by_framework[framework.value] = {
                    "total_checks": len(framework_results),
                    "latest_score": latest.compliance_score,
                    "is_compliant": latest.is_compliant,
                    "policies_defined": len([
                        p for p in policies.values()
                        if p.framework == framework
                    ]),
                }

        return {
            "total_policies": len(policies),
            "total_checks": len(results),
            "frameworks_configured": len(by_framework),
            "by_framework": by_framework,
        }
