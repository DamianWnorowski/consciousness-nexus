"""Container Security Observability Module

Security scanning and compliance:
- SBOM (Software Bill of Materials) generation
- Vulnerability scanning and CVE detection
- Trivy scanner integration
- Compliance verification
"""

from .sbom_generator import (
    SBOMGenerator,
    SBOMComponent,
    SBOMDocument,
    PackageType,
    LicenseInfo,
)
from .vulnerability_scan import (
    VulnerabilityScanner,
    Vulnerability,
    ScanResult,
    Severity,
    CVSSScore,
)
from .trivy_integration import (
    TrivyScanner,
    TrivyResult,
    TrivyVulnerability,
    TrivyMisconfiguration,
    TrivySecret,
)
from .compliance_check import (
    ComplianceChecker,
    ComplianceResult,
    CompliancePolicy,
    PolicyViolation,
    ComplianceFramework,
)

__all__ = [
    # SBOM
    "SBOMGenerator",
    "SBOMComponent",
    "SBOMDocument",
    "PackageType",
    "LicenseInfo",
    # Vulnerability
    "VulnerabilityScanner",
    "Vulnerability",
    "ScanResult",
    "Severity",
    "CVSSScore",
    # Trivy
    "TrivyScanner",
    "TrivyResult",
    "TrivyVulnerability",
    "TrivyMisconfiguration",
    "TrivySecret",
    # Compliance
    "ComplianceChecker",
    "ComplianceResult",
    "CompliancePolicy",
    "PolicyViolation",
    "ComplianceFramework",
]
