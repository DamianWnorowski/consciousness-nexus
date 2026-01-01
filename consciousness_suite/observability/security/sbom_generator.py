"""SBOM Generator

Software Bill of Materials generation and management.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum
import json
import hashlib
import logging
import uuid

logger = logging.getLogger(__name__)


class SBOMFormat(str, Enum):
    """SBOM output formats."""
    SPDX_JSON = "spdx_json"
    SPDX_TV = "spdx_tagvalue"
    CYCLONEDX_JSON = "cyclonedx_json"
    CYCLONEDX_XML = "cyclonedx_xml"


class PackageType(str, Enum):
    """Package/component types."""
    LIBRARY = "library"
    APPLICATION = "application"
    FRAMEWORK = "framework"
    OPERATING_SYSTEM = "operating_system"
    CONTAINER = "container"
    DEVICE = "device"
    FILE = "file"
    DATA = "data"


class HashAlgorithm(str, Enum):
    """Hash algorithms for checksums."""
    SHA256 = "SHA256"
    SHA512 = "SHA512"
    SHA1 = "SHA1"
    MD5 = "MD5"


@dataclass
class LicenseInfo:
    """License information for a component."""
    license_id: str
    license_name: str
    is_osi_approved: bool = False
    is_fsf_libre: bool = False
    url: Optional[str] = None
    text: Optional[str] = None


@dataclass
class ExternalReference:
    """External reference for a component."""
    ref_type: str  # vcs, website, distribution, documentation, etc.
    url: str
    comment: Optional[str] = None


@dataclass
class SBOMComponent:
    """A component in the SBOM.

    Usage:
        component = SBOMComponent(
            name="requests",
            version="2.31.0",
            package_type=PackageType.LIBRARY,
            purl="pkg:pypi/requests@2.31.0",
            licenses=[LicenseInfo("Apache-2.0", "Apache License 2.0", True)],
        )
    """
    name: str
    version: str
    package_type: PackageType = PackageType.LIBRARY
    purl: Optional[str] = None  # Package URL
    cpe: Optional[str] = None  # Common Platform Enumeration
    supplier: Optional[str] = None
    author: Optional[str] = None
    description: Optional[str] = None
    licenses: List[LicenseInfo] = field(default_factory=list)
    hashes: Dict[HashAlgorithm, str] = field(default_factory=dict)
    external_refs: List[ExternalReference] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)  # List of PURLs
    properties: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.purl:
            self.purl = self._generate_purl()

    def _generate_purl(self) -> str:
        """Generate Package URL."""
        # Simplified PURL generation
        name = self.name.lower().replace("-", "_")
        return f"pkg:generic/{name}@{self.version}"

    def to_spdx(self) -> Dict[str, Any]:
        """Convert to SPDX format."""
        spdx_id = f"SPDXRef-Package-{self.name.replace('-', '').replace('_', '')}"

        result = {
            "SPDXID": spdx_id,
            "name": self.name,
            "versionInfo": self.version,
            "downloadLocation": "NOASSERTION",
            "filesAnalyzed": False,
            "primaryPackagePurpose": self.package_type.value.upper(),
        }

        if self.purl:
            result["externalRefs"] = [{
                "referenceCategory": "PACKAGE-MANAGER",
                "referenceType": "purl",
                "referenceLocator": self.purl,
            }]

        if self.supplier:
            result["supplier"] = f"Organization: {self.supplier}"

        if self.licenses:
            # Use first license for simplicity
            result["licenseConcluded"] = self.licenses[0].license_id
            result["licenseDeclared"] = self.licenses[0].license_id
        else:
            result["licenseConcluded"] = "NOASSERTION"
            result["licenseDeclared"] = "NOASSERTION"

        result["copyrightText"] = "NOASSERTION"

        if self.hashes:
            result["checksums"] = [
                {"algorithm": algo.value, "checksumValue": value}
                for algo, value in self.hashes.items()
            ]

        return result

    def to_cyclonedx(self) -> Dict[str, Any]:
        """Convert to CycloneDX format."""
        result = {
            "type": self.package_type.value,
            "bom-ref": self.purl or f"{self.name}@{self.version}",
            "name": self.name,
            "version": self.version,
        }

        if self.supplier:
            result["supplier"] = {"name": self.supplier}

        if self.author:
            result["author"] = self.author

        if self.description:
            result["description"] = self.description

        if self.purl:
            result["purl"] = self.purl

        if self.cpe:
            result["cpe"] = self.cpe

        if self.licenses:
            result["licenses"] = [
                {"license": {"id": lic.license_id}}
                for lic in self.licenses
            ]

        if self.hashes:
            result["hashes"] = [
                {"alg": algo.value, "content": value}
                for algo, value in self.hashes.items()
            ]

        if self.external_refs:
            result["externalReferences"] = [
                {"type": ref.ref_type, "url": ref.url}
                for ref in self.external_refs
            ]

        if self.properties:
            result["properties"] = [
                {"name": k, "value": v}
                for k, v in self.properties.items()
            ]

        return result


@dataclass
class SBOMDocument:
    """SBOM document containing components."""
    name: str
    version: str
    components: List[SBOMComponent] = field(default_factory=list)
    created: datetime = field(default_factory=datetime.now)
    creator_tool: str = "consciousness-sbom-generator"
    creator_organization: Optional[str] = None
    namespace: Optional[str] = None
    document_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def component_count(self) -> int:
        return len(self.components)

    @property
    def license_summary(self) -> Dict[str, int]:
        """Get license usage summary."""
        licenses: Dict[str, int] = {}
        for comp in self.components:
            for lic in comp.licenses:
                licenses[lic.license_id] = licenses.get(lic.license_id, 0) + 1
        return licenses

    def to_spdx_json(self) -> Dict[str, Any]:
        """Convert to SPDX JSON format."""
        doc_namespace = self.namespace or f"https://spdx.org/spdxdocs/{self.name}-{self.version}"

        packages = [comp.to_spdx() for comp in self.components]

        # Add root package
        root_package = {
            "SPDXID": "SPDXRef-DOCUMENT",
            "name": self.name,
            "versionInfo": self.version,
            "downloadLocation": "NOASSERTION",
            "filesAnalyzed": False,
            "licenseConcluded": "NOASSERTION",
            "licenseDeclared": "NOASSERTION",
            "copyrightText": "NOASSERTION",
        }

        # Build relationships
        relationships = [
            {
                "spdxElementId": "SPDXRef-DOCUMENT",
                "relationshipType": "DESCRIBES",
                "relatedSpdxElement": "SPDXRef-DOCUMENT",
            }
        ]

        for comp in self.components:
            spdx_id = f"SPDXRef-Package-{comp.name.replace('-', '').replace('_', '')}"
            relationships.append({
                "spdxElementId": "SPDXRef-DOCUMENT",
                "relationshipType": "CONTAINS",
                "relatedSpdxElement": spdx_id,
            })

            for dep_purl in comp.dependencies:
                # Find dependency
                for other in self.components:
                    if other.purl == dep_purl:
                        other_id = f"SPDXRef-Package-{other.name.replace('-', '').replace('_', '')}"
                        relationships.append({
                            "spdxElementId": spdx_id,
                            "relationshipType": "DEPENDS_ON",
                            "relatedSpdxElement": other_id,
                        })
                        break

        return {
            "spdxVersion": "SPDX-2.3",
            "dataLicense": "CC0-1.0",
            "SPDXID": "SPDXRef-DOCUMENT",
            "name": self.name,
            "documentNamespace": doc_namespace,
            "creationInfo": {
                "created": self.created.isoformat(),
                "creators": [
                    f"Tool: {self.creator_tool}",
                    f"Organization: {self.creator_organization or 'Unknown'}",
                ],
            },
            "packages": [root_package] + packages,
            "relationships": relationships,
        }

    def to_cyclonedx_json(self) -> Dict[str, Any]:
        """Convert to CycloneDX JSON format."""
        components = [comp.to_cyclonedx() for comp in self.components]

        # Build dependencies
        dependencies = []
        for comp in self.components:
            if comp.dependencies:
                dep_refs = []
                for dep_purl in comp.dependencies:
                    for other in self.components:
                        if other.purl == dep_purl:
                            dep_refs.append(other.purl or f"{other.name}@{other.version}")
                            break

                dependencies.append({
                    "ref": comp.purl or f"{comp.name}@{comp.version}",
                    "dependsOn": dep_refs,
                })

        return {
            "$schema": "http://cyclonedx.org/schema/bom-1.5.schema.json",
            "bomFormat": "CycloneDX",
            "specVersion": "1.5",
            "serialNumber": f"urn:uuid:{self.document_id}",
            "version": 1,
            "metadata": {
                "timestamp": self.created.isoformat(),
                "tools": [
                    {"name": self.creator_tool, "version": "1.0.0"}
                ],
                "component": {
                    "type": "application",
                    "name": self.name,
                    "version": self.version,
                },
            },
            "components": components,
            "dependencies": dependencies,
        }


class SBOMGenerator:
    """Generates Software Bill of Materials.

    Usage:
        generator = SBOMGenerator()

        # From pip requirements
        sbom = await generator.from_pip_freeze()

        # From package.json
        sbom = await generator.from_package_json("package.json")

        # Manual
        generator.add_component(SBOMComponent(
            name="mylib",
            version="1.0.0",
        ))
        sbom = generator.generate("myapp", "1.0.0")

        # Export
        json_data = sbom.to_cyclonedx_json()
    """

    def __init__(self):
        self._components: List[SBOMComponent] = []
        self._known_licenses: Dict[str, LicenseInfo] = {
            "MIT": LicenseInfo("MIT", "MIT License", True, True),
            "Apache-2.0": LicenseInfo("Apache-2.0", "Apache License 2.0", True, True),
            "GPL-3.0": LicenseInfo("GPL-3.0", "GNU General Public License v3.0", True, True),
            "BSD-3-Clause": LicenseInfo("BSD-3-Clause", "BSD 3-Clause License", True, True),
            "ISC": LicenseInfo("ISC", "ISC License", True, True),
        }

    def add_component(self, component: SBOMComponent):
        """Add a component to the SBOM.

        Args:
            component: Component to add
        """
        self._components.append(component)

    def clear(self):
        """Clear all components."""
        self._components.clear()

    def generate(
        self,
        name: str,
        version: str,
        organization: Optional[str] = None,
    ) -> SBOMDocument:
        """Generate SBOM document.

        Args:
            name: Application/project name
            version: Version
            organization: Creator organization

        Returns:
            SBOMDocument
        """
        return SBOMDocument(
            name=name,
            version=version,
            components=list(self._components),
            creator_organization=organization,
        )

    async def from_pip_freeze(self) -> List[SBOMComponent]:
        """Parse pip freeze output.

        Returns:
            List of components
        """
        import subprocess

        try:
            result = subprocess.run(
                ["pip", "freeze"],
                capture_output=True,
                text=True,
                timeout=60,
            )

            components = []

            for line in result.stdout.strip().split("\n"):
                if "==" in line:
                    name, version = line.split("==", 1)
                    comp = SBOMComponent(
                        name=name.strip(),
                        version=version.strip(),
                        package_type=PackageType.LIBRARY,
                        purl=f"pkg:pypi/{name.lower()}@{version}",
                    )
                    components.append(comp)
                    self._components.append(comp)

            logger.info(f"Parsed {len(components)} packages from pip freeze")
            return components

        except Exception as e:
            logger.error(f"Failed to parse pip freeze: {e}")
            return []

    async def from_package_json(self, path: str) -> List[SBOMComponent]:
        """Parse package.json file.

        Args:
            path: Path to package.json

        Returns:
            List of components
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                pkg = json.load(f)

            components = []

            for deps_key in ["dependencies", "devDependencies"]:
                deps = pkg.get(deps_key, {})
                for name, version_spec in deps.items():
                    # Clean version spec
                    version = version_spec.lstrip("^~>=<")

                    comp = SBOMComponent(
                        name=name,
                        version=version,
                        package_type=PackageType.LIBRARY,
                        purl=f"pkg:npm/{name}@{version}",
                        properties={"dev": str(deps_key == "devDependencies")},
                    )
                    components.append(comp)
                    self._components.append(comp)

            logger.info(f"Parsed {len(components)} packages from {path}")
            return components

        except Exception as e:
            logger.error(f"Failed to parse package.json: {e}")
            return []

    async def from_requirements_txt(self, path: str) -> List[SBOMComponent]:
        """Parse requirements.txt file.

        Args:
            path: Path to requirements.txt

        Returns:
            List of components
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            components = []

            for line in lines:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("-"):
                    continue

                # Parse package==version or package>=version etc.
                for op in ["==", ">=", "<=", ">", "<", "~="]:
                    if op in line:
                        name, version = line.split(op, 1)
                        name = name.strip()
                        version = version.strip().split(",")[0]  # First version constraint

                        comp = SBOMComponent(
                            name=name,
                            version=version,
                            package_type=PackageType.LIBRARY,
                            purl=f"pkg:pypi/{name.lower()}@{version}",
                        )
                        components.append(comp)
                        self._components.append(comp)
                        break

            logger.info(f"Parsed {len(components)} packages from {path}")
            return components

        except Exception as e:
            logger.error(f"Failed to parse requirements.txt: {e}")
            return []

    async def from_go_mod(self, path: str) -> List[SBOMComponent]:
        """Parse go.mod file.

        Args:
            path: Path to go.mod

        Returns:
            List of components
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            components = []
            in_require = False

            for line in content.split("\n"):
                line = line.strip()

                if line.startswith("require ("):
                    in_require = True
                    continue
                elif line == ")":
                    in_require = False
                    continue

                if in_require or line.startswith("require "):
                    # Parse module path version
                    parts = line.replace("require ", "").split()
                    if len(parts) >= 2:
                        module = parts[0]
                        version = parts[1].lstrip("v")

                        # Extract name from module path
                        name = module.split("/")[-1]

                        comp = SBOMComponent(
                            name=name,
                            version=version,
                            package_type=PackageType.LIBRARY,
                            purl=f"pkg:golang/{module}@{version}",
                            properties={"module": module},
                        )
                        components.append(comp)
                        self._components.append(comp)

            logger.info(f"Parsed {len(components)} modules from {path}")
            return components

        except Exception as e:
            logger.error(f"Failed to parse go.mod: {e}")
            return []

    def export(
        self,
        sbom: SBOMDocument,
        format: SBOMFormat,
        path: str,
    ):
        """Export SBOM to file.

        Args:
            sbom: SBOM document
            format: Output format
            path: Output path
        """
        if format == SBOMFormat.CYCLONEDX_JSON:
            data = sbom.to_cyclonedx_json()
        elif format == SBOMFormat.SPDX_JSON:
            data = sbom.to_spdx_json()
        else:
            raise ValueError(f"Unsupported format: {format}")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Exported SBOM to {path} ({format.value})")

    def get_summary(self) -> Dict[str, Any]:
        """Get SBOM summary.

        Returns:
            Summary dictionary
        """
        package_types: Dict[str, int] = {}
        licenses: Dict[str, int] = {}

        for comp in self._components:
            pt = comp.package_type.value
            package_types[pt] = package_types.get(pt, 0) + 1

            for lic in comp.licenses:
                licenses[lic.license_id] = licenses.get(lic.license_id, 0) + 1

        return {
            "component_count": len(self._components),
            "by_type": package_types,
            "by_license": licenses,
            "unlicensed_count": sum(1 for c in self._components if not c.licenses),
        }
