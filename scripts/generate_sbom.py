#!/usr/bin/env python3
"""
Consciousness Nexus - Software Bill of Materials Generator
==========================================================

Generates SBOM (Software Bill of Materials) for the consciousness computing suite.
"""

import json
import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path


def get_python_packages(include_dev=True):
    """Get Python packages using pip list"""
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'list', '--format', 'json'
        ], capture_output=True, text=True, check=True)

        packages = json.loads(result.stdout)

        # Convert to our format
        formatted_packages = []
        for pkg in packages:
            package_info = {
                'name': pkg['name'],
                'version': pkg['version'],
                'source': 'pip',
                'type': 'runtime',
                'license': 'Unknown',
                'description': '',
                'homepage': '',
                'dependencies': [],
                'vulnerabilities': []
            }

            # Classify as dev or runtime (basic heuristic)
            dev_indicators = ['pytest', 'black', 'mypy', 'flake8', 'playwright', 'types-']
            if any(indicator.lower() in pkg['name'].lower() for indicator in dev_indicators):
                package_info['type'] = 'development'

            # Only include dev deps if requested
            if include_dev or package_info['type'] == 'runtime':
                formatted_packages.append(package_info)

        return formatted_packages

    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        print(f"Warning: Could not enumerate pip packages: {e}")
        return []


def get_rust_dependencies():
    """Get Rust dependencies (placeholder for future implementation)"""
    # Placeholder for Cargo.toml analysis
    return [{
        'name': 'consciousness-nexus-rust',
        'version': '0.1.0',
        'source': 'cargo',
        'type': 'application',
        'license': 'MIT',
        'description': 'Rust components for consciousness computing',
        'homepage': 'https://github.com/consciousness-nexus/rust',
        'dependencies': ['tokio', 'serde', 'anyhow'],
        'vulnerabilities': []
    }]


def create_sbom_document(python_packages, rust_packages, timestamp):
    """Create SBOM document following SPDX-like format"""
    sbom = {
        'sbom': {
            'version': '1.0',
            'format': 'Consciousness-Nexus-SBOM',
            'specVersion': '1.4',
            'creationInfo': {
                'created': timestamp,
                'creators': ['Tool: Consciousness Nexus SBOM Generator v1.0']
            }
        },
        'components': []
    }

    # Add Python components
    for pkg in python_packages:
        component = {
            'type': 'library',
            'name': pkg['name'],
            'version': pkg['version'],
            'purl': f"pkg:pypi/{pkg['name']}@{pkg['version']}",
            'properties': [
                {'name': 'source', 'value': pkg['source']},
                {'name': 'type', 'value': pkg['type']}
            ]
        }

        if pkg.get('license') and pkg['license'] != 'Unknown':
            component['properties'].append({
                'name': 'license',
                'value': pkg['license']
            })

        sbom['components'].append(component)

    # Add Rust components
    for pkg in rust_packages:
        component = {
            'type': 'library',
            'name': pkg['name'],
            'version': pkg['version'],
            'purl': f"pkg:cargo/{pkg['name']}@{pkg['version']}",
            'properties': [
                {'name': 'source', 'value': pkg['source']},
                {'name': 'type', 'value': pkg['type']},
                {'name': 'language', 'value': 'rust'}
            ]
        }

        sbom['components'].append(component)

    return sbom


def main():
    """Main SBOM generation function"""
    print("CONSCIOUSNESS NEXUS - SBOM GENERATOR")
    print("=" * 45)

    # Parse arguments
    include_dev = '--include-dev' in sys.argv
    quiet = '--quiet' in sys.argv

    if not quiet:
        if include_dev:
            print("Status: Including development dependencies")
        else:
            print("Status: Production dependencies only")

    # Create output directory
    output_dir = Path("MASTER_INDEX/sbom")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not quiet:
        print(f"Output directory: {output_dir}")

    # Generate timestamp
    timestamp = datetime.now().isoformat() + 'Z'

    # Get dependencies
    python_packages = get_python_packages(include_dev)
    rust_packages = get_rust_dependencies()

    if not quiet:
        print(f"Metrics: Found {len(python_packages)} Python packages")
        print(f"Metrics: Found {len(rust_packages)} Rust packages")

    # Create SBOM document
    sbom = create_sbom_document(python_packages, rust_packages, timestamp)

    # Create Python-specific SBOM
    python_sbom = {
        'metadata': {
            'timestamp': timestamp,
            'generator': 'Consciousness Nexus SBOM Generator',
            'includeDev': include_dev
        },
        'packages': python_packages
    }

    # Save files
    sbom_file = output_dir / "sbom.json"
    python_file = output_dir / "python_packages.json"
    rust_file = output_dir / "rust_packages.json"

    with open(sbom_file, 'w') as f:
        json.dump(sbom, f, indent=2)

    with open(python_file, 'w') as f:
        json.dump(python_sbom, f, indent=2)

    # Save Rust packages
    rust_sbom = {
        'metadata': {
            'timestamp': timestamp,
            'generator': 'Consciousness Nexus SBOM Generator',
            'note': 'Rust SBOM generation not fully implemented - placeholder'
        },
        'packages': rust_packages
    }

    with open(rust_file, 'w') as f:
        json.dump(rust_sbom, f, indent=2)

    if not quiet:
        print("")
        print("SUCCESS: SBOM Generation Complete")
        print(f"Files saved to: {output_dir}")
        print("   --- sbom.json (Complete SBOM)")
        print("   --- python_packages.json (Python dependencies)")
        print("   --- rust_packages.json (Rust dependencies)")
        print("")
        print(f"Total components: {len(sbom['components'])}")
        print(f"Python packages: {len(python_packages)}")
        print(f"Rust packages: {len(rust_packages)}")


if __name__ == '__main__':
    main()
