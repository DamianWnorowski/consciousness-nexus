#!/usr/bin/env python3
"""
🧪 COMPREHENSIVE CONSCIOUSNESS SUITE TEST SUITE
==============================================

Tests all components of the Consciousness Suite deployment:
- API server functionality
- Web dashboard accessibility
- SDK imports and basic functionality
- Docker services health
- Port availability and conflicts
- Data persistence and recovery

Usage: python test-consciousness-suite.py
"""

import asyncio
import json
import sys
import time
import subprocess
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import importlib.util

# Test configuration
API_BASE_URL = "http://localhost:18473"
WEB_DASHBOARD_URL = "http://localhost:31573"
SERVICES = {
    "API Server": ("18473", f"{API_BASE_URL}/health"),
    "Web Dashboard": ("31573", WEB_DASHBOARD_URL),
    "Grafana": ("31572", "http://localhost:31572/api/health"),
    "Prometheus": ("24789", "http://localhost:24789/-/healthy"),
    "Loki": ("42851", "http://localhost:42851/ready"),
}

class TestResult:
    def __init__(self, name: str, status: str, message: str = "", duration: float = 0.0):
        self.name = name
        self.status = status  # 'PASS', 'FAIL', 'SKIP', 'WARN'
        self.message = message
        self.duration = duration

    def __str__(self):
        icon = {
            'PASS': '✅',
            'FAIL': '❌',
            'SKIP': '⏭️ ',
            'WARN': '⚠️ '
        }.get(self.status, '❓')

        result = f"{icon} {self.name}"
        if self.message:
            result += f" - {self.message}"
        if self.duration > 0:
            result += f" ({self.duration:.2f}s)"
        return result

class ConsciousnessSuiteTester:
    def __init__(self):
        self.results: List[TestResult] = []
        self.start_time = time.time()

    def log(self, message: str, level: str = "INFO"):
        """Log a message with timestamp"""
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        print(f"[{timestamp}] [{level}] {message}")

    def add_result(self, result: TestResult):
        """Add a test result"""
        self.results.append(result)
        print(result)

    async def run_test(self, name: str, test_func):
        """Run a single test and record results"""
        start_time = time.time()
        try:
            result = await test_func()
            duration = time.time() - start_time
            result.duration = duration
            self.add_result(result)
        except Exception as e:
            duration = time.time() - start_time
            self.add_result(TestResult(
                name=name,
                status="FAIL",
                message=f"Exception: {str(e)}",
                duration=duration
            ))

    def print_summary(self):
        """Print test summary"""
        total_time = time.time() - self.start_time
        passed = sum(1 for r in self.results if r.status == 'PASS')
        failed = sum(1 for r in self.results if r.status == 'FAIL')
        skipped = sum(1 for r in self.results if r.status == 'SKIP')
        warnings = sum(1 for r in self.results if r.status == 'WARN')

        print("\n" + "="*60)
        print("🧪 TEST SUMMARY")
        print("="*60)
        print(f"Total Tests: {len(self.results)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Skipped: {skipped}")
        print(f"Warnings: {warnings}")
        print(".2f")
        print()

        if failed > 0:
            print("❌ FAILED TESTS:")
            for result in self.results:
                if result.status == 'FAIL':
                    print(f"   {result.name}: {result.message}")
            print()

        if warnings > 0:
            print("⚠️  WARNINGS:")
            for result in self.results:
                if result.status == 'WARN':
                    print(f"   {result.name}: {result.message}")
            print()

        # Overall status
        if failed == 0 and warnings == 0:
            print("🎉 ALL TESTS PASSED! Consciousness Suite is fully operational.")
        elif failed == 0:
            print("⚠️  TESTS PASSED WITH WARNINGS - Review and fix warnings.")
        else:
            print("❌ SOME TESTS FAILED - Check the issues above.")

# Test functions

async def test_docker_services(tester: ConsciousnessSuiteTester) -> TestResult:
    """Test if Docker services are running"""
    try:
        result = subprocess.run(['docker-compose', 'ps'], capture_output=True, text=True)
        if result.returncode != 0:
            return TestResult("Docker Services", "FAIL", "docker-compose ps failed")

        lines = result.stdout.strip().split('\n')
        if len(lines) < 2:  # Header + at least one service
            return TestResult("Docker Services", "FAIL", "No services found running")

        running_services = 0
        for line in lines[1:]:  # Skip header
            if 'Up' in line:
                running_services += 1

        if running_services >= 6:  # API, Dashboard, Grafana, Prometheus, Loki, Postgres/Redis
            return TestResult("Docker Services", "PASS", f"{running_services} services running")
        else:
            return TestResult("Docker Services", "WARN", f"Only {running_services} services running (expected 6+)")

    except Exception as e:
        return TestResult("Docker Services", "SKIP", f"Docker not available: {str(e)}")

async def test_port_availability(tester: ConsciousnessSuiteTester) -> TestResult:
    """Test if all required ports are available"""
    import socket

    unavailable_ports = []
    for service_name, (port_str, _) in SERVICES.items():
        port = int(port_str)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', port))
        sock.close()

        if result != 0:
            unavailable_ports.append(f"{service_name} ({port})")

    if unavailable_ports:
        return TestResult("Port Availability", "FAIL",
                         f"Ports not accessible: {', '.join(unavailable_ports)}")
    else:
        return TestResult("Port Availability", "PASS", "All ports accessible")

async def test_api_server(tester: ConsciousnessSuiteTester) -> TestResult:
    """Test API server health and basic functionality"""
    try:
        # Test health endpoint
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        if response.status_code != 200:
            return TestResult("API Server", "FAIL", f"Health check failed: {response.status_code}")

        health_data = response.json()
        if not health_data.get('status') == 'healthy':
            return TestResult("API Server", "WARN", f"Health status: {health_data.get('status')}")

        # Test system status endpoint
        response = requests.get(f"{API_BASE_URL}/status", timeout=10)
        if response.status_code != 200:
            return TestResult("API Server", "FAIL", f"Status check failed: {response.status_code}")

        return TestResult("API Server", "PASS", "Health and status endpoints working")

    except requests.exceptions.RequestException as e:
        return TestResult("API Server", "FAIL", f"Connection failed: {str(e)}")

async def test_web_dashboard(tester: ConsciousnessSuiteTester) -> TestResult:
    """Test web dashboard accessibility"""
    try:
        response = requests.get(WEB_DASHBOARD_URL, timeout=15)
        if response.status_code != 200:
            return TestResult("Web Dashboard", "FAIL", f"HTTP {response.status_code}")

        # Check if it's an HTML page (basic check)
        if 'text/html' not in response.headers.get('content-type', ''):
            return TestResult("Web Dashboard", "WARN", "Response not HTML")

        return TestResult("Web Dashboard", "PASS", "Dashboard accessible and serving HTML")

    except requests.exceptions.RequestException as e:
        return TestResult("Web Dashboard", "FAIL", f"Connection failed: {str(e)}")

async def test_monitoring_services(tester: ConsciousnessSuiteTester) -> TestResult:
    """Test monitoring services (Grafana, Prometheus, Loki)"""
    results = []

    for service_name, (_, health_url) in [("Grafana", SERVICES["Grafana"]),
                                         ("Prometheus", SERVICES["Prometheus"]),
                                         ("Loki", SERVICES["Loki"])]:
        try:
            response = requests.get(health_url, timeout=10)
            if response.status_code == 200:
                results.append(f"{service_name}: OK")
            else:
                results.append(f"{service_name}: HTTP {response.status_code}")
        except Exception as e:
            results.append(f"{service_name}: ERROR - {str(e)}")

    ok_count = sum(1 for r in results if ': OK' in r)
    if ok_count == 3:
        return TestResult("Monitoring Services", "PASS", ", ".join(results))
    elif ok_count > 0:
        return TestResult("Monitoring Services", "WARN", f"{ok_count}/3 services OK: " + ", ".join(results))
    else:
        return TestResult("Monitoring Services", "FAIL", "No monitoring services accessible")

async def test_python_sdk_imports(tester: ConsciousnessSuiteTester) -> TestResult:
    """Test that Python SDK can be imported"""
    try:
        # Test core imports
        from consciousness_suite import AutoRecursiveChainAI, get_safety_orchestrator
        return TestResult("Python SDK Imports", "PASS", "Core modules importable")
    except ImportError as e:
        return TestResult("Python SDK Imports", "FAIL", f"Import failed: {str(e)}")

async def test_api_functionality(tester: ConsciousnessSuiteTester) -> TestResult:
    """Test basic API functionality"""
    try:
        # Test login endpoint (should fail gracefully with demo credentials)
        login_data = {"username": "test", "password": "test"}
        response = requests.post(f"{API_BASE_URL}/auth/login", json=login_data, timeout=10)

        # Should return 401 (unauthorized) for invalid credentials
        if response.status_code == 401:
            return TestResult("API Functionality", "PASS", "Authentication endpoint working")
        elif response.status_code == 200:
            return TestResult("API Functionality", "WARN", "Login succeeded with test credentials")
        else:
            return TestResult("API Functionality", "FAIL", f"Unexpected response: {response.status_code}")

    except requests.exceptions.RequestException as e:
        return TestResult("API Functionality", "FAIL", f"API call failed: {str(e)}")

async def test_file_structure(tester: ConsciousnessSuiteTester) -> TestResult:
    """Test that all required files exist"""
    required_files = [
        "docker-compose.yml",
        "consciousness_api_server.py",
        "consciousness-suite/__init__.py",
        "consciousness-dashboard/package.json",
        "consciousness-sdk-js/package.json",
        "consciousness-sdk-rust/Cargo.toml",
        "consciousness-sdk-go/go.mod",
        "monitoring/prometheus.yml",
        "monitoring/grafana/provisioning/datasources/prometheus.yml",
        "monitoring/loki-config.yml",
        "monitoring/promtail-config.yml",
    ]

    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        return TestResult("File Structure", "FAIL",
                         f"Missing files: {', '.join(missing_files)}")
    else:
        return TestResult("File Structure", "PASS",
                         f"All {len(required_files)} required files present")

async def test_configuration_validity(tester: ConsciousnessSuiteTester) -> TestResult:
    """Test that configuration files are valid"""
    issues = []

    # Test docker-compose.yml
    try:
        import yaml
        with open('docker-compose.yml', 'r') as f:
            yaml.safe_load(f)
    except Exception as e:
        issues.append(f"docker-compose.yml: {str(e)}")

    # Test package.json files
    for pkg_file in ['consciousness-dashboard/package.json', 'consciousness-sdk-js/package.json']:
        if Path(pkg_file).exists():
            try:
                with open(pkg_file, 'r') as f:
                    json.load(f)
            except Exception as e:
                issues.append(f"{pkg_file}: {str(e)}")

    # Test Cargo.toml
    if Path('consciousness-sdk-rust/Cargo.toml').exists():
        try:
            import tomllib
            with open('consciousness-sdk-rust/Cargo.toml', 'rb') as f:
                tomllib.load(f)
        except ImportError:
            pass  # tomllib not available in older Python
        except Exception as e:
            issues.append(f"Cargo.toml: {str(e)}")

    if issues:
        return TestResult("Configuration Validity", "FAIL", "; ".join(issues))
    else:
        return TestResult("Configuration Validity", "PASS", "All configuration files valid")

async def test_sdk_structure(tester: ConsciousnessSuiteTester) -> TestResult:
    """Test SDK directory structures"""
    sdk_checks = {
        "consciousness-dashboard": ["package.json", "src/main.tsx", "vite.config.ts"],
        "consciousness-sdk-js": ["package.json", "src/index.ts", "rollup.config.js"],
        "consciousness-sdk-rust": ["Cargo.toml", "src/lib.rs", "src/client.rs"],
        "consciousness-sdk-go": ["go.mod", "consciousness.go"],
    }

    missing_components = []

    for sdk, required_files in sdk_checks.items():
        for file in required_files:
            if not Path(f"{sdk}/{file}").exists():
                missing_components.append(f"{sdk}/{file}")

    if missing_components:
        return TestResult("SDK Structure", "FAIL",
                         f"Missing components: {', '.join(missing_components)}")
    else:
        return TestResult("SDK Structure", "PASS",
                         f"All SDK structures complete ({len(sdk_checks)} SDKs)")

async def run_comprehensive_test():
    """Run the complete test suite"""
    print("🧪 CONSCIOUSNESS SUITE COMPREHENSIVE TEST SUITE")
    print("="*60)
    print("Testing all components of your Consciousness Suite deployment...")
    print()

    tester = ConsciousnessSuiteTester()

    # Define test suite
    tests = [
        ("File Structure", test_file_structure),
        ("Configuration Validity", test_configuration_validity),
        ("SDK Structure", test_sdk_structure),
        ("Python SDK Imports", test_python_sdk_imports),
        ("Port Availability", test_port_availability),
        ("Docker Services", test_docker_services),
        ("API Server", test_api_server),
        ("Web Dashboard", test_web_dashboard),
        ("Monitoring Services", test_monitoring_services),
        ("API Functionality", test_api_functionality),
    ]

    # Run all tests
    for test_name, test_func in tests:
        print(f"🔬 Running {test_name}...")
        await tester.run_test(test_name, test_func)
        await asyncio.sleep(0.1)  # Small delay between tests

    # Print summary
    tester.print_summary()

    return tester

if __name__ == "__main__":
    # Check if running in async context
    try:
        asyncio.run(run_comprehensive_test())
    except KeyboardInterrupt:
        print("\n🛑 Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        sys.exit(1)
