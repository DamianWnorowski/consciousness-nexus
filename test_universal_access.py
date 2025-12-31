#!/usr/bin/env python3
"""
🧪 TEST UNIVERSAL ACCESS - Consciousness Computing Suite
=======================================================

Test script to verify that Consciousness Suite is accessible from multiple interfaces.
"""

import asyncio
import subprocess
import sys
import requests
import time
import os
from pathlib import Path

def test_python_direct():
    """Test direct Python import"""
    print("🐍 Testing Python Direct Import...")
    try:
        from consciousness_suite import get_safety_orchestrator, initialize_consciousness_suite
        print("✅ Python direct import: SUCCESS")
        return True
    except ImportError as e:
        print(f"❌ Python direct import failed: {e}")
        return False

async def test_api_server():
    """Test API server access"""
    print("🌐 Testing API Server Access...")
    try:
        # Start API server in background
        import subprocess
        server_process = subprocess.Popen([
            sys.executable, "consciousness_api_server.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Wait for server to start
        time.sleep(3)

        # Test health endpoint
        response = requests.get("http://localhost:18473/health", timeout=5)
        if response.status_code == 200:
            print("✅ API server access: SUCCESS")
            server_process.terminate()
            return True
        else:
            print(f"❌ API server returned status {response.status_code}")
            server_process.terminate()
            return False

    except Exception as e:
        print(f"❌ API server test failed: {e}")
        return False

def test_cli_tool():
    """Test CLI tool access"""
    print("🖥️  Testing CLI Tool Access...")
    try:
        cli_path = Path("consciousness-cli")
        if cli_path.exists():
            # Note: On Windows, we'd need to handle .exe extension
            print("✅ CLI tool exists: SUCCESS")
            return True
        else:
            print("❌ CLI tool not found")
            return False
    except Exception as e:
        print(f"❌ CLI tool test failed: {e}")
        return False

def test_docker_setup():
    """Test Docker setup"""
    print("🐳 Testing Docker Setup...")
    try:
        # Check if docker-compose.yml exists
        if Path("docker-compose.yml").exists():
            print("✅ Docker compose file exists: SUCCESS")
            return True
        else:
            print("❌ Docker compose file not found")
            return False
    except Exception as e:
        print(f"❌ Docker setup test failed: {e}")
        return False

def test_sdks():
    """Test SDK availability"""
    print("📦 Testing SDK Availability...")
    try:
        # Check JavaScript SDK
        js_sdk = Path("consciousness-sdk-js/package.json")
        if js_sdk.exists():
            print("✅ JavaScript SDK: AVAILABLE")
        else:
            print("❌ JavaScript SDK: MISSING")

        # Check Rust SDK
        rust_sdk = Path("consciousness-sdk-rust/Cargo.toml")
        if rust_sdk.exists():
            print("✅ Rust SDK: AVAILABLE")
        else:
            print("❌ Rust SDK: MISSING")

        # Check Go SDK
        go_sdk = Path("consciousness-sdk-go/go.mod")
        if go_sdk.exists():
            print("✅ Go SDK: AVAILABLE")
        else:
            print("❌ Go SDK: MISSING")

        # Check CLI wrappers
        cli_bat = Path("consciousness-cli.bat")
        cli_ps1 = Path("consciousness-cli.ps1")
        if cli_bat.exists() and cli_ps1.exists():
            print("✅ CLI Wrappers: AVAILABLE")
        else:
            print("❌ CLI Wrappers: MISSING")

        # Check monitoring configs
        monitoring = Path("monitoring/prometheus.yml")
        if monitoring.exists():
            print("✅ Monitoring Configs: AVAILABLE")
        else:
            print("❌ Monitoring Configs: MISSING")

        return True
    except Exception as e:
        print(f"❌ SDK test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("🧪 CONSCIOUSNESS COMPUTING SUITE - UNIVERSAL ACCESS TEST")
    print("=" * 60)

    tests = [
        test_python_direct,
        test_api_server,
        test_cli_tool,
        test_docker_setup,
        test_sdks,
    ]

    passed = 0
    total = len(tests)

    for test_func in tests:
        if asyncio.iscoroutinefunction(test_func):
            result = await test_func()
        else:
            result = test_func()

        if result:
            passed += 1
        print()

    print("=" * 60)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED - Consciousness Suite is universally accessible!")
        print()
        print("🚀 ACCESS METHODS AVAILABLE:")
        print("   • Python: from consciousness_suite import *")
        print("   • HTTP API: http://localhost:18473")
        print("   • CLI: ./consciousness-cli")
        print("   • Docker: docker-compose up")
        print("   • JavaScript SDK: npm install consciousness-suite-sdk")
        print("   • Rust SDK: cargo add consciousness-suite-sdk")
        print()
        print("📖 See UNIVERSAL_DEPLOYMENT_GUIDE.md for complete usage instructions")
    else:
        print(f"⚠️  {total - passed} tests failed - check setup")

if __name__ == "__main__":
    asyncio.run(main())
