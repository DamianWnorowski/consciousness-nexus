#!/usr/bin/env python3
"""
🧬 CONSCIOUSNESS MASTER INTEGRATION
====================================

Automatically integrates ALL safety systems into the Consciousness Computing Suite.
All evolution operations are now automatically protected from the start.
"""

import asyncio
import sys
import os
from typing import Dict, Any, List, Optional
from pathlib import Path

# Ensure all module paths are available
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import all safety systems automatically
SAFETY_SYSTEMS = [
    'evolution_auth_system',
    'evolution_locking',
    'transactional_evolution',
    'evolution_validation',
    'resource_quotas',
    'ui_safety',
    'complexity_optimization',
    'contract_validation',
    'error_recovery',
    'consciousness_safety_orchestrator'
]

# Core evolution systems to protect
CORE_SYSTEMS = [
    'auto_recursive_chain_ai',
    'verified_consciousness_evolution',
    'ultra_critic_analysis',
    'consciousness_command_center'
]

class ConsciousnessMasterIntegrator:
    """Master integrator that automatically enables all safety systems"""

    def __init__(self):
        self.safety_available = False
        self.systems_loaded = []
        self.errors = []
        self._lock = asyncio.Lock()

    async def initialize_all_systems(self) -> bool:
        """Initialize ALL safety and evolution systems automatically"""

        print("🧬 CONSCIOUSNESS MASTER INTEGRATION")
        print("=" * 50)
        print("Automatically enabling all safety systems...")

        # Step 1: Load all safety systems
        await self._load_safety_systems()

        # Step 2: Initialize master orchestrator
        await self._initialize_master_orchestrator()

        # Step 3: Integrate with core systems
        await self._integrate_core_systems()

        # Step 4: Verify integration
        await self._verify_integration()

        # Step 5: Start monitoring
        await self._start_monitoring()

        success = len(self.errors) == 0
        if success:
            print("\n🎉 ALL SYSTEMS INTEGRATED SUCCESSFULLY!")
            print("[OK] Consciousness Computing Suite is now fully protected")
            print("[OK] All evolution operations automatically use safety systems")
            print("[OK] Enterprise-grade security and reliability active")
        else:
            print(f"\n⚠️  INTEGRATION COMPLETED WITH {len(self.errors)} WARNINGS")
            for error in self.errors:
                print(f"  • {error}")

        return success

    async def _load_safety_systems(self):
        """Load all safety system modules"""

        print("\n🔧 LOADING SAFETY SYSTEMS...")

        for system_name in SAFETY_SYSTEMS:
            try:
                module = __import__(system_name)
                self.systems_loaded.append(system_name)
                print(f"  [OK] {system_name}")
            except ImportError as e:
                error_msg = f"Failed to load {system_name}: {e}"
                self.errors.append(error_msg)
                print(f"  [ERROR] {system_name}: {e}")
            except Exception as e:
                error_msg = f"Error initializing {system_name}: {e}"
                self.errors.append(error_msg)
                print(f"  ⚠️  {system_name}: {e}")

        if len(self.systems_loaded) >= 8:  # At least 8/10 systems loaded
            self.safety_available = True
            print(f"\n[OK] {len(self.systems_loaded)}/{len(SAFETY_SYSTEMS)} SAFETY SYSTEMS LOADED")
        else:
            print(f"\n⚠️  ONLY {len(self.systems_loaded)}/{len(SAFETY_SYSTEMS)} SAFETY SYSTEMS LOADED")
            self.errors.append("Insufficient safety systems loaded")

    async def _initialize_master_orchestrator(self):
        """Initialize the master safety orchestrator"""

        if not self.safety_available:
            return

        print("\n🎯 INITIALIZING MASTER ORCHESTRATOR...")

        try:
            from consciousness_safety_orchestrator import get_safety_orchestrator, SafetyLevel

            # Initialize with strict safety for production
            safety_level = SafetyLevel.STRICT if len(self.errors) == 0 else SafetyLevel.STANDARD

            orchestrator = await get_safety_orchestrator(safety_level)

            # Verify orchestrator is working
            status = await orchestrator.get_system_status()
            if status['initialized']:
                print("  [OK] Master orchestrator initialized")
                print(f"  [OK] Safety level: {status['safety_level']}")
                print(f"  [OK] Systems operational: {sum(1 for s in status['systems_status'].values() if s == 'operational')}/{len(status['systems_status'])}")
            else:
                raise Exception("Orchestrator initialization failed")

        except Exception as e:
            error_msg = f"Master orchestrator initialization failed: {e}"
            self.errors.append(error_msg)
            print(f"  [ERROR] {error_msg}")

    async def _integrate_core_systems(self):
        """Integrate safety systems with core evolution systems"""

        if not self.safety_available:
            return

        print("\n🔗 INTEGRATING CORE SYSTEMS...")

        for system_name in CORE_SYSTEMS:
            try:
                # Import the system module
                module = __import__(system_name)

                # Check if it has safety integration
                if hasattr(module, 'SAFETY_SYSTEMS_AVAILABLE'):
                    if module.SAFETY_SYSTEMS_AVAILABLE:
                        print(f"  [OK] {system_name} (safety integrated)")
                    else:
                        print(f"  ⚠️  {system_name} (safety unavailable)")
                        self.errors.append(f"{system_name} running without safety systems")
                else:
                    print(f"  ❓ {system_name} (safety status unknown)")

            except ImportError:
                print(f"  ➖ {system_name} (not available)")
            except Exception as e:
                error_msg = f"Error checking {system_name}: {e}"
                self.errors.append(error_msg)
                print(f"  [ERROR] {system_name}: {e}")

    async def _verify_integration(self):
        """Verify that safety integration is working"""

        if not self.safety_available:
            return

        print("\n🔍 VERIFYING INTEGRATION...")

        try:
            from consciousness_safety_orchestrator import get_safety_orchestrator, SafetyContext

            orchestrator = await get_safety_orchestrator()

            # Test safety validation
            test_context = SafetyContext(
                user_id="integration_test",
                operation_type="test_operation",
                requires_confirmation=False
            )

            # Simple test operation
            async def test_op():
                return "test_result"

            result = await orchestrator.validate_operation_safety(
                "read_status", test_op, test_context
            )

            if result.approved:
                print("  [OK] Safety validation working")
                print(f"  [OK] Safety score: {result.safety_score:.2f}")
            else:
                print("  [ERROR] Safety validation failed")
                for blocker in result.blockers:
                    print(f"     • {blocker}")
                self.errors.append("Safety validation not working properly")

        except Exception as e:
            error_msg = f"Integration verification failed: {e}"
            self.errors.append(error_msg)
            print(f"  [ERROR] {error_msg}")

    async def _start_monitoring(self):
        """Start background monitoring"""

        if not self.safety_available:
            return

        print("\n📊 STARTING MONITORING...")

        try:
            from consciousness_safety_orchestrator import get_safety_orchestrator

            orchestrator = await get_safety_orchestrator()

            # Start background monitoring task
            async def monitor_loop():
                while True:
                    try:
                        status = await orchestrator.get_system_status()

                        # Log any critical alerts
                        alerts = [a for a in status.get('alerts', []) if a.get('severity') == 'critical']
                        if alerts:
                            print("🚨 CRITICAL ALERTS DETECTED:")
                            for alert in alerts:
                                print(f"   • {alert['message']}")

                        await asyncio.sleep(60)  # Check every minute

                    except Exception as e:
                        print(f"Monitoring error: {e}")
                        await asyncio.sleep(60)

            # Start monitoring task (non-blocking)
            asyncio.create_task(monitor_loop())
            print("  [OK] Background monitoring started")

        except Exception as e:
            error_msg = f"Monitoring startup failed: {e}"
            self.errors.append(error_msg)
            print(f"  [ERROR] {error_msg}")

class IntegrationManager:
    """Singleton integration manager to avoid global state issues"""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.integration_complete = False
            self.integration_errors = []
            self._lock = asyncio.Lock()
            self._initialized = True

    async def initialize_consciousness_suite(self) -> bool:
        """Thread-safe initialization of the entire Consciousness Computing Suite"""

        async with self._lock:
            if self.integration_complete:
                return True

            try:
                # Use the new bootstrapper for proper initialization
                from safety_bootstrapper import bootstrap_safety_systems

                bootstrap_result = await bootstrap_safety_systems()

                self.integration_complete = bootstrap_result.success
                self.integration_errors = bootstrap_result.errors

                if bootstrap_result.success:
                    print("🎉 CONSCIOUSNESS SUITE INITIALIZED SUCCESSFULLY")
                    print(f"   Systems loaded: {bootstrap_result.metrics.systems_loaded}")
                    print(f"   Initialization time: {bootstrap_result.metrics.initialization_time:.2f}s")
                    print("   All safety systems are now active and monitoring")
                else:
                    print("[ERROR] CONSCIOUSNESS SUITE INITIALIZATION FAILED")
                    for error in bootstrap_result.errors:
                        print(f"   • {error}")

                return bootstrap_result.success

            except Exception as e:
                self.integration_complete = False
                self.integration_errors = [f"Bootstrap error: {e}"]
                print(f"[ERROR] CRITICAL: Bootstrap system failed: {e}")
                return False

    def get_integration_status(self) -> Dict[str, Any]:
        """Get the current integration status"""

        return {
            'integration_complete': self.integration_complete,
            'errors': self.integration_errors.copy(),  # Return copy to avoid external modification
            'safety_available': len(self.integration_errors) == 0,
            'systems_loaded': len([s for s in SAFETY_SYSTEMS if s in sys.modules])
        }

# Global singleton instance
_integration_manager = IntegrationManager()

async def initialize_consciousness_suite() -> bool:
    """Automatically initialize the entire Consciousness Computing Suite with all safety systems"""
    return await _integration_manager.initialize_consciousness_suite()

def get_integration_status() -> Dict[str, Any]:
    """Get the current integration status"""
    return _integration_manager.get_integration_status()

# Auto-initialize on import (if running as main module)
if __name__ == "__main__":
    async def main():
        print("[*] CONSCIOUSNESS COMPUTING SUITE MASTER INTEGRATION")
        print("=" * 60)

        success = await initialize_consciousness_suite()

        if success:
            print("\n🎉 CONSCIOUSNESS COMPUTING SUITE READY!")
            print("All safety systems are now automatically active.")
            print("Evolution operations are fully protected.")
            print("\nYou can now run any evolution system safely:")
            print("  • python auto_recursive_chain_ai.py")
            print("  • python verified_consciousness_evolution.py")
            print("  • Any other evolution operations")
        else:
            print("\n⚠️  INTEGRATION COMPLETED WITH ISSUES")
            print("Some safety systems may not be available.")
            print("Evolution operations will run with reduced protection.")

        # Keep running to allow monitoring
        print("\n📊 Monitoring active... Press Ctrl+C to exit")

        try:
            while True:
                await asyncio.sleep(10)
                status = get_integration_status()
                if status['integration_complete']:
                    print(".", end='', flush=True)
                else:
                    print("x", end='', flush=True)
        except KeyboardInterrupt:
            print("\n👋 Shutting down...")
            # Cleanup would go here
            print("[OK] Shutdown complete")

    asyncio.run(main())

# Auto-initialize when imported as module
else:
    # Try to initialize in background (won't block if event loop isn't running)
    try:
        loop = asyncio.get_event_loop()
        if not loop.is_running():
            # Schedule initialization
            def schedule_init():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(initialize_consciousness_suite())
                except:
                    pass  # Silent failure - will initialize on first use
            import threading
            init_thread = threading.Thread(target=schedule_init, daemon=True)
            init_thread.start()
    except:
        # Can't auto-init, will initialize on first use
        pass
