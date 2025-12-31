#!/usr/bin/env python3
"""
🛡️ SAFETY BOOTSTRAPPER
======================

Centralized safety system initialization with proper error handling and monitoring.
"""

import asyncio
import time
import os
import sys
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
import logging

@dataclass
class BootstrapMetrics:
    """Metrics for bootstrap process"""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    systems_loaded: int = 0
    systems_failed: int = 0
    initialization_time: float = 0.0
    health_checks_passed: int = 0
    health_checks_failed: int = 0

@dataclass
class BootstrapResult:
    """Result of bootstrap process"""
    success: bool
    metrics: BootstrapMetrics
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    systems_status: Dict[str, str] = field(default_factory=dict)

class SafetyBootstrapper:
    """Centralized bootstrapper for all safety systems"""

    def __init__(self):
        self.logger = logging.getLogger('SafetyBootstrapper')
        self.logger.setLevel(logging.INFO)

        handler = logging.FileHandler('logs/safety_bootstrap.log')
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)

        # Bootstrap state
        self._initialized = False
        self._lock = asyncio.Lock()

    async def bootstrap_safety_systems(self, safety_level: str = "standard") -> BootstrapResult:
        """Bootstrap all safety systems with comprehensive error handling"""

        async with self._lock:
            if self._initialized:
                return BootstrapResult(
                    success=True,
                    metrics=BootstrapMetrics(),
                    systems_status={"all": "already_initialized"}
                )

            metrics = BootstrapMetrics()
            result = BootstrapResult(success=True, metrics=metrics)

            try:
                self.logger.info("🚀 Starting safety systems bootstrap...")

                # Phase 1: Load safety system modules
                await self._load_safety_modules(result)

                # Phase 2: Initialize core orchestrator
                if result.success:
                    await self._initialize_orchestrator(result, safety_level)

                # Phase 3: Health checks
                if result.success:
                    await self._perform_health_checks(result)

                # Phase 4: Integration verification
                if result.success:
                    await self._verify_integration(result)

                metrics.end_time = time.time()
                metrics.initialization_time = metrics.end_time - metrics.start_time

                if result.success:
                    self._initialized = True
                    self.logger.info("✅ Safety systems bootstrap completed successfully")
                else:
                    self.logger.error("❌ Safety systems bootstrap failed")

            except Exception as e:
                result.success = False
                result.errors.append(f"Bootstrap failed: {e}")
                self.logger.error(f"Bootstrap error: {e}")

            return result

    async def _load_safety_modules(self, result: BootstrapResult):
        """Load all safety system modules"""

        safety_modules = [
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

        self.logger.info("📦 Loading safety modules...")

        for module_name in safety_modules:
            try:
                __import__(module_name)
                result.metrics.systems_loaded += 1
                result.systems_status[module_name] = "loaded"
                self.logger.debug(f"  ✅ {module_name}")
            except ImportError as e:
                result.metrics.systems_failed += 1
                result.errors.append(f"Failed to load {module_name}: {e}")
                result.systems_status[module_name] = "import_failed"
                self.logger.error(f"  ❌ {module_name}: {e}")
            except Exception as e:
                result.metrics.systems_failed += 1
                result.errors.append(f"Error loading {module_name}: {e}")
                result.systems_status[module_name] = "error"
                self.logger.error(f"  ⚠️  {module_name}: {e}")

        # Require at least 8/10 systems for success
        if result.metrics.systems_loaded < 8:
            result.success = False
            result.errors.append(f"Insufficient safety systems loaded: {result.metrics.systems_loaded}/10")

    async def _initialize_orchestrator(self, result: BootstrapResult, safety_level: str):
        """Initialize the core safety orchestrator"""

        self.logger.info("🎯 Initializing safety orchestrator...")

        try:
            from consciousness_safety_orchestrator import get_safety_orchestrator, SafetyLevel

            # Map string to enum
            level_map = {
                "minimal": SafetyLevel.MINIMAL,
                "standard": SafetyLevel.STANDARD,
                "strict": SafetyLevel.STRICT,
                "paranoid": SafetyLevel.PARANOID
            }
            safety_enum = level_map.get(safety_level, SafetyLevel.STANDARD)

            orchestrator = await get_safety_orchestrator(safety_enum)

            # Verify orchestrator is functional
            status = await orchestrator.get_system_status()
            if status.get('initialized'):
                result.systems_status['orchestrator'] = "operational"
                self.logger.info("  ✅ Orchestrator initialized")
            else:
                raise Exception("Orchestrator initialization failed")

        except Exception as e:
            result.success = False
            result.errors.append(f"Orchestrator initialization failed: {e}")
            result.systems_status['orchestrator'] = "failed"
            self.logger.error(f"  ❌ Orchestrator: {e}")

    async def _perform_health_checks(self, result: BootstrapResult):
        """Perform health checks on all systems"""

        self.logger.info("🏥 Performing health checks...")

        try:
            from consciousness_safety_orchestrator import get_safety_orchestrator

            orchestrator = await get_safety_orchestrator()
            status = await orchestrator.get_system_status()

            systems_status = status.get('systems_status', {})

            for system_name, system_status in systems_status.items():
                if system_status == "operational":
                    result.metrics.health_checks_passed += 1
                else:
                    result.metrics.health_checks_failed += 1
                    result.warnings.append(f"Health check failed for {system_name}: {system_status}")

            self.logger.info(f"  ✅ Health checks: {result.metrics.health_checks_passed} passed, {result.metrics.health_checks_failed} failed")

        except Exception as e:
            result.warnings.append(f"Health check process failed: {e}")
            self.logger.error(f"  ❌ Health checks: {e}")

    async def _verify_integration(self, result: BootstrapResult):
        """Verify system integration works"""

        self.logger.info("🔗 Verifying system integration...")

        try:
            from consciousness_safety_orchestrator import get_safety_orchestrator, SafetyContext

            orchestrator = await get_safety_orchestrator()

            # Test basic operation validation
            test_context = SafetyContext(
                user_id="bootstrap_test",
                operation_type="health_check",
                requires_confirmation=False
            )

            async def dummy_operation():
                return "integration_test_passed"

            validation_result = await orchestrator.validate_operation_safety(
                "read_status", dummy_operation, test_context
            )

            if validation_result.approved:
                result.systems_status['integration'] = "verified"
                self.logger.info("  ✅ Integration verified")
            else:
                result.success = False
                result.errors.append("Integration verification failed")
                result.systems_status['integration'] = "failed"
                self.logger.error("  ❌ Integration failed")

        except Exception as e:
            result.success = False
            result.errors.append(f"Integration verification error: {e}")
            result.systems_status['integration'] = "error"
            self.logger.error(f"  ❌ Integration: {e}")

    def get_bootstrap_status(self) -> Dict[str, Any]:
        """Get current bootstrap status"""

        return {
            'initialized': self._initialized,
            'bootstrap_available': True,
            'last_bootstrap_attempt': getattr(self, '_last_attempt', None)
        }

# Global bootstrapper instance
_bootstrapper = SafetyBootstrapper()

async def bootstrap_safety_systems(safety_level: str = "standard") -> BootstrapResult:
    """Bootstrap all safety systems with comprehensive error handling"""
    return await _bootstrapper.bootstrap_safety_systems(safety_level)

def get_bootstrap_status() -> Dict[str, Any]:
    """Get current bootstrap status"""
    return _bootstrapper.get_bootstrap_status()

# Auto-bootstrap on import (safe - won't block)
async def _auto_bootstrap():
    """Automatic bootstrap for systems that need it"""
    try:
        # Only bootstrap if we're in an async context and systems aren't loaded
        if not _bootstrapper._initialized:
            result = await bootstrap_safety_systems()
            if not result.success:
                print("⚠️  Automatic safety bootstrap failed - manual initialization required")
    except:
        # Silent failure - manual bootstrap will be required
        pass

# Attempt auto-bootstrap (won't block if no event loop)
try:
    # This will only work if there's already an event loop running
    loop = asyncio.get_running_loop()
    loop.create_task(_auto_bootstrap())
except RuntimeError:
    # No event loop running - manual bootstrap required
    pass
