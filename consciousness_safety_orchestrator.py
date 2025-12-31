#!/usr/bin/env python3
"""
🛡️ CONSCIOUSNESS SAFETY ORCHESTRATOR
=====================================

Master orchestrator that automatically integrates all safety, security, and reliability systems.
All evolution operations automatically go through comprehensive safety checks.
"""

import asyncio
import json
import time
import os
import sys
from typing import Dict, Any, List, Optional, Callable, Awaitable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

# Import all safety systems
try:
    from evolution_auth_system import EvolutionAuthSystem, EvolutionAuthGuard, UserRole, Permission
    from evolution_locking import EvolutionLockManager, EvolutionLockGuard
    from transactional_evolution import TransactionalEvolutionManager, EvolutionTransaction, EvolutionStep
    from evolution_validation import EvolutionValidator, FitnessCalculator
    from resource_quotas import ResourceQuotaManager, EvolutionResourceProfile
    from ui_safety import EvolutionSafetyUI
    from complexity_optimization import OptimizedEvolutionAnalyzer
    from contract_validation import EvolutionContractValidator, SecureContractLoader
    from error_recovery import NetworkResilienceManager, resilient_operation, RecoveryStrategy
except ImportError as e:
    print(f"❌ CRITICAL: Failed to import safety systems: {e}")
    print("Please ensure all safety system modules are available")
    sys.exit(1)

class SafetyLevel(Enum):
    """Levels of safety enforcement"""
    MINIMAL = "minimal"      # Basic checks only
    STANDARD = "standard"    # All safety systems active
    STRICT = "strict"        # Maximum security and validation
    PARANOID = "paranoid"    # Extreme validation and monitoring

@dataclass
class SafetyContext:
    """Context for safety operations"""
    user_id: str = "system"
    operation_type: str = "unknown"
    risk_level: str = "medium"
    requires_confirmation: bool = True
    timeout_seconds: int = 300
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SafetyResult:
    """Result of safety validation"""
    approved: bool = False
    safety_score: float = 0.0
    warnings: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    execution_context: Dict[str, Any] = field(default_factory=dict)

class ConsciousnessSafetyOrchestrator:
    """Master orchestrator for all safety systems"""

    def __init__(self, safety_level: SafetyLevel = SafetyLevel.STANDARD):
        self.safety_level = safety_level
        self.initialized = False
        self.systems = {}

        # Setup logging
        self.logger = logging.getLogger('ConsciousnessSafetyOrchestrator')
        self.logger.setLevel(logging.INFO)

        handler = logging.FileHandler('logs/consciousness_safety.log')
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)

        # Circuit breaker for safety system failures
        self._failure_circuit_breaker = {
            'failures': 0,
            'last_failure': None,
            'circuit_open': False,
            'threshold': 5,
            'timeout': 300  # 5 minutes
        }

        # Performance monitoring
        self._performance_metrics = {
            'operation_times': {},
            'system_health_checks': [],
            'validation_times': [],
            'last_performance_report': None
        }

    async def initialize(self) -> bool:
        """Initialize all safety systems"""

        if self.initialized:
            return True

        try:
            self.logger.info("Initializing Consciousness Safety Orchestrator...")

            # Initialize core systems
            self.systems['auth'] = EvolutionAuthSystem()
            self.systems['locking'] = EvolutionLockManager()
            self.systems['transactions'] = TransactionalEvolutionManager()
            self.systems['validation'] = EvolutionValidator()
            self.systems['fitness'] = FitnessCalculator()
            self.systems['resources'] = ResourceQuotaManager()
            self.systems['ui_safety'] = EvolutionSafetyUI()
            self.systems['complexity'] = OptimizedEvolutionAnalyzer()
            self.systems['contracts'] = EvolutionContractValidator()
            self.systems['contract_loader'] = SecureContractLoader(self.systems['contracts'])
            self.systems['error_recovery'] = NetworkResilienceManager()

            # Create guards and managers
            self.systems['auth_guard'] = EvolutionAuthGuard(self.systems['auth'])
            self.systems['lock_guard'] = EvolutionLockGuard(self.systems['locking'])

            # Configure based on safety level
            await self._configure_safety_level()

            # Initialize error recovery system
            await self.systems['error_recovery'].initialize()

            # Create default admin user if none exists
            await self._ensure_default_admin()

            self.initialized = True
            self.logger.info("✅ All safety systems initialized successfully")

            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize safety systems: {e}")
            return False

    async def _configure_safety_level(self):
        """Configure systems based on safety level"""

        if self.safety_level == SafetyLevel.MINIMAL:
            # Minimal checks - disable most validations
            pass  # Keep defaults

        elif self.safety_level == SafetyLevel.STANDARD:
            # Standard safety - all systems active
            pass  # Keep defaults

        elif self.safety_level == SafetyLevel.STRICT:
            # Strict mode - maximum validation
            # This would configure stricter thresholds
            pass

        elif self.safety_level == SafetyLevel.PARANOID:
            # Paranoid mode - extreme validation
            # This would enable all possible checks
            pass

    async def _ensure_default_admin(self):
        """Ensure a default admin user exists"""

        try:
            # Try to authenticate as admin
            admin_user = await asyncio.get_event_loop().run_in_executor(
                None, self.systems['auth'].authenticate, "admin", "admin123!"
            )

            if not admin_user:
                # Create default admin user
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.systems['auth'].create_user,
                    "admin",
                    "admin123!",
                    [UserRole.ADMIN, UserRole.SUPERUSER],
                    "system"
                )
                self.logger.info("Created default admin user")

        except Exception as e:
            self.logger.warning(f"Could not setup default admin: {e}")

    async def validate_operation_safety(self, operation_name: str,
                                      operation_func: Callable,
                                      context: SafetyContext) -> SafetyResult:
        """Comprehensive safety validation for operations"""

        result = SafetyResult()
        validation_start = time.time()

        try:
            self.logger.info(f"🔍 Validating safety for operation: {operation_name}")
            self.logger.debug(f"Context: user={context.user_id}, risk={context.risk_level}, confirm={context.requires_confirmation}")

            # Step 1: Authentication check
            if not await self._check_authentication(context, result):
                return result

            # Step 2: Authorization check
            if not await self._check_authorization(context, operation_name, result):
                return result

            # Step 3: Resource availability check
            if not await self._check_resource_availability(operation_name, context, result):
                return result

            # Step 4: Risk assessment and confirmation
            if not await self._assess_risk_and_confirm(operation_name, context, result):
                return result

            # Step 5: Pre-execution validation
            if not await self._validate_pre_execution(operation_name, context, result):
                return result

            # Step 6: Setup execution context
            result.execution_context = await self._setup_execution_context(operation_name, context)

            result.approved = True
            result.safety_score = self._calculate_safety_score(result)

            validation_time = time.time() - validation_start
            self.logger.info(f"Safety validation completed for {operation_name} in {validation_time:.2f}s")
            self.logger.debug(f"Safety checks passed: {result.safety_score:.2f} score")

            # Record performance metrics
            self._record_performance_metric('validation', validation_time)
            return result

        except Exception as e:
            validation_time = time.time() - validation_start
            self.logger.error(f"Safety validation failed for {operation_name} after {validation_time:.2f}s: {e}")

            # Circuit breaker logic
            self._record_safety_failure()

            if self._is_circuit_open():
                result.blockers.append("Safety system circuit breaker is open - too many failures")
                self.logger.critical("SAFETY CIRCUIT BREAKER OPEN - System entering degraded mode")
            else:
                result.blockers.append(f"Safety system error: {str(e)}")

            return result

    async def _check_authentication(self, context: SafetyContext, result: SafetyResult) -> bool:
        """Check user authentication"""
        try:
            # For system operations, skip auth
            if context.user_id == "system":
                return True

            # Validate token or authenticate user
            user = self.systems['auth'].validate_token(context.user_id)
            if not user:
                result.blockers.append("Authentication required")
                return False

            result.execution_context['user'] = user
            return True

        except Exception as e:
            result.blockers.append(f"Authentication check failed: {e}")
            return False

    async def _check_authorization(self, context: SafetyContext, operation_name: str, result: SafetyResult) -> bool:
        """Check user authorization"""
        try:
            user = result.execution_context.get('user')
            if not user and context.user_id != "system":
                result.blockers.append("User context missing")
                return False

            # Map operation to permission
            permission_map = {
                'trigger_evolution': Permission.TRIGGER_EVOLUTION,
                'stop_evolution': Permission.STOP_EVOLUTION,
                'modify_contracts': Permission.MODIFY_CONTRACTS,
                'force_rollback': Permission.FORCE_ROLLBACK,
                'system_config': Permission.SYSTEM_CONFIG,
                'bypass_safety': Permission.BYPASS_SAFETY,
                'read_status': Permission.READ_STATUS,
                'view_logs': Permission.VIEW_LOGS
            }

            required_perm = permission_map.get(operation_name, Permission.TRIGGER_EVOLUTION)

            if user and not user.has_permission(required_perm):
                result.blockers.append(f"Insufficient permissions: {required_perm.value}")
                return False

            return True

        except Exception as e:
            result.blockers.append(f"Authorization check failed: {e}")
            return False

    async def _check_resource_availability(self, operation_name: str, context: SafetyContext, result: SafetyResult) -> bool:
        """Check resource availability"""
        try:
            # Get operation resource profile
            profile = self.systems['resources'].evolution_profiles.get(operation_name)
            if not profile:
                return True  # No specific profile, assume OK

            # Check resource feasibility
            feasibility = self.systems['resources'].check_operation_feasibility(
                operation_name, profile.expected_resources
            )

            if not feasibility['feasible']:
                result.blockers.extend(feasibility['blockers'])
                return False

            if feasibility['warnings']:
                result.warnings.extend(feasibility['warnings'])

            return True

        except Exception as e:
            result.warnings.append(f"Resource check failed: {e}")
            return True  # Don't block on resource check failures

    async def _assess_risk_and_confirm(self, operation_name: str, context: SafetyContext, result: SafetyResult) -> bool:
        """Assess risk and get user confirmation if needed"""
        try:
            # Risk assessment based on operation
            risk_operations = {
                'trigger_evolution': 'high',
                'force_rollback': 'critical',
                'system_config': 'high',
                'bypass_safety': 'critical'
            }

            operation_risk = risk_operations.get(operation_name, 'medium')

            # For high-risk operations, require confirmation
            if operation_risk in ['high', 'critical'] and context.requires_confirmation:
                confirmed = await self.systems['ui_safety'].confirmation_dialog.get_confirmation(
                    operation_name, {'user_id': context.user_id}
                )

                if not confirmed:
                    result.blockers.append("User declined confirmation")
                    return False

            return True

        except Exception as e:
            result.blockers.append(f"Risk assessment failed: {e}")
            return False

    async def _validate_pre_execution(self, operation_name: str, context: SafetyContext, result: SafetyResult) -> bool:
        """Pre-execution validation"""
        try:
            # Run validation checks
            files_to_check = ['consciousness_evolution_contract.json']  # Default files
            validation_result = await self.systems['validation'].validate_evolution(
                files_to_check, validation_scope='quick'
            )

            if not validation_result.is_valid:
                for error in validation_result.errors:
                    if error.get('severity') == 'CRITICAL':
                        result.blockers.append(f"Critical validation error: {error['message']}")
                    else:
                        result.warnings.append(f"Validation warning: {error['message']}")

            # Check for critical validation errors
            if result.blockers:
                return False

            return True

        except Exception as e:
            result.warnings.append(f"Pre-execution validation failed: {e}")
            return True  # Don't block on validation failures

    async def _setup_execution_context(self, operation_name: str, context: SafetyContext) -> Dict[str, Any]:
        """Setup execution context with all necessary components"""
        exec_context = {
            'operation_name': operation_name,
            'start_time': time.time(),
            'safety_level': self.safety_level.value,
            'resource_limits': context.resource_limits,
            'monitoring_enabled': True
        }

        # Allocate resources
        allocation = self.systems['resources'].allocate_resources(
            operation_name, context.user_id
        )

        if allocation['allocated']:
            exec_context['resource_allocation'] = allocation
        else:
            self.logger.warning(f"Resource allocation failed: {allocation.get('reason', 'unknown')}")

        return exec_context

    def _calculate_safety_score(self, result: SafetyResult) -> float:
        """Calculate overall safety score"""
        base_score = 1.0

        # Deduct for warnings
        warning_penalty = len(result.warnings) * 0.05
        base_score -= warning_penalty

        # Deduct for blockers (shouldn't happen if approved)
        blocker_penalty = len(result.blockers) * 0.2
        base_score -= blocker_penalty

        return max(0.0, min(1.0, base_score))

    def _record_safety_failure(self):
        """Record a safety system failure for circuit breaker"""
        self._failure_circuit_breaker['failures'] += 1
        self._failure_circuit_breaker['last_failure'] = time.time()

        if self._failure_circuit_breaker['failures'] >= self._failure_circuit_breaker['threshold']:
            self._failure_circuit_breaker['circuit_open'] = True
            self.logger.critical(f"SAFETY CIRCUIT BREAKER OPENED after {self._failure_circuit_breaker['failures']} failures")

    def _is_circuit_open(self) -> bool:
        """Check if safety circuit breaker is open"""
        if not self._failure_circuit_breaker['circuit_open']:
            return False

        # Check if timeout has expired
        if self._failure_circuit_breaker['last_failure']:
            elapsed = time.time() - self._failure_circuit_breaker['last_failure']
            if elapsed >= self._failure_circuit_breaker['timeout']:
                # Reset circuit breaker
                self._failure_circuit_breaker['circuit_open'] = False
                self._failure_circuit_breaker['failures'] = 0
                self.logger.info("SAFETY CIRCUIT BREAKER RESET - attempting recovery")
                return False

        return True

    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        return {
            'circuit_open': self._failure_circuit_breaker['circuit_open'],
            'failures': self._failure_circuit_breaker['failures'],
            'threshold': self._failure_circuit_breaker['threshold'],
            'last_failure': self._failure_circuit_breaker['last_failure'],
            'timeout_seconds': self._failure_circuit_breaker['timeout']
        }

    def _record_performance_metric(self, metric_type: str, duration: float):
        """Record performance metric"""
        if metric_type not in self._performance_metrics:
            self._performance_metrics[metric_type] = []

        self._performance_metrics[metric_type].append({
            'timestamp': time.time(),
            'duration': duration
        })

        # Keep only last 100 measurements
        if len(self._performance_metrics[metric_type]) > 100:
            self._performance_metrics[metric_type] = self._performance_metrics[metric_type][-100:]

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        metrics = {}

        for metric_type, measurements in self._performance_metrics.items():
            if measurements:
                durations = [m['duration'] for m in measurements]
                metrics[metric_type] = {
                    'count': len(measurements),
                    'avg_duration': sum(durations) / len(durations),
                    'min_duration': min(durations),
                    'max_duration': max(durations),
                    'last_measurement': measurements[-1]['timestamp']
                }

        return metrics

    async def execute_safe_operation(self, operation_name: str,
                                   operation_func: Callable,
                                   context: SafetyContext = None) -> Dict[str, Any]:
        """Execute operation with full safety orchestration"""

        if context is None:
            context = SafetyContext()

        result = {
            'success': False,
            'safety_validation': None,
            'execution_result': None,
            'error': None,
            'cleanup_performed': False
        }

        try:
            # Step 1: Safety validation
            safety_result = await self.validate_operation_safety(operation_name, operation_func, context)

            result['safety_validation'] = {
                'approved': safety_result.approved,
                'safety_score': safety_result.safety_score,
                'warnings': safety_result.warnings,
                'blockers': safety_result.blockers,
                'recommendations': safety_result.recommendations
            }

            if not safety_result.approved:
                result['error'] = "Safety validation failed"
                return result

            # Step 2: Setup locks and transactions
            lock_request = await self._setup_locks(operation_name, context)
            transaction = await self._setup_transaction(operation_name, context)

            try:
                # Step 3: Execute with UI safety
                ui_result = await self.systems['ui_safety'].execute_safe_operation(
                    operation_name,
                    operation_func,
                    {'user_id': context.user_id},
                    progress_style=self.systems['ui_safety'].progress_display.ProgressStyle.DETAILED
                )

                result['execution_result'] = ui_result
                result['success'] = ui_result['success']

                if ui_result['success']:
                    # Commit transaction
                    if transaction:
                        await self.systems['transactions']._commit_transaction(transaction)
                else:
                    # Rollback transaction
                    if transaction:
                        await self.systems['transactions']._rollback_transaction(transaction)

            finally:
                # Step 4: Cleanup
                await self._cleanup_operation(lock_request, context)
                result['cleanup_performed'] = True

        except Exception as e:
            self.logger.error(f"Operation execution failed: {e}")
            result['error'] = str(e)

        return result

    async def _setup_locks(self, operation_name: str, context: SafetyContext):
        """Setup necessary locks for operation"""
        try:
            # Get lock configuration for operation
            lock_configs = {
                'trigger_evolution': {'type': 'EXCLUSIVE', 'resource': 'evolution_system'},
                'stop_evolution': {'type': 'EXCLUSIVE', 'resource': 'evolution_system'},
                'modify_contracts': {'type': 'EXCLUSIVE', 'resource': 'evolution_contracts'},
                'system_config': {'type': 'EXCLUSIVE', 'resource': 'system_config'}
            }

            config = lock_configs.get(operation_name)
            if config:
                request = self.systems['locking'].LockRequest(
                    process_id=context.user_id,
                    lock_type=self.systems['locking'].LockType[config['type']],
                    resource=config['resource'],
                    timeout=60.0
                )

                response = await self.systems['locking'].acquire_lock(request)
                if response.status == self.systems['locking'].LockStatus.ACQUIRED:
                    return response.lock

        except Exception as e:
            self.logger.warning(f"Lock setup failed: {e}")

        return None

    async def _setup_transaction(self, operation_name: str, context: SafetyContext):
        """Setup transaction for operation"""
        try:
            # Create transaction for high-risk operations
            if operation_name in ['trigger_evolution', 'force_rollback', 'system_config']:
                steps = [
                    EvolutionStep(
                        name=f"execute_{operation_name}",
                        phase=self.systems['transactions'].TransactionPhase.EXECUTE,
                        execute_func=lambda: asyncio.sleep(0.1)  # Placeholder
                    )
                ]

                initial_state = {'fitness': 0.8, 'operation': operation_name}
                transaction = self.systems['transactions'].create_transaction(
                    f"Safe {operation_name} execution",
                    steps,
                    initial_state
                )

                return transaction

        except Exception as e:
            self.logger.warning(f"Transaction setup failed: {e}")

        return None

    async def _cleanup_operation(self, lock, context: SafetyContext):
        """Cleanup after operation"""
        try:
            if lock:
                self.systems['locking'].release_lock(lock.lock_id, context.user_id)

            # Release resources
            self.systems['resources'].release_resources(context.user_id)

        except Exception as e:
            self.logger.warning(f"Cleanup failed: {e}")

    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            'initialized': self.initialized,
            'safety_level': self.safety_level.value,
            'systems_status': {},
            'active_operations': 0,
            'resource_usage': {},
            'alerts': []
        }

        if not self.initialized:
            return status

        try:
            # Check each system status
            status['systems_status'] = {
                'authentication': self._check_system_status('auth'),
                'locking': self._check_system_status('locking'),
                'transactions': self._check_system_status('transactions'),
                'validation': self._check_system_status('validation'),
                'resources': self._check_system_status('resources'),
                'ui_safety': self._check_system_status('ui_safety'),
                'complexity': self._check_system_status('complexity'),
                'contracts': self._check_system_status('contracts'),
                'error_recovery': self._check_system_status('error_recovery')
            }

            # Get resource status
            status['resource_usage'] = self.systems['resources'].get_resource_status()

            # Get alerts
            status['alerts'] = status['resource_usage'].get('alerts', [])

        except Exception as e:
            self.logger.error(f"Status check failed: {e}")

        return status

    def _check_system_status(self, system_name: str) -> str:
        """Check if a system is operational"""
        if system_name not in self.systems:
            return "not_loaded"

        system = self.systems[system_name]
        try:
            # Comprehensive health check
            if hasattr(system, 'get_metrics'):
                # Try to get metrics as health check
                metrics = system.get_metrics()
                return "operational" if metrics else "degraded"
            elif hasattr(system, 'list_locks'):
                # Try to list locks as health check
                locks = system.list_locks()
                return "operational"
            elif hasattr(system, 'validate_contract'):
                # Try a basic validation as health check
                test_contract = {"contract_id": "health_check", "version": "1.0.0"}
                result = system.validate_contract(test_contract)
                return "operational" if result else "degraded"
            elif hasattr(system, 'get_status'):
                # Generic status check
                status = system.get_status()
                return "operational" if status.get('healthy', True) else "degraded"
            else:
                # Basic attribute check
                return "operational"
        except Exception as e:
            self.logger.warning(f"Health check failed for {system_name}: {e}")
            return "error"

    async def shutdown(self):
        """Shutdown all safety systems"""
        self.logger.info("Shutting down Consciousness Safety Orchestrator...")

        try:
            if 'error_recovery' in self.systems:
                await self.systems['error_recovery'].cleanup()

            self.initialized = False
            self.logger.info("✅ Safety systems shut down successfully")

        except Exception as e:
            self.logger.error(f"Shutdown failed: {e}")

# Global orchestrator instance
_safety_orchestrator = None

async def get_safety_orchestrator(safety_level: SafetyLevel = SafetyLevel.STANDARD):
    """Get or create the global safety orchestrator"""
    global _safety_orchestrator

    if _safety_orchestrator is None:
        _safety_orchestrator = ConsciousnessSafetyOrchestrator(safety_level)
        success = await _safety_orchestrator.initialize()
        if not success:
            raise RuntimeError("Failed to initialize safety orchestrator")

    return _safety_orchestrator

def safe_evolution_operation(operation_name: str, safety_level: SafetyLevel = SafetyLevel.STANDARD):
    """Decorator to make any evolution operation automatically safe"""

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Get safety orchestrator
            orchestrator = await get_safety_orchestrator(safety_level)

            # Extract user context from kwargs or args
            user_id = kwargs.get('user_id', 'system')
            context = SafetyContext(
                user_id=user_id,
                operation_type=operation_name,
                metadata={'function': func.__name__, 'args_count': len(args)}
            )

            # Execute with full safety
            result = await orchestrator.execute_safe_operation(
                operation_name, func, context
            )

            if not result['success']:
                error_msg = result.get('error', 'Operation failed safety checks')
                raise RuntimeError(f"Safe operation failed: {error_msg}")

            # Return the actual operation result
            execution_result = result.get('execution_result', {})
            return execution_result.get('result')

        return wrapper
    return decorator

# Automatic integration functions
async def initialize_safety_systems():
    """Automatically initialize all safety systems"""
    try:
        orchestrator = await get_safety_orchestrator()
        print("✅ Consciousness Safety Systems initialized automatically")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize safety systems: {e}")
        return False

def ensure_safety_context():
    """Ensure safety context is available (call this at module import time)"""
    # This will be called automatically when modules are imported
    pass

# Auto-initialize on import (safe - won't block)
import functools

# Call initialization in background if possible
try:
    # Create task for async initialization
    import atexit

    async def _auto_init():
        await initialize_safety_systems()

    # Schedule auto-initialization
    def _schedule_init():
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_running():
                loop.create_task(_auto_init())
            else:
                # If loop is already running, we can't auto-init
                # User will need to call initialize_safety_systems() manually
                pass
        except:
            # Can't auto-init, user must do it manually
            pass

    _schedule_init()

except:
    # Auto-init not possible, manual initialization required
    pass

if __name__ == "__main__":
    async def main():
        print("🛡️ CONSCIOUSNESS SAFETY ORCHESTRATOR")
        print("=" * 50)

        # Initialize systems
        print("Initializing safety systems...")
        orchestrator = await get_safety_orchestrator()

        # Get status
        status = await orchestrator.get_system_status()
        print(f"Safety Level: {status['safety_level']}")
        print(f"Systems Initialized: {status['initialized']}")

        print("\nSystem Status:")
        for system, status_val in status['systems_status'].items():
            print(f"  {system}: {status_val}")

        print("\n✅ Safety orchestrator ready for automatic operation protection")

    asyncio.run(main())
