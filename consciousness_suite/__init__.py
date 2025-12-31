"""
Consciousness Computing Suite - Enterprise AI Safety & Evolution Platform

This package provides a comprehensive suite of tools for safe, intelligent AI evolution
with built-in security, validation, and monitoring systems.
"""

__version__ = "2.0.0"
__author__ = "Consciousness AI"
__description__ = "Enterprise-grade Consciousness Computing Suite with automatic safety systems"

# Core imports for easy access
def __getattr__(name):
    """Lazy imports to avoid circular dependencies"""
    if name == "AutoRecursiveChainAI":
        from auto_recursive_chain_ai import AutoRecursiveChainAI
        return AutoRecursiveChainAI
    elif name == "VerifiedEvolutionEngine":
        from verified_consciousness_evolution import VerifiedEvolutionEngine
        return VerifiedEvolutionEngine
    elif name == "ConsciousnessSafetyOrchestrator":
        from consciousness_safety_orchestrator import ConsciousnessSafetyOrchestrator
        return ConsciousnessSafetyOrchestrator
    elif name == "get_safety_orchestrator":
        from consciousness_safety_orchestrator import get_safety_orchestrator
        return get_safety_orchestrator
    elif name == "initialize_consciousness_suite":
        from consciousness_master_integration import initialize_consciousness_suite
        return initialize_consciousness_suite
    elif name == "EvolutionAuthSystem":
        from evolution_auth_system import EvolutionAuthSystem
        return EvolutionAuthSystem
    elif name == "EvolutionAuthGuard":
        from evolution_auth_system import EvolutionAuthGuard
        return EvolutionAuthGuard
    elif name == "EvolutionLockManager":
        from evolution_locking import EvolutionLockManager
        return EvolutionLockManager
    elif name == "EvolutionLockGuard":
        from evolution_locking import EvolutionLockGuard
        return EvolutionLockGuard
    elif name == "TransactionalEvolutionManager":
        from transactional_evolution import TransactionalEvolutionManager
        return TransactionalEvolutionManager
    elif name == "EvolutionValidator":
        from evolution_validation import EvolutionValidator
        return EvolutionValidator
    elif name == "FitnessCalculator":
        from evolution_validation import FitnessCalculator
        return FitnessCalculator
    elif name == "ResourceQuotaManager":
        from resource_quotas import ResourceQuotaManager
        return ResourceQuotaManager
    elif name == "EvolutionSafetyUI":
        from ui_safety import EvolutionSafetyUI
        return EvolutionSafetyUI
    elif name == "OptimizedEvolutionAnalyzer":
        from complexity_optimization import OptimizedEvolutionAnalyzer
        return OptimizedEvolutionAnalyzer
    elif name == "EvolutionContractValidator":
        from contract_validation import EvolutionContractValidator
        return EvolutionContractValidator
    elif name == "SecureContractLoader":
        from contract_validation import SecureContractLoader
        return SecureContractLoader
    elif name == "NetworkResilienceManager":
        from error_recovery import NetworkResilienceManager
        return NetworkResilienceManager
    elif name == "resilient_operation":
        from error_recovery import resilient_operation
        return resilient_operation
    # Add other imports as needed
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    # Main classes
    'AutoRecursiveChainAI',
    'VerifiedEvolutionEngine',
    'ConsciousnessSafetyOrchestrator',

    # Safety systems
    'EvolutionAuthSystem',
    'EvolutionAuthGuard',
    'EvolutionLockManager',
    'EvolutionLockGuard',
    'TransactionalEvolutionManager',
    'EvolutionValidator',
    'FitnessCalculator',
    'ResourceQuotaManager',
    'EvolutionSafetyUI',
    'OptimizedEvolutionAnalyzer',
    'EvolutionContractValidator',
    'SecureContractLoader',
    'NetworkResilienceManager',

    # Functions
    'get_safety_orchestrator',
    'initialize_consciousness_suite',
    'resilient_operation',
]
