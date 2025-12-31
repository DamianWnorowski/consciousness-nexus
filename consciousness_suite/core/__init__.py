"""
Consciousness Computing Suite - Core Infrastructure
==================================================

Core utilities, base classes, and shared functionality for all consciousness
computing systems.

Components:
- Base classes for all analyzers and orchestrators
- Common data structures and models
- Utility functions for async operations
- Configuration management
- Logging and monitoring infrastructure
- Error handling and resilience patterns
"""

from .async_utils import AsyncTaskManager, RateLimiter
from .base import BaseAnalyzer, BaseOrchestrator, BaseProcessor
from .config import ConfigManager
from .data_models import (
    AnalysisResult,
    ConfidenceScore,
    ProcessingContext,
    ProcessingMetadata,
    SystemMetrics,
)
from .logging import ConsciousnessLogger

__all__ = [
    'BaseAnalyzer', 'BaseOrchestrator', 'BaseProcessor',
    'ConfigManager', 'ConsciousnessLogger',
    'AsyncTaskManager', 'RateLimiter',
    'AnalysisResult', 'ProcessingContext', 'SystemMetrics',
    'ConfidenceScore', 'ProcessingMetadata'
]
