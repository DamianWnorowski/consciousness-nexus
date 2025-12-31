"""
Base Classes for Consciousness Computing Suite
===============================================

Provides the foundational classes that all consciousness computing systems
inherit from, ensuring consistency, monitoring, and error handling across
all components.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .config import ConfigManager
from .logging import ConsciousnessLogger


@dataclass
class ProcessingContext:
    """Context information for processing operations"""
    session_id: str
    correlation_id: str
    start_time: datetime
    user_id: Optional[str] = None
    system_version: str = "v1.0.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemMetrics:
    """System performance and health metrics"""
    cpu_usage: float
    memory_usage: float
    processing_time: float
    error_count: int
    success_rate: float
    throughput: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ConfidenceScore:
    """Confidence scoring for analysis results"""
    value: float  # 0.0 to 1.0
    factors: List[str]
    uncertainty_reasons: List[str]
    calibration_date: datetime = field(default_factory=datetime.now)

@dataclass
class ProcessingMetadata:
    """Metadata for processing operations"""
    processor_name: str
    operation_type: str
    input_size: int
    output_size: int
    processing_steps: List[str]
    warnings: List[str]
    recommendations: List[str]

@dataclass
class AnalysisResult:
    """Standardized result structure for all analysis operations"""
    success: bool
    data: Dict[str, Any]
    confidence: ConfidenceScore
    metadata: ProcessingMetadata
    context: ProcessingContext
    processing_time: float
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

class BaseProcessor(ABC):
    """
    Base class for all processing components in the consciousness suite.

    Provides common functionality for:
    - Configuration management
    - Logging and monitoring
    - Error handling and resilience
    - Performance tracking
    - Context management
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.logger = ConsciousnessLogger(name)
        self.config_manager = ConfigManager()
        self._metrics = []
        self._is_initialized = False

    async def initialize(self) -> bool:
        """Initialize the processor"""
        try:
            await self._initialize_components()
            self._is_initialized = True
            self.logger.info("Processor initialized successfully", {
                'processor_name': self.name,
                'config_keys': list(self.config.keys())
            })
            return True
        except Exception as e:
            self.logger.error("Failed to initialize processor", {
                'processor_name': self.name,
                'error': str(e)
            })
            return False

    @abstractmethod
    async def _initialize_components(self):
        """Initialize processor-specific components"""
        pass

    async def process(self, input_data: Any, context: Optional[ProcessingContext] = None) -> AnalysisResult:
        """Main processing method with comprehensive error handling and monitoring"""

        if not self._is_initialized:
            raise RuntimeError(f"Processor {self.name} not initialized")

        # Create processing context if not provided
        if context is None:
            context = ProcessingContext(
                session_id=f"session_{int(time.time())}",
                correlation_id=f"corr_{int(time.time() * 1000)}",
                start_time=datetime.now()
            )

        start_time = time.time()
        processing_metadata = ProcessingMetadata(
            processor_name=self.name,
            operation_type=self._get_operation_type(),
            input_size=self._calculate_input_size(input_data),
            output_size=0,
            processing_steps=[],
            warnings=[],
            recommendations=[]
        )

        try:
            # Pre-processing validation
            await self._validate_input(input_data)

            # Core processing
            result_data = await self._process_core(input_data, context)

            # Post-processing validation
            validated_result = await self._validate_output(result_data)

            # Calculate confidence
            confidence = await self._calculate_confidence(validated_result, input_data)

            # Build metadata
            processing_metadata.output_size = self._calculate_output_size(validated_result)
            processing_metadata.processing_steps = await self._get_processing_steps()

            processing_time = time.time() - start_time

            # Record metrics
            await self._record_metrics(processing_time, True)

            result = AnalysisResult(
                success=True,
                data=validated_result,
                confidence=confidence,
                metadata=processing_metadata,
                context=context,
                processing_time=processing_time
            )

            self.logger.info("Processing completed successfully", {
                'processor_name': self.name,
                'processing_time': processing_time,
                'confidence': confidence.value,
                'correlation_id': context.correlation_id
            })

            return result

        except Exception as e:
            processing_time = time.time() - start_time

            # Record failure metrics
            await self._record_metrics(processing_time, False)

            error_result = AnalysisResult(
                success=False,
                data={},
                confidence=ConfidenceScore(0.0, [], ["Processing failed"]),
                metadata=processing_metadata,
                context=context,
                processing_time=processing_time,
                errors=[str(e)]
            )

            self.logger.error("Processing failed", {
                'processor_name': self.name,
                'error': str(e),
                'processing_time': processing_time,
                'correlation_id': context.correlation_id
            })

            return error_result

    @abstractmethod
    async def _process_core(self, input_data: Any, context: ProcessingContext) -> Dict[str, Any]:
        """Core processing logic - implemented by subclasses"""
        pass

    @abstractmethod
    def _get_operation_type(self) -> str:
        """Return the operation type for this processor"""
        pass

    async def _validate_input(self, input_data: Any):
        """Validate input data - can be overridden"""
        pass

    async def _validate_output(self, output_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate output data - can be overridden"""
        return output_data

    async def _calculate_confidence(self, result: Dict[str, Any], input_data: Any) -> ConfidenceScore:
        """Calculate confidence score - can be overridden"""
        return ConfidenceScore(0.8, ["Default confidence"], [])

    def _calculate_input_size(self, input_data: Any) -> int:
        """Calculate input data size"""
        if isinstance(input_data, str):
            return len(input_data)
        elif isinstance(input_data, (list, dict)):
            return len(str(input_data))
        return 0

    def _calculate_output_size(self, output_data: Dict[str, Any]) -> int:
        """Calculate output data size"""
        return len(str(output_data))

    async def _get_processing_steps(self) -> List[str]:
        """Get processing steps - can be overridden"""
        return ["Input validation", "Core processing", "Output validation"]

    async def _record_metrics(self, processing_time: float, success: bool):
        """Record performance metrics"""
        metrics = SystemMetrics(
            cpu_usage=0.0,  # Would be measured in real implementation
            memory_usage=0.0,
            processing_time=processing_time,
            error_count=0 if success else 1,
            success_rate=1.0 if success else 0.0,
            throughput=1.0 / processing_time if processing_time > 0 else 0.0
        )

        self._metrics.append(metrics)

        # Keep only last 1000 metrics
        if len(self._metrics) > 1000:
            self._metrics = self._metrics[-1000:]

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of recent performance metrics"""
        if not self._metrics:
            return {}

        recent_metrics = self._metrics[-100:]  # Last 100 operations

        return {
            'total_operations': len(recent_metrics),
            'avg_processing_time': sum(m.processing_time for m in recent_metrics) / len(recent_metrics),
            'success_rate': sum(m.success_rate for m in recent_metrics) / len(recent_metrics),
            'avg_throughput': sum(m.throughput for m in recent_metrics) / len(recent_metrics),
            'error_rate': sum(m.error_count for m in recent_metrics) / len(recent_metrics)
        }

class BaseAnalyzer(BaseProcessor):
    """
    Base class for analysis components.

    Provides common functionality for:
    - Data preprocessing and validation
    - Analysis result formatting
    - Confidence scoring
    - Pattern recognition utilities
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.analysis_patterns = {}
        self.confidence_calibrators = {}

    def _get_operation_type(self) -> str:
        return "analysis"

    async def _calculate_confidence(self, result: Dict[str, Any], input_data: Any) -> ConfidenceScore:
        """Calculate analysis-specific confidence"""
        factors = []
        uncertainty_reasons = []

        # Analyze result completeness
        if 'insights' in result and len(result['insights']) > 0:
            factors.append(f"Generated {len(result['insights'])} insights")
        else:
            uncertainty_reasons.append("No insights generated")

        # Analyze data quality
        if hasattr(input_data, '__len__') and len(input_data) < 10:
            uncertainty_reasons.append("Limited input data")

        # Calculate confidence score
        base_confidence = 0.8
        if uncertainty_reasons:
            base_confidence -= len(uncertainty_reasons) * 0.1
        if factors:
            base_confidence += len(factors) * 0.05

        confidence_value = max(0.0, min(1.0, base_confidence))

        return ConfidenceScore(
            value=confidence_value,
            factors=factors,
            uncertainty_reasons=uncertainty_reasons
        )

class BaseOrchestrator(BaseProcessor):
    """
    Base class for orchestration components.

    Provides common functionality for:
    - Multi-component coordination
    - Workflow management
    - Resource allocation
    - Progress tracking
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.coordination_patterns = {}
        self.workflow_templates = {}
        self.resource_allocators = {}

    def _get_operation_type(self) -> str:
        return "orchestration"

    async def coordinate_components(self, components: List[BaseProcessor],
                                  input_data: Any, context: ProcessingContext) -> Dict[str, Any]:
        """Coordinate multiple components in a workflow"""

        results = {}
        component_tasks = []

        # Create tasks for parallel execution where possible
        for component in components:
            if self._can_execute_parallel(component, components):
                task = asyncio.create_task(
                    component.process(input_data, context)
                )
                component_tasks.append((component.name, task))
            else:
                # Execute sequentially
                result = await component.process(input_data, context)
                results[component.name] = result

        # Wait for parallel tasks
        if component_tasks:
            parallel_results = await asyncio.gather(
                *[task for _, task in component_tasks],
                return_exceptions=True
            )

            # Process parallel results
            for (component_name, _), result in zip(component_tasks, parallel_results):
                if isinstance(result, Exception):
                    results[component_name] = AnalysisResult(
                        success=False,
                        data={},
                        confidence=ConfidenceScore(0.0, [], [str(result)]),
                        metadata=ProcessingMetadata(component_name, "orchestration", 0, 0, [], [], []),
                        context=context,
                        processing_time=0.0,
                        errors=[str(result)]
                    )
                else:
                    results[component_name] = result

        # Synthesize results
        synthesized = await self._synthesize_component_results(results, context)

        return synthesized

    def _can_execute_parallel(self, component: BaseProcessor, all_components: List[BaseProcessor]) -> bool:
        """Determine if a component can execute in parallel"""
        # Simple implementation - can be overridden for complex dependency logic
        return True

    async def _synthesize_component_results(self, results: Dict[str, AnalysisResult],
                                          context: ProcessingContext) -> Dict[str, Any]:
        """Synthesize results from multiple components"""
        synthesized = {
            'component_results': results,
            'overall_success': all(r.success for r in results.values()),
            'total_processing_time': sum(r.processing_time for r in results.values()),
            'average_confidence': sum(r.confidence.value for r in results.values()) / len(results),
            'error_summary': [error for r in results.values() for error in r.errors],
            'warning_summary': [warning for r in results.values() for warning in r.warnings]
        }

        return synthesized
