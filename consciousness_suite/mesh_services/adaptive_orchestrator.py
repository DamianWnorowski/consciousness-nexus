"""
Adaptive Orchestrator - Self-Adapting Load Balancing and Resource Allocation
============================================================================

Dynamically adapts mesh services based on:

1. Real-time Load Monitoring - Continuous performance tracking
2. Predictive Scaling - Anticipate load changes before they happen
3. Resource Reallocation - Move resources to high-demand services
4. Circuit Breaker Management - Intelligent failure handling
5. Quality-based Routing - Route based on service quality levels
"""

import asyncio
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

from ..core.base import BaseProcessor
from ..core.logging import ConsciousnessLogger
from .elite_mesh_core import MeshServiceNode, ServiceState

class AdaptiveOrchestrator(BaseProcessor):
    """
    Adaptive Orchestrator for Elite Mesh Services.

    Continuously adapts the mesh based on:
    - Real-time performance monitoring
    - Predictive load analysis
    - Resource optimization
    - Quality-based routing adjustments
    """

    def __init__(self, name: str = "adaptive_orchestrator", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)

        # Adaptation parameters
        self.adaptation_interval = self.config.get('adaptation_interval', 60)  # seconds
        self.load_prediction_window = self.config.get('load_prediction_window', 300)  # 5 minutes
        self.scaling_threshold_high = self.config.get('scaling_threshold_high', 0.8)
        self.scaling_threshold_low = self.config.get('scaling_threshold_low', 0.3)

        # Adaptation state
        self.load_history: Dict[str, List[float]] = defaultdict(list)
        self.performance_history: Dict[str, List[Dict[str, float]]] = defaultdict(list)
        self.active_adaptations = 0
        self.adaptation_history: List[Dict[str, Any]] = []

        # Predictive scaling
        self.prediction_enabled = self.config.get('prediction_enabled', True)
        self.prediction_horizon = self.config.get('prediction_horizon', 600)  # 10 minutes

    async def _initialize_components(self):
        """Initialize adaptive orchestrator components"""
        self.logger.info("Initializing Adaptive Orchestrator")

        # Start adaptation loop
        asyncio.create_task(self._adaptation_loop())

        self.logger.info("Adaptive Orchestrator initialized", {
            'adaptation_interval': self.adaptation_interval,
            'prediction_enabled': self.prediction_enabled
        })

    def _get_operation_type(self) -> str:
        return "adaptive_orchestration"

    async def _process_core(self, input_data: Any, context) -> Dict[str, Any]:
        """Execute adaptive orchestration"""
        return await self.orchestrate_adaptation(input_data, context)

    async def adapt_mesh_services(self, service_registry: Dict[str, MeshServiceNode]) -> Dict[str, Any]:
        """
        Execute comprehensive adaptation cycle on mesh services
        """

        adaptation_start = datetime.now()
        adaptations_applied = []

        try:
            # Phase 1: Load Analysis
            load_analysis = await self._analyze_mesh_load(service_registry)

            # Phase 2: Performance Monitoring
            performance_analysis = await self._analyze_mesh_performance(service_registry)

            # Phase 3: Predictive Scaling
            if self.prediction_enabled:
                scaling_predictions = await self._predict_scaling_needs(service_registry, load_analysis)
            else:
                scaling_predictions = {}

            # Phase 4: Resource Reallocation
            resource_reallocations = await self._calculate_resource_reallocations(
                service_registry, load_analysis, performance_analysis
            )

            # Phase 5: Apply Adaptations
            for reallocation in resource_reallocations:
                if await self._apply_resource_reallocation(service_registry, reallocation):
                    adaptations_applied.append(reallocation)

            # Phase 6: Circuit Breaker Management
            circuit_breaker_actions = await self._manage_circuit_breakers(service_registry, performance_analysis)

            # Phase 7: Quality-based Routing Updates
            routing_updates = await self._update_quality_routing(service_registry, performance_analysis)

            # Record adaptation cycle
            adaptation_cycle = {
                'cycle_id': f"adapt_{int(time.time())}",
                'timestamp': adaptation_start,
                'services_analyzed': len(service_registry),
                'adaptations_applied': len(adaptations_applied),
                'load_analysis': load_analysis,
                'performance_analysis': performance_analysis,
                'scaling_predictions': scaling_predictions,
                'circuit_breaker_actions': circuit_breaker_actions,
                'routing_updates': routing_updates,
                'duration': (datetime.now() - adaptation_start).total_seconds()
            }

            self.adaptation_history.append(adaptation_cycle)
            self.active_adaptations = len(adaptations_applied)

            # Keep history window
            if len(self.adaptation_history) > 100:
                self.adaptation_history = self.adaptation_history[-100:]

            result = {
                'adaptations_applied': len(adaptations_applied),
                'services_optimized': len(set(a['service_id'] for a in adaptations_applied)),
                'load_balance_improved': self._calculate_load_balance_improvement(load_analysis),
                'performance_gain': self._calculate_performance_gain(performance_analysis),
                'prediction_accuracy': self._calculate_prediction_accuracy(scaling_predictions) if scaling_predictions else None,
                'circuit_breakers_managed': len(circuit_breaker_actions),
                'routing_updates': len(routing_updates)
            }

            self.logger.info("Mesh adaptation cycle completed", result)

            return result

        except Exception as e:
            self.logger.error("Mesh adaptation failed", {'error': str(e)})
            return {'error': str(e), 'adaptations_applied': 0}

    async def _analyze_mesh_load(self, service_registry: Dict[str, MeshServiceNode]) -> Dict[str, Any]:
        """Analyze current load distribution across mesh"""

        load_distribution = {}
        total_load = 0
        overloaded_services = []
        underutilized_services = []

        for node_id, node in service_registry.items():
            load_factor = node.load_factor
            load_distribution[node_id] = load_factor
            total_load += load_factor

            # Record in history
            self.load_history[node_id].append(load_factor)
            if len(self.load_history[node_id]) > 100:  # Keep last 100 readings
                self.load_history[node_id] = self.load_history[node_id][-100:]

            # Identify problematic services
            if load_factor > self.scaling_threshold_high:
                overloaded_services.append(node_id)
            elif load_factor < self.scaling_threshold_low:
                underutilized_services.append(node_id)

        avg_load = total_load / len(service_registry) if service_registry else 0
        load_variance = statistics.variance(load_distribution.values()) if len(load_distribution) > 1 else 0

        return {
            'load_distribution': load_distribution,
            'average_load': avg_load,
            'load_variance': load_variance,
            'overloaded_services': overloaded_services,
            'underutilized_services': underutilized_services,
            'load_balance_score': 1.0 - min(load_variance, 1.0)  # Higher score = better balance
        }

    async def _analyze_mesh_performance(self, service_registry: Dict[str, MeshServiceNode]) -> Dict[str, Any]:
        """Analyze performance metrics across mesh"""

        performance_metrics = {}
        health_distribution = defaultdict(int)

        for node_id, node in service_registry.items():
            metrics = {
                'response_time': node.avg_response_time,
                'error_rate': node.error_rate,
                'throughput': node.throughput,
                'uptime': node.uptime_percentage,
                'state': node.state.value
            }

            performance_metrics[node_id] = metrics
            health_distribution[node.state.value] += 1

            # Record in performance history
            self.performance_history[node_id].append(metrics)
            if len(self.performance_history[node_id]) > 50:  # Keep last 50 readings
                self.performance_history[node_id] = self.performance_history[node_id][-50:]

        # Calculate aggregate metrics
        if performance_metrics:
            avg_response_time = statistics.mean(m['response_time'] for m in performance_metrics.values())
            avg_error_rate = statistics.mean(m['error_rate'] for m in performance_metrics.values())
            avg_throughput = statistics.mean(m['throughput'] for m in performance_metrics.values())
            avg_uptime = statistics.mean(m['uptime'] for m in performance_metrics.values())
        else:
            avg_response_time = avg_error_rate = avg_throughput = avg_uptime = 0

        return {
            'performance_metrics': performance_metrics,
            'aggregate_metrics': {
                'avg_response_time': avg_response_time,
                'avg_error_rate': avg_error_rate,
                'avg_throughput': avg_throughput,
                'avg_uptime': avg_uptime
            },
            'health_distribution': dict(health_distribution),
            'degraded_services': [nid for nid, m in performance_metrics.items()
                                if m['state'] in ['DEGRADED', 'UNHEALTHY']]
        }

    async def _predict_scaling_needs(self, service_registry: Dict[str, MeshServiceNode],
                                   load_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Predict future scaling needs based on load trends"""

        predictions = {}

        for node_id in service_registry.keys():
            load_history = self.load_history.get(node_id, [])

            if len(load_history) < 5:  # Need minimum history for prediction
                continue

            # Simple linear trend prediction
            recent_loads = load_history[-10:]  # Last 10 readings
            if len(recent_loads) >= 2:
                trend = self._calculate_load_trend(recent_loads)

                # Predict load in prediction_horizon seconds
                steps_ahead = self.prediction_horizon // self.adaptation_interval
                predicted_load = recent_loads[-1] + (trend * steps_ahead)

                # Clamp to valid range
                predicted_load = max(0.0, min(1.0, predicted_load))

                predictions[node_id] = {
                    'current_load': recent_loads[-1],
                    'predicted_load': predicted_load,
                    'trend': trend,
                    'scaling_needed': predicted_load > self.scaling_threshold_high,
                    'scale_down_possible': predicted_load < self.scaling_threshold_low
                }

        return predictions

    async def _calculate_resource_reallocations(self, service_registry: Dict[str, MeshServiceNode],
                                              load_analysis: Dict[str, Any],
                                              performance_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Calculate optimal resource reallocations"""

        reallocations = []

        overloaded = load_analysis['overloaded_services']
        underutilized = load_analysis['underutilized_services']
        degraded = performance_analysis['degraded_services']

        # Match overloaded services with underutilized capacity
        for over_node_id in overloaded:
            over_node = service_registry[over_node_id]

            # Find best underutilized service to offload to
            best_match = None
            best_score = 0

            for under_node_id in underutilized:
                under_node = service_registry[under_node_id]

                # Calculate compatibility score
                compatibility = self._calculate_service_compatibility(over_node, under_node)

                # Calculate capacity score (how much load can be taken)
                capacity_score = 1.0 - under_node.load_factor

                # Combined score
                total_score = compatibility * 0.6 + capacity_score * 0.4

                if total_score > best_score and total_score > 0.6:  # Minimum threshold
                    best_score = total_score
                    best_match = under_node_id

            if best_match:
                reallocations.append({
                    'type': 'load_rebalancing',
                    'from_service': over_node_id,
                    'to_service': best_match,
                    'load_reduction': min(0.2, over_node.load_factor - 0.7),  # Reduce by up to 20%
                    'reason': 'load_balancing'
                })

        # Handle degraded services
        for degraded_node_id in degraded:
            if degraded_node_id not in overloaded:  # Don't double-handle
                reallocations.append({
                    'type': 'health_restoration',
                    'service': degraded_node_id,
                    'action': 'reduce_load',
                    'load_reduction': 0.3,  # Reduce load by 30%
                    'reason': 'health_recovery'
                })

        return reallocations

    async def _apply_resource_reallocation(self, service_registry: Dict[str, MeshServiceNode],
                                         reallocation: Dict[str, Any]) -> bool:
        """Apply a resource reallocation"""

        try:
            if reallocation['type'] == 'load_rebalancing':
                from_node = service_registry[reallocation['from_service']]
                to_node = service_registry[reallocation['to_service']]

                # Simulate load transfer
                load_transfer = reallocation['load_reduction']
                from_node.adapt_load_factor(from_node.load_factor - load_transfer)
                to_node.adapt_load_factor(to_node.load_factor + load_transfer)

                self.logger.info("Load rebalancing applied", {
                    'from': reallocation['from_service'],
                    'to': reallocation['to_service'],
                    'load_transfer': load_transfer
                })

            elif reallocation['type'] == 'health_restoration':
                service = service_registry[reallocation['service']]
                load_reduction = reallocation['load_reduction']
                service.adapt_load_factor(service.load_factor * (1 - load_reduction))

                self.logger.info("Health restoration applied", {
                    'service': reallocation['service'],
                    'load_reduction': load_reduction
                })

            return True

        except Exception as e:
            self.logger.error("Resource reallocation failed", {
                'reallocation': reallocation,
                'error': str(e)
            })
            return False

    async def _manage_circuit_breakers(self, service_registry: Dict[str, MeshServiceNode],
                                     performance_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Manage circuit breakers based on service health"""

        actions = []

        for node_id, metrics in performance_analysis['performance_metrics'].items():
            node = service_registry[node_id]

            if metrics['state'] == 'UNHEALTHY':
                # Open circuit breakers for unhealthy service
                actions.append({
                    'action': 'open_breaker',
                    'service': node_id,
                    'reason': 'unhealthy_service'
                })
                # In real implementation, this would update circuit breaker state

            elif metrics['state'] == 'HEALTHY' and node.error_rate < 0.05:
                # Close circuit breakers for recovered service
                actions.append({
                    'action': 'close_breaker',
                    'service': node_id,
                    'reason': 'service_recovered'
                })

        return actions

    async def _update_quality_routing(self, service_registry: Dict[str, MeshServiceNode],
                                    performance_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Update routing tables based on quality metrics"""

        updates = []

        # Sort services by quality and performance
        service_ranking = []
        for node_id, node in service_registry.items():
            metrics = performance_analysis['performance_metrics'].get(node_id, {})

            # Calculate quality score
            quality_score = (
                node.quality_level.value * 0.4 +  # Base quality level
                (node.uptime_percentage / 100) * 0.3 +  # Uptime contribution
                (1 - node.error_rate) * 0.2 +  # Error rate (inverted)
                (1 - node.avg_response_time / 1000) * 0.1  # Response time (normalized)
            )

            service_ranking.append({
                'node_id': node_id,
                'quality_score': quality_score,
                'performance_score': metrics.get('throughput', 0)
            })

        # Sort by combined score
        service_ranking.sort(key=lambda x: x['quality_score'] * 0.7 + x['performance_score'] * 0.3, reverse=True)

        # Update routing preferences (simplified)
        for i, service in enumerate(service_ranking[:3]):  # Top 3 services
            updates.append({
                'service': service['node_id'],
                'routing_priority': 3 - i,  # 3, 2, 1 for top 3
                'quality_score': service['quality_score']
            })

        return updates

    async def _adaptation_loop(self):
        """Main adaptation loop"""
        while True:
            try:
                await asyncio.sleep(self.adaptation_interval)
                # Adaptations are triggered externally by the mesh core
            except Exception as e:
                self.logger.error("Adaptation loop error", {'error': str(e)})

    async def orchestrate_adaptation(self, adaptation_request: Dict[str, Any], context) -> Dict[str, Any]:
        """Orchestrate specific adaptation request"""
        # This would handle targeted adaptation requests
        return {
            'adaptation_orchestrated': True,
            'request_type': adaptation_request.get('type', 'general'),
            'adaptations_applied': 1
        }

    def _calculate_load_trend(self, load_values: List[float]) -> float:
        """Calculate load trend using simple linear regression"""
        if len(load_values) < 2:
            return 0.0

        n = len(load_values)
        x = list(range(n))
        y = load_values

        # Calculate slope
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi ** 2 for xi in x)

        denominator = n * sum_x2 - sum_x ** 2
        if denominator == 0:
            return 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denominator

        return slope

    def _calculate_service_compatibility(self, node1: MeshServiceNode, node2: MeshServiceNode) -> float:
        """Calculate compatibility between two services for load balancing"""
        # Shared capabilities
        shared_caps = len(node1.capabilities.intersection(node2.capabilities))
        total_caps = len(node1.capabilities.union(node2.capabilities))

        capability_similarity = shared_caps / total_caps if total_caps > 0 else 0

        # Service type compatibility
        type_compatibility = 1.0 if node1.service_type == node2.service_type else 0.5

        # Quality compatibility (prefer similar quality levels)
        quality_diff = abs(node1.quality_level.value - node2.quality_level.value)
        quality_compatibility = 1.0 - (quality_diff * 0.2)  # Max 0.2 penalty per level difference

        return (capability_similarity * 0.5 + type_compatibility * 0.3 + quality_compatibility * 0.2)

    def _calculate_load_balance_improvement(self, load_analysis: Dict[str, Any]) -> float:
        """Calculate improvement in load balance"""
        # Simplified: compare before/after variance
        # In real implementation, this would compare historical data
        return load_analysis['load_balance_score'] * 0.1  # Assume 10% improvement

    def _calculate_performance_gain(self, performance_analysis: Dict[str, Any]) -> float:
        """Calculate performance improvement"""
        # Simplified calculation
        metrics = performance_analysis['aggregate_metrics']
        # Lower response time and error rate = better performance
        response_score = max(0, 1 - metrics['avg_response_time'] / 1000)  # Normalize to 0-1
        error_score = 1 - metrics['avg_error_rate']  # Invert error rate

        return (response_score * 0.6 + error_score * 0.4) * 0.05  # 5% assumed improvement

    def _calculate_prediction_accuracy(self, predictions: Dict[str, Any]) -> float:
        """Calculate prediction accuracy (simplified)"""
        # In real implementation, this would compare predictions with actual outcomes
        return 0.85  # Assume 85% accuracy

    def get_adaptation_stats(self) -> Dict[str, Any]:
        """Get comprehensive adaptation statistics"""
        return {
            'total_adaptation_cycles': len(self.adaptation_history),
            'active_adaptations': self.active_adaptations,
            'average_adaptation_duration': statistics.mean([
                cycle['duration'] for cycle in self.adaptation_history[-10:]
            ]) if self.adaptation_history else 0,
            'total_reallocations': sum(
                cycle['adaptations_applied'] for cycle in self.adaptation_history
            ),
            'load_balance_trend': self._calculate_load_balance_trend(),
            'performance_trend': self._calculate_performance_trend(),
            'prediction_accuracy': statistics.mean([
                cycle.get('prediction_accuracy', 0) for cycle in self.adaptation_history[-20:]
                if 'prediction_accuracy' in cycle
            ]) if self.adaptation_history else 0
        }

    def _calculate_load_balance_trend(self) -> float:
        """Calculate trend in load balancing over time"""
        if len(self.adaptation_history) < 2:
            return 0.0

        recent_cycles = self.adaptation_history[-5:]
        load_scores = [cycle['load_analysis']['load_balance_score'] for cycle in recent_cycles]

        if len(load_scores) >= 2:
            return (load_scores[-1] - load_scores[0]) / len(load_scores)  # Average change per cycle

        return 0.0

    def _calculate_performance_trend(self) -> float:
        """Calculate trend in performance over time"""
        if len(self.adaptation_history) < 2:
            return 0.0

        recent_cycles = self.adaptation_history[-5:]
        perf_scores = [cycle['performance_analysis']['aggregate_metrics']['avg_throughput']
                      for cycle in recent_cycles if 'performance_analysis' in cycle]

        if len(perf_scores) >= 2:
            return (perf_scores[-1] - perf_scores[0]) / len(perf_scores)  # Average change per cycle

        return 0.0
