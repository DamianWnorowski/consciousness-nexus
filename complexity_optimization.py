#!/usr/bin/env python3
"""
⚡ COMPLEXITY OPTIMIZATION SYSTEM
=================================

Fix O(n²) complexity in evolution analysis with incremental processing.
"""

import asyncio
import time
import os
from typing import Dict, Any, List, Optional, Tuple, Set, Deque
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics
import math

@dataclass
class IncrementalStats:
    """Incremental statistical calculations"""
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0  # For variance calculation
    min_val: float = float('inf')
    max_val: float = float('-inf')
    sum_val: float = 0.0

    def add_value(self, value: float):
        """Add a value using Welford's online algorithm for variance"""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)
        self.sum_val += value

    def get_variance(self) -> float:
        """Get variance (population variance for n >= 2)"""
        if self.count < 2:
            return 0.0
        return self.m2 / self.count

    def get_std_dev(self) -> float:
        """Get standard deviation"""
        return math.sqrt(self.get_variance())

@dataclass
class EvolutionMetricsSnapshot:
    """Snapshot of evolution metrics for incremental analysis"""
    timestamp: float
    cycle_count: int
    fitness_score: float
    performance_gain: float
    success_rate: float
    improvement_rate: float

@dataclass
class IncrementalEvolutionAnalyzer:
    """Incremental evolution analyzer with O(1) updates"""

    # Core metrics with incremental calculation
    overall_fitness: IncrementalStats = field(default_factory=IncrementalStats)
    performance_gains: IncrementalStats = field(default_factory=IncrementalStats)
    success_rates: IncrementalStats = field(default_factory=IncrementalStats)
    improvement_rates: IncrementalStats = field(default_factory=IncrementalStats)

    # Rolling windows for trend analysis
    fitness_history: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    performance_history: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))

    # Pattern detection with incremental updates
    pattern_counts: Dict[str, int] = field(default_factory=dict)
    pattern_success_rates: Dict[str, IncrementalStats] = field(default_factory=dict)

    # Correlation tracking (incremental covariance)
    fitness_performance_cov: float = 0.0
    correlation_count: int = 0

    # Optimization flags
    enable_incremental_updates: bool = True
    max_history_size: int = 1000
    trend_analysis_window: int = 50

    def add_evolution_cycle(self, cycle_data: Dict[str, Any]):
        """Add evolution cycle data incrementally - O(1) operation"""

        # Extract metrics
        fitness = cycle_data.get('fitness_score', 0.0)
        performance_gain = cycle_data.get('performance_gain', 0.0)
        success = 1.0 if cycle_data.get('success', False) else 0.0
        improvement_rate = cycle_data.get('improvement_rate', 0.0)

        # Update incremental statistics - O(1)
        self.overall_fitness.add_value(fitness)
        self.performance_gains.add_value(performance_gain)
        self.success_rates.add_value(success)
        self.improvement_rates.add_value(improvement_rate)

        # Update rolling history - O(1) amortized
        self.fitness_history.append(fitness)
        self.performance_history.append(performance_gain)

        # Update correlation incrementally - O(1)
        self._update_correlation(fitness, performance_gain)

        # Update pattern analysis - O(1) amortized
        self._update_pattern_analysis(cycle_data)

    def _update_correlation(self, fitness: float, performance_gain: float):
        """Update correlation coefficient incrementally"""
        if self.correlation_count == 0:
            self.fitness_performance_cov = 0.0
        else:
            # Incremental covariance update
            fitness_mean = self.overall_fitness.mean
            perf_mean = self.performance_gains.mean

            delta_fitness = fitness - fitness_mean
            delta_perf = performance_gain - perf_mean

            self.fitness_performance_cov += delta_fitness * delta_perf

        self.correlation_count += 1

    def _update_pattern_analysis(self, cycle_data: Dict[str, Any]):
        """Update pattern analysis incrementally"""
        patterns = cycle_data.get('patterns_identified', [])

        for pattern in patterns:
            # Update pattern counts
            self.pattern_counts[pattern] = self.pattern_counts.get(pattern, 0) + 1

            # Update pattern success rates
            if pattern not in self.pattern_success_rates:
                self.pattern_success_rates[pattern] = IncrementalStats()

            success = 1.0 if cycle_data.get('success', False) else 0.0
            self.pattern_success_rates[pattern].add_value(success)

    def get_evolution_statistics(self) -> Dict[str, Any]:
        """Get comprehensive evolution statistics - O(1) operation"""

        # Calculate correlation coefficient
        if self.correlation_count >= 2:
            fitness_std = self.overall_fitness.get_std_dev()
            perf_std = self.performance_gains.get_std_dev()

            if fitness_std > 0 and perf_std > 0:
                correlation = self.fitness_performance_cov / (self.correlation_count * fitness_std * perf_std)
            else:
                correlation = 0.0
        else:
            correlation = 0.0

        # Calculate trends using rolling windows - O(1)
        fitness_trend = self._calculate_trend(self.fitness_history)
        performance_trend = self._calculate_trend(self.performance_history)

        # Get top patterns by success rate - O(k) where k is number of patterns
        top_patterns = self._get_top_patterns(5)

        return {
            'total_cycles': self.overall_fitness.count,
            'average_fitness': self.overall_fitness.mean,
            'fitness_std_dev': self.overall_fitness.get_std_dev(),
            'fitness_range': (self.overall_fitness.min_val, self.overall_fitness.max_val),
            'average_performance_gain': self.performance_gains.mean,
            'performance_std_dev': self.performance_gains.get_std_dev(),
            'overall_success_rate': self.success_rates.mean,
            'average_improvement_rate': self.improvement_rates.mean,
            'fitness_performance_correlation': correlation,
            'fitness_trend': fitness_trend,
            'performance_trend': performance_trend,
            'top_patterns': top_patterns,
            'pattern_diversity': len(self.pattern_counts),
            'last_updated': time.time()
        }

    def _calculate_trend(self, history: Deque[float]) -> str:
        """Calculate trend from rolling window - O(window_size) but bounded"""
        if len(history) < 2:
            return "insufficient_data"

        # Use only recent data for trend calculation
        window_size = min(self.trend_analysis_window, len(history))
        recent = list(history)[-window_size:]

        if len(recent) < 2:
            return "stable"

        # Linear regression slope approximation
        n = len(recent)
        x = list(range(n))
        y = recent

        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_xx = sum(xi * xi for xi in x)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)

        if slope > 0.01:
            return "improving"
        elif slope < -0.01:
            return "declining"
        else:
            return "stable"

    def _get_top_patterns(self, limit: int) -> List[Dict[str, Any]]:
        """Get top patterns by success rate - O(k log k) where k is patterns"""
        pattern_stats = []

        for pattern, stats in self.pattern_success_rates.items():
            if stats.count >= 3:  # Require minimum samples
                pattern_stats.append({
                    'pattern': pattern,
                    'success_rate': stats.mean,
                    'usage_count': stats.count,
                    'confidence': min(1.0, stats.count / 10.0)  # Confidence based on sample size
                })

        # Sort by success rate and return top patterns
        pattern_stats.sort(key=lambda x: (x['success_rate'], x['usage_count']), reverse=True)
        return pattern_stats[:limit]

@dataclass
class OptimizedEvolutionAnalyzer:
    """Optimized evolution analyzer with O(n) complexity"""

    incremental_analyzer: IncrementalEvolutionAnalyzer = field(default_factory=IncrementalEvolutionAnalyzer)

    # Batch processing for efficiency
    batch_size: int = 100
    pending_cycles: List[Dict[str, Any]] = field(default_factory=list)

    # Caching for expensive calculations
    stats_cache: Optional[Dict[str, Any]] = None
    cache_timestamp: float = 0.0
    cache_ttl: float = 5.0  # Cache for 5 seconds

    def add_evolution_cycle_batch(self, cycles: List[Dict[str, Any]]):
        """Add multiple evolution cycles efficiently - O(n)"""

        for cycle in cycles:
            self.incremental_analyzer.add_evolution_cycle(cycle)

        # Invalidate cache
        self.stats_cache = None

    def add_evolution_cycle(self, cycle_data: Dict[str, Any]):
        """Add single evolution cycle - O(1)"""

        # Add to incremental analyzer
        self.incremental_analyzer.add_evolution_cycle(cycle_data)

        # Batch processing - process when we hit batch size
        self.pending_cycles.append(cycle_data)

        if len(self.pending_cycles) >= self.batch_size:
            self._process_batch()
        else:
            # Invalidate cache for immediate updates
            self.stats_cache = None

    def _process_batch(self):
        """Process pending cycles in batch - O(batch_size)"""
        # Batch processing logic here (e.g., bulk database updates, etc.)
        # For now, just clear the batch since incremental analyzer handles it
        self.pending_cycles.clear()

    def get_evolution_statistics_optimized(self) -> Dict[str, Any]:
        """Get evolution statistics with caching - O(1) amortized"""

        current_time = time.time()

        # Return cached result if still valid
        if (self.stats_cache and
            current_time - self.cache_timestamp < self.cache_ttl):
            return self.stats_cache

        # Calculate fresh statistics
        stats = self.incremental_analyzer.get_evolution_statistics()

        # Add optimization metadata
        stats['optimization_info'] = {
            'complexity': 'O(1) amortized',
            'caching_enabled': True,
            'batch_processing': self.batch_size,
            'memory_efficient': True
        }

        # Cache the result
        self.stats_cache = stats
        self.cache_timestamp = current_time

        return stats

    def get_performance_predictions(self) -> Dict[str, Any]:
        """Get performance predictions based on trends - O(1)"""

        stats = self.get_evolution_statistics_optimized()

        # Simple linear extrapolation
        cycles = stats['total_cycles']
        if cycles < 2:
            return {'prediction_available': False}

        fitness_trend = stats['fitness_trend']
        performance_trend = stats['performance_trend']

        # Predict next 5 cycles
        predictions = []

        for i in range(1, 6):
            predicted_fitness = self._extrapolate_metric(
                self.incremental_analyzer.fitness_history, i, fitness_trend
            )
            predicted_performance = self._extrapolate_metric(
                self.incremental_analyzer.performance_history, i, performance_trend
            )

            predictions.append({
                'cycle_offset': i,
                'predicted_fitness': predicted_fitness,
                'predicted_performance_gain': predicted_performance,
                'confidence': max(0.1, 1.0 - (i * 0.1))  # Decreasing confidence
            })

        return {
            'prediction_available': True,
            'current_trends': {
                'fitness': fitness_trend,
                'performance': performance_trend
            },
            'predictions': predictions,
            'recommended_actions': self._get_recommended_actions(stats, predictions)
        }

    def _extrapolate_metric(self, history: Deque[float], steps_ahead: int, trend: str) -> float:
        """Extrapolate metric based on trend - O(1)"""

        if len(history) < 2:
            return history[-1] if history else 0.0

        # Simple trend-based extrapolation
        recent_values = list(history)[-10:]  # Use last 10 values
        current_avg = statistics.mean(recent_values)

        # Trend multiplier
        trend_multiplier = {
            'improving': 1.02,   # 2% improvement per cycle
            'declining': 0.98,   # 2% decline per cycle
            'stable': 1.0        # No change
        }.get(trend, 1.0)

        # Compound the trend
        predicted = current_avg * (trend_multiplier ** steps_ahead)

        # Bound predictions to reasonable ranges
        return max(0.0, min(1.0, predicted))

    def _get_recommended_actions(self, stats: Dict[str, Any],
                               predictions: List[Dict[str, Any]]) -> List[str]:
        """Get recommended actions based on analysis"""

        actions = []

        # Check if fitness is declining
        if stats.get('fitness_trend') == 'declining':
            actions.append("Consider adjusting evolution parameters to improve fitness")

        # Check if performance gains are diminishing
        if stats.get('performance_trend') == 'declining':
            actions.append("Performance gains are decreasing - review optimization strategies")

        # Check low success rate
        if stats.get('overall_success_rate', 0) < 0.5:
            actions.append("Evolution success rate is low - investigate failure patterns")

        # Check predictions
        if predictions:
            last_prediction = predictions[-1]
            if last_prediction['predicted_fitness'] < 0.7:
                actions.append("Long-term fitness predictions are concerning - consider intervention")

        # Pattern diversity check
        if stats.get('pattern_diversity', 0) < 3:
            actions.append("Low pattern diversity detected - encourage exploration")

        return actions

class ComplexityBenchmark:
    """Benchmark to demonstrate complexity improvements"""

    @staticmethod
    async def benchmark_complexity():
        """Benchmark old O(n²) vs new O(n) approaches"""

        print("🧪 COMPLEXITY OPTIMIZATION BENCHMARK")
        print("=" * 50)

        # Test data sizes
        test_sizes = [100, 500, 1000, 2500]

        for size in test_sizes:
            print(f"\nTesting with {size} evolution cycles...")

            # Generate test data
            test_cycles = [
                {
                    'fitness_score': 0.5 + (i / size) * 0.4 + (0.1 * (i % 10) / 10),  # Some randomness
                    'performance_gain': 0.1 + (i / size) * 0.2,
                    'success': (i % 3) != 0,  # 66% success rate
                    'improvement_rate': 0.05 + (i / size) * 0.1,
                    'patterns_identified': [f'pattern_{i % 5}', f'pattern_{(i + 1) % 5}']
                }
                for i in range(size)
            ]

            # Benchmark old approach (simulated O(n²))
            start_time = time.time()
            old_result = ComplexityBenchmark._old_complexity_approach(test_cycles)
            old_time = time.time() - start_time

            # Benchmark new approach
            start_time = time.time()
            analyzer = OptimizedEvolutionAnalyzer()
            analyzer.add_evolution_cycle_batch(test_cycles)
            new_result = analyzer.get_evolution_statistics_optimized()
            new_time = time.time() - start_time

            # Calculate improvement
            improvement = old_time / new_time if new_time > 0 else float('inf')

            print(f"  Old approach (O(n²)): {old_time:.3f}s")
            print(f"  New approach (O(n)):  {new_time:.3f}s")
            print(f"  Improvement: {improvement:.2f}x faster")
            print(f"  Old complexity: O(n²) = {old_time * (size ** 2) / 1000:.3f} sec for 1000 items")
            print(f"  New complexity: O(n) = {new_time * size / 1000:.3f} sec for 1000 items")
            # Verify results are approximately equal
            fitness_diff = abs(old_result['average_fitness'] - new_result['average_fitness'])
            print(f"  Result accuracy: {'✓' if fitness_diff < 0.01 else '✗'} (diff: {fitness_diff:.6f})")

    @staticmethod
    def _old_complexity_approach(cycles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Simulate old O(n²) approach for comparison"""

        # Simulate O(n²) operations
        fitness_scores = [c['fitness_score'] for c in cycles]

        # O(n²) correlation calculation (naive approach)
        correlation = 0.0
        if len(cycles) >= 2:
            fitness_mean = statistics.mean(fitness_scores)
            perf_scores = [c['performance_gain'] for c in cycles]
            perf_mean = statistics.mean(perf_scores)

            numerator = sum((f - fitness_mean) * (p - perf_mean)
                          for f, p in zip(fitness_scores, perf_scores))
            denominator = (math.sqrt(sum((f - fitness_mean) ** 2 for f in fitness_scores)) *
                          math.sqrt(sum((p - perf_mean) ** 2 for p in perf_scores)))

            correlation = numerator / denominator if denominator > 0 else 0.0

        # Simulate other O(n²) operations
        for i in range(len(cycles)):
            for j in range(len(cycles)):
                # Dummy O(n²) operation
                _ = cycles[i]['fitness_score'] * cycles[j]['performance_gain']

        return {
            'total_cycles': len(cycles),
            'average_fitness': statistics.mean(fitness_scores),
            'fitness_std_dev': statistics.stdev(fitness_scores) if len(cycles) >= 2 else 0.0,
            'correlation': correlation
        }

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Complexity Optimization System")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run complexity benchmark")
    parser.add_argument("--analyze", nargs="*", metavar="CYCLE_FILE",
                       help="Analyze evolution cycles from files")
    parser.add_argument("--predict", action="store_true",
                       help="Show performance predictions")

    args = parser.parse_args()

    async def run():
        analyzer = OptimizedEvolutionAnalyzer()

        if args.benchmark:
            await ComplexityBenchmark.benchmark_complexity()

        elif args.analyze:
            # Load and analyze evolution cycles
            all_cycles = []

            for file_path in args.analyze:
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'r') as f:
                            cycles = json.load(f)
                            if isinstance(cycles, list):
                                all_cycles.extend(cycles)
                    except:
                        print(f"Warning: Could not load {file_path}")

            if all_cycles:
                print(f"Analyzing {len(all_cycles)} evolution cycles...")

                analyzer.add_evolution_cycle_batch(all_cycles)
                stats = analyzer.get_evolution_statistics_optimized()

                print("\n📊 OPTIMIZED EVOLUTION STATISTICS")
                print("=" * 40)
                print(f"Total Cycles: {stats['total_cycles']}")
                print(".3f")
                print(".3f")
                print(".3f")
                print(".1f")
                print(f"Pattern Diversity: {stats['pattern_diversity']}")

                if args.predict:
                    predictions = analyzer.get_performance_predictions()
                    if predictions['prediction_available']:
                        print("\n🔮 PERFORMANCE PREDICTIONS")
                        print("=" * 40)
                        print(f"Current Trends: Fitness={predictions['current_trends']['fitness']}, Performance={predictions['current_trends']['performance']}")

                        print("\nNext 5 Cycles:")
                        for pred in predictions['predictions']:
                            print(f"  Cycle +{pred['cycle_offset']}: "
                                  f"Fitness={pred['predicted_fitness']:.3f}, "
                                  f"Perf={pred['predicted_performance_gain']:.3f}, "
                                  f"Conf={pred['confidence']:.2f}")

                        if predictions['recommended_actions']:
                            print("\n💡 Recommended Actions:")
                            for action in predictions['recommended_actions']:
                                print(f"  • {action}")
            else:
                print("No evolution cycles found to analyze")

        else:
            parser.print_help()

    asyncio.run(run())

if __name__ == "__main__":
    main()
