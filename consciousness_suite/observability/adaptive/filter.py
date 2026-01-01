"""Low-Value Data Filtering

Filters out telemetry data with low observability value to reduce costs.
Implements pattern-based, entropy-based, and value-based filtering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Pattern
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import threading
import logging
import re
import math
import time

from prometheus_client import Counter, Gauge

logger = logging.getLogger(__name__)


class FilterAction(str, Enum):
    """Filter action to take."""
    PASS = "pass"           # Allow data through
    DROP = "drop"           # Drop data
    AGGREGATE = "aggregate" # Convert to aggregate
    DOWNSAMPLE = "downsample"  # Reduce frequency
    TAG_LOW_VALUE = "tag"   # Tag as low value but keep


class FilterType(str, Enum):
    """Types of filters."""
    PATTERN = "pattern"         # Regex/glob pattern matching
    ENTROPY = "entropy"         # Information entropy based
    FREQUENCY = "frequency"     # High-frequency redundancy
    VALUE_RANGE = "value_range" # Static/boring values
    SEMANTIC = "semantic"       # Semantic value analysis
    COMPOSITE = "composite"     # Multiple conditions


@dataclass
class FilterRule:
    """Rule for filtering telemetry data.

    Usage:
        rule = FilterRule(
            name="debug_logs",
            filter_type=FilterType.PATTERN,
            pattern="*.debug.*",
            action=FilterAction.DROP,
            description="Drop debug-level logs",
        )
    """
    name: str
    filter_type: FilterType
    action: FilterAction = FilterAction.DROP
    pattern: Optional[str] = None
    regex: Optional[str] = None
    min_entropy: Optional[float] = None          # Minimum entropy to keep
    max_frequency_per_minute: Optional[int] = None
    value_range: Optional[tuple] = None          # (min, max) for boring values
    static_value_threshold: int = 100            # Consecutive same values = static
    semantic_keywords: List[str] = field(default_factory=list)
    conditions: List["FilterRule"] = field(default_factory=list)  # For composite
    description: str = ""
    enabled: bool = True
    priority: int = 0                            # Higher = checked first
    dry_run: bool = False                        # Log but don't filter


@dataclass
class FilterContext:
    """Context for a filter decision."""
    data_type: str                              # metric, log, trace, event
    name: str                                   # Metric/log/trace name
    value: Any                                  # Data value
    labels: Dict[str, str] = field(default_factory=dict)
    attributes: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""
    service: str = ""


@dataclass
class FilterResult:
    """Result of a filter decision."""
    action: FilterAction
    rule_name: str
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class DataFilter:
    """Filters low-value telemetry data.

    Usage:
        filter = DataFilter(namespace="consciousness")

        # Add filter rules
        filter.add_rule(FilterRule(
            name="static_metrics",
            filter_type=FilterType.VALUE_RANGE,
            action=FilterAction.DROP,
            static_value_threshold=50,
        ))

        # Filter data
        result = filter.should_filter(FilterContext(
            data_type="metric",
            name="cpu_usage",
            value=0.0,
        ))

        if result.action != FilterAction.DROP:
            send_telemetry(data)
    """

    def __init__(
        self,
        namespace: str = "consciousness",
        default_action: FilterAction = FilterAction.PASS,
    ):
        self.namespace = namespace
        self.default_action = default_action
        self._lock = threading.Lock()

        # Rules
        self._rules: Dict[str, FilterRule] = {}
        self._compiled_patterns: Dict[str, Pattern] = {}

        # Tracking state
        self._value_history: Dict[str, List[Any]] = defaultdict(list)
        self._frequency_counters: Dict[str, List[datetime]] = defaultdict(list)
        self._entropy_cache: Dict[str, float] = {}

        # Statistics
        self._total_evaluated: int = 0
        self._total_passed: int = 0
        self._total_filtered: int = 0
        self._filtered_by_rule: Dict[str, int] = defaultdict(int)

        # Prometheus metrics
        self.filter_decisions = Counter(
            f"{namespace}_filter_decisions_total",
            "Filter decisions made",
            ["action", "rule", "data_type"],
        )

        self.filter_rate = Gauge(
            f"{namespace}_filter_rate",
            "Current filter rate (0-1)",
        )

        self.data_entropy = Gauge(
            f"{namespace}_data_entropy",
            "Data entropy by source",
            ["source"],
        )

        self.low_value_detected = Counter(
            f"{namespace}_low_value_detected_total",
            "Low-value data detected",
            ["reason", "data_type"],
        )

    def add_rule(self, rule: FilterRule):
        """Add a filter rule.

        Args:
            rule: Filter rule
        """
        with self._lock:
            self._rules[rule.name] = rule

            # Compile regex if provided
            if rule.regex:
                try:
                    self._compiled_patterns[rule.name] = re.compile(rule.regex)
                except re.error as e:
                    logger.error(f"Invalid regex in rule {rule.name}: {e}")

        logger.info(
            f"Added filter rule: {rule.name} "
            f"({rule.filter_type.value} -> {rule.action.value})"
        )

    def remove_rule(self, rule_name: str):
        """Remove a filter rule.

        Args:
            rule_name: Rule name
        """
        with self._lock:
            if rule_name in self._rules:
                del self._rules[rule_name]
            if rule_name in self._compiled_patterns:
                del self._compiled_patterns[rule_name]

    def should_filter(self, context: FilterContext) -> FilterResult:
        """Determine if data should be filtered.

        Args:
            context: Filter context

        Returns:
            FilterResult with action
        """
        self._total_evaluated += 1

        with self._lock:
            # Sort rules by priority
            sorted_rules = sorted(
                self._rules.values(),
                key=lambda r: r.priority,
                reverse=True,
            )

        for rule in sorted_rules:
            if not rule.enabled:
                continue

            result = self._evaluate_rule(rule, context)

            if result.action != FilterAction.PASS:
                # Rule matched
                if not rule.dry_run:
                    self._total_filtered += 1
                    self._filtered_by_rule[rule.name] += 1
                    self._update_metrics(result, context)
                    return result
                else:
                    logger.debug(
                        f"[DRY RUN] Would filter {context.name} by rule {rule.name}"
                    )

        # No rule matched, default action
        self._total_passed += 1
        self._update_filter_rate()

        return FilterResult(
            action=self.default_action,
            rule_name="default",
            reason="no_rule_matched",
        )

    def _evaluate_rule(self, rule: FilterRule, context: FilterContext) -> FilterResult:
        """Evaluate a single filter rule."""
        if rule.filter_type == FilterType.PATTERN:
            return self._evaluate_pattern(rule, context)
        elif rule.filter_type == FilterType.ENTROPY:
            return self._evaluate_entropy(rule, context)
        elif rule.filter_type == FilterType.FREQUENCY:
            return self._evaluate_frequency(rule, context)
        elif rule.filter_type == FilterType.VALUE_RANGE:
            return self._evaluate_value_range(rule, context)
        elif rule.filter_type == FilterType.SEMANTIC:
            return self._evaluate_semantic(rule, context)
        elif rule.filter_type == FilterType.COMPOSITE:
            return self._evaluate_composite(rule, context)
        else:
            return FilterResult(
                action=FilterAction.PASS,
                rule_name=rule.name,
                reason="unknown_filter_type",
            )

    def _evaluate_pattern(self, rule: FilterRule, context: FilterContext) -> FilterResult:
        """Evaluate pattern-based filter."""
        matched = False

        # Check glob pattern
        if rule.pattern:
            import fnmatch
            if fnmatch.fnmatch(context.name, rule.pattern):
                matched = True

        # Check regex
        if rule.regex and rule.name in self._compiled_patterns:
            pattern = self._compiled_patterns[rule.name]
            if pattern.search(context.name):
                matched = True

        if matched:
            return FilterResult(
                action=rule.action,
                rule_name=rule.name,
                reason="pattern_match",
                metadata={"pattern": rule.pattern or rule.regex},
            )

        return FilterResult(
            action=FilterAction.PASS,
            rule_name=rule.name,
            reason="pattern_no_match",
        )

    def _evaluate_entropy(self, rule: FilterRule, context: FilterContext) -> FilterResult:
        """Evaluate entropy-based filter.

        Low entropy = predictable/redundant data = low value.
        """
        key = f"{context.data_type}:{context.name}"

        with self._lock:
            # Track value history
            history = self._value_history[key]
            history.append(str(context.value))

            # Keep limited history
            if len(history) > 1000:
                self._value_history[key] = history[-500:]
                history = self._value_history[key]

        # Calculate Shannon entropy
        if len(history) < 10:
            # Not enough data
            return FilterResult(
                action=FilterAction.PASS,
                rule_name=rule.name,
                reason="insufficient_history",
            )

        entropy = self._calculate_entropy(history)
        self._entropy_cache[key] = entropy
        self.data_entropy.labels(source=context.source or "unknown").set(entropy)

        min_entropy = rule.min_entropy or 0.5

        if entropy < min_entropy:
            self.low_value_detected.labels(
                reason="low_entropy",
                data_type=context.data_type,
            ).inc()

            return FilterResult(
                action=rule.action,
                rule_name=rule.name,
                reason="low_entropy",
                metadata={"entropy": entropy, "threshold": min_entropy},
            )

        return FilterResult(
            action=FilterAction.PASS,
            rule_name=rule.name,
            reason="sufficient_entropy",
            metadata={"entropy": entropy},
        )

    def _calculate_entropy(self, values: List[str]) -> float:
        """Calculate Shannon entropy of value distribution."""
        if not values:
            return 0.0

        # Count frequencies
        freq: Dict[str, int] = defaultdict(int)
        for v in values:
            freq[v] += 1

        # Calculate entropy
        total = len(values)
        entropy = 0.0

        for count in freq.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)

        # Normalize by max possible entropy
        max_entropy = math.log2(len(freq)) if len(freq) > 1 else 1.0

        return entropy / max_entropy if max_entropy > 0 else 0.0

    def _evaluate_frequency(self, rule: FilterRule, context: FilterContext) -> FilterResult:
        """Evaluate frequency-based filter.

        High-frequency identical data = redundant.
        """
        key = f"{context.data_type}:{context.name}"
        now = datetime.now()
        window = timedelta(minutes=1)

        with self._lock:
            # Add current timestamp
            self._frequency_counters[key].append(now)

            # Clean old entries
            cutoff = now - window
            self._frequency_counters[key] = [
                ts for ts in self._frequency_counters[key]
                if ts > cutoff
            ]

            frequency = len(self._frequency_counters[key])

        max_freq = rule.max_frequency_per_minute or 60

        if frequency > max_freq:
            self.low_value_detected.labels(
                reason="high_frequency",
                data_type=context.data_type,
            ).inc()

            # Downsample instead of dropping
            if rule.action == FilterAction.DROP:
                # Keep 1 in N samples
                keep_ratio = max_freq / frequency
                import random
                if random.random() > keep_ratio:
                    return FilterResult(
                        action=FilterAction.DROP,
                        rule_name=rule.name,
                        reason="frequency_exceeded_sampled",
                        metadata={"frequency": frequency, "max": max_freq},
                    )

            return FilterResult(
                action=rule.action,
                rule_name=rule.name,
                reason="frequency_exceeded",
                metadata={"frequency": frequency, "max": max_freq},
            )

        return FilterResult(
            action=FilterAction.PASS,
            rule_name=rule.name,
            reason="frequency_ok",
            metadata={"frequency": frequency},
        )

    def _evaluate_value_range(self, rule: FilterRule, context: FilterContext) -> FilterResult:
        """Evaluate value-range filter.

        Static/boring values (always 0, always 100) = low value.
        """
        key = f"{context.data_type}:{context.name}"

        with self._lock:
            history = self._value_history.get(key, [])

        # Check for static value
        if len(history) >= rule.static_value_threshold:
            recent = history[-rule.static_value_threshold:]
            if len(set(recent)) == 1:
                self.low_value_detected.labels(
                    reason="static_value",
                    data_type=context.data_type,
                ).inc()

                return FilterResult(
                    action=rule.action,
                    rule_name=rule.name,
                    reason="static_value",
                    metadata={
                        "value": recent[0],
                        "consecutive_count": len(recent),
                    },
                )

        # Check value range (boring values like 0.0 or 100.0)
        if rule.value_range:
            min_val, max_val = rule.value_range
            try:
                value = float(context.value)
                if min_val <= value <= max_val:
                    self.low_value_detected.labels(
                        reason="boring_value",
                        data_type=context.data_type,
                    ).inc()

                    return FilterResult(
                        action=rule.action,
                        rule_name=rule.name,
                        reason="value_in_boring_range",
                        metadata={"value": value, "range": rule.value_range},
                    )
            except (TypeError, ValueError):
                pass

        return FilterResult(
            action=FilterAction.PASS,
            rule_name=rule.name,
            reason="value_interesting",
        )

    def _evaluate_semantic(self, rule: FilterRule, context: FilterContext) -> FilterResult:
        """Evaluate semantic-based filter.

        Check for semantic keywords indicating low value.
        """
        # Check name for keywords
        name_lower = context.name.lower()
        value_str = str(context.value).lower()

        low_value_keywords = rule.semantic_keywords or [
            "debug", "trace", "verbose", "internal",
            "healthcheck", "heartbeat", "ping", "test",
        ]

        for keyword in low_value_keywords:
            if keyword in name_lower or keyword in value_str:
                self.low_value_detected.labels(
                    reason="semantic_keyword",
                    data_type=context.data_type,
                ).inc()

                return FilterResult(
                    action=rule.action,
                    rule_name=rule.name,
                    reason=f"semantic_keyword:{keyword}",
                    metadata={"keyword": keyword},
                )

        # Check labels for low-value indicators
        for label, value in context.labels.items():
            if any(kw in label.lower() or kw in value.lower() for kw in low_value_keywords):
                return FilterResult(
                    action=rule.action,
                    rule_name=rule.name,
                    reason="semantic_label",
                    metadata={"label": label, "value": value},
                )

        return FilterResult(
            action=FilterAction.PASS,
            rule_name=rule.name,
            reason="semantic_ok",
        )

    def _evaluate_composite(self, rule: FilterRule, context: FilterContext) -> FilterResult:
        """Evaluate composite filter (multiple conditions)."""
        if not rule.conditions:
            return FilterResult(
                action=FilterAction.PASS,
                rule_name=rule.name,
                reason="no_conditions",
            )

        # All conditions must match for composite to match
        matched_conditions = []

        for condition in rule.conditions:
            result = self._evaluate_rule(condition, context)
            if result.action != FilterAction.PASS:
                matched_conditions.append(condition.name)
            else:
                # One condition didn't match
                return FilterResult(
                    action=FilterAction.PASS,
                    rule_name=rule.name,
                    reason="composite_partial_match",
                    metadata={"matched": matched_conditions},
                )

        # All conditions matched
        return FilterResult(
            action=rule.action,
            rule_name=rule.name,
            reason="composite_all_matched",
            metadata={"conditions": matched_conditions},
        )

    def _update_metrics(self, result: FilterResult, context: FilterContext):
        """Update Prometheus metrics."""
        self.filter_decisions.labels(
            action=result.action.value,
            rule=result.rule_name,
            data_type=context.data_type,
        ).inc()

        self._update_filter_rate()

    def _update_filter_rate(self):
        """Update filter rate gauge."""
        if self._total_evaluated > 0:
            rate = self._total_filtered / self._total_evaluated
            self.filter_rate.set(rate)

    def get_statistics(self) -> Dict[str, Any]:
        """Get filter statistics.

        Returns:
            Statistics dictionary
        """
        with self._lock:
            rule_stats = {
                name: count
                for name, count in self._filtered_by_rule.items()
            }

        return {
            "total_evaluated": self._total_evaluated,
            "total_passed": self._total_passed,
            "total_filtered": self._total_filtered,
            "filter_rate": (
                self._total_filtered / self._total_evaluated
                if self._total_evaluated > 0 else 0.0
            ),
            "rules": len(self._rules),
            "filtered_by_rule": rule_stats,
        }

    def get_entropy_report(self) -> Dict[str, float]:
        """Get entropy values for tracked sources.

        Returns:
            Dictionary of source to entropy
        """
        return dict(self._entropy_cache)

    def reset_statistics(self):
        """Reset filter statistics."""
        with self._lock:
            self._total_evaluated = 0
            self._total_passed = 0
            self._total_filtered = 0
            self._filtered_by_rule.clear()


class FilterChain:
    """Chain multiple filters together.

    Usage:
        chain = FilterChain()
        chain.add_filter(pattern_filter)
        chain.add_filter(entropy_filter)
        chain.add_filter(frequency_filter)

        result = chain.filter(context)
    """

    def __init__(self, namespace: str = "consciousness"):
        self.namespace = namespace
        self._filters: List[DataFilter] = []
        self._lock = threading.Lock()

    def add_filter(self, filter_instance: DataFilter):
        """Add a filter to the chain.

        Args:
            filter_instance: DataFilter instance
        """
        with self._lock:
            self._filters.append(filter_instance)

    def filter(self, context: FilterContext) -> FilterResult:
        """Run context through filter chain.

        Args:
            context: Filter context

        Returns:
            First non-PASS result, or PASS if all pass
        """
        for filter_instance in self._filters:
            result = filter_instance.should_filter(context)

            if result.action != FilterAction.PASS:
                return result

        return FilterResult(
            action=FilterAction.PASS,
            rule_name="chain_default",
            reason="all_filters_passed",
        )

    def get_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics from all filters."""
        return {
            f"filter_{i}": f.get_statistics()
            for i, f in enumerate(self._filters)
        }
