"""SLI (Service Level Indicator) Calculator

Calculates and tracks Service Level Indicators from various data sources.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
from datetime import datetime, timedelta
from enum import Enum
import threading
import logging
import statistics

from prometheus_client import Gauge, Histogram

logger = logging.getLogger(__name__)


class SLIType(str, Enum):
    """Types of SLI metrics."""
    REQUEST_SUCCESS_RATE = "request_success_rate"
    REQUEST_LATENCY = "request_latency"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    AVAILABILITY = "availability"
    SATURATION = "saturation"
    FRESHNESS = "freshness"
    COVERAGE = "coverage"
    CORRECTNESS = "correctness"
    CUSTOM = "custom"


class AggregationType(str, Enum):
    """Types of aggregation for SLI calculation."""
    COUNT = "count"
    SUM = "sum"
    AVERAGE = "average"
    PERCENTILE = "percentile"
    RATIO = "ratio"
    RATE = "rate"


@dataclass
class TimeWindow:
    """Time window for SLI calculation."""
    duration: timedelta
    alignment: str = "none"  # none, hour, day
    offset: timedelta = field(default_factory=lambda: timedelta())

    @classmethod
    def minutes(cls, n: int) -> TimeWindow:
        return cls(duration=timedelta(minutes=n))

    @classmethod
    def hours(cls, n: int) -> TimeWindow:
        return cls(duration=timedelta(hours=n))

    @classmethod
    def days(cls, n: int) -> TimeWindow:
        return cls(duration=timedelta(days=n))


@dataclass
class SLIMetric:
    """Definition of an SLI metric.

    Usage:
        sli = SLIMetric(
            id="api-latency-p99",
            name="API Latency P99",
            sli_type=SLIType.REQUEST_LATENCY,
            aggregation=AggregationType.PERCENTILE,
            percentile=99,
            good_threshold=500,  # ms
            unit="ms",
        )
    """
    id: str
    name: str
    sli_type: SLIType
    aggregation: AggregationType = AggregationType.RATIO
    good_threshold: Optional[float] = None
    bad_threshold: Optional[float] = None
    percentile: float = 99.0
    unit: str = ""
    description: str = ""
    labels: Dict[str, str] = field(default_factory=dict)

    # For ratio-based SLIs
    numerator_query: Optional[str] = None
    denominator_query: Optional[str] = None


@dataclass
class SLIResult:
    """Result of an SLI calculation."""
    metric_id: str
    value: float
    good_events: int
    total_events: int
    window_start: datetime
    window_end: datetime
    calculation_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def ratio(self) -> float:
        """Get value as ratio (0-1)."""
        return self.good_events / self.total_events if self.total_events > 0 else 0

    @property
    def percentage(self) -> float:
        """Get value as percentage."""
        return self.ratio * 100


class SLICalculator:
    """Calculates SLI values from raw metrics.

    Usage:
        calc = SLICalculator()

        # Define SLI
        calc.define_sli(SLIMetric(
            id="api-success-rate",
            name="API Success Rate",
            sli_type=SLIType.REQUEST_SUCCESS_RATE,
        ))

        # Record events
        calc.record_event("api-success-rate", good=True, latency_ms=50)
        calc.record_event("api-success-rate", good=False, latency_ms=5000)

        # Calculate SLI
        result = calc.calculate("api-success-rate", TimeWindow.hours(1))
    """

    def __init__(self, namespace: str = "consciousness"):
        self.namespace = namespace
        self._slis: Dict[str, SLIMetric] = {}
        self._events: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self._max_events = 1000000

        # Prometheus metrics
        self.sli_value = Gauge(
            f"{namespace}_sli_value",
            "Current SLI value",
            ["sli_id", "sli_type"],
        )

        self.sli_good_events = Gauge(
            f"{namespace}_sli_good_events_total",
            "Total good events in current window",
            ["sli_id"],
        )

        self.sli_total_events = Gauge(
            f"{namespace}_sli_total_events_total",
            "Total events in current window",
            ["sli_id"],
        )

    def define_sli(self, sli: SLIMetric):
        """Define an SLI metric.

        Args:
            sli: SLI definition
        """
        with self._lock:
            self._slis[sli.id] = sli
            self._events[sli.id] = []

        logger.info(f"Defined SLI: {sli.id}")

    def remove_sli(self, sli_id: str):
        """Remove an SLI definition."""
        with self._lock:
            if sli_id in self._slis:
                del self._slis[sli_id]
                del self._events[sli_id]

    def record_event(
        self,
        sli_id: str,
        good: bool,
        value: Optional[float] = None,
        latency_ms: Optional[float] = None,
        timestamp: Optional[datetime] = None,
        labels: Optional[Dict[str, str]] = None,
    ):
        """Record an SLI event.

        Args:
            sli_id: SLI metric ID
            good: Whether this was a "good" event
            value: Event value (for aggregations)
            latency_ms: Latency in milliseconds
            timestamp: Event timestamp
            labels: Additional labels
        """
        if sli_id not in self._slis:
            return

        event = {
            "good": good,
            "value": value,
            "latency_ms": latency_ms,
            "timestamp": timestamp or datetime.now(),
            "labels": labels or {},
        }

        with self._lock:
            self._events[sli_id].append(event)

            # Trim old events
            if len(self._events[sli_id]) > self._max_events:
                self._events[sli_id] = self._events[sli_id][-self._max_events // 2:]

    def record_batch(
        self,
        sli_id: str,
        events: List[Dict[str, Any]],
    ):
        """Record multiple events.

        Args:
            sli_id: SLI metric ID
            events: List of event dictionaries
        """
        if sli_id not in self._slis:
            return

        with self._lock:
            for event in events:
                if "timestamp" not in event:
                    event["timestamp"] = datetime.now()
                self._events[sli_id].append(event)

            # Trim
            if len(self._events[sli_id]) > self._max_events:
                self._events[sli_id] = self._events[sli_id][-self._max_events // 2:]

    def calculate(
        self,
        sli_id: str,
        window: TimeWindow,
        labels: Optional[Dict[str, str]] = None,
    ) -> Optional[SLIResult]:
        """Calculate SLI value for a time window.

        Args:
            sli_id: SLI metric ID
            window: Time window
            labels: Filter by labels

        Returns:
            SLIResult or None
        """
        import time
        start = time.perf_counter()

        if sli_id not in self._slis:
            return None

        sli = self._slis[sli_id]

        now = datetime.now()
        window_start = now - window.duration - window.offset
        window_end = now - window.offset

        with self._lock:
            events = [
                e for e in self._events.get(sli_id, [])
                if window_start <= e["timestamp"] <= window_end
            ]

        # Apply label filters
        if labels:
            events = [
                e for e in events
                if all(e.get("labels", {}).get(k) == v for k, v in labels.items())
            ]

        if not events:
            return SLIResult(
                metric_id=sli_id,
                value=1.0,
                good_events=0,
                total_events=0,
                window_start=window_start,
                window_end=window_end,
                calculation_time_ms=(time.perf_counter() - start) * 1000,
            )

        # Calculate based on SLI type
        good_events = 0
        total_events = len(events)
        value = 0.0

        if sli.aggregation == AggregationType.RATIO:
            good_events = sum(1 for e in events if e.get("good", False))
            value = good_events / total_events

        elif sli.aggregation == AggregationType.PERCENTILE:
            values = [e.get("latency_ms", e.get("value", 0)) for e in events]
            if values:
                values.sort()
                idx = int(len(values) * sli.percentile / 100)
                value = values[min(idx, len(values) - 1)]

                # Count good if under threshold
                if sli.good_threshold:
                    good_events = sum(1 for v in values if v <= sli.good_threshold)

        elif sli.aggregation == AggregationType.AVERAGE:
            values = [e.get("value", 0) for e in events if e.get("value") is not None]
            if values:
                value = statistics.mean(values)

                if sli.good_threshold:
                    good_events = sum(1 for v in values if v <= sli.good_threshold)

        elif sli.aggregation == AggregationType.COUNT:
            good_events = sum(1 for e in events if e.get("good", False))
            value = good_events

        elif sli.aggregation == AggregationType.SUM:
            values = [e.get("value", 0) for e in events if e.get("value") is not None]
            value = sum(values)

        elif sli.aggregation == AggregationType.RATE:
            window_seconds = window.duration.total_seconds()
            value = total_events / window_seconds if window_seconds > 0 else 0

        # Update Prometheus metrics
        self.sli_value.labels(
            sli_id=sli_id,
            sli_type=sli.sli_type.value,
        ).set(value if sli.aggregation != AggregationType.RATIO else value * 100)

        self.sli_good_events.labels(sli_id=sli_id).set(good_events)
        self.sli_total_events.labels(sli_id=sli_id).set(total_events)

        return SLIResult(
            metric_id=sli_id,
            value=value,
            good_events=good_events,
            total_events=total_events,
            window_start=window_start,
            window_end=window_end,
            calculation_time_ms=(time.perf_counter() - start) * 1000,
        )

    def calculate_all(
        self,
        window: TimeWindow,
    ) -> Dict[str, SLIResult]:
        """Calculate all SLIs.

        Args:
            window: Time window

        Returns:
            Dictionary mapping SLI ID to result
        """
        return {
            sli_id: self.calculate(sli_id, window)
            for sli_id in self._slis
        }

    def get_sli_definition(self, sli_id: str) -> Optional[SLIMetric]:
        """Get SLI definition.

        Args:
            sli_id: SLI ID

        Returns:
            SLIMetric or None
        """
        return self._slis.get(sli_id)

    def list_slis(self) -> List[SLIMetric]:
        """List all defined SLIs.

        Returns:
            List of SLI definitions
        """
        return list(self._slis.values())

    def get_event_count(self, sli_id: str) -> int:
        """Get total event count for an SLI.

        Args:
            sli_id: SLI ID

        Returns:
            Event count
        """
        with self._lock:
            return len(self._events.get(sli_id, []))


# Predefined SLI templates

def availability_sli(service: str) -> SLIMetric:
    """Create an availability SLI."""
    return SLIMetric(
        id=f"{service}-availability",
        name=f"{service} Availability",
        sli_type=SLIType.AVAILABILITY,
        aggregation=AggregationType.RATIO,
        description=f"Availability of {service}",
    )


def latency_p99_sli(service: str, threshold_ms: float = 500) -> SLIMetric:
    """Create a P99 latency SLI."""
    return SLIMetric(
        id=f"{service}-latency-p99",
        name=f"{service} Latency P99",
        sli_type=SLIType.REQUEST_LATENCY,
        aggregation=AggregationType.PERCENTILE,
        percentile=99,
        good_threshold=threshold_ms,
        unit="ms",
        description=f"P99 latency of {service}",
    )


def error_rate_sli(service: str) -> SLIMetric:
    """Create an error rate SLI."""
    return SLIMetric(
        id=f"{service}-error-rate",
        name=f"{service} Error Rate",
        sli_type=SLIType.ERROR_RATE,
        aggregation=AggregationType.RATIO,
        description=f"Error rate of {service}",
    )


def throughput_sli(service: str) -> SLIMetric:
    """Create a throughput SLI."""
    return SLIMetric(
        id=f"{service}-throughput",
        name=f"{service} Throughput",
        sli_type=SLIType.THROUGHPUT,
        aggregation=AggregationType.RATE,
        unit="req/s",
        description=f"Throughput of {service}",
    )
