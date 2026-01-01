"""Event Router

Event routing and fanout for event-driven observability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Pattern
from datetime import datetime
from enum import Enum
import asyncio
import re
import threading
import logging

from prometheus_client import Counter, Gauge, Histogram

from .cloudevents import CloudEvent

logger = logging.getLogger(__name__)


class RoutingAction(str, Enum):
    """Actions for routed events."""
    DELIVER = "deliver"
    DROP = "drop"
    TRANSFORM = "transform"
    DUPLICATE = "duplicate"


@dataclass
class EventFilter:
    """Filter for matching events."""
    event_types: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    subjects: List[str] = field(default_factory=list)
    type_pattern: Optional[str] = None
    source_pattern: Optional[str] = None
    data_filters: Dict[str, Any] = field(default_factory=dict)

    def matches(self, event: CloudEvent) -> bool:
        """Check if event matches filter."""
        # Type check
        if self.event_types and event.type not in self.event_types:
            return False

        if self.type_pattern:
            if not re.match(self.type_pattern, event.type):
                return False

        # Source check
        if self.sources and event.source not in self.sources:
            return False

        if self.source_pattern:
            if not re.match(self.source_pattern, event.source):
                return False

        # Subject check
        if self.subjects:
            if not event.subject or event.subject not in self.subjects:
                return False

        # Data filters
        if self.data_filters and isinstance(event.data, dict):
            for key, expected in self.data_filters.items():
                actual = event.data.get(key)
                if actual != expected:
                    return False

        return True


@dataclass
class RoutingRule:
    """A routing rule for events."""
    rule_id: str
    name: str
    filter: EventFilter
    action: RoutingAction = RoutingAction.DELIVER
    priority: int = 0
    subscribers: List[str] = field(default_factory=list)
    transform_fn: Optional[Callable[[CloudEvent], CloudEvent]] = None
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EventSubscriber:
    """An event subscriber."""
    subscriber_id: str
    name: str
    handler: Callable[[CloudEvent], None]
    async_handler: Optional[Callable[[CloudEvent], Any]] = None
    filter: Optional[EventFilter] = None
    max_concurrent: int = 10
    retry_attempts: int = 3
    timeout_seconds: float = 30.0
    enabled: bool = True

    @property
    def is_async(self) -> bool:
        return self.async_handler is not None


class EventRouter:
    """Routes events to subscribers.

    Usage:
        router = EventRouter()

        # Add subscriber
        router.add_subscriber(EventSubscriber(
            subscriber_id="alerts",
            name="Alert Handler",
            handler=handle_alert,
            filter=EventFilter(event_types=["io.consciousness.alert.*"]),
        ))

        # Add routing rule
        router.add_rule(RoutingRule(
            rule_id="critical-alerts",
            name="Critical Alert Router",
            filter=EventFilter(data_filters={"severity": "critical"}),
            subscribers=["alerts", "pagerduty"],
        ))

        # Route event
        await router.route(event)
    """

    def __init__(self, namespace: str = "consciousness"):
        self.namespace = namespace

        self._subscribers: Dict[str, EventSubscriber] = {}
        self._rules: Dict[str, RoutingRule] = {}
        self._lock = threading.Lock()

        # Metrics
        self.events_routed = Counter(
            f"{namespace}_events_routed_total",
            "Total events routed",
            ["event_type", "subscriber"],
        )

        self.events_dropped = Counter(
            f"{namespace}_events_dropped_total",
            "Total events dropped",
            ["reason"],
        )

        self.routing_latency = Histogram(
            f"{namespace}_event_routing_latency_seconds",
            "Event routing latency",
            ["subscriber"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
        )

        self.subscriber_queue = Gauge(
            f"{namespace}_event_subscriber_queue_size",
            "Subscriber queue size",
            ["subscriber"],
        )

        self.active_subscribers = Gauge(
            f"{namespace}_event_active_subscribers",
            "Active subscribers",
        )

    def add_subscriber(self, subscriber: EventSubscriber):
        """Add an event subscriber.

        Args:
            subscriber: Subscriber to add
        """
        with self._lock:
            self._subscribers[subscriber.subscriber_id] = subscriber

        self.active_subscribers.set(len(self._subscribers))
        logger.info(f"Added subscriber: {subscriber.subscriber_id}")

    def remove_subscriber(self, subscriber_id: str):
        """Remove a subscriber.

        Args:
            subscriber_id: Subscriber ID
        """
        with self._lock:
            self._subscribers.pop(subscriber_id, None)

        self.active_subscribers.set(len(self._subscribers))

    def add_rule(self, rule: RoutingRule):
        """Add a routing rule.

        Args:
            rule: Routing rule
        """
        with self._lock:
            self._rules[rule.rule_id] = rule

        logger.info(f"Added routing rule: {rule.rule_id}")

    def remove_rule(self, rule_id: str):
        """Remove a routing rule.

        Args:
            rule_id: Rule ID
        """
        with self._lock:
            self._rules.pop(rule_id, None)

    async def route(self, event: CloudEvent) -> Dict[str, Any]:
        """Route an event to subscribers.

        Args:
            event: Event to route

        Returns:
            Routing result
        """
        import time
        start_time = time.time()

        # Find matching rules
        with self._lock:
            rules = sorted(
                [r for r in self._rules.values() if r.enabled],
                key=lambda r: r.priority,
                reverse=True,
            )

        matched_subscribers: set = set()
        should_drop = False
        transformed_event = event

        for rule in rules:
            if rule.filter.matches(event):
                if rule.action == RoutingAction.DROP:
                    should_drop = True
                    self.events_dropped.labels(reason="rule_drop").inc()
                    break

                if rule.action == RoutingAction.TRANSFORM and rule.transform_fn:
                    transformed_event = rule.transform_fn(event)

                matched_subscribers.update(rule.subscribers)

        if should_drop:
            return {"status": "dropped", "rule": rule.rule_id}

        # Add subscribers that match event directly
        with self._lock:
            subscribers = dict(self._subscribers)

        for sub in subscribers.values():
            if sub.enabled and sub.filter and sub.filter.matches(event):
                matched_subscribers.add(sub.subscriber_id)

        # Deliver to matched subscribers
        results = {}
        tasks = []

        for sub_id in matched_subscribers:
            subscriber = subscribers.get(sub_id)
            if not subscriber or not subscriber.enabled:
                continue

            if subscriber.is_async:
                task = self._deliver_async(subscriber, transformed_event)
                tasks.append((sub_id, task))
            else:
                result = self._deliver_sync(subscriber, transformed_event)
                results[sub_id] = result

        # Wait for async deliveries
        if tasks:
            async_results = await asyncio.gather(
                *[t[1] for t in tasks],
                return_exceptions=True,
            )
            for (sub_id, _), result in zip(tasks, async_results):
                if isinstance(result, Exception):
                    results[sub_id] = {"status": "error", "error": str(result)}
                else:
                    results[sub_id] = result

        duration = time.time() - start_time

        return {
            "status": "routed",
            "event_id": event.id,
            "event_type": event.type,
            "subscribers_notified": len(matched_subscribers),
            "duration_seconds": duration,
            "results": results,
        }

    def _deliver_sync(
        self,
        subscriber: EventSubscriber,
        event: CloudEvent,
    ) -> Dict[str, Any]:
        """Deliver event synchronously."""
        import time
        start_time = time.time()

        try:
            subscriber.handler(event)
            duration = time.time() - start_time

            self.events_routed.labels(
                event_type=event.type[:50],
                subscriber=subscriber.subscriber_id,
            ).inc()

            self.routing_latency.labels(
                subscriber=subscriber.subscriber_id,
            ).observe(duration)

            return {"status": "delivered", "duration": duration}

        except Exception as e:
            logger.error(f"Delivery error to {subscriber.subscriber_id}: {e}")
            return {"status": "error", "error": str(e)}

    async def _deliver_async(
        self,
        subscriber: EventSubscriber,
        event: CloudEvent,
    ) -> Dict[str, Any]:
        """Deliver event asynchronously."""
        import time
        start_time = time.time()

        try:
            await asyncio.wait_for(
                subscriber.async_handler(event),
                timeout=subscriber.timeout_seconds,
            )
            duration = time.time() - start_time

            self.events_routed.labels(
                event_type=event.type[:50],
                subscriber=subscriber.subscriber_id,
            ).inc()

            self.routing_latency.labels(
                subscriber=subscriber.subscriber_id,
            ).observe(duration)

            return {"status": "delivered", "duration": duration}

        except asyncio.TimeoutError:
            logger.error(f"Delivery timeout to {subscriber.subscriber_id}")
            return {"status": "timeout"}

        except Exception as e:
            logger.error(f"Delivery error to {subscriber.subscriber_id}: {e}")
            return {"status": "error", "error": str(e)}

    def route_sync(self, event: CloudEvent) -> Dict[str, Any]:
        """Route an event synchronously (for non-async contexts).

        Args:
            event: Event to route

        Returns:
            Routing result
        """
        # Find matching subscribers
        with self._lock:
            subscribers = dict(self._subscribers)
            rules = list(self._rules.values())

        matched_subscribers: set = set()

        for rule in rules:
            if rule.enabled and rule.filter.matches(event):
                matched_subscribers.update(rule.subscribers)

        for sub in subscribers.values():
            if sub.enabled and sub.filter and sub.filter.matches(event):
                matched_subscribers.add(sub.subscriber_id)

        # Deliver to matched subscribers
        results = {}

        for sub_id in matched_subscribers:
            subscriber = subscribers.get(sub_id)
            if not subscriber or not subscriber.enabled:
                continue

            if not subscriber.is_async:
                result = self._deliver_sync(subscriber, event)
                results[sub_id] = result

        return {
            "status": "routed",
            "event_id": event.id,
            "subscribers_notified": len(results),
            "results": results,
        }

    def get_subscribers(self) -> List[EventSubscriber]:
        """Get all subscribers."""
        with self._lock:
            return list(self._subscribers.values())

    def get_rules(self) -> List[RoutingRule]:
        """Get all rules."""
        with self._lock:
            return list(self._rules.values())

    def get_summary(self) -> Dict[str, Any]:
        """Get router summary."""
        with self._lock:
            subscribers = list(self._subscribers.values())
            rules = list(self._rules.values())

        return {
            "total_subscribers": len(subscribers),
            "active_subscribers": sum(1 for s in subscribers if s.enabled),
            "total_rules": len(rules),
            "active_rules": sum(1 for r in rules if r.enabled),
            "subscribers": [
                {
                    "id": s.subscriber_id,
                    "name": s.name,
                    "enabled": s.enabled,
                    "async": s.is_async,
                }
                for s in subscribers
            ],
        }
