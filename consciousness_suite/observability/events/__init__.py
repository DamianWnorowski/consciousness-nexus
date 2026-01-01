"""Events & CloudEvents Module

Event-driven observability:
- CloudEvents SDK integration
- Kubernetes events collection
- Event routing and fanout
- Event persistence and querying
- Event-trace correlation
"""

from .cloudevents import (
    CloudEventBuilder,
    CloudEventParser,
    CloudEventBatch,
    EventType,
    EventSource,
)
from .k8s_events import (
    K8sEventCollector,
    K8sEvent,
    K8sEventType,
    EventReason,
)
from .event_router import (
    EventRouter,
    EventSubscriber,
    RoutingRule,
    EventFilter,
)
from .event_store import (
    EventStore,
    StoredEvent,
    EventQuery,
    EventIndex,
)
from .correlation import (
    EventCorrelator,
    CorrelatedEvents,
    CorrelationRule,
    TraceEventLink,
)

__all__ = [
    # CloudEvents
    "CloudEventBuilder",
    "CloudEventParser",
    "CloudEventBatch",
    "EventType",
    "EventSource",
    # Kubernetes
    "K8sEventCollector",
    "K8sEvent",
    "K8sEventType",
    "EventReason",
    # Router
    "EventRouter",
    "EventSubscriber",
    "RoutingRule",
    "EventFilter",
    # Store
    "EventStore",
    "StoredEvent",
    "EventQuery",
    "EventIndex",
    # Correlation
    "EventCorrelator",
    "CorrelatedEvents",
    "CorrelationRule",
    "TraceEventLink",
]
