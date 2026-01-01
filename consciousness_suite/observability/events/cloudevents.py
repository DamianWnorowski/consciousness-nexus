"""CloudEvents Integration

CloudEvents SDK integration for event-driven observability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from enum import Enum
import json
import uuid
import logging

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Standard event types."""
    # System events
    SYSTEM_STARTUP = "io.consciousness.system.startup"
    SYSTEM_SHUTDOWN = "io.consciousness.system.shutdown"
    SYSTEM_HEALTH_CHANGED = "io.consciousness.system.health.changed"

    # Deployment events
    DEPLOYMENT_STARTED = "io.consciousness.deployment.started"
    DEPLOYMENT_COMPLETED = "io.consciousness.deployment.completed"
    DEPLOYMENT_FAILED = "io.consciousness.deployment.failed"
    DEPLOYMENT_ROLLED_BACK = "io.consciousness.deployment.rolledback"

    # Incident events
    INCIDENT_CREATED = "io.consciousness.incident.created"
    INCIDENT_ACKNOWLEDGED = "io.consciousness.incident.acknowledged"
    INCIDENT_RESOLVED = "io.consciousness.incident.resolved"

    # Alert events
    ALERT_TRIGGERED = "io.consciousness.alert.triggered"
    ALERT_RESOLVED = "io.consciousness.alert.resolved"

    # Security events
    SECURITY_VULNERABILITY_DETECTED = "io.consciousness.security.vulnerability"
    SECURITY_POLICY_VIOLATION = "io.consciousness.security.policy.violation"

    # Custom
    CUSTOM = "io.consciousness.custom"


class ContentType(str, Enum):
    """Content types for event data."""
    JSON = "application/json"
    TEXT = "text/plain"
    PROTOBUF = "application/protobuf"
    AVRO = "application/avro"


@dataclass
class EventSource:
    """Event source identifier."""
    service: str
    instance: str = ""
    version: str = ""

    def to_uri(self) -> str:
        """Convert to URI format."""
        uri = f"//{self.service}"
        if self.instance:
            uri += f"/{self.instance}"
        if self.version:
            uri += f"@{self.version}"
        return uri

    @classmethod
    def from_uri(cls, uri: str) -> "EventSource":
        """Parse from URI."""
        # Remove leading //
        uri = uri.lstrip("/")

        version = ""
        if "@" in uri:
            uri, version = uri.rsplit("@", 1)

        parts = uri.split("/")
        service = parts[0]
        instance = parts[1] if len(parts) > 1 else ""

        return cls(service=service, instance=instance, version=version)


@dataclass
class CloudEvent:
    """CloudEvents specification v1.0 event.

    Usage:
        event = CloudEventBuilder() \\
            .with_type(EventType.DEPLOYMENT_COMPLETED) \\
            .with_source("deployment-service") \\
            .with_data({"environment": "production"}) \\
            .build()
    """
    # Required attributes
    id: str
    source: str
    type: str
    specversion: str = "1.0"

    # Optional attributes
    datacontenttype: str = "application/json"
    dataschema: Optional[str] = None
    subject: Optional[str] = None
    time: Optional[datetime] = None

    # Data
    data: Optional[Union[Dict[str, Any], str, bytes]] = None

    # Extension attributes
    extensions: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (JSON format)."""
        result = {
            "id": self.id,
            "source": self.source,
            "type": self.type,
            "specversion": self.specversion,
        }

        if self.datacontenttype:
            result["datacontenttype"] = self.datacontenttype

        if self.dataschema:
            result["dataschema"] = self.dataschema

        if self.subject:
            result["subject"] = self.subject

        if self.time:
            result["time"] = self.time.isoformat() + "Z"

        if self.data is not None:
            result["data"] = self.data

        # Add extensions
        for key, value in self.extensions.items():
            result[key] = value

        return result

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)

    def to_http_headers(self) -> Dict[str, str]:
        """Convert to HTTP headers (binary format)."""
        headers = {
            "ce-id": self.id,
            "ce-source": self.source,
            "ce-type": self.type,
            "ce-specversion": self.specversion,
            "Content-Type": self.datacontenttype,
        }

        if self.dataschema:
            headers["ce-dataschema"] = self.dataschema

        if self.subject:
            headers["ce-subject"] = self.subject

        if self.time:
            headers["ce-time"] = self.time.isoformat() + "Z"

        # Add extensions
        for key, value in self.extensions.items():
            headers[f"ce-{key}"] = str(value)

        return headers

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CloudEvent":
        """Create from dictionary."""
        # Extract known fields
        event_id = data.pop("id")
        source = data.pop("source")
        event_type = data.pop("type")
        specversion = data.pop("specversion", "1.0")

        datacontenttype = data.pop("datacontenttype", "application/json")
        dataschema = data.pop("dataschema", None)
        subject = data.pop("subject", None)

        time_str = data.pop("time", None)
        time = datetime.fromisoformat(time_str.rstrip("Z")) if time_str else None

        event_data = data.pop("data", None)

        # Remaining fields are extensions
        extensions = data

        return cls(
            id=event_id,
            source=source,
            type=event_type,
            specversion=specversion,
            datacontenttype=datacontenttype,
            dataschema=dataschema,
            subject=subject,
            time=time,
            data=event_data,
            extensions=extensions,
        )


class CloudEventBuilder:
    """Builder for CloudEvents.

    Usage:
        event = CloudEventBuilder() \\
            .with_type(EventType.DEPLOYMENT_COMPLETED) \\
            .with_source(EventSource("deploy-svc", "prod-1")) \\
            .with_subject("deployment/nginx") \\
            .with_data({"status": "success"}) \\
            .with_extension("traceparent", trace_id) \\
            .build()
    """

    def __init__(self):
        self._id: Optional[str] = None
        self._source: Optional[str] = None
        self._type: Optional[str] = None
        self._datacontenttype: str = "application/json"
        self._dataschema: Optional[str] = None
        self._subject: Optional[str] = None
        self._time: Optional[datetime] = None
        self._data: Optional[Any] = None
        self._extensions: Dict[str, Any] = {}

    def with_id(self, event_id: str) -> "CloudEventBuilder":
        """Set event ID."""
        self._id = event_id
        return self

    def with_source(self, source: Union[str, EventSource]) -> "CloudEventBuilder":
        """Set event source."""
        if isinstance(source, EventSource):
            self._source = source.to_uri()
        else:
            self._source = source
        return self

    def with_type(self, event_type: Union[str, EventType]) -> "CloudEventBuilder":
        """Set event type."""
        if isinstance(event_type, EventType):
            self._type = event_type.value
        else:
            self._type = event_type
        return self

    def with_data(
        self,
        data: Any,
        content_type: str = "application/json",
    ) -> "CloudEventBuilder":
        """Set event data."""
        self._data = data
        self._datacontenttype = content_type
        return self

    def with_subject(self, subject: str) -> "CloudEventBuilder":
        """Set event subject."""
        self._subject = subject
        return self

    def with_time(self, time: datetime) -> "CloudEventBuilder":
        """Set event time."""
        self._time = time
        return self

    def with_schema(self, schema_uri: str) -> "CloudEventBuilder":
        """Set data schema URI."""
        self._dataschema = schema_uri
        return self

    def with_extension(self, key: str, value: Any) -> "CloudEventBuilder":
        """Add extension attribute."""
        self._extensions[key] = value
        return self

    def with_trace_context(
        self,
        trace_id: str,
        span_id: Optional[str] = None,
    ) -> "CloudEventBuilder":
        """Add W3C trace context."""
        if span_id:
            traceparent = f"00-{trace_id}-{span_id}-01"
        else:
            traceparent = f"00-{trace_id}-0000000000000000-01"
        self._extensions["traceparent"] = traceparent
        return self

    def build(self) -> CloudEvent:
        """Build the CloudEvent."""
        if not self._source:
            raise ValueError("Event source is required")
        if not self._type:
            raise ValueError("Event type is required")

        return CloudEvent(
            id=self._id or str(uuid.uuid4()),
            source=self._source,
            type=self._type,
            datacontenttype=self._datacontenttype,
            dataschema=self._dataschema,
            subject=self._subject,
            time=self._time or datetime.utcnow(),
            data=self._data,
            extensions=self._extensions,
        )


class CloudEventParser:
    """Parses CloudEvents from various formats."""

    @staticmethod
    def from_json(json_str: str) -> CloudEvent:
        """Parse from JSON string."""
        data = json.loads(json_str)
        return CloudEvent.from_dict(data)

    @staticmethod
    def from_http(
        headers: Dict[str, str],
        body: Optional[bytes] = None,
    ) -> CloudEvent:
        """Parse from HTTP request (binary mode)."""
        # Check if structured or binary mode
        content_type = headers.get("Content-Type", headers.get("content-type", ""))

        if "application/cloudevents" in content_type:
            # Structured mode
            data = json.loads(body.decode() if body else "{}")
            return CloudEvent.from_dict(data)

        # Binary mode - parse from headers
        event_id = headers.get("ce-id")
        source = headers.get("ce-source")
        event_type = headers.get("ce-type")
        specversion = headers.get("ce-specversion", "1.0")

        if not all([event_id, source, event_type]):
            raise ValueError("Missing required CloudEvents headers")

        # Optional headers
        dataschema = headers.get("ce-dataschema")
        subject = headers.get("ce-subject")
        time_str = headers.get("ce-time")
        time = datetime.fromisoformat(time_str.rstrip("Z")) if time_str else None

        # Parse data
        event_data = None
        if body:
            if "json" in content_type:
                event_data = json.loads(body.decode())
            else:
                event_data = body.decode()

        # Extract extensions
        extensions = {}
        for key, value in headers.items():
            if key.startswith("ce-") and key not in [
                "ce-id", "ce-source", "ce-type", "ce-specversion",
                "ce-dataschema", "ce-subject", "ce-time",
            ]:
                ext_name = key[3:]  # Remove "ce-" prefix
                extensions[ext_name] = value

        return CloudEvent(
            id=event_id,
            source=source,
            type=event_type,
            specversion=specversion,
            datacontenttype=content_type,
            dataschema=dataschema,
            subject=subject,
            time=time,
            data=event_data,
            extensions=extensions,
        )


@dataclass
class CloudEventBatch:
    """A batch of CloudEvents.

    Usage:
        batch = CloudEventBatch()
        batch.add(event1)
        batch.add(event2)

        # Send batch
        json_batch = batch.to_json()
    """
    events: List[CloudEvent] = field(default_factory=list)

    def add(self, event: CloudEvent):
        """Add event to batch."""
        self.events.append(event)

    def clear(self):
        """Clear all events."""
        self.events.clear()

    @property
    def size(self) -> int:
        """Get batch size."""
        return len(self.events)

    def to_list(self) -> List[Dict[str, Any]]:
        """Convert to list of dictionaries."""
        return [e.to_dict() for e in self.events]

    def to_json(self) -> str:
        """Convert to JSON array."""
        return json.dumps(self.to_list(), default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "CloudEventBatch":
        """Parse batch from JSON array."""
        data = json.loads(json_str)
        events = [CloudEvent.from_dict(e) for e in data]
        return cls(events=events)

    def filter_by_type(self, event_type: str) -> "CloudEventBatch":
        """Filter events by type."""
        filtered = [e for e in self.events if e.type == event_type]
        return CloudEventBatch(events=filtered)

    def filter_by_source(self, source: str) -> "CloudEventBatch":
        """Filter events by source."""
        filtered = [e for e in self.events if source in e.source]
        return CloudEventBatch(events=filtered)
