"""Security Event Generation

Generates structured security events with severity levels for SIEM consumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
from enum import Enum
import uuid
import threading
import logging

from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)


class EventSeverity(str, Enum):
    """Security event severity levels following SIEM standards."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


class EventCategory(str, Enum):
    """Security event categories following ECS standards."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    CONFIGURATION = "configuration"
    DATABASE = "database"
    FILE = "file"
    HOST = "host"
    IAM = "iam"
    INTRUSION_DETECTION = "intrusion_detection"
    MALWARE = "malware"
    NETWORK = "network"
    PACKAGE = "package"
    PROCESS = "process"
    SESSION = "session"
    THREAT = "threat"
    WEB = "web"


class EventOutcome(str, Enum):
    """Event outcome status."""
    SUCCESS = "success"
    FAILURE = "failure"
    UNKNOWN = "unknown"


class EventKind(str, Enum):
    """Event kind classification."""
    ALERT = "alert"
    ENRICHMENT = "enrichment"
    EVENT = "event"
    METRIC = "metric"
    STATE = "state"
    SIGNAL = "signal"


@dataclass
class EventSource:
    """Source of a security event."""
    ip: Optional[str] = None
    port: Optional[int] = None
    domain: Optional[str] = None
    user: Optional[str] = None
    process: Optional[str] = None
    geo: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result: Dict[str, Any] = {}
        if self.ip:
            result["ip"] = self.ip
        if self.port:
            result["port"] = self.port
        if self.domain:
            result["domain"] = self.domain
        if self.user:
            result["user"] = self.user
        if self.process:
            result["process"] = self.process
        if self.geo:
            result["geo"] = self.geo
        return result


@dataclass
class EventDestination:
    """Destination of a security event."""
    ip: Optional[str] = None
    port: Optional[int] = None
    domain: Optional[str] = None
    user: Optional[str] = None
    resource: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result: Dict[str, Any] = {}
        if self.ip:
            result["ip"] = self.ip
        if self.port:
            result["port"] = self.port
        if self.domain:
            result["domain"] = self.domain
        if self.user:
            result["user"] = self.user
        if self.resource:
            result["resource"] = self.resource
        return result


@dataclass
class ThreatIndicator:
    """Threat intelligence indicator."""
    type: str  # ip, domain, hash, url, etc.
    value: str
    provider: str
    confidence: float = 0.0
    severity: EventSeverity = EventSeverity.MEDIUM
    description: Optional[str] = None
    reference: Optional[str] = None
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type,
            "value": self.value,
            "provider": self.provider,
            "confidence": self.confidence,
            "severity": self.severity.value,
            "description": self.description,
            "reference": self.reference,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
        }


@dataclass
class SecurityEvent:
    """A security event following ECS (Elastic Common Schema) format."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    kind: EventKind = EventKind.EVENT
    category: EventCategory = EventCategory.NETWORK
    severity: EventSeverity = EventSeverity.INFORMATIONAL
    outcome: EventOutcome = EventOutcome.UNKNOWN
    action: str = ""
    message: str = ""
    source: Optional[EventSource] = None
    destination: Optional[EventDestination] = None
    threat: Optional[ThreatIndicator] = None
    rule_id: Optional[str] = None
    rule_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    risk_score: float = 0.0

    def __post_init__(self):
        """Calculate risk score if not provided."""
        if self.risk_score == 0.0:
            self.risk_score = self._calculate_risk_score()

    def _calculate_risk_score(self) -> float:
        """Calculate risk score based on severity and other factors."""
        severity_scores = {
            EventSeverity.CRITICAL: 100.0,
            EventSeverity.HIGH: 75.0,
            EventSeverity.MEDIUM: 50.0,
            EventSeverity.LOW: 25.0,
            EventSeverity.INFORMATIONAL: 10.0,
        }
        base_score = severity_scores.get(self.severity, 0.0)

        # Adjust based on outcome
        if self.outcome == EventOutcome.SUCCESS:
            base_score *= 1.2  # Successful attacks are more serious

        # Adjust based on threat indicator
        if self.threat:
            base_score *= (1.0 + self.threat.confidence)

        return min(100.0, base_score)

    @property
    def severity_value(self) -> int:
        """Get numeric severity for sorting."""
        values = {
            EventSeverity.CRITICAL: 5,
            EventSeverity.HIGH: 4,
            EventSeverity.MEDIUM: 3,
            EventSeverity.LOW: 2,
            EventSeverity.INFORMATIONAL: 1,
        }
        return values.get(self.severity, 0)

    def to_ecs(self) -> Dict[str, Any]:
        """Convert to Elastic Common Schema format."""
        ecs = {
            "@timestamp": self.timestamp.isoformat(),
            "event": {
                "id": self.event_id,
                "kind": self.kind.value,
                "category": [self.category.value],
                "type": [self.action] if self.action else [],
                "outcome": self.outcome.value,
                "severity": self.severity_value * 20,  # ECS severity 0-100
                "risk_score": self.risk_score,
            },
            "message": self.message,
            "tags": self.tags,
            "labels": self.labels,
        }

        if self.rule_id:
            ecs["rule"] = {
                "id": self.rule_id,
                "name": self.rule_name,
            }

        if self.source:
            ecs["source"] = self.source.to_dict()

        if self.destination:
            ecs["destination"] = self.destination.to_dict()

        if self.threat:
            ecs["threat"] = {
                "indicator": self.threat.to_dict(),
            }

        return ecs

    def to_cef(self) -> str:
        """Convert to Common Event Format (CEF) string."""
        cef_severity = {
            EventSeverity.CRITICAL: 10,
            EventSeverity.HIGH: 8,
            EventSeverity.MEDIUM: 5,
            EventSeverity.LOW: 3,
            EventSeverity.INFORMATIONAL: 1,
        }

        severity = cef_severity.get(self.severity, 0)

        # CEF Header
        header = (
            f"CEF:0|Consciousness|SecurityNexus|1.0|"
            f"{self.action}|{self.message[:128]}|{severity}"
        )

        # Extension fields
        extensions = [f"eventId={self.event_id}"]

        if self.source:
            if self.source.ip:
                extensions.append(f"src={self.source.ip}")
            if self.source.port:
                extensions.append(f"spt={self.source.port}")
            if self.source.user:
                extensions.append(f"suser={self.source.user}")

        if self.destination:
            if self.destination.ip:
                extensions.append(f"dst={self.destination.ip}")
            if self.destination.port:
                extensions.append(f"dpt={self.destination.port}")
            if self.destination.user:
                extensions.append(f"duser={self.destination.user}")

        extensions.append(f"cs1={self.category.value}")
        extensions.append(f"cs1Label=Category")
        extensions.append(f"cn1={self.risk_score}")
        extensions.append(f"cn1Label=RiskScore")

        return f"{header}|{' '.join(extensions)}"

    def to_leef(self) -> str:
        """Convert to Log Event Extended Format (LEEF) for QRadar."""
        leef_header = (
            f"LEEF:1.0|Consciousness|SecurityNexus|1.0|{self.action}|"
        )

        fields = [
            f"eventId={self.event_id}",
            f"cat={self.category.value}",
            f"sev={self.severity.value}",
            f"msg={self.message}",
        ]

        if self.source:
            if self.source.ip:
                fields.append(f"src={self.source.ip}")
            if self.source.port:
                fields.append(f"srcPort={self.source.port}")

        if self.destination:
            if self.destination.ip:
                fields.append(f"dst={self.destination.ip}")
            if self.destination.port:
                fields.append(f"dstPort={self.destination.port}")

        return leef_header + "\t".join(fields)


class SecurityEventGenerator:
    """Generates and manages security events.

    Usage:
        generator = SecurityEventGenerator()

        # Generate a security event
        event = generator.create_event(
            category=EventCategory.AUTHENTICATION,
            severity=EventSeverity.HIGH,
            action="login_failure",
            message="Failed login attempt from suspicious IP",
            source=EventSource(ip="192.168.1.100", user="admin"),
        )

        # Register callbacks
        generator.on_event(lambda e: print(f"New event: {e.event_id}"))

        # Emit event
        generator.emit(event)
    """

    def __init__(self, namespace: str = "consciousness"):
        self.namespace = namespace

        self._events: List[SecurityEvent] = []
        self._callbacks: List[Callable[[SecurityEvent], None]] = []
        self._lock = threading.Lock()

        # Severity thresholds for alerts
        self._alert_thresholds: Dict[EventSeverity, int] = {
            EventSeverity.CRITICAL: 0,  # Alert immediately
            EventSeverity.HIGH: 3,
        }

        # Metrics
        self.events_generated = Counter(
            f"{namespace}_siem_events_generated_total",
            "Total security events generated",
            ["category", "severity"],
        )

        self.events_by_category = Gauge(
            f"{namespace}_siem_events_by_category",
            "Events by category (last hour)",
            ["category"],
        )

        self.events_by_severity = Gauge(
            f"{namespace}_siem_events_by_severity",
            "Events by severity (last hour)",
            ["severity"],
        )

        self.risk_score_distribution = Histogram(
            f"{namespace}_siem_risk_score_distribution",
            "Risk score distribution",
            buckets=[10, 25, 50, 75, 90, 95, 100],
        )

        self.event_queue_size = Gauge(
            f"{namespace}_siem_event_queue_size",
            "Current event queue size",
        )

    def create_event(
        self,
        category: EventCategory,
        severity: EventSeverity,
        action: str,
        message: str,
        source: Optional[EventSource] = None,
        destination: Optional[EventDestination] = None,
        threat: Optional[ThreatIndicator] = None,
        rule_id: Optional[str] = None,
        rule_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        labels: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        kind: EventKind = EventKind.EVENT,
        outcome: EventOutcome = EventOutcome.UNKNOWN,
    ) -> SecurityEvent:
        """Create a new security event.

        Args:
            category: Event category
            severity: Event severity
            action: Action that triggered the event
            message: Human-readable message
            source: Event source
            destination: Event destination
            threat: Associated threat indicator
            rule_id: Detection rule ID
            rule_name: Detection rule name
            tags: Event tags
            labels: Event labels
            metadata: Additional metadata
            kind: Event kind
            outcome: Event outcome

        Returns:
            SecurityEvent instance
        """
        event = SecurityEvent(
            category=category,
            severity=severity,
            action=action,
            message=message,
            source=source,
            destination=destination,
            threat=threat,
            rule_id=rule_id,
            rule_name=rule_name,
            tags=tags or [],
            labels=labels or {},
            metadata=metadata or {},
            kind=kind,
            outcome=outcome,
        )

        return event

    def emit(self, event: SecurityEvent):
        """Emit a security event.

        Args:
            event: Event to emit
        """
        with self._lock:
            self._events.append(event)

            # Keep only recent events
            if len(self._events) > 10000:
                self._events = self._events[-5000:]

        # Update metrics
        self.events_generated.labels(
            category=event.category.value,
            severity=event.severity.value,
        ).inc()

        self.risk_score_distribution.observe(event.risk_score)
        self.event_queue_size.set(len(self._events))

        # Call registered callbacks
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")

        logger.debug(
            f"Emitted security event: {event.event_id} "
            f"[{event.severity.value}] {event.action}"
        )

    def on_event(self, callback: Callable[[SecurityEvent], None]):
        """Register an event callback.

        Args:
            callback: Function to call when event is emitted
        """
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[SecurityEvent], None]):
        """Remove an event callback.

        Args:
            callback: Callback to remove
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def get_events(
        self,
        category: Optional[EventCategory] = None,
        severity: Optional[EventSeverity] = None,
        min_risk_score: float = 0.0,
        limit: int = 100,
        since: Optional[datetime] = None,
    ) -> List[SecurityEvent]:
        """Get filtered events.

        Args:
            category: Filter by category
            severity: Filter by severity
            min_risk_score: Minimum risk score
            limit: Maximum events to return
            since: Only events after this time

        Returns:
            List of matching events
        """
        with self._lock:
            events = list(self._events)

        if category:
            events = [e for e in events if e.category == category]

        if severity:
            events = [e for e in events if e.severity == severity]

        if min_risk_score > 0:
            events = [e for e in events if e.risk_score >= min_risk_score]

        if since:
            events = [e for e in events if e.timestamp >= since]

        # Sort by timestamp descending
        events.sort(key=lambda e: e.timestamp, reverse=True)

        return events[:limit]

    def get_high_risk_events(
        self,
        threshold: float = 75.0,
        limit: int = 50,
    ) -> List[SecurityEvent]:
        """Get high-risk events.

        Args:
            threshold: Minimum risk score
            limit: Maximum events

        Returns:
            List of high-risk events
        """
        return self.get_events(min_risk_score=threshold, limit=limit)

    def get_statistics(self) -> Dict[str, Any]:
        """Get event statistics.

        Returns:
            Statistics dictionary
        """
        with self._lock:
            events = list(self._events)

        if not events:
            return {
                "total_events": 0,
                "by_category": {},
                "by_severity": {},
                "avg_risk_score": 0.0,
            }

        by_category: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}
        total_risk = 0.0

        for event in events:
            cat = event.category.value
            sev = event.severity.value

            by_category[cat] = by_category.get(cat, 0) + 1
            by_severity[sev] = by_severity.get(sev, 0) + 1
            total_risk += event.risk_score

        avg_risk = total_risk / len(events)

        # Update metrics
        for cat, count in by_category.items():
            self.events_by_category.labels(category=cat).set(count)

        for sev, count in by_severity.items():
            self.events_by_severity.labels(severity=sev).set(count)

        return {
            "total_events": len(events),
            "by_category": by_category,
            "by_severity": by_severity,
            "avg_risk_score": avg_risk,
            "high_risk_count": sum(
                1 for e in events if e.risk_score >= 75.0
            ),
            "critical_count": by_severity.get("critical", 0),
        }

    def clear_events(self):
        """Clear all events."""
        with self._lock:
            self._events.clear()
        self.event_queue_size.set(0)


# Convenience functions for common event types
def create_authentication_event(
    action: str,
    success: bool,
    username: str,
    source_ip: Optional[str] = None,
    message: Optional[str] = None,
) -> SecurityEvent:
    """Create an authentication event.

    Args:
        action: Auth action (login, logout, mfa_challenge, etc.)
        success: Whether auth succeeded
        username: Username
        source_ip: Source IP address
        message: Optional message

    Returns:
        SecurityEvent
    """
    severity = EventSeverity.INFORMATIONAL if success else EventSeverity.MEDIUM
    outcome = EventOutcome.SUCCESS if success else EventOutcome.FAILURE

    if not message:
        message = f"Authentication {action} {'succeeded' if success else 'failed'} for user {username}"

    return SecurityEvent(
        category=EventCategory.AUTHENTICATION,
        severity=severity,
        action=action,
        message=message,
        outcome=outcome,
        source=EventSource(ip=source_ip, user=username),
    )


def create_authorization_event(
    action: str,
    resource: str,
    username: str,
    allowed: bool,
    source_ip: Optional[str] = None,
) -> SecurityEvent:
    """Create an authorization event.

    Args:
        action: Access action
        resource: Resource being accessed
        username: Username
        allowed: Whether access was allowed
        source_ip: Source IP

    Returns:
        SecurityEvent
    """
    severity = EventSeverity.INFORMATIONAL if allowed else EventSeverity.MEDIUM
    outcome = EventOutcome.SUCCESS if allowed else EventOutcome.FAILURE

    return SecurityEvent(
        category=EventCategory.AUTHORIZATION,
        severity=severity,
        action=action,
        message=f"Access to {resource} {'allowed' if allowed else 'denied'} for {username}",
        outcome=outcome,
        source=EventSource(ip=source_ip, user=username),
        destination=EventDestination(resource=resource),
    )


def create_intrusion_event(
    attack_type: str,
    source_ip: str,
    target_ip: Optional[str] = None,
    target_port: Optional[int] = None,
    severity: EventSeverity = EventSeverity.HIGH,
    threat: Optional[ThreatIndicator] = None,
) -> SecurityEvent:
    """Create an intrusion detection event.

    Args:
        attack_type: Type of attack detected
        source_ip: Attacking IP
        target_ip: Target IP
        target_port: Target port
        severity: Event severity
        threat: Associated threat indicator

    Returns:
        SecurityEvent
    """
    return SecurityEvent(
        kind=EventKind.ALERT,
        category=EventCategory.INTRUSION_DETECTION,
        severity=severity,
        action=attack_type,
        message=f"Intrusion attempt detected: {attack_type} from {source_ip}",
        source=EventSource(ip=source_ip),
        destination=EventDestination(ip=target_ip, port=target_port),
        threat=threat,
    )


def create_malware_event(
    detection_type: str,
    file_path: Optional[str] = None,
    file_hash: Optional[str] = None,
    malware_name: Optional[str] = None,
    severity: EventSeverity = EventSeverity.CRITICAL,
) -> SecurityEvent:
    """Create a malware detection event.

    Args:
        detection_type: Type of detection
        file_path: Affected file path
        file_hash: File hash
        malware_name: Name of malware
        severity: Event severity

    Returns:
        SecurityEvent
    """
    message = f"Malware detected: {malware_name or detection_type}"
    if file_path:
        message += f" in {file_path}"

    metadata: Dict[str, Any] = {}
    if file_path:
        metadata["file_path"] = file_path
    if file_hash:
        metadata["file_hash"] = file_hash

    return SecurityEvent(
        kind=EventKind.ALERT,
        category=EventCategory.MALWARE,
        severity=severity,
        action=detection_type,
        message=message,
        metadata=metadata,
        tags=["malware", malware_name] if malware_name else ["malware"],
    )
