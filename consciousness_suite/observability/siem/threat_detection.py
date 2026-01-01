"""Threat Pattern Detection

Detects threat patterns and anomalies in security events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Set
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import re
import threading
import logging

from prometheus_client import Counter, Gauge, Histogram

from .security_events import (
    SecurityEvent,
    EventSeverity,
    EventCategory,
    EventKind,
    EventOutcome,
    ThreatIndicator,
)

logger = logging.getLogger(__name__)


class ThreatType(str, Enum):
    """Types of detected threats."""
    BRUTE_FORCE = "brute_force"
    CREDENTIAL_STUFFING = "credential_stuffing"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    LATERAL_MOVEMENT = "lateral_movement"
    DATA_EXFILTRATION = "data_exfiltration"
    RECONNAISSANCE = "reconnaissance"
    COMMAND_AND_CONTROL = "command_and_control"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    POLICY_VIOLATION = "policy_violation"
    MALWARE_ACTIVITY = "malware_activity"
    INJECTION_ATTACK = "injection_attack"
    DOS_ATTACK = "dos_attack"


class PatternMatchType(str, Enum):
    """How patterns are matched."""
    EXACT = "exact"
    REGEX = "regex"
    THRESHOLD = "threshold"
    SEQUENCE = "sequence"
    CORRELATION = "correlation"


@dataclass
class ThreatPattern:
    """A threat detection pattern."""
    pattern_id: str
    name: str
    description: str
    threat_type: ThreatType
    severity: EventSeverity
    match_type: PatternMatchType
    conditions: Dict[str, Any] = field(default_factory=dict)
    threshold: int = 1
    window_seconds: int = 300  # 5 minutes
    enabled: bool = True
    tags: List[str] = field(default_factory=list)
    mitre_tactics: List[str] = field(default_factory=list)
    mitre_techniques: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_id": self.pattern_id,
            "name": self.name,
            "description": self.description,
            "threat_type": self.threat_type.value,
            "severity": self.severity.value,
            "match_type": self.match_type.value,
            "conditions": self.conditions,
            "threshold": self.threshold,
            "window_seconds": self.window_seconds,
            "enabled": self.enabled,
            "tags": self.tags,
            "mitre_tactics": self.mitre_tactics,
            "mitre_techniques": self.mitre_techniques,
        }


@dataclass
class ThreatMatch:
    """A threat pattern match."""
    match_id: str
    pattern: ThreatPattern
    matched_events: List[SecurityEvent]
    match_time: datetime = field(default_factory=datetime.utcnow)
    confidence: float = 0.0
    context: Dict[str, Any] = field(default_factory=dict)

    @property
    def severity(self) -> EventSeverity:
        """Get severity from pattern."""
        return self.pattern.severity

    @property
    def threat_type(self) -> ThreatType:
        """Get threat type from pattern."""
        return self.pattern.threat_type

    def to_security_event(self) -> SecurityEvent:
        """Convert match to a security event."""
        return SecurityEvent(
            kind=EventKind.ALERT,
            category=EventCategory.THREAT,
            severity=self.pattern.severity,
            action=self.pattern.threat_type.value,
            message=f"Threat detected: {self.pattern.name}",
            rule_id=self.pattern.pattern_id,
            rule_name=self.pattern.name,
            tags=self.pattern.tags + ["threat_detection"],
            metadata={
                "matched_events": len(self.matched_events),
                "confidence": self.confidence,
                "threat_type": self.pattern.threat_type.value,
                "mitre_tactics": self.pattern.mitre_tactics,
                "mitre_techniques": self.pattern.mitre_techniques,
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "match_id": self.match_id,
            "pattern_id": self.pattern.pattern_id,
            "pattern_name": self.pattern.name,
            "threat_type": self.pattern.threat_type.value,
            "severity": self.pattern.severity.value,
            "matched_events": len(self.matched_events),
            "match_time": self.match_time.isoformat(),
            "confidence": self.confidence,
            "context": self.context,
        }


@dataclass
class AnomalyScore:
    """Anomaly detection score."""
    entity: str  # IP, user, etc.
    entity_type: str
    score: float  # 0-100
    baseline: float
    current: float
    deviation: float
    factors: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_anomalous(self) -> bool:
        """Check if score indicates anomaly."""
        return self.score >= 75.0


class ThreatDetector:
    """Detects threats based on patterns and anomalies.

    Usage:
        detector = ThreatDetector()

        # Add built-in patterns
        detector.add_default_patterns()

        # Process events
        matches = detector.analyze(event)

        # Register threat callback
        detector.on_threat(lambda m: send_alert(m))

        # Get threat summary
        summary = detector.get_summary()
    """

    def __init__(self, namespace: str = "consciousness"):
        self.namespace = namespace

        self._patterns: Dict[str, ThreatPattern] = {}
        self._matches: List[ThreatMatch] = []
        self._event_window: List[SecurityEvent] = []
        self._callbacks: List[Callable[[ThreatMatch], None]] = []
        self._lock = threading.Lock()

        # Behavioral baselines for anomaly detection
        self._baselines: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._current_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Metrics
        self.threats_detected = Counter(
            f"{namespace}_siem_threats_detected_total",
            "Total threats detected",
            ["threat_type", "severity"],
        )

        self.patterns_matched = Counter(
            f"{namespace}_siem_patterns_matched_total",
            "Total patterns matched",
            ["pattern_id"],
        )

        self.anomaly_scores = Histogram(
            f"{namespace}_siem_anomaly_scores",
            "Anomaly score distribution",
            ["entity_type"],
            buckets=[25, 50, 75, 85, 90, 95, 100],
        )

        self.active_patterns = Gauge(
            f"{namespace}_siem_active_patterns",
            "Number of active patterns",
        )

        self.events_analyzed = Counter(
            f"{namespace}_siem_events_analyzed_total",
            "Total events analyzed",
        )

    def add_pattern(self, pattern: ThreatPattern):
        """Add a detection pattern.

        Args:
            pattern: Pattern to add
        """
        with self._lock:
            self._patterns[pattern.pattern_id] = pattern

        self.active_patterns.set(
            sum(1 for p in self._patterns.values() if p.enabled)
        )
        logger.info(f"Added threat pattern: {pattern.pattern_id}")

    def remove_pattern(self, pattern_id: str):
        """Remove a pattern.

        Args:
            pattern_id: Pattern ID to remove
        """
        with self._lock:
            self._patterns.pop(pattern_id, None)

        self.active_patterns.set(
            sum(1 for p in self._patterns.values() if p.enabled)
        )

    def enable_pattern(self, pattern_id: str, enabled: bool = True):
        """Enable or disable a pattern.

        Args:
            pattern_id: Pattern ID
            enabled: Whether to enable
        """
        with self._lock:
            if pattern_id in self._patterns:
                self._patterns[pattern_id].enabled = enabled

        self.active_patterns.set(
            sum(1 for p in self._patterns.values() if p.enabled)
        )

    def add_default_patterns(self):
        """Add default threat detection patterns."""
        defaults = [
            ThreatPattern(
                pattern_id="brute_force_ssh",
                name="SSH Brute Force Attack",
                description="Multiple failed SSH login attempts from same source",
                threat_type=ThreatType.BRUTE_FORCE,
                severity=EventSeverity.HIGH,
                match_type=PatternMatchType.THRESHOLD,
                conditions={
                    "category": EventCategory.AUTHENTICATION.value,
                    "action": "ssh_login",
                    "outcome": EventOutcome.FAILURE.value,
                    "group_by": "source.ip",
                },
                threshold=10,
                window_seconds=300,
                mitre_tactics=["TA0006"],  # Credential Access
                mitre_techniques=["T1110.001"],  # Brute Force: Password Guessing
            ),
            ThreatPattern(
                pattern_id="credential_stuffing",
                name="Credential Stuffing Attack",
                description="Multiple failed logins with different usernames from same IP",
                threat_type=ThreatType.CREDENTIAL_STUFFING,
                severity=EventSeverity.HIGH,
                match_type=PatternMatchType.THRESHOLD,
                conditions={
                    "category": EventCategory.AUTHENTICATION.value,
                    "outcome": EventOutcome.FAILURE.value,
                    "group_by": "source.ip",
                    "unique_field": "source.user",
                    "min_unique": 5,
                },
                threshold=20,
                window_seconds=600,
                mitre_tactics=["TA0006"],
                mitre_techniques=["T1110.004"],  # Credential Stuffing
            ),
            ThreatPattern(
                pattern_id="privilege_escalation",
                name="Privilege Escalation Attempt",
                description="User attempting to access resources above their privilege level",
                threat_type=ThreatType.PRIVILEGE_ESCALATION,
                severity=EventSeverity.CRITICAL,
                match_type=PatternMatchType.THRESHOLD,
                conditions={
                    "category": EventCategory.AUTHORIZATION.value,
                    "outcome": EventOutcome.FAILURE.value,
                    "action_pattern": "admin|root|sudo|elevate",
                },
                threshold=3,
                window_seconds=180,
                mitre_tactics=["TA0004"],  # Privilege Escalation
                mitre_techniques=["T1068"],  # Exploitation for Privilege Escalation
            ),
            ThreatPattern(
                pattern_id="port_scan",
                name="Port Scanning Activity",
                description="Connection attempts to multiple ports from same source",
                threat_type=ThreatType.RECONNAISSANCE,
                severity=EventSeverity.MEDIUM,
                match_type=PatternMatchType.THRESHOLD,
                conditions={
                    "category": EventCategory.NETWORK.value,
                    "group_by": "source.ip",
                    "unique_field": "destination.port",
                    "min_unique": 10,
                },
                threshold=50,
                window_seconds=60,
                mitre_tactics=["TA0043"],  # Reconnaissance
                mitre_techniques=["T1046"],  # Network Service Scanning
            ),
            ThreatPattern(
                pattern_id="sql_injection",
                name="SQL Injection Attempt",
                description="SQL injection patterns detected in web requests",
                threat_type=ThreatType.INJECTION_ATTACK,
                severity=EventSeverity.HIGH,
                match_type=PatternMatchType.REGEX,
                conditions={
                    "category": EventCategory.WEB.value,
                    "payload_pattern": r"(?i)(union\s+select|or\s+1\s*=\s*1|;\s*drop|--\s*$|/\*.*\*/)",
                },
                threshold=1,
                window_seconds=1,
                mitre_tactics=["TA0001"],  # Initial Access
                mitre_techniques=["T1190"],  # Exploit Public-Facing Application
            ),
            ThreatPattern(
                pattern_id="data_exfil_volume",
                name="Large Data Transfer (Potential Exfiltration)",
                description="Unusually large outbound data transfer",
                threat_type=ThreatType.DATA_EXFILTRATION,
                severity=EventSeverity.HIGH,
                match_type=PatternMatchType.THRESHOLD,
                conditions={
                    "category": EventCategory.NETWORK.value,
                    "direction": "outbound",
                    "bytes_threshold": 100_000_000,  # 100MB
                },
                threshold=1,
                window_seconds=3600,
                mitre_tactics=["TA0010"],  # Exfiltration
                mitre_techniques=["T1048"],  # Exfiltration Over Alternative Protocol
            ),
            ThreatPattern(
                pattern_id="lateral_movement",
                name="Lateral Movement Detection",
                description="Internal connections to multiple systems",
                threat_type=ThreatType.LATERAL_MOVEMENT,
                severity=EventSeverity.HIGH,
                match_type=PatternMatchType.THRESHOLD,
                conditions={
                    "category": EventCategory.NETWORK.value,
                    "network_type": "internal",
                    "group_by": "source.ip",
                    "unique_field": "destination.ip",
                    "min_unique": 5,
                },
                threshold=10,
                window_seconds=600,
                mitre_tactics=["TA0008"],  # Lateral Movement
                mitre_techniques=["T1021"],  # Remote Services
            ),
            ThreatPattern(
                pattern_id="c2_beacon",
                name="Command and Control Beacon",
                description="Regular periodic connections to same destination",
                threat_type=ThreatType.COMMAND_AND_CONTROL,
                severity=EventSeverity.CRITICAL,
                match_type=PatternMatchType.SEQUENCE,
                conditions={
                    "category": EventCategory.NETWORK.value,
                    "group_by": ["source.ip", "destination.ip"],
                    "periodic": True,
                    "period_variance_max": 0.1,  # 10% variance
                },
                threshold=10,
                window_seconds=3600,
                mitre_tactics=["TA0011"],  # Command and Control
                mitre_techniques=["T1071"],  # Application Layer Protocol
            ),
        ]

        for pattern in defaults:
            self.add_pattern(pattern)

        logger.info(f"Added {len(defaults)} default threat patterns")

    def analyze(self, event: SecurityEvent) -> List[ThreatMatch]:
        """Analyze an event for threats.

        Args:
            event: Event to analyze

        Returns:
            List of threat matches
        """
        self.events_analyzed.inc()

        # Add event to window
        with self._lock:
            self._event_window.append(event)

            # Clean old events from window
            cutoff = datetime.utcnow() - timedelta(hours=1)
            self._event_window = [
                e for e in self._event_window
                if e.timestamp >= cutoff
            ]

        matches = []

        # Check each pattern
        with self._lock:
            patterns = [p for p in self._patterns.values() if p.enabled]

        for pattern in patterns:
            match = self._check_pattern(pattern, event)
            if match:
                matches.append(match)

                # Update metrics
                self.threats_detected.labels(
                    threat_type=match.threat_type.value,
                    severity=match.severity.value,
                ).inc()

                self.patterns_matched.labels(
                    pattern_id=pattern.pattern_id,
                ).inc()

                # Trigger callbacks
                for callback in self._callbacks:
                    try:
                        callback(match)
                    except Exception as e:
                        logger.error(f"Threat callback error: {e}")

        return matches

    def _check_pattern(
        self,
        pattern: ThreatPattern,
        event: SecurityEvent,
    ) -> Optional[ThreatMatch]:
        """Check if event matches a pattern."""
        # First check if event matches basic conditions
        if not self._matches_conditions(pattern, event):
            return None

        if pattern.match_type == PatternMatchType.THRESHOLD:
            return self._check_threshold_pattern(pattern, event)

        elif pattern.match_type == PatternMatchType.REGEX:
            return self._check_regex_pattern(pattern, event)

        elif pattern.match_type == PatternMatchType.SEQUENCE:
            return self._check_sequence_pattern(pattern, event)

        return None

    def _matches_conditions(
        self,
        pattern: ThreatPattern,
        event: SecurityEvent,
    ) -> bool:
        """Check if event matches pattern conditions."""
        conditions = pattern.conditions

        # Category check
        if "category" in conditions:
            if event.category.value != conditions["category"]:
                return False

        # Action check
        if "action" in conditions:
            if event.action != conditions["action"]:
                return False

        # Action pattern check
        if "action_pattern" in conditions:
            if not re.search(conditions["action_pattern"], event.action, re.I):
                return False

        # Outcome check
        if "outcome" in conditions:
            if event.outcome.value != conditions["outcome"]:
                return False

        return True

    def _check_threshold_pattern(
        self,
        pattern: ThreatPattern,
        event: SecurityEvent,
    ) -> Optional[ThreatMatch]:
        """Check threshold-based pattern."""
        conditions = pattern.conditions
        group_by = conditions.get("group_by", "source.ip")

        # Get grouping value from event
        group_value = self._get_event_field(event, group_by)
        if not group_value:
            return None

        # Get events in window
        cutoff = datetime.utcnow() - timedelta(seconds=pattern.window_seconds)

        with self._lock:
            window_events = [
                e for e in self._event_window
                if e.timestamp >= cutoff
                and self._matches_conditions(pattern, e)
                and self._get_event_field(e, group_by) == group_value
            ]

        # Check unique field constraint
        if "unique_field" in conditions and "min_unique" in conditions:
            unique_field = conditions["unique_field"]
            min_unique = conditions["min_unique"]

            unique_values: Set[str] = set()
            for e in window_events:
                val = self._get_event_field(e, unique_field)
                if val:
                    unique_values.add(str(val))

            if len(unique_values) < min_unique:
                return None

        # Check threshold
        if len(window_events) >= pattern.threshold:
            confidence = min(1.0, len(window_events) / (pattern.threshold * 2))

            match = ThreatMatch(
                match_id=f"{pattern.pattern_id}_{event.event_id}",
                pattern=pattern,
                matched_events=window_events,
                confidence=confidence,
                context={
                    "group_value": group_value,
                    "event_count": len(window_events),
                    "threshold": pattern.threshold,
                },
            )

            with self._lock:
                self._matches.append(match)

            return match

        return None

    def _check_regex_pattern(
        self,
        pattern: ThreatPattern,
        event: SecurityEvent,
    ) -> Optional[ThreatMatch]:
        """Check regex-based pattern."""
        conditions = pattern.conditions
        payload_pattern = conditions.get("payload_pattern")

        if not payload_pattern:
            return None

        # Check message and metadata for pattern
        text_to_check = event.message

        if event.metadata:
            text_to_check += " " + str(event.metadata)

        if re.search(payload_pattern, text_to_check, re.I):
            match = ThreatMatch(
                match_id=f"{pattern.pattern_id}_{event.event_id}",
                pattern=pattern,
                matched_events=[event],
                confidence=0.9,
                context={
                    "matched_pattern": payload_pattern,
                },
            )

            with self._lock:
                self._matches.append(match)

            return match

        return None

    def _check_sequence_pattern(
        self,
        pattern: ThreatPattern,
        event: SecurityEvent,
    ) -> Optional[ThreatMatch]:
        """Check sequence-based pattern (e.g., periodic beacons)."""
        conditions = pattern.conditions
        group_by = conditions.get("group_by", ["source.ip", "destination.ip"])

        if isinstance(group_by, str):
            group_by = [group_by]

        # Build group key
        group_values = []
        for field in group_by:
            val = self._get_event_field(event, field)
            if val:
                group_values.append(str(val))

        if len(group_values) != len(group_by):
            return None

        group_key = ":".join(group_values)

        # Get events in window for this group
        cutoff = datetime.utcnow() - timedelta(seconds=pattern.window_seconds)

        with self._lock:
            window_events = [
                e for e in self._event_window
                if e.timestamp >= cutoff
                and self._matches_conditions(pattern, e)
            ]

        # Filter to matching group
        matching_events = []
        for e in window_events:
            event_values = []
            for field in group_by:
                val = self._get_event_field(e, field)
                if val:
                    event_values.append(str(val))
            if ":".join(event_values) == group_key:
                matching_events.append(e)

        if len(matching_events) < pattern.threshold:
            return None

        # Check for periodicity
        if conditions.get("periodic"):
            if self._check_periodicity(matching_events, conditions):
                match = ThreatMatch(
                    match_id=f"{pattern.pattern_id}_{event.event_id}",
                    pattern=pattern,
                    matched_events=matching_events,
                    confidence=0.85,
                    context={
                        "group_key": group_key,
                        "event_count": len(matching_events),
                        "periodic": True,
                    },
                )

                with self._lock:
                    self._matches.append(match)

                return match

        return None

    def _check_periodicity(
        self,
        events: List[SecurityEvent],
        conditions: Dict[str, Any],
    ) -> bool:
        """Check if events occur at regular intervals."""
        if len(events) < 3:
            return False

        # Sort by timestamp
        sorted_events = sorted(events, key=lambda e: e.timestamp)

        # Calculate intervals
        intervals = []
        for i in range(1, len(sorted_events)):
            delta = (
                sorted_events[i].timestamp - sorted_events[i - 1].timestamp
            ).total_seconds()
            intervals.append(delta)

        if not intervals:
            return False

        # Calculate mean and variance
        mean_interval = sum(intervals) / len(intervals)
        if mean_interval == 0:
            return False

        variance = sum((i - mean_interval) ** 2 for i in intervals) / len(intervals)
        std_dev = variance ** 0.5
        coefficient_of_variation = std_dev / mean_interval

        max_variance = conditions.get("period_variance_max", 0.1)

        return coefficient_of_variation <= max_variance

    def _get_event_field(
        self,
        event: SecurityEvent,
        field_path: str,
    ) -> Optional[Any]:
        """Get a field value from an event using dot notation."""
        parts = field_path.split(".")

        if parts[0] == "source" and event.source:
            return getattr(event.source, parts[1], None) if len(parts) > 1 else None

        elif parts[0] == "destination" and event.destination:
            return getattr(event.destination, parts[1], None) if len(parts) > 1 else None

        elif parts[0] == "metadata":
            return event.metadata.get(parts[1]) if len(parts) > 1 else None

        else:
            return getattr(event, parts[0], None)

    def calculate_anomaly_score(
        self,
        entity: str,
        entity_type: str,
        current_value: float,
    ) -> AnomalyScore:
        """Calculate anomaly score for an entity.

        Args:
            entity: Entity identifier (IP, user, etc.)
            entity_type: Type of entity
            current_value: Current metric value

        Returns:
            AnomalyScore
        """
        baseline = self._baselines[entity_type].get(entity, current_value)

        if baseline == 0:
            deviation = current_value
        else:
            deviation = abs(current_value - baseline) / baseline

        # Calculate score (0-100)
        score = min(100.0, deviation * 100)

        # Update baseline with exponential moving average
        alpha = 0.1
        new_baseline = alpha * current_value + (1 - alpha) * baseline
        self._baselines[entity_type][entity] = new_baseline

        factors = []
        if score >= 75:
            factors.append(f"Value {current_value:.2f} deviates {deviation:.1%} from baseline {baseline:.2f}")

        anomaly = AnomalyScore(
            entity=entity,
            entity_type=entity_type,
            score=score,
            baseline=baseline,
            current=current_value,
            deviation=deviation,
            factors=factors,
        )

        self.anomaly_scores.labels(entity_type=entity_type).observe(score)

        return anomaly

    def on_threat(self, callback: Callable[[ThreatMatch], None]):
        """Register a threat detection callback.

        Args:
            callback: Function to call when threat detected
        """
        self._callbacks.append(callback)

    def get_matches(
        self,
        threat_type: Optional[ThreatType] = None,
        severity: Optional[EventSeverity] = None,
        limit: int = 100,
        since: Optional[datetime] = None,
    ) -> List[ThreatMatch]:
        """Get threat matches.

        Args:
            threat_type: Filter by threat type
            severity: Filter by severity
            limit: Maximum matches
            since: Only matches after this time

        Returns:
            List of threat matches
        """
        with self._lock:
            matches = list(self._matches)

        if threat_type:
            matches = [m for m in matches if m.threat_type == threat_type]

        if severity:
            matches = [m for m in matches if m.severity == severity]

        if since:
            matches = [m for m in matches if m.match_time >= since]

        matches.sort(key=lambda m: m.match_time, reverse=True)

        return matches[:limit]

    def get_patterns(self) -> List[ThreatPattern]:
        """Get all patterns.

        Returns:
            List of patterns
        """
        with self._lock:
            return list(self._patterns.values())

    def get_summary(self) -> Dict[str, Any]:
        """Get threat detection summary.

        Returns:
            Summary dictionary
        """
        with self._lock:
            matches = list(self._matches)
            patterns = list(self._patterns.values())

        by_type: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}

        for match in matches:
            tt = match.threat_type.value
            sev = match.severity.value

            by_type[tt] = by_type.get(tt, 0) + 1
            by_severity[sev] = by_severity.get(sev, 0) + 1

        return {
            "total_matches": len(matches),
            "active_patterns": sum(1 for p in patterns if p.enabled),
            "total_patterns": len(patterns),
            "by_threat_type": by_type,
            "by_severity": by_severity,
            "recent_critical": sum(
                1 for m in matches
                if m.severity == EventSeverity.CRITICAL
                and m.match_time >= datetime.utcnow() - timedelta(hours=1)
            ),
        }

    def clear_matches(self):
        """Clear all matches."""
        with self._lock:
            self._matches.clear()
