"""Escalation Policies

Incident escalation policy management.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import threading
import logging

from prometheus_client import Counter, Gauge

logger = logging.getLogger(__name__)


class TargetType(str, Enum):
    """Escalation target types."""
    USER = "user"
    TEAM = "team"
    SCHEDULE = "schedule"
    WEBHOOK = "webhook"
    EMAIL = "email"
    SLACK = "slack"
    PAGERDUTY = "pagerduty"


class EscalationStatus(str, Enum):
    """Escalation status."""
    PENDING = "pending"
    NOTIFIED = "notified"
    ACKNOWLEDGED = "acknowledged"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
    TIMEOUT = "timeout"


@dataclass
class EscalationTarget:
    """Target for escalation notification."""
    target_id: str
    target_type: TargetType
    name: str
    contact: str  # Email, phone, webhook URL, etc.
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "target_type": self.target_type.value,
            "name": self.name,
            "contact": self.contact,
            "priority": self.priority,
        }


@dataclass
class EscalationLevel:
    """A level in an escalation policy."""
    level: int
    targets: List[EscalationTarget]
    timeout_minutes: int = 15
    notify_all: bool = False  # Notify all targets at once
    repeat_count: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "targets": [t.to_dict() for t in self.targets],
            "timeout_minutes": self.timeout_minutes,
            "notify_all": self.notify_all,
            "repeat_count": self.repeat_count,
        }


@dataclass
class EscalationPolicy:
    """An escalation policy."""
    policy_id: str
    name: str
    description: str = ""
    levels: List[EscalationLevel] = field(default_factory=list)
    enabled: bool = True
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "name": self.name,
            "description": self.description,
            "levels": [l.to_dict() for l in self.levels],
            "enabled": self.enabled,
            "tags": self.tags,
        }


@dataclass
class EscalationState:
    """State of an active escalation."""
    escalation_id: str
    incident_id: str
    policy: EscalationPolicy
    current_level: int = 0
    current_repeat: int = 0
    status: EscalationStatus = EscalationStatus.PENDING
    started_at: datetime = field(default_factory=datetime.now)
    last_notification_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    notifications_sent: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class EscalationManager:
    """Manages incident escalation.

    Usage:
        manager = EscalationManager()

        # Create policy
        policy = EscalationPolicy(
            policy_id="critical-alerts",
            name="Critical Alert Escalation",
            levels=[
                EscalationLevel(
                    level=1,
                    targets=[EscalationTarget(...)],
                    timeout_minutes=5,
                ),
                EscalationLevel(
                    level=2,
                    targets=[EscalationTarget(...)],
                    timeout_minutes=10,
                ),
            ],
        )
        manager.add_policy(policy)

        # Start escalation
        state = await manager.start_escalation("incident-123", "critical-alerts")

        # Acknowledge
        await manager.acknowledge("incident-123", "user@example.com")
    """

    def __init__(self, namespace: str = "consciousness"):
        self.namespace = namespace

        self._policies: Dict[str, EscalationPolicy] = {}
        self._escalations: Dict[str, EscalationState] = {}
        self._lock = threading.Lock()

        # Notification handlers
        self._notifiers: Dict[TargetType, Callable] = {}

        # Callbacks
        self._on_escalate: List[Callable[[EscalationState], None]] = []
        self._on_acknowledge: List[Callable[[EscalationState], None]] = []
        self._on_timeout: List[Callable[[EscalationState], None]] = []

        # Metrics
        self.escalations_started = Counter(
            f"{namespace}_escalations_started_total",
            "Total escalations started",
            ["policy"],
        )

        self.escalations_acknowledged = Counter(
            f"{namespace}_escalations_acknowledged_total",
            "Total escalations acknowledged",
            ["policy", "level"],
        )

        self.escalations_resolved = Counter(
            f"{namespace}_escalations_resolved_total",
            "Total escalations resolved",
            ["policy"],
        )

        self.escalation_level_reached = Counter(
            f"{namespace}_escalation_level_reached_total",
            "Escalation levels reached",
            ["policy", "level"],
        )

        self.active_escalations = Gauge(
            f"{namespace}_active_escalations",
            "Number of active escalations",
        )

        self.time_to_acknowledge = Gauge(
            f"{namespace}_escalation_time_to_acknowledge_seconds",
            "Time to acknowledge escalation",
            ["policy"],
        )

    def add_policy(self, policy: EscalationPolicy):
        """Add an escalation policy.

        Args:
            policy: Escalation policy
        """
        with self._lock:
            self._policies[policy.policy_id] = policy

        logger.info(f"Added escalation policy: {policy.policy_id}")

    def remove_policy(self, policy_id: str):
        """Remove an escalation policy.

        Args:
            policy_id: Policy ID
        """
        with self._lock:
            self._policies.pop(policy_id, None)

    def get_policy(self, policy_id: str) -> Optional[EscalationPolicy]:
        """Get an escalation policy.

        Args:
            policy_id: Policy ID

        Returns:
            EscalationPolicy or None
        """
        with self._lock:
            return self._policies.get(policy_id)

    def register_notifier(
        self,
        target_type: TargetType,
        handler: Callable[[EscalationTarget, str, Dict[str, Any]], bool],
    ):
        """Register a notification handler.

        Args:
            target_type: Target type to handle
            handler: Async function(target, incident_id, context) -> success
        """
        self._notifiers[target_type] = handler

    async def start_escalation(
        self,
        incident_id: str,
        policy_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[EscalationState]:
        """Start an escalation for an incident.

        Args:
            incident_id: Incident ID
            policy_id: Escalation policy ID
            context: Additional context

        Returns:
            EscalationState or None
        """
        policy = self.get_policy(policy_id)
        if not policy or not policy.enabled:
            logger.warning(f"Policy not found or disabled: {policy_id}")
            return None

        import uuid

        escalation_id = str(uuid.uuid4())

        state = EscalationState(
            escalation_id=escalation_id,
            incident_id=incident_id,
            policy=policy,
            metadata=context or {},
        )

        with self._lock:
            self._escalations[incident_id] = state

        # Start at first level
        await self._notify_level(state)

        self.escalations_started.labels(policy=policy_id).inc()
        self._update_active_count()

        return state

    async def _notify_level(self, state: EscalationState):
        """Notify current escalation level.

        Args:
            state: Escalation state
        """
        if state.current_level >= len(state.policy.levels):
            state.status = EscalationStatus.TIMEOUT
            self._trigger_timeout(state)
            return

        level = state.policy.levels[state.current_level]
        state.last_notification_at = datetime.now()
        state.status = EscalationStatus.NOTIFIED

        self.escalation_level_reached.labels(
            policy=state.policy.policy_id,
            level=str(state.current_level),
        ).inc()

        # Notify targets
        if level.notify_all:
            for target in level.targets:
                await self._notify_target(target, state)
        else:
            # Notify one by one based on priority
            sorted_targets = sorted(level.targets, key=lambda t: t.priority)
            for target in sorted_targets:
                success = await self._notify_target(target, state)
                if success:
                    break

        for callback in self._on_escalate:
            try:
                callback(state)
            except Exception as e:
                logger.error(f"Escalate callback error: {e}")

    async def _notify_target(
        self,
        target: EscalationTarget,
        state: EscalationState,
    ) -> bool:
        """Notify a specific target.

        Args:
            target: Target to notify
            state: Escalation state

        Returns:
            True if notification was successful
        """
        handler = self._notifiers.get(target.target_type)
        if not handler:
            logger.warning(f"No handler for target type: {target.target_type}")
            return False

        try:
            success = await handler(target, state.incident_id, state.metadata)

            state.notifications_sent.append({
                "target_id": target.target_id,
                "target_type": target.target_type.value,
                "timestamp": datetime.now().isoformat(),
                "success": success,
            })

            return success

        except Exception as e:
            logger.error(f"Notification error: {e}")
            return False

    async def escalate(self, incident_id: str) -> bool:
        """Escalate to next level.

        Args:
            incident_id: Incident ID

        Returns:
            True if escalated
        """
        with self._lock:
            state = self._escalations.get(incident_id)

        if not state:
            return False

        if state.status in [EscalationStatus.ACKNOWLEDGED, EscalationStatus.RESOLVED]:
            return False

        # Check if we should repeat current level
        level = state.policy.levels[state.current_level]
        if state.current_repeat < level.repeat_count - 1:
            state.current_repeat += 1
        else:
            state.current_level += 1
            state.current_repeat = 0
            state.status = EscalationStatus.ESCALATED

        await self._notify_level(state)

        return True

    async def acknowledge(
        self,
        incident_id: str,
        acknowledged_by: str,
    ) -> bool:
        """Acknowledge an escalation.

        Args:
            incident_id: Incident ID
            acknowledged_by: User who acknowledged

        Returns:
            True if acknowledged
        """
        with self._lock:
            state = self._escalations.get(incident_id)

        if not state:
            return False

        if state.status in [EscalationStatus.ACKNOWLEDGED, EscalationStatus.RESOLVED]:
            return False

        state.status = EscalationStatus.ACKNOWLEDGED
        state.acknowledged_at = datetime.now()
        state.acknowledged_by = acknowledged_by

        # Calculate time to acknowledge
        tta = (state.acknowledged_at - state.started_at).total_seconds()
        self.time_to_acknowledge.labels(policy=state.policy.policy_id).set(tta)

        self.escalations_acknowledged.labels(
            policy=state.policy.policy_id,
            level=str(state.current_level),
        ).inc()

        for callback in self._on_acknowledge:
            try:
                callback(state)
            except Exception as e:
                logger.error(f"Acknowledge callback error: {e}")

        return True

    async def resolve(self, incident_id: str) -> bool:
        """Resolve an escalation.

        Args:
            incident_id: Incident ID

        Returns:
            True if resolved
        """
        with self._lock:
            state = self._escalations.get(incident_id)

        if not state:
            return False

        state.status = EscalationStatus.RESOLVED
        state.resolved_at = datetime.now()

        self.escalations_resolved.labels(policy=state.policy.policy_id).inc()
        self._update_active_count()

        return True

    def _trigger_timeout(self, state: EscalationState):
        """Trigger timeout callbacks.

        Args:
            state: Escalation state
        """
        for callback in self._on_timeout:
            try:
                callback(state)
            except Exception as e:
                logger.error(f"Timeout callback error: {e}")

    def _update_active_count(self):
        """Update active escalations gauge."""
        with self._lock:
            active = sum(
                1 for s in self._escalations.values()
                if s.status not in [EscalationStatus.RESOLVED, EscalationStatus.TIMEOUT]
            )
        self.active_escalations.set(active)

    async def check_timeouts(self):
        """Check for escalation timeouts."""
        now = datetime.now()

        with self._lock:
            states = list(self._escalations.values())

        for state in states:
            if state.status in [EscalationStatus.ACKNOWLEDGED, EscalationStatus.RESOLVED, EscalationStatus.TIMEOUT]:
                continue

            if state.current_level >= len(state.policy.levels):
                continue

            level = state.policy.levels[state.current_level]
            timeout = timedelta(minutes=level.timeout_minutes)

            if state.last_notification_at and now - state.last_notification_at > timeout:
                await self.escalate(state.incident_id)

    def on_escalate(self, callback: Callable[[EscalationState], None]):
        """Register escalation callback.

        Args:
            callback: Function to call on escalation
        """
        self._on_escalate.append(callback)

    def on_acknowledge(self, callback: Callable[[EscalationState], None]):
        """Register acknowledgment callback.

        Args:
            callback: Function to call on acknowledgment
        """
        self._on_acknowledge.append(callback)

    def on_timeout(self, callback: Callable[[EscalationState], None]):
        """Register timeout callback.

        Args:
            callback: Function to call on timeout
        """
        self._on_timeout.append(callback)

    def get_escalation(self, incident_id: str) -> Optional[EscalationState]:
        """Get escalation state for an incident.

        Args:
            incident_id: Incident ID

        Returns:
            EscalationState or None
        """
        with self._lock:
            return self._escalations.get(incident_id)

    def get_active_escalations(self) -> List[EscalationState]:
        """Get all active escalations.

        Returns:
            List of active escalation states
        """
        with self._lock:
            return [
                s for s in self._escalations.values()
                if s.status not in [EscalationStatus.RESOLVED, EscalationStatus.TIMEOUT]
            ]

    def get_summary(self) -> Dict[str, Any]:
        """Get escalation manager summary.

        Returns:
            Summary dictionary
        """
        with self._lock:
            states = list(self._escalations.values())
            policies = list(self._policies.values())

        # Count by status
        by_status: Dict[str, int] = {}
        for s in states:
            status = s.status.value
            by_status[status] = by_status.get(status, 0) + 1

        return {
            "total_policies": len(policies),
            "total_escalations": len(states),
            "by_status": by_status,
            "active_count": sum(1 for s in states if s.status not in [EscalationStatus.RESOLVED, EscalationStatus.TIMEOUT]),
        }
