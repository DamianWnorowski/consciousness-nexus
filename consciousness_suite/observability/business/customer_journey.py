"""Customer Journey Tracking

Track and analyze customer experience across the product journey.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from enum import Enum
import threading
import logging
import hashlib

from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)


class JourneyStage(str, Enum):
    """Stages of customer journey."""
    AWARENESS = "awareness"
    ACQUISITION = "acquisition"
    ACTIVATION = "activation"
    RETENTION = "retention"
    REVENUE = "revenue"
    REFERRAL = "referral"


class EventType(str, Enum):
    """Types of journey events."""
    PAGE_VIEW = "page_view"
    CLICK = "click"
    FORM_SUBMIT = "form_submit"
    SIGN_UP = "sign_up"
    LOGIN = "login"
    FEATURE_USE = "feature_use"
    PURCHASE = "purchase"
    ERROR = "error"
    SUPPORT_CONTACT = "support_contact"
    CHURN = "churn"


@dataclass
class JourneyEvent:
    """An event in the customer journey."""
    user_id: str
    event_type: EventType
    stage: JourneyStage
    timestamp: datetime = field(default_factory=datetime.now)
    properties: Dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class JourneyMetrics:
    """Metrics for a journey stage."""
    stage: JourneyStage
    total_users: int
    conversion_rate: float
    avg_time_in_stage_hours: float
    drop_off_rate: float
    error_rate: float
    satisfaction_score: float  # 0-100
    revenue_generated: float = 0.0


@dataclass
class ExperienceScore:
    """Customer experience score."""
    user_id: str
    overall_score: float  # 0-100
    reliability_score: float
    performance_score: float
    usability_score: float
    support_score: float
    calculated_at: datetime = field(default_factory=datetime.now)
    factors: Dict[str, float] = field(default_factory=dict)

    @property
    def grade(self) -> str:
        """Get letter grade."""
        if self.overall_score >= 90:
            return "A"
        elif self.overall_score >= 80:
            return "B"
        elif self.overall_score >= 70:
            return "C"
        elif self.overall_score >= 60:
            return "D"
        else:
            return "F"


class CustomerJourneyTracker:
    """Tracks customer journey and experience.

    Usage:
        tracker = CustomerJourneyTracker()

        # Track events
        tracker.track_event(JourneyEvent(
            user_id="user-123",
            event_type=EventType.SIGN_UP,
            stage=JourneyStage.ACQUISITION,
        ))

        # Get stage metrics
        metrics = tracker.get_stage_metrics(JourneyStage.ACTIVATION)

        # Calculate experience score
        score = tracker.calculate_experience_score("user-123")
    """

    def __init__(self, namespace: str = "consciousness"):
        self.namespace = namespace
        self._events: Dict[str, List[JourneyEvent]] = {}
        self._user_stages: Dict[str, JourneyStage] = {}
        self._lock = threading.Lock()
        self._max_events_per_user = 1000

        # Stage progression (for funnel analysis)
        self._stage_order = [
            JourneyStage.AWARENESS,
            JourneyStage.ACQUISITION,
            JourneyStage.ACTIVATION,
            JourneyStage.RETENTION,
            JourneyStage.REVENUE,
            JourneyStage.REFERRAL,
        ]

        # Prometheus metrics
        self.journey_events = Counter(
            f"{namespace}_journey_events_total",
            "Total journey events",
            ["stage", "event_type", "success"],
        )

        self.stage_users = Gauge(
            f"{namespace}_journey_stage_users",
            "Users in each stage",
            ["stage"],
        )

        self.conversion_rate = Gauge(
            f"{namespace}_journey_conversion_rate",
            "Conversion rate between stages",
            ["from_stage", "to_stage"],
        )

        self.experience_score = Histogram(
            f"{namespace}_customer_experience_score",
            "Customer experience scores",
            buckets=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        )

        self.stage_duration = Histogram(
            f"{namespace}_journey_stage_duration_hours",
            "Time spent in each stage",
            ["stage"],
            buckets=[0.1, 0.5, 1, 2, 6, 12, 24, 48, 168],  # Up to 1 week
        )

    def track_event(self, event: JourneyEvent):
        """Track a journey event.

        Args:
            event: Journey event
        """
        user_id = event.user_id

        with self._lock:
            if user_id not in self._events:
                self._events[user_id] = []

            self._events[user_id].append(event)

            # Trim old events
            if len(self._events[user_id]) > self._max_events_per_user:
                self._events[user_id] = self._events[user_id][-self._max_events_per_user // 2:]

            # Update user stage
            self._user_stages[user_id] = event.stage

        # Update metrics
        self.journey_events.labels(
            stage=event.stage.value,
            event_type=event.event_type.value,
            success="true" if event.success else "false",
        ).inc()

        self._update_stage_counts()

    def track_stage_transition(
        self,
        user_id: str,
        from_stage: JourneyStage,
        to_stage: JourneyStage,
        properties: Optional[Dict[str, Any]] = None,
    ):
        """Track a stage transition.

        Args:
            user_id: User identifier
            from_stage: Previous stage
            to_stage: New stage
            properties: Transition properties
        """
        # Record time in previous stage
        with self._lock:
            events = self._events.get(user_id, [])
            from_stage_events = [e for e in events if e.stage == from_stage]

        if from_stage_events:
            first_in_stage = min(e.timestamp for e in from_stage_events)
            duration_hours = (datetime.now() - first_in_stage).total_seconds() / 3600
            self.stage_duration.labels(stage=from_stage.value).observe(duration_hours)

        # Track transition event
        self.track_event(JourneyEvent(
            user_id=user_id,
            event_type=EventType.FEATURE_USE,
            stage=to_stage,
            properties={
                "transition": True,
                "from_stage": from_stage.value,
                **(properties or {}),
            },
        ))

        logger.debug(f"User {user_id} transitioned {from_stage.value} -> {to_stage.value}")

    def _update_stage_counts(self):
        """Update stage user counts."""
        with self._lock:
            counts: Dict[JourneyStage, int] = {}
            for stage in self._user_stages.values():
                counts[stage] = counts.get(stage, 0) + 1

        for stage in JourneyStage:
            self.stage_users.labels(stage=stage.value).set(counts.get(stage, 0))

    def get_stage_metrics(self, stage: JourneyStage) -> JourneyMetrics:
        """Get metrics for a journey stage.

        Args:
            stage: Journey stage

        Returns:
            JourneyMetrics
        """
        with self._lock:
            # Count users in stage
            users_in_stage = sum(
                1 for s in self._user_stages.values()
                if s == stage
            )

            # Get all events for this stage
            stage_events = []
            for events in self._events.values():
                stage_events.extend([e for e in events if e.stage == stage])

            # Calculate error rate
            if stage_events:
                error_count = sum(1 for e in stage_events if not e.success)
                error_rate = error_count / len(stage_events)
            else:
                error_rate = 0

            # Calculate average time in stage
            stage_durations = []
            for user_id, events in self._events.items():
                user_stage_events = [e for e in events if e.stage == stage]
                if user_stage_events:
                    first = min(e.timestamp for e in user_stage_events)
                    last = max(e.timestamp for e in user_stage_events)
                    duration_hours = (last - first).total_seconds() / 3600
                    stage_durations.append(duration_hours)

        avg_time = sum(stage_durations) / len(stage_durations) if stage_durations else 0

        # Calculate conversion rate to next stage
        stage_idx = self._stage_order.index(stage)
        if stage_idx < len(self._stage_order) - 1:
            next_stage = self._stage_order[stage_idx + 1]
            with self._lock:
                users_next = sum(
                    1 for s in self._user_stages.values()
                    if s == next_stage
                )
            conversion = users_next / users_in_stage if users_in_stage > 0 else 0
        else:
            conversion = 0

        # Drop-off rate (users who haven't progressed in 7 days)
        drop_off_threshold = datetime.now() - timedelta(days=7)
        with self._lock:
            inactive_users = sum(
                1 for user_id, s in self._user_stages.items()
                if s == stage and (
                    not self._events.get(user_id) or
                    max(e.timestamp for e in self._events[user_id]) < drop_off_threshold
                )
            )
        drop_off_rate = inactive_users / users_in_stage if users_in_stage > 0 else 0

        # Simple satisfaction proxy (success rate * 100)
        satisfaction = (1 - error_rate) * 100

        return JourneyMetrics(
            stage=stage,
            total_users=users_in_stage,
            conversion_rate=conversion,
            avg_time_in_stage_hours=avg_time,
            drop_off_rate=drop_off_rate,
            error_rate=error_rate,
            satisfaction_score=satisfaction,
        )

    def calculate_experience_score(self, user_id: str) -> ExperienceScore:
        """Calculate experience score for a user.

        Args:
            user_id: User identifier

        Returns:
            ExperienceScore
        """
        with self._lock:
            events = self._events.get(user_id, [])

        if not events:
            return ExperienceScore(
                user_id=user_id,
                overall_score=50,
                reliability_score=50,
                performance_score=50,
                usability_score=50,
                support_score=50,
            )

        # Calculate reliability score (based on error rate)
        error_count = sum(1 for e in events if not e.success)
        reliability = (1 - error_count / len(events)) * 100

        # Calculate performance score (based on durations)
        durations = [e.duration_ms for e in events if e.duration_ms is not None]
        if durations:
            avg_duration = sum(durations) / len(durations)
            # Score: 100 for < 100ms, 0 for > 5000ms
            performance = max(0, min(100, 100 - (avg_duration - 100) / 49))
        else:
            performance = 75  # Default

        # Calculate usability score (based on journey progression)
        current_stage = self._user_stages.get(user_id, JourneyStage.AWARENESS)
        stage_idx = self._stage_order.index(current_stage)
        usability = (stage_idx + 1) / len(self._stage_order) * 100

        # Calculate support score (based on support contacts)
        support_contacts = sum(
            1 for e in events
            if e.event_type == EventType.SUPPORT_CONTACT
        )
        # Fewer contacts = better experience
        support = max(0, 100 - support_contacts * 10)

        # Overall weighted score
        overall = (
            reliability * 0.35 +
            performance * 0.25 +
            usability * 0.25 +
            support * 0.15
        )

        score = ExperienceScore(
            user_id=user_id,
            overall_score=overall,
            reliability_score=reliability,
            performance_score=performance,
            usability_score=usability,
            support_score=support,
            factors={
                "total_events": len(events),
                "error_count": error_count,
                "support_contacts": support_contacts,
                "current_stage": current_stage.value,
            },
        )

        # Update metric
        self.experience_score.observe(overall)

        return score

    def get_funnel_metrics(self) -> List[JourneyMetrics]:
        """Get metrics for entire funnel.

        Returns:
            List of JourneyMetrics for each stage
        """
        return [
            self.get_stage_metrics(stage)
            for stage in self._stage_order
        ]

    def get_user_journey(
        self,
        user_id: str,
        limit: int = 100,
    ) -> List[JourneyEvent]:
        """Get journey events for a user.

        Args:
            user_id: User identifier
            limit: Maximum events to return

        Returns:
            List of journey events
        """
        with self._lock:
            events = self._events.get(user_id, [])

        return list(reversed(events[-limit:]))

    def get_cohort_analysis(
        self,
        cohort_start: datetime,
        cohort_end: datetime,
    ) -> Dict[str, Any]:
        """Analyze a cohort of users.

        Args:
            cohort_start: Cohort start time
            cohort_end: Cohort end time

        Returns:
            Cohort analysis
        """
        with self._lock:
            # Find users who started in the cohort period
            cohort_users = []
            for user_id, events in self._events.items():
                first_event = min(e.timestamp for e in events) if events else None
                if first_event and cohort_start <= first_event <= cohort_end:
                    cohort_users.append(user_id)

        # Analyze cohort
        stage_counts: Dict[JourneyStage, int] = {}
        for user_id in cohort_users:
            stage = self._user_stages.get(user_id, JourneyStage.AWARENESS)
            stage_counts[stage] = stage_counts.get(stage, 0) + 1

        return {
            "cohort_size": len(cohort_users),
            "cohort_start": cohort_start.isoformat(),
            "cohort_end": cohort_end.isoformat(),
            "stage_distribution": {
                s.value: stage_counts.get(s, 0) / len(cohort_users) * 100
                if cohort_users else 0
                for s in self._stage_order
            },
            "retention_rate": (
                sum(1 for s in stage_counts if s != JourneyStage.AWARENESS) /
                len(cohort_users) if cohort_users else 0
            ),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get journey tracking summary.

        Returns:
            Summary dictionary
        """
        with self._lock:
            total_users = len(self._user_stages)
            total_events = sum(len(events) for events in self._events.values())

        funnel = self.get_funnel_metrics()

        return {
            "total_users": total_users,
            "total_events": total_events,
            "funnel": {
                stage.stage.value: {
                    "users": stage.total_users,
                    "conversion_rate": stage.conversion_rate,
                    "drop_off_rate": stage.drop_off_rate,
                }
                for stage in funnel
            },
            "overall_conversion": (
                funnel[-1].total_users / funnel[0].total_users * 100
                if funnel and funnel[0].total_users > 0 else 0
            ),
        }
