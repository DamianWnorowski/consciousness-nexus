"""On-Call Schedule Management

On-call schedule tracking and management.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta, time
from enum import Enum
import threading
import logging

from prometheus_client import Gauge, Counter

logger = logging.getLogger(__name__)


class RotationType(str, Enum):
    """Schedule rotation types."""
    DAILY = "daily"
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"


class ShiftType(str, Enum):
    """Shift types."""
    PRIMARY = "primary"
    SECONDARY = "secondary"
    BACKUP = "backup"


@dataclass
class OnCallUser:
    """An on-call user."""
    user_id: str
    name: str
    email: str
    phone: Optional[str] = None
    slack_id: Optional[str] = None
    timezone: str = "UTC"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OnCallShift:
    """An on-call shift."""
    shift_id: str
    user: OnCallUser
    shift_type: ShiftType
    start_time: datetime
    end_time: datetime
    schedule_id: str = ""
    overridden: bool = False
    override_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_hours(self) -> float:
        """Get shift duration in hours."""
        return (self.end_time - self.start_time).total_seconds() / 3600

    @property
    def is_active(self) -> bool:
        """Check if shift is currently active."""
        now = datetime.now()
        return self.start_time <= now <= self.end_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shift_id": self.shift_id,
            "user_id": self.user.user_id,
            "user_name": self.user.name,
            "shift_type": self.shift_type.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_hours": self.duration_hours,
            "is_active": self.is_active,
            "overridden": self.overridden,
        }


@dataclass
class ScheduleRotation:
    """Rotation configuration for a schedule."""
    rotation_type: RotationType
    users: List[OnCallUser]
    handoff_time: time = field(default_factory=lambda: time(9, 0))  # 9 AM
    handoff_day: int = 0  # Day of week (0=Monday) or day of month
    start_date: Optional[datetime] = None
    timezone: str = "UTC"


@dataclass
class OnCallSchedule:
    """An on-call schedule."""
    schedule_id: str
    name: str
    description: str = ""
    team_id: Optional[str] = None
    rotation: Optional[ScheduleRotation] = None
    shifts: List[OnCallShift] = field(default_factory=list)
    enabled: bool = True
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_current_oncall(
        self,
        shift_type: ShiftType = ShiftType.PRIMARY,
    ) -> Optional[OnCallShift]:
        """Get current on-call shift.

        Args:
            shift_type: Shift type to get

        Returns:
            OnCallShift or None
        """
        now = datetime.now()
        for shift in self.shifts:
            if shift.shift_type == shift_type and shift.start_time <= now <= shift.end_time:
                return shift
        return None

    def to_dict(self) -> Dict[str, Any]:
        current = self.get_current_oncall()
        return {
            "schedule_id": self.schedule_id,
            "name": self.name,
            "description": self.description,
            "team_id": self.team_id,
            "enabled": self.enabled,
            "current_oncall": current.user.name if current else None,
            "shift_count": len(self.shifts),
        }


class OnCallManager:
    """Manages on-call schedules.

    Usage:
        manager = OnCallManager()

        # Create schedule
        schedule = OnCallSchedule(
            schedule_id="platform-team",
            name="Platform Team On-Call",
            rotation=ScheduleRotation(
                rotation_type=RotationType.WEEKLY,
                users=[user1, user2, user3],
                handoff_time=time(9, 0),
                handoff_day=0,  # Monday
            ),
        )
        manager.add_schedule(schedule)

        # Get current on-call
        current = manager.get_current_oncall("platform-team")

        # Create override
        manager.create_override(
            "platform-team",
            user,
            start_time,
            end_time,
            reason="PTO coverage",
        )
    """

    def __init__(self, namespace: str = "consciousness"):
        self.namespace = namespace

        self._schedules: Dict[str, OnCallSchedule] = {}
        self._overrides: List[OnCallShift] = []
        self._lock = threading.Lock()

        # Callbacks
        self._on_handoff: List[Callable[[OnCallShift, OnCallShift], None]] = []

        # Metrics
        self.oncall_shifts_total = Counter(
            f"{namespace}_oncall_shifts_total",
            "Total on-call shifts",
            ["schedule", "shift_type"],
        )

        self.oncall_overrides_total = Counter(
            f"{namespace}_oncall_overrides_total",
            "Total on-call overrides",
            ["schedule"],
        )

        self.oncall_current_user = Gauge(
            f"{namespace}_oncall_current_user",
            "Current on-call user (by hash)",
            ["schedule", "shift_type"],
        )

        self.oncall_shift_hours_remaining = Gauge(
            f"{namespace}_oncall_shift_hours_remaining",
            "Hours remaining in current shift",
            ["schedule", "user"],
        )

    def add_schedule(self, schedule: OnCallSchedule):
        """Add an on-call schedule.

        Args:
            schedule: On-call schedule
        """
        with self._lock:
            self._schedules[schedule.schedule_id] = schedule

        # Generate initial shifts if rotation configured
        if schedule.rotation:
            self._generate_rotation_shifts(schedule)

        logger.info(f"Added on-call schedule: {schedule.schedule_id}")

    def remove_schedule(self, schedule_id: str):
        """Remove an on-call schedule.

        Args:
            schedule_id: Schedule ID
        """
        with self._lock:
            self._schedules.pop(schedule_id, None)

    def get_schedule(self, schedule_id: str) -> Optional[OnCallSchedule]:
        """Get an on-call schedule.

        Args:
            schedule_id: Schedule ID

        Returns:
            OnCallSchedule or None
        """
        with self._lock:
            return self._schedules.get(schedule_id)

    def _generate_rotation_shifts(
        self,
        schedule: OnCallSchedule,
        days_ahead: int = 30,
    ):
        """Generate rotation shifts.

        Args:
            schedule: Schedule with rotation
            days_ahead: Days to generate ahead
        """
        if not schedule.rotation:
            return

        rotation = schedule.rotation
        now = datetime.now()
        start_date = rotation.start_date or now

        # Calculate rotation period
        if rotation.rotation_type == RotationType.DAILY:
            period = timedelta(days=1)
        elif rotation.rotation_type == RotationType.WEEKLY:
            period = timedelta(weeks=1)
        elif rotation.rotation_type == RotationType.BIWEEKLY:
            period = timedelta(weeks=2)
        elif rotation.rotation_type == RotationType.MONTHLY:
            period = timedelta(days=30)
        else:
            return

        # Generate shifts
        current = start_date
        user_index = 0
        num_users = len(rotation.users)

        while current <= now + timedelta(days=days_ahead):
            user = rotation.users[user_index % num_users]

            shift_start = datetime.combine(
                current.date(),
                rotation.handoff_time,
            )
            shift_end = shift_start + period

            shift = OnCallShift(
                shift_id=f"{schedule.schedule_id}-{current.strftime('%Y%m%d')}",
                user=user,
                shift_type=ShiftType.PRIMARY,
                start_time=shift_start,
                end_time=shift_end,
                schedule_id=schedule.schedule_id,
            )

            schedule.shifts.append(shift)

            self.oncall_shifts_total.labels(
                schedule=schedule.schedule_id,
                shift_type=ShiftType.PRIMARY.value,
            ).inc()

            current += period
            user_index += 1

    def get_current_oncall(
        self,
        schedule_id: str,
        shift_type: ShiftType = ShiftType.PRIMARY,
    ) -> Optional[OnCallUser]:
        """Get current on-call user.

        Args:
            schedule_id: Schedule ID
            shift_type: Shift type

        Returns:
            OnCallUser or None
        """
        schedule = self.get_schedule(schedule_id)
        if not schedule:
            return None

        # Check for active override first
        now = datetime.now()
        for override in self._overrides:
            if (override.schedule_id == schedule_id and
                override.shift_type == shift_type and
                override.start_time <= now <= override.end_time):
                return override.user

        # Check regular shifts
        shift = schedule.get_current_oncall(shift_type)
        return shift.user if shift else None

    def get_current_shift(
        self,
        schedule_id: str,
        shift_type: ShiftType = ShiftType.PRIMARY,
    ) -> Optional[OnCallShift]:
        """Get current on-call shift.

        Args:
            schedule_id: Schedule ID
            shift_type: Shift type

        Returns:
            OnCallShift or None
        """
        schedule = self.get_schedule(schedule_id)
        if not schedule:
            return None

        # Check for active override first
        now = datetime.now()
        for override in self._overrides:
            if (override.schedule_id == schedule_id and
                override.shift_type == shift_type and
                override.start_time <= now <= override.end_time):
                return override

        return schedule.get_current_oncall(shift_type)

    def create_override(
        self,
        schedule_id: str,
        user: OnCallUser,
        start_time: datetime,
        end_time: datetime,
        shift_type: ShiftType = ShiftType.PRIMARY,
        reason: Optional[str] = None,
    ) -> OnCallShift:
        """Create an on-call override.

        Args:
            schedule_id: Schedule ID
            user: User taking on-call
            start_time: Override start
            end_time: Override end
            shift_type: Shift type
            reason: Reason for override

        Returns:
            Override shift
        """
        import uuid

        override = OnCallShift(
            shift_id=f"override-{uuid.uuid4().hex[:8]}",
            user=user,
            shift_type=shift_type,
            start_time=start_time,
            end_time=end_time,
            schedule_id=schedule_id,
            overridden=True,
            override_reason=reason,
        )

        with self._lock:
            self._overrides.append(override)

        self.oncall_overrides_total.labels(schedule=schedule_id).inc()

        logger.info(f"Created on-call override for {schedule_id}: {user.name}")

        return override

    def remove_override(self, shift_id: str) -> bool:
        """Remove an override.

        Args:
            shift_id: Override shift ID

        Returns:
            True if removed
        """
        with self._lock:
            for i, override in enumerate(self._overrides):
                if override.shift_id == shift_id:
                    self._overrides.pop(i)
                    return True
        return False

    def get_upcoming_shifts(
        self,
        schedule_id: str,
        days: int = 7,
    ) -> List[OnCallShift]:
        """Get upcoming shifts.

        Args:
            schedule_id: Schedule ID
            days: Days ahead

        Returns:
            List of upcoming shifts
        """
        schedule = self.get_schedule(schedule_id)
        if not schedule:
            return []

        now = datetime.now()
        end = now + timedelta(days=days)

        return [
            shift for shift in schedule.shifts
            if now <= shift.start_time <= end
        ]

    def get_user_shifts(
        self,
        user_id: str,
        days_back: int = 30,
        days_ahead: int = 30,
    ) -> List[OnCallShift]:
        """Get shifts for a user.

        Args:
            user_id: User ID
            days_back: Days in the past
            days_ahead: Days in the future

        Returns:
            List of shifts
        """
        now = datetime.now()
        start = now - timedelta(days=days_back)
        end = now + timedelta(days=days_ahead)

        shifts = []

        with self._lock:
            schedules = list(self._schedules.values())

        for schedule in schedules:
            for shift in schedule.shifts:
                if shift.user.user_id == user_id:
                    if start <= shift.start_time <= end:
                        shifts.append(shift)

        # Include overrides
        for override in self._overrides:
            if override.user.user_id == user_id:
                if start <= override.start_time <= end:
                    shifts.append(override)

        return sorted(shifts, key=lambda s: s.start_time)

    async def check_handoffs(self):
        """Check for shift handoffs and trigger callbacks."""
        now = datetime.now()

        with self._lock:
            schedules = list(self._schedules.values())

        for schedule in schedules:
            if not schedule.enabled:
                continue

            current_shift = schedule.get_current_oncall()
            if not current_shift:
                continue

            # Check if handoff just occurred (within last minute)
            if (current_shift.start_time <= now and
                now - current_shift.start_time < timedelta(minutes=1)):

                # Find previous shift
                previous_shift = None
                for shift in schedule.shifts:
                    if shift.end_time == current_shift.start_time:
                        previous_shift = shift
                        break

                if previous_shift:
                    for callback in self._on_handoff:
                        try:
                            callback(previous_shift, current_shift)
                        except Exception as e:
                            logger.error(f"Handoff callback error: {e}")

    def on_handoff(
        self,
        callback: Callable[[OnCallShift, OnCallShift], None],
    ):
        """Register handoff callback.

        Args:
            callback: Function(old_shift, new_shift) to call on handoff
        """
        self._on_handoff.append(callback)

    def update_metrics(self):
        """Update Prometheus metrics."""
        with self._lock:
            schedules = list(self._schedules.values())

        now = datetime.now()

        for schedule in schedules:
            current = schedule.get_current_oncall()
            if current:
                # Update current user metric (using hash as value)
                user_hash = hash(current.user.user_id) % 1000000
                self.oncall_current_user.labels(
                    schedule=schedule.schedule_id,
                    shift_type=ShiftType.PRIMARY.value,
                ).set(user_hash)

                # Update hours remaining
                hours_remaining = (current.end_time - now).total_seconds() / 3600
                self.oncall_shift_hours_remaining.labels(
                    schedule=schedule.schedule_id,
                    user=current.user.name,
                ).set(max(0, hours_remaining))

    def get_all_schedules(self) -> List[OnCallSchedule]:
        """Get all schedules.

        Returns:
            List of schedules
        """
        with self._lock:
            return list(self._schedules.values())

    def get_summary(self) -> Dict[str, Any]:
        """Get on-call manager summary.

        Returns:
            Summary dictionary
        """
        with self._lock:
            schedules = list(self._schedules.values())
            overrides = list(self._overrides)

        now = datetime.now()
        active_overrides = [o for o in overrides if o.start_time <= now <= o.end_time]

        current_oncalls = {}
        for schedule in schedules:
            if schedule.enabled:
                current = schedule.get_current_oncall()
                if current:
                    current_oncalls[schedule.schedule_id] = current.user.name

        return {
            "total_schedules": len(schedules),
            "enabled_schedules": sum(1 for s in schedules if s.enabled),
            "total_overrides": len(overrides),
            "active_overrides": len(active_overrides),
            "current_oncalls": current_oncalls,
        }
