"""Probe Scheduler

Schedules and manages synthetic probe execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Type
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import threading
import logging

from prometheus_client import Gauge, Counter

from .probes import SyntheticProbe, ProbeConfig, ProbeResult, ProbeStatus

logger = logging.getLogger(__name__)


class ScheduleState(str, Enum):
    """Scheduled probe state."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"


@dataclass
class ScheduleConfig:
    """Configuration for probe scheduling."""
    interval_seconds: int = 60
    jitter_seconds: int = 5
    max_concurrent: int = 10
    retry_on_failure: bool = True
    alert_on_consecutive_failures: int = 3
    maintenance_windows: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ScheduledProbe:
    """A scheduled probe instance."""
    probe_id: str
    config: ProbeConfig
    probe_instance: SyntheticProbe
    schedule_config: ScheduleConfig
    state: ScheduleState = ScheduleState.PENDING
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    last_result: Optional[ProbeResult] = None
    consecutive_failures: int = 0
    total_runs: int = 0
    total_failures: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProbeScheduler:
    """Schedules and manages synthetic probes.

    Usage:
        scheduler = ProbeScheduler()

        # Register probe
        scheduler.register(
            probe_id="api-health",
            config=ProbeConfig(...),
            probe_class=HTTPProbe,
        )

        # Start scheduler
        await scheduler.start()

        # Stop scheduler
        await scheduler.stop()
    """

    def __init__(self, namespace: str = "consciousness"):
        self.namespace = namespace

        self._probes: Dict[str, ScheduledProbe] = {}
        self._lock = threading.Lock()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._default_schedule = ScheduleConfig()

        # Alert callbacks
        self._alert_callbacks: List[Callable[[str, ProbeResult], None]] = []

        # Metrics
        self.scheduled_probes = Gauge(
            f"{namespace}_synthetic_scheduled_probes",
            "Number of scheduled probes",
            ["state"],
        )

        self.probe_runs = Counter(
            f"{namespace}_synthetic_probe_runs_total",
            "Total probe runs by scheduler",
            ["probe_id", "status"],
        )

        self.scheduler_cycles = Counter(
            f"{namespace}_synthetic_scheduler_cycles_total",
            "Total scheduler cycles",
        )

        self.consecutive_failures = Gauge(
            f"{namespace}_synthetic_consecutive_failures",
            "Consecutive probe failures",
            ["probe_id"],
        )

    def register(
        self,
        probe_id: str,
        config: ProbeConfig,
        probe_class: Type[SyntheticProbe],
        schedule_config: Optional[ScheduleConfig] = None,
    ):
        """Register a probe for scheduling.

        Args:
            probe_id: Probe ID
            config: Probe configuration
            probe_class: Probe class to instantiate
            schedule_config: Schedule configuration
        """
        probe_instance = probe_class(namespace=self.namespace)
        schedule = schedule_config or self._default_schedule

        scheduled = ScheduledProbe(
            probe_id=probe_id,
            config=config,
            probe_instance=probe_instance,
            schedule_config=schedule,
            next_run=datetime.now(),
        )

        with self._lock:
            self._probes[probe_id] = scheduled

        self._update_metrics()
        logger.info(f"Registered probe: {probe_id}")

    def unregister(self, probe_id: str):
        """Unregister a probe.

        Args:
            probe_id: Probe ID
        """
        with self._lock:
            if probe_id in self._probes:
                self._probes[probe_id].state = ScheduleState.STOPPED
                del self._probes[probe_id]

        self._update_metrics()
        logger.info(f"Unregistered probe: {probe_id}")

    def pause(self, probe_id: str):
        """Pause a probe.

        Args:
            probe_id: Probe ID
        """
        with self._lock:
            if probe_id in self._probes:
                self._probes[probe_id].state = ScheduleState.PAUSED

        self._update_metrics()

    def resume(self, probe_id: str):
        """Resume a paused probe.

        Args:
            probe_id: Probe ID
        """
        with self._lock:
            if probe_id in self._probes:
                scheduled = self._probes[probe_id]
                if scheduled.state == ScheduleState.PAUSED:
                    scheduled.state = ScheduleState.PENDING
                    scheduled.next_run = datetime.now()

        self._update_metrics()

    async def start(self, max_concurrent: int = 10):
        """Start the scheduler.

        Args:
            max_concurrent: Maximum concurrent probe executions
        """
        if self._running:
            return

        self._running = True
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._task = asyncio.create_task(self._scheduler_loop())

        logger.info("Probe scheduler started")

    async def stop(self):
        """Stop the scheduler."""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info("Probe scheduler stopped")

    async def _scheduler_loop(self):
        """Main scheduler loop."""
        while self._running:
            try:
                await self._run_cycle()
                self.scheduler_cycles.inc()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler cycle error: {e}")

            await asyncio.sleep(1)  # Check every second

    async def _run_cycle(self):
        """Run a scheduler cycle."""
        now = datetime.now()
        tasks = []

        with self._lock:
            for probe_id, scheduled in self._probes.items():
                if scheduled.state != ScheduleState.PENDING:
                    continue

                if scheduled.next_run and scheduled.next_run <= now:
                    if not self._in_maintenance_window(scheduled):
                        tasks.append(self._run_probe(scheduled))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _in_maintenance_window(self, scheduled: ScheduledProbe) -> bool:
        """Check if probe is in maintenance window."""
        now = datetime.now()

        for window in scheduled.schedule_config.maintenance_windows:
            start = window.get("start")
            end = window.get("end")

            if start and end:
                # Parse as time or datetime
                if isinstance(start, str) and ":" in start:
                    # Time-based window (daily)
                    start_time = datetime.strptime(start, "%H:%M").time()
                    end_time = datetime.strptime(end, "%H:%M").time()
                    if start_time <= now.time() <= end_time:
                        return True
                else:
                    # Datetime-based window
                    if isinstance(start, str):
                        start = datetime.fromisoformat(start)
                    if isinstance(end, str):
                        end = datetime.fromisoformat(end)
                    if start <= now <= end:
                        return True

        return False

    async def _run_probe(self, scheduled: ScheduledProbe):
        """Run a single probe.

        Args:
            scheduled: Scheduled probe
        """
        async with self._semaphore:
            scheduled.state = ScheduleState.RUNNING
            self._update_metrics()

            try:
                # Add jitter
                import random
                jitter = random.uniform(0, scheduled.schedule_config.jitter_seconds)
                await asyncio.sleep(jitter)

                # Execute probe for each location
                locations = scheduled.config.locations or ["default"]

                for location in locations:
                    result = await scheduled.probe_instance.run(
                        scheduled.config,
                        location=location,
                    )

                    # Update scheduled probe state
                    scheduled.last_run = datetime.now()
                    scheduled.last_result = result
                    scheduled.total_runs += 1

                    if result.is_success:
                        scheduled.consecutive_failures = 0
                    else:
                        scheduled.consecutive_failures += 1
                        scheduled.total_failures += 1

                    # Update metrics
                    self.probe_runs.labels(
                        probe_id=scheduled.probe_id,
                        status=result.status.value,
                    ).inc()

                    self.consecutive_failures.labels(
                        probe_id=scheduled.probe_id,
                    ).set(scheduled.consecutive_failures)

                    # Check for alert threshold
                    if scheduled.consecutive_failures >= scheduled.schedule_config.alert_on_consecutive_failures:
                        self._trigger_alert(scheduled.probe_id, result)

            except Exception as e:
                logger.error(f"Probe {scheduled.probe_id} execution error: {e}")
                scheduled.consecutive_failures += 1
                scheduled.total_failures += 1

            finally:
                # Schedule next run
                scheduled.next_run = datetime.now() + timedelta(
                    seconds=scheduled.schedule_config.interval_seconds
                )
                scheduled.state = ScheduleState.PENDING
                self._update_metrics()

    def _trigger_alert(self, probe_id: str, result: ProbeResult):
        """Trigger alert for failed probe.

        Args:
            probe_id: Probe ID
            result: Last probe result
        """
        logger.warning(f"Probe {probe_id} alert: {result.consecutive_failures} consecutive failures")

        for callback in self._alert_callbacks:
            try:
                callback(probe_id, result)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

    def on_alert(self, callback: Callable[[str, ProbeResult], None]):
        """Register alert callback.

        Args:
            callback: Function to call on probe alerts
        """
        self._alert_callbacks.append(callback)

    def _update_metrics(self):
        """Update scheduler metrics."""
        states: Dict[str, int] = {}
        with self._lock:
            for scheduled in self._probes.values():
                s = scheduled.state.value
                states[s] = states.get(s, 0) + 1

        for state, count in states.items():
            self.scheduled_probes.labels(state=state).set(count)

    async def run_once(self, probe_id: str) -> Optional[ProbeResult]:
        """Run a probe once immediately.

        Args:
            probe_id: Probe ID

        Returns:
            ProbeResult or None
        """
        with self._lock:
            scheduled = self._probes.get(probe_id)

        if not scheduled:
            return None

        return await scheduled.probe_instance.run(scheduled.config)

    def get_probe(self, probe_id: str) -> Optional[ScheduledProbe]:
        """Get a scheduled probe.

        Args:
            probe_id: Probe ID

        Returns:
            ScheduledProbe or None
        """
        with self._lock:
            return self._probes.get(probe_id)

    def get_all_probes(self) -> List[ScheduledProbe]:
        """Get all scheduled probes.

        Returns:
            List of scheduled probes
        """
        with self._lock:
            return list(self._probes.values())

    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status.

        Returns:
            Status dictionary
        """
        with self._lock:
            probes = list(self._probes.values())

        running = sum(1 for p in probes if p.state == ScheduleState.RUNNING)
        pending = sum(1 for p in probes if p.state == ScheduleState.PENDING)
        paused = sum(1 for p in probes if p.state == ScheduleState.PAUSED)

        failing = sum(1 for p in probes if p.consecutive_failures > 0)

        return {
            "running": self._running,
            "total_probes": len(probes),
            "running_probes": running,
            "pending_probes": pending,
            "paused_probes": paused,
            "failing_probes": failing,
            "probes": [
                {
                    "probe_id": p.probe_id,
                    "state": p.state.value,
                    "last_run": p.last_run.isoformat() if p.last_run else None,
                    "next_run": p.next_run.isoformat() if p.next_run else None,
                    "consecutive_failures": p.consecutive_failures,
                    "total_runs": p.total_runs,
                    "total_failures": p.total_failures,
                    "last_status": p.last_result.status.value if p.last_result else None,
                }
                for p in probes
            ],
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get scheduler summary.

        Returns:
            Summary dictionary
        """
        status = self.get_status()

        # Calculate success rates
        with self._lock:
            probes = list(self._probes.values())

        success_rates = []
        for p in probes:
            if p.total_runs > 0:
                rate = (p.total_runs - p.total_failures) / p.total_runs
                success_rates.append(rate)

        avg_success_rate = sum(success_rates) / len(success_rates) if success_rates else 1.0

        return {
            "running": status["running"],
            "total_probes": status["total_probes"],
            "failing_probes": status["failing_probes"],
            "average_success_rate": avg_success_rate,
        }
