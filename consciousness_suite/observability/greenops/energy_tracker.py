"""Energy Tracker

Track energy consumption of infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from enum import Enum
import threading
import logging

from prometheus_client import Gauge, Counter, Histogram

logger = logging.getLogger(__name__)


class EnergySource(str, Enum):
    """Energy sources."""
    GRID = "grid"
    SOLAR = "solar"
    WIND = "wind"
    HYDRO = "hydro"
    NUCLEAR = "nuclear"
    NATURAL_GAS = "natural_gas"
    COAL = "coal"
    OTHER = "other"


@dataclass
class EnergyReading:
    """Energy reading from a source."""
    source_id: str
    source_type: str
    power_watts: float
    energy_kwh: float
    timestamp: datetime = field(default_factory=datetime.now)
    source: EnergySource = EnergySource.GRID
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PowerUsage:
    """Power usage statistics."""
    avg_power_watts: float
    max_power_watts: float
    min_power_watts: float
    total_energy_kwh: float
    period_start: datetime
    period_end: datetime
    by_source: Dict[str, float] = field(default_factory=dict)
    by_type: Dict[EnergySource, float] = field(default_factory=dict)


@dataclass
class EnergyEfficiency:
    """Energy efficiency metrics."""
    pue: float  # Power Usage Effectiveness
    wue: float  # Water Usage Effectiveness
    cue: float  # Carbon Usage Effectiveness
    dcie: float  # Data Center Infrastructure Efficiency
    compute_efficiency: float  # Useful compute per kWh
    timestamp: datetime = field(default_factory=datetime.now)


class EnergyTracker:
    """Tracks energy consumption and efficiency.

    Usage:
        tracker = EnergyTracker()

        # Record energy readings
        tracker.record_reading(EnergyReading(
            source_id="server-rack-1",
            source_type="compute",
            power_watts=5000,
            energy_kwh=120,
        ))

        # Get usage statistics
        usage = tracker.get_usage()
        print(f"Total energy: {usage.total_energy_kwh} kWh")

        # Calculate efficiency
        efficiency = tracker.calculate_efficiency()
    """

    def __init__(self, namespace: str = "consciousness"):
        self.namespace = namespace
        self._readings: List[EnergyReading] = []
        self._efficiency_history: List[EnergyEfficiency] = []
        self._lock = threading.Lock()
        self._max_readings = 100000

        # Baseline for efficiency calculations
        self._it_load_watts = 0.0
        self._total_facility_watts = 0.0
        self._water_usage_liters = 0.0

        # Prometheus metrics
        self.power_consumption = Gauge(
            f"{namespace}_greenops_power_consumption_watts",
            "Current power consumption in watts",
            ["source_id", "source_type"],
        )

        self.energy_consumption = Counter(
            f"{namespace}_greenops_energy_consumption_kwh_total",
            "Total energy consumption in kWh",
            ["source_type", "energy_source"],
        )

        self.pue_metric = Gauge(
            f"{namespace}_greenops_pue",
            "Power Usage Effectiveness",
        )

        self.energy_by_source = Gauge(
            f"{namespace}_greenops_energy_by_source_kwh",
            "Energy by source type",
            ["source"],
        )

        self.power_histogram = Histogram(
            f"{namespace}_greenops_power_distribution_watts",
            "Power distribution",
            buckets=[100, 500, 1000, 2000, 5000, 10000, 20000, 50000],
        )

    def record_reading(self, reading: EnergyReading):
        """Record an energy reading.

        Args:
            reading: Energy reading
        """
        with self._lock:
            self._readings.append(reading)

            if len(self._readings) > self._max_readings:
                self._readings = self._readings[-self._max_readings // 2:]

        # Update metrics
        self.power_consumption.labels(
            source_id=reading.source_id,
            source_type=reading.source_type,
        ).set(reading.power_watts)

        self.energy_consumption.labels(
            source_type=reading.source_type,
            energy_source=reading.source.value,
        ).inc(reading.energy_kwh)

        self.power_histogram.observe(reading.power_watts)

    def record_batch(self, readings: List[EnergyReading]):
        """Record multiple readings.

        Args:
            readings: List of readings
        """
        for reading in readings:
            self.record_reading(reading)

    def set_facility_metrics(
        self,
        it_load_watts: float,
        total_facility_watts: float,
        water_usage_liters: float = 0,
    ):
        """Set facility-level metrics for PUE calculation.

        Args:
            it_load_watts: IT equipment power
            total_facility_watts: Total facility power
            water_usage_liters: Water usage (for WUE)
        """
        self._it_load_watts = it_load_watts
        self._total_facility_watts = total_facility_watts
        self._water_usage_liters = water_usage_liters

    def get_usage(
        self,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
    ) -> PowerUsage:
        """Get power usage statistics.

        Args:
            period_start: Start of period
            period_end: End of period

        Returns:
            PowerUsage statistics
        """
        now = datetime.now()
        period_start = period_start or now - timedelta(hours=24)
        period_end = period_end or now

        with self._lock:
            readings = [
                r for r in self._readings
                if period_start <= r.timestamp <= period_end
            ]

        if not readings:
            return PowerUsage(
                avg_power_watts=0,
                max_power_watts=0,
                min_power_watts=0,
                total_energy_kwh=0,
                period_start=period_start,
                period_end=period_end,
            )

        powers = [r.power_watts for r in readings]
        total_energy = sum(r.energy_kwh for r in readings)

        # By source
        by_source: Dict[str, float] = {}
        by_type: Dict[EnergySource, float] = {}

        for r in readings:
            by_source[r.source_id] = by_source.get(r.source_id, 0) + r.energy_kwh
            by_type[r.source] = by_type.get(r.source, 0) + r.energy_kwh

        # Update source metrics
        for source, kwh in by_type.items():
            self.energy_by_source.labels(source=source.value).set(kwh)

        return PowerUsage(
            avg_power_watts=sum(powers) / len(powers),
            max_power_watts=max(powers),
            min_power_watts=min(powers),
            total_energy_kwh=total_energy,
            period_start=period_start,
            period_end=period_end,
            by_source=by_source,
            by_type=by_type,
        )

    def calculate_efficiency(
        self,
        compute_operations: int = 0,
    ) -> EnergyEfficiency:
        """Calculate energy efficiency metrics.

        Args:
            compute_operations: Number of compute operations (for compute efficiency)

        Returns:
            EnergyEfficiency metrics
        """
        # PUE = Total Facility Power / IT Equipment Power
        pue = (
            self._total_facility_watts / self._it_load_watts
            if self._it_load_watts > 0 else 1.0
        )

        # WUE = Water Usage (L) / IT Equipment Power (kW)
        it_load_kw = self._it_load_watts / 1000
        wue = (
            self._water_usage_liters / it_load_kw
            if it_load_kw > 0 else 0
        )

        # CUE would need carbon data - placeholder
        cue = 0.0

        # DCiE = 1 / PUE * 100
        dcie = (1 / pue * 100) if pue > 0 else 0

        # Compute efficiency = operations per kWh
        usage = self.get_usage()
        compute_efficiency = (
            compute_operations / usage.total_energy_kwh
            if usage.total_energy_kwh > 0 else 0
        )

        efficiency = EnergyEfficiency(
            pue=pue,
            wue=wue,
            cue=cue,
            dcie=dcie,
            compute_efficiency=compute_efficiency,
        )

        with self._lock:
            self._efficiency_history.append(efficiency)

        # Update PUE metric
        self.pue_metric.set(pue)

        return efficiency

    def get_efficiency_trend(
        self,
        count: int = 30,
    ) -> List[EnergyEfficiency]:
        """Get efficiency trend.

        Args:
            count: Number of readings

        Returns:
            List of efficiency readings
        """
        with self._lock:
            return list(reversed(self._efficiency_history[-count:]))

    def get_renewable_percentage(self) -> float:
        """Get percentage of energy from renewable sources.

        Returns:
            Renewable percentage (0-100)
        """
        usage = self.get_usage()

        if usage.total_energy_kwh == 0:
            return 0

        renewable_sources = {
            EnergySource.SOLAR,
            EnergySource.WIND,
            EnergySource.HYDRO,
        }

        renewable = sum(
            kwh for source, kwh in usage.by_type.items()
            if source in renewable_sources
        )

        return renewable / usage.total_energy_kwh * 100

    def get_summary(self) -> Dict[str, Any]:
        """Get energy tracking summary.

        Returns:
            Summary dictionary
        """
        usage = self.get_usage()
        efficiency = self.calculate_efficiency()

        return {
            "avg_power_watts": usage.avg_power_watts,
            "total_energy_kwh_24h": usage.total_energy_kwh,
            "pue": efficiency.pue,
            "dcie_percentage": efficiency.dcie,
            "renewable_percentage": self.get_renewable_percentage(),
            "top_consumer": max(
                usage.by_source.items(),
                key=lambda x: x[1]
            )[0] if usage.by_source else None,
        }
