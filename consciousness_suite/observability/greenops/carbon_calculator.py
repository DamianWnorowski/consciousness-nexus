"""Carbon Calculator

Calculate carbon footprint of cloud infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from enum import Enum
import threading
import logging

from prometheus_client import Gauge, Counter

logger = logging.getLogger(__name__)


class CloudProvider(str, Enum):
    """Cloud providers."""
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    ON_PREMISE = "on_premise"


@dataclass
class CloudRegion:
    """Cloud region with carbon intensity data."""
    provider: CloudProvider
    region_code: str
    region_name: str
    carbon_intensity_gco2_kwh: float  # gCO2eq per kWh
    renewable_percentage: float  # 0-100
    pue: float = 1.2  # Power Usage Effectiveness

    @property
    def is_green(self) -> bool:
        """Check if region is considered green (>80% renewable)."""
        return self.renewable_percentage >= 80


@dataclass
class EmissionFactor:
    """Emission factor for a resource type."""
    resource_type: str
    energy_kwh_per_hour: float  # kWh per hour of usage
    embodied_carbon_gco2: float  # gCO2eq for manufacturing
    lifetime_hours: float  # Expected lifetime


@dataclass
class CarbonFootprint:
    """Carbon footprint calculation result."""
    total_gco2eq: float
    operational_gco2eq: float
    embodied_gco2eq: float
    energy_kwh: float
    renewable_energy_kwh: float
    period_start: datetime
    period_end: datetime
    by_resource_type: Dict[str, float] = field(default_factory=dict)
    by_region: Dict[str, float] = field(default_factory=dict)

    @property
    def renewable_percentage(self) -> float:
        """Get percentage of renewable energy used."""
        return (self.renewable_energy_kwh / self.energy_kwh * 100
                if self.energy_kwh > 0 else 0)


# Default region carbon intensity data (gCO2eq/kWh)
# Based on cloud provider sustainability reports
DEFAULT_REGIONS: Dict[str, CloudRegion] = {
    # AWS regions
    "us-east-1": CloudRegion(CloudProvider.AWS, "us-east-1", "N. Virginia", 379.0, 40, 1.2),
    "us-west-2": CloudRegion(CloudProvider.AWS, "us-west-2", "Oregon", 87.0, 90, 1.1),
    "eu-west-1": CloudRegion(CloudProvider.AWS, "eu-west-1", "Ireland", 316.0, 60, 1.15),
    "eu-north-1": CloudRegion(CloudProvider.AWS, "eu-north-1", "Stockholm", 8.0, 100, 1.1),
    "ap-northeast-1": CloudRegion(CloudProvider.AWS, "ap-northeast-1", "Tokyo", 471.0, 20, 1.2),

    # GCP regions
    "us-central1": CloudRegion(CloudProvider.GCP, "us-central1", "Iowa", 417.0, 30, 1.1),
    "us-west1": CloudRegion(CloudProvider.GCP, "us-west1", "Oregon", 87.0, 90, 1.1),
    "europe-north1": CloudRegion(CloudProvider.GCP, "europe-north1", "Finland", 72.0, 95, 1.1),
    "europe-west1": CloudRegion(CloudProvider.GCP, "europe-west1", "Belgium", 164.0, 75, 1.12),

    # Azure regions
    "eastus": CloudRegion(CloudProvider.AZURE, "eastus", "East US", 379.0, 40, 1.18),
    "westus2": CloudRegion(CloudProvider.AZURE, "westus2", "West US 2", 87.0, 90, 1.12),
    "northeurope": CloudRegion(CloudProvider.AZURE, "northeurope", "Ireland", 316.0, 60, 1.15),
    "swedencentral": CloudRegion(CloudProvider.AZURE, "swedencentral", "Sweden", 8.0, 100, 1.1),
}

# Default emission factors by resource type
DEFAULT_EMISSION_FACTORS: Dict[str, EmissionFactor] = {
    "compute_small": EmissionFactor("compute_small", 0.05, 50000, 35040),  # ~4 years
    "compute_medium": EmissionFactor("compute_medium", 0.15, 100000, 35040),
    "compute_large": EmissionFactor("compute_large", 0.35, 200000, 35040),
    "compute_xlarge": EmissionFactor("compute_xlarge", 0.75, 400000, 35040),
    "gpu_small": EmissionFactor("gpu_small", 0.5, 300000, 26280),  # ~3 years
    "gpu_large": EmissionFactor("gpu_large", 1.5, 600000, 26280),
    "storage_ssd": EmissionFactor("storage_ssd", 0.002, 10000, 43800),  # per TB
    "storage_hdd": EmissionFactor("storage_hdd", 0.005, 5000, 43800),
    "network_egress": EmissionFactor("network_egress", 0.001, 0, 87600),  # per GB
}


class CarbonCalculator:
    """Calculates carbon footprint of infrastructure.

    Usage:
        calculator = CarbonCalculator()

        # Track resource usage
        calculator.record_usage(
            resource_type="compute_medium",
            region="us-west-2",
            hours=720,
            quantity=5,
        )

        # Calculate footprint
        footprint = calculator.calculate_footprint()
        print(f"Total emissions: {footprint.total_gco2eq:.2f} gCO2eq")
    """

    def __init__(self, namespace: str = "consciousness"):
        self.namespace = namespace
        self._regions = DEFAULT_REGIONS.copy()
        self._emission_factors = DEFAULT_EMISSION_FACTORS.copy()
        self._usage_records: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

        # Prometheus metrics
        self.carbon_emissions = Gauge(
            f"{namespace}_greenops_carbon_emissions_gco2eq",
            "Carbon emissions in gCO2eq",
            ["region", "resource_type"],
        )

        self.energy_consumption = Gauge(
            f"{namespace}_greenops_energy_consumption_kwh",
            "Energy consumption in kWh",
            ["region"],
        )

        self.renewable_percentage = Gauge(
            f"{namespace}_greenops_renewable_percentage",
            "Percentage of renewable energy",
        )

        self.carbon_intensity = Gauge(
            f"{namespace}_greenops_carbon_intensity_gco2_per_kwh",
            "Carbon intensity per kWh",
            ["region"],
        )

    def add_region(self, region: CloudRegion):
        """Add or update a region.

        Args:
            region: Cloud region configuration
        """
        with self._lock:
            self._regions[region.region_code] = region

        self.carbon_intensity.labels(
            region=region.region_code
        ).set(region.carbon_intensity_gco2_kwh)

    def add_emission_factor(self, factor: EmissionFactor):
        """Add or update an emission factor.

        Args:
            factor: Emission factor configuration
        """
        with self._lock:
            self._emission_factors[factor.resource_type] = factor

    def record_usage(
        self,
        resource_type: str,
        region: str,
        hours: float,
        quantity: int = 1,
        timestamp: Optional[datetime] = None,
    ):
        """Record resource usage.

        Args:
            resource_type: Type of resource
            region: Region code
            hours: Hours of usage
            quantity: Number of instances
            timestamp: Usage timestamp
        """
        with self._lock:
            self._usage_records.append({
                "resource_type": resource_type,
                "region": region,
                "hours": hours,
                "quantity": quantity,
                "timestamp": timestamp or datetime.now(),
            })

    def calculate_footprint(
        self,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
    ) -> CarbonFootprint:
        """Calculate carbon footprint.

        Args:
            period_start: Start of period
            period_end: End of period

        Returns:
            CarbonFootprint calculation
        """
        now = datetime.now()
        period_start = period_start or now - timedelta(days=30)
        period_end = period_end or now

        with self._lock:
            records = [
                r for r in self._usage_records
                if period_start <= r["timestamp"] <= period_end
            ]

        total_energy = 0.0
        total_operational = 0.0
        total_embodied = 0.0
        renewable_energy = 0.0
        by_resource_type: Dict[str, float] = {}
        by_region: Dict[str, float] = {}

        for record in records:
            resource_type = record["resource_type"]
            region_code = record["region"]
            hours = record["hours"]
            quantity = record["quantity"]

            # Get emission factor
            factor = self._emission_factors.get(
                resource_type,
                EmissionFactor(resource_type, 0.1, 50000, 35040)
            )

            # Get region
            region = self._regions.get(
                region_code,
                CloudRegion(CloudProvider.AWS, region_code, region_code, 400, 30, 1.2)
            )

            # Calculate energy (kWh)
            energy = factor.energy_kwh_per_hour * hours * quantity * region.pue
            total_energy += energy

            # Calculate renewable portion
            renewable_energy += energy * (region.renewable_percentage / 100)

            # Calculate operational emissions
            operational = energy * region.carbon_intensity_gco2_kwh
            total_operational += operational

            # Calculate embodied emissions (amortized over lifetime)
            embodied = (factor.embodied_carbon_gco2 * hours / factor.lifetime_hours) * quantity
            total_embodied += embodied

            # Track by type and region
            total_for_record = operational + embodied
            by_resource_type[resource_type] = by_resource_type.get(
                resource_type, 0
            ) + total_for_record
            by_region[region_code] = by_region.get(
                region_code, 0
            ) + total_for_record

            # Update metrics
            self.carbon_emissions.labels(
                region=region_code,
                resource_type=resource_type,
            ).set(total_for_record)

            self.energy_consumption.labels(region=region_code).set(energy)

        # Update renewable percentage metric
        if total_energy > 0:
            self.renewable_percentage.set(renewable_energy / total_energy * 100)

        return CarbonFootprint(
            total_gco2eq=total_operational + total_embodied,
            operational_gco2eq=total_operational,
            embodied_gco2eq=total_embodied,
            energy_kwh=total_energy,
            renewable_energy_kwh=renewable_energy,
            period_start=period_start,
            period_end=period_end,
            by_resource_type=by_resource_type,
            by_region=by_region,
        )

    def estimate_annual_footprint(self) -> CarbonFootprint:
        """Estimate annual carbon footprint based on current usage.

        Returns:
            Annualized CarbonFootprint
        """
        monthly = self.calculate_footprint()

        return CarbonFootprint(
            total_gco2eq=monthly.total_gco2eq * 12,
            operational_gco2eq=monthly.operational_gco2eq * 12,
            embodied_gco2eq=monthly.embodied_gco2eq * 12,
            energy_kwh=monthly.energy_kwh * 12,
            renewable_energy_kwh=monthly.renewable_energy_kwh * 12,
            period_start=datetime.now(),
            period_end=datetime.now() + timedelta(days=365),
            by_resource_type={k: v * 12 for k, v in monthly.by_resource_type.items()},
            by_region={k: v * 12 for k, v in monthly.by_region.items()},
        )

    def get_green_regions(self) -> List[CloudRegion]:
        """Get list of green regions (>80% renewable).

        Returns:
            List of green regions
        """
        return [r for r in self._regions.values() if r.is_green]

    def suggest_region_migration(
        self,
        current_region: str,
    ) -> Optional[CloudRegion]:
        """Suggest a greener region for migration.

        Args:
            current_region: Current region code

        Returns:
            Suggested region or None
        """
        current = self._regions.get(current_region)
        if not current:
            return None

        # Find same-provider regions with lower carbon intensity
        better = [
            r for r in self._regions.values()
            if r.provider == current.provider
            and r.carbon_intensity_gco2_kwh < current.carbon_intensity_gco2_kwh * 0.5
        ]

        if better:
            return min(better, key=lambda r: r.carbon_intensity_gco2_kwh)

        return None

    def get_summary(self) -> Dict[str, Any]:
        """Get carbon footprint summary.

        Returns:
            Summary dictionary
        """
        footprint = self.calculate_footprint()
        annual = self.estimate_annual_footprint()

        return {
            "monthly_gco2eq": footprint.total_gco2eq,
            "annual_gco2eq_estimate": annual.total_gco2eq,
            "monthly_energy_kwh": footprint.energy_kwh,
            "renewable_percentage": footprint.renewable_percentage,
            "top_emitting_region": max(
                footprint.by_region.items(),
                key=lambda x: x[1]
            )[0] if footprint.by_region else None,
            "top_emitting_resource": max(
                footprint.by_resource_type.items(),
                key=lambda x: x[1]
            )[0] if footprint.by_resource_type else None,
            "green_regions_available": len(self.get_green_regions()),
        }
