"""FOCUS Standard Exporter

Export cost data in FinOps FOCUS (FinOps Open Cost and Usage Specification) format.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TextIO
from datetime import datetime, date
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ChargeCategory(str, Enum):
    """FOCUS charge categories."""
    USAGE = "Usage"
    PURCHASE = "Purchase"
    TAX = "Tax"
    CREDIT = "Credit"
    ADJUSTMENT = "Adjustment"
    SUPPORT = "Support"


class ChargeClass(str, Enum):
    """FOCUS charge classes."""
    REGULAR = "Regular"
    CORRECTION = "Correction"


class PricingCategory(str, Enum):
    """FOCUS pricing categories."""
    ON_DEMAND = "On-Demand"
    COMMITMENT = "Commitment"
    SPOT = "Spot"
    OTHER = "Other"


@dataclass
class BillingPeriod:
    """Billing period definition."""
    start: date
    end: date

    def to_string(self) -> str:
        return f"{self.start.isoformat()}/{self.end.isoformat()}"


@dataclass
class FOCUSRecord:
    """A FOCUS-compliant cost record.

    Based on FinOps FOCUS specification v1.0.

    Usage:
        record = FOCUSRecord(
            billing_account_id="123456789",
            billing_account_name="Production Account",
            billing_period_start=date(2024, 1, 1),
            billing_period_end=date(2024, 1, 31),
            charge_category=ChargeCategory.USAGE,
            charge_description="Compute instance usage",
            provider="aws",
            service_name="Amazon EC2",
            resource_id="i-12345678",
            billed_cost=100.50,
            effective_cost=95.00,
            usage_quantity=720,
            usage_unit="Hours",
        )
    """
    # Required columns
    billing_account_id: str
    billing_account_name: str
    billing_period_start: date
    billing_period_end: date
    charge_category: ChargeCategory
    charge_description: str
    provider: str
    service_name: str
    billed_cost: float

    # Recommended columns
    effective_cost: Optional[float] = None
    charge_class: ChargeClass = ChargeClass.REGULAR
    pricing_category: PricingCategory = PricingCategory.ON_DEMAND

    # Resource identification
    resource_id: Optional[str] = None
    resource_name: Optional[str] = None
    resource_type: Optional[str] = None

    # Usage information
    usage_quantity: Optional[float] = None
    usage_unit: Optional[str] = None
    pricing_quantity: Optional[float] = None
    pricing_unit: Optional[str] = None

    # Geographic information
    region: Optional[str] = None
    availability_zone: Optional[str] = None

    # Cost allocation
    sub_account_id: Optional[str] = None
    sub_account_name: Optional[str] = None

    # Tags
    tags: Dict[str, str] = field(default_factory=dict)

    # Timestamps
    charge_period_start: Optional[datetime] = None
    charge_period_end: Optional[datetime] = None

    # Commitment information
    commitment_discount_id: Optional[str] = None
    commitment_discount_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "BillingAccountId": self.billing_account_id,
            "BillingAccountName": self.billing_account_name,
            "BillingPeriodStart": self.billing_period_start.isoformat(),
            "BillingPeriodEnd": self.billing_period_end.isoformat(),
            "ChargeCategory": self.charge_category.value,
            "ChargeClass": self.charge_class.value,
            "ChargeDescription": self.charge_description,
            "Provider": self.provider,
            "ServiceName": self.service_name,
            "BilledCost": self.billed_cost,
            "EffectiveCost": self.effective_cost or self.billed_cost,
            "PricingCategory": self.pricing_category.value,
        }

        # Optional fields
        if self.resource_id:
            result["ResourceId"] = self.resource_id
        if self.resource_name:
            result["ResourceName"] = self.resource_name
        if self.resource_type:
            result["ResourceType"] = self.resource_type

        if self.usage_quantity is not None:
            result["UsageQuantity"] = self.usage_quantity
        if self.usage_unit:
            result["UsageUnit"] = self.usage_unit
        if self.pricing_quantity is not None:
            result["PricingQuantity"] = self.pricing_quantity
        if self.pricing_unit:
            result["PricingUnit"] = self.pricing_unit

        if self.region:
            result["Region"] = self.region
        if self.availability_zone:
            result["AvailabilityZone"] = self.availability_zone

        if self.sub_account_id:
            result["SubAccountId"] = self.sub_account_id
        if self.sub_account_name:
            result["SubAccountName"] = self.sub_account_name

        if self.charge_period_start:
            result["ChargePeriodStart"] = self.charge_period_start.isoformat()
        if self.charge_period_end:
            result["ChargePeriodEnd"] = self.charge_period_end.isoformat()

        if self.commitment_discount_id:
            result["CommitmentDiscountId"] = self.commitment_discount_id
        if self.commitment_discount_name:
            result["CommitmentDiscountName"] = self.commitment_discount_name

        # Add tags with standard prefix
        for key, value in self.tags.items():
            result[f"Tag/{key}"] = value

        return result


class FOCUSExporter:
    """Exports cost data in FOCUS format.

    Usage:
        exporter = FOCUSExporter()

        # Add records
        exporter.add_record(focus_record)

        # Export to CSV
        exporter.export_csv("costs.csv")

        # Export to JSON
        exporter.export_json("costs.json")
    """

    def __init__(self):
        self._records: List[FOCUSRecord] = []

    def add_record(self, record: FOCUSRecord):
        """Add a FOCUS record.

        Args:
            record: FOCUS-compliant record
        """
        self._records.append(record)

    def add_records(self, records: List[FOCUSRecord]):
        """Add multiple records.

        Args:
            records: List of FOCUS records
        """
        self._records.extend(records)

    def clear(self):
        """Clear all records."""
        self._records.clear()

    def export_csv(self, path: str):
        """Export records to CSV.

        Args:
            path: Output file path
        """
        if not self._records:
            logger.warning("No records to export")
            return

        # Get all columns from all records
        all_keys = set()
        for record in self._records:
            all_keys.update(record.to_dict().keys())

        # Sort columns: required first, then optional, then tags
        required = [
            "BillingAccountId", "BillingAccountName",
            "BillingPeriodStart", "BillingPeriodEnd",
            "ChargeCategory", "ChargeClass", "ChargeDescription",
            "Provider", "ServiceName", "BilledCost", "EffectiveCost",
            "PricingCategory",
        ]
        optional = sorted([k for k in all_keys if k not in required and not k.startswith("Tag/")])
        tags = sorted([k for k in all_keys if k.startswith("Tag/")])
        columns = required + optional + tags

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()

            for record in self._records:
                writer.writerow(record.to_dict())

        logger.info(f"Exported {len(self._records)} records to {path}")

    def export_csv_to_stream(self, stream: TextIO):
        """Export records to CSV stream.

        Args:
            stream: Output stream
        """
        if not self._records:
            return

        all_keys = set()
        for record in self._records:
            all_keys.update(record.to_dict().keys())

        columns = sorted(all_keys)
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()

        for record in self._records:
            writer.writerow(record.to_dict())

    def export_json(self, path: str, indent: int = 2):
        """Export records to JSON.

        Args:
            path: Output file path
            indent: JSON indentation
        """
        data = {
            "schema_version": "1.0",
            "generated_at": datetime.now().isoformat(),
            "record_count": len(self._records),
            "records": [r.to_dict() for r in self._records],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, default=str)

        logger.info(f"Exported {len(self._records)} records to {path}")

    def export_jsonl(self, path: str):
        """Export records to JSON Lines format.

        Args:
            path: Output file path
        """
        with open(path, "w", encoding="utf-8") as f:
            for record in self._records:
                f.write(json.dumps(record.to_dict(), default=str))
                f.write("\n")

        logger.info(f"Exported {len(self._records)} records to {path}")

    def get_summary(self) -> Dict[str, Any]:
        """Get export summary.

        Returns:
            Summary dictionary
        """
        if not self._records:
            return {
                "record_count": 0,
                "total_billed_cost": 0,
                "total_effective_cost": 0,
            }

        total_billed = sum(r.billed_cost for r in self._records)
        total_effective = sum(r.effective_cost or r.billed_cost for r in self._records)

        # By provider
        by_provider: Dict[str, float] = {}
        for r in self._records:
            by_provider[r.provider] = by_provider.get(r.provider, 0) + r.billed_cost

        # By service
        by_service: Dict[str, float] = {}
        for r in self._records:
            by_service[r.service_name] = by_service.get(r.service_name, 0) + r.billed_cost

        # By category
        by_category: Dict[str, float] = {}
        for r in self._records:
            cat = r.charge_category.value
            by_category[cat] = by_category.get(cat, 0) + r.billed_cost

        # Date range
        periods = [(r.billing_period_start, r.billing_period_end) for r in self._records]
        min_start = min(p[0] for p in periods)
        max_end = max(p[1] for p in periods)

        return {
            "record_count": len(self._records),
            "total_billed_cost": total_billed,
            "total_effective_cost": total_effective,
            "savings": total_billed - total_effective,
            "by_provider": by_provider,
            "by_service": by_service,
            "by_category": by_category,
            "period_start": min_start.isoformat(),
            "period_end": max_end.isoformat(),
        }

    def filter_records(
        self,
        provider: Optional[str] = None,
        service: Optional[str] = None,
        category: Optional[ChargeCategory] = None,
        min_cost: Optional[float] = None,
        max_cost: Optional[float] = None,
    ) -> List[FOCUSRecord]:
        """Filter records.

        Args:
            provider: Filter by provider
            service: Filter by service name
            category: Filter by charge category
            min_cost: Minimum cost filter
            max_cost: Maximum cost filter

        Returns:
            Filtered records
        """
        records = self._records

        if provider:
            records = [r for r in records if r.provider == provider]
        if service:
            records = [r for r in records if r.service_name == service]
        if category:
            records = [r for r in records if r.charge_category == category]
        if min_cost is not None:
            records = [r for r in records if r.billed_cost >= min_cost]
        if max_cost is not None:
            records = [r for r in records if r.billed_cost <= max_cost]

        return records


# Helper functions for creating common records

def create_compute_record(
    account_id: str,
    account_name: str,
    provider: str,
    service: str,
    resource_id: str,
    cost: float,
    hours: float,
    period: BillingPeriod,
    region: Optional[str] = None,
) -> FOCUSRecord:
    """Create a compute usage record."""
    return FOCUSRecord(
        billing_account_id=account_id,
        billing_account_name=account_name,
        billing_period_start=period.start,
        billing_period_end=period.end,
        charge_category=ChargeCategory.USAGE,
        charge_description=f"Compute usage for {resource_id}",
        provider=provider,
        service_name=service,
        resource_id=resource_id,
        resource_type="Compute",
        billed_cost=cost,
        usage_quantity=hours,
        usage_unit="Hours",
        region=region,
    )


def create_storage_record(
    account_id: str,
    account_name: str,
    provider: str,
    service: str,
    resource_id: str,
    cost: float,
    storage_gb: float,
    period: BillingPeriod,
    region: Optional[str] = None,
) -> FOCUSRecord:
    """Create a storage usage record."""
    return FOCUSRecord(
        billing_account_id=account_id,
        billing_account_name=account_name,
        billing_period_start=period.start,
        billing_period_end=period.end,
        charge_category=ChargeCategory.USAGE,
        charge_description=f"Storage usage for {resource_id}",
        provider=provider,
        service_name=service,
        resource_id=resource_id,
        resource_type="Storage",
        billed_cost=cost,
        usage_quantity=storage_gb,
        usage_unit="GB-Mo",
        region=region,
    )
