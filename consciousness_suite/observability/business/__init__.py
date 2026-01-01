"""Business Observability Module

Business-focused observability including:
- SLO/SLI management
- Error budget tracking
- Revenue impact analysis
- Customer journey tracking
"""

from .slo_manager import (
    SLOManager,
    SLODefinition,
    SLOStatus,
    SLOViolation,
    SLOType,
)
from .sli_calculator import (
    SLICalculator,
    SLIMetric,
    SLIResult,
    TimeWindow,
)
from .error_budget import (
    ErrorBudgetManager,
    ErrorBudget,
    BudgetConsumption,
    BurnRate,
)
from .revenue_impact import (
    RevenueImpactAnalyzer,
    IncidentCost,
    RevenueCorrelation,
)
from .customer_journey import (
    CustomerJourneyTracker,
    JourneyStage,
    JourneyMetrics,
    ExperienceScore,
)

__all__ = [
    # SLO
    "SLOManager",
    "SLODefinition",
    "SLOStatus",
    "SLOViolation",
    "SLOType",
    # SLI
    "SLICalculator",
    "SLIMetric",
    "SLIResult",
    "TimeWindow",
    # Error Budget
    "ErrorBudgetManager",
    "ErrorBudget",
    "BudgetConsumption",
    "BurnRate",
    # Revenue
    "RevenueImpactAnalyzer",
    "IncidentCost",
    "RevenueCorrelation",
    # Customer Journey
    "CustomerJourneyTracker",
    "JourneyStage",
    "JourneyMetrics",
    "ExperienceScore",
]
