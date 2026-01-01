"""Cost Per Invocation Tracking

Tracks and calculates costs for serverless invocations.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class LambdaPricing:
    region: str
    architecture: str
    price_per_gb_second: float = 0.0000166667
    price_per_invocation: float = 0.0000002

@dataclass
class InvocationCost:
    invocation_id: str
    duration_ms: int
    memory_mb: int
    cost_usd: float

class InvocationCostTracker:
    def __init__(self, pricing: Optional[LambdaPricing] = None):
        self.pricing = pricing or LambdaPricing("us-east-1", "x86_64")

    def calculate_cost(self, duration_ms: int, memory_mb: int) -> float:
        gb_seconds = (memory_mb / 1024) * (duration_ms / 1000)
        return self.pricing.price_per_invocation + (gb_seconds * self.pricing.price_per_gb_second)

# Stubbed for integration
CostReport = Any
