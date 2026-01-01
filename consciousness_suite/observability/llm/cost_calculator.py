"""LLM Cost Calculator

Real-time cost tracking and forecasting for LLM API calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import threading
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelPricing:
    """Pricing configuration for a model."""
    model: str
    provider: str
    input_cost_per_1k: float  # Cost per 1000 input tokens
    output_cost_per_1k: float  # Cost per 1000 output tokens
    context_window: int = 128000
    max_output_tokens: int = 4096

    def calculate(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for given token counts."""
        input_cost = (input_tokens / 1000) * self.input_cost_per_1k
        output_cost = (output_tokens / 1000) * self.output_cost_per_1k
        return input_cost + output_cost


@dataclass
class CostRecord:
    """Record of a single cost event."""
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


# Default pricing database (as of 2024)
DEFAULT_PRICING: Dict[str, ModelPricing] = {
    # OpenAI
    "gpt-4": ModelPricing("gpt-4", "openai", 0.03, 0.06),
    "gpt-4-turbo": ModelPricing("gpt-4-turbo", "openai", 0.01, 0.03),
    "gpt-4o": ModelPricing("gpt-4o", "openai", 0.005, 0.015),
    "gpt-4o-mini": ModelPricing("gpt-4o-mini", "openai", 0.00015, 0.0006),
    "gpt-3.5-turbo": ModelPricing("gpt-3.5-turbo", "openai", 0.0005, 0.0015),
    "o1-preview": ModelPricing("o1-preview", "openai", 0.015, 0.06),
    "o1-mini": ModelPricing("o1-mini", "openai", 0.003, 0.012),
    # Anthropic
    "claude-3-opus": ModelPricing("claude-3-opus", "anthropic", 0.015, 0.075),
    "claude-3-sonnet": ModelPricing("claude-3-sonnet", "anthropic", 0.003, 0.015),
    "claude-3-haiku": ModelPricing("claude-3-haiku", "anthropic", 0.00025, 0.00125),
    "claude-3.5-sonnet": ModelPricing("claude-3.5-sonnet", "anthropic", 0.003, 0.015),
    "claude-opus-4": ModelPricing("claude-opus-4", "anthropic", 0.015, 0.075),
    "claude-sonnet-4": ModelPricing("claude-sonnet-4", "anthropic", 0.003, 0.015),
    # Google
    "gemini-1.5-pro": ModelPricing("gemini-1.5-pro", "google", 0.00125, 0.005),
    "gemini-1.5-flash": ModelPricing("gemini-1.5-flash", "google", 0.000075, 0.0003),
    "gemini-2.0-flash": ModelPricing("gemini-2.0-flash", "google", 0.0001, 0.0004),
    # Mistral
    "mistral-large": ModelPricing("mistral-large", "mistral", 0.004, 0.012),
    "mistral-small": ModelPricing("mistral-small", "mistral", 0.001, 0.003),
    # Cohere
    "command-r": ModelPricing("command-r", "cohere", 0.0005, 0.0015),
    "command-r-plus": ModelPricing("command-r-plus", "cohere", 0.003, 0.015),
}


class CostCalculator:
    """LLM cost calculator with history tracking.

    Usage:
        calc = CostCalculator()

        # Calculate cost
        cost = calc.calculate("gpt-4", input_tokens=1000, output_tokens=500)

        # Track cost
        calc.track("gpt-4", 1000, 500, user_id="user123")

        # Get summary
        summary = calc.get_summary()
    """

    def __init__(
        self,
        pricing: Optional[Dict[str, ModelPricing]] = None,
        max_history: int = 10000,
    ):
        self._pricing = pricing or DEFAULT_PRICING.copy()
        self._history: List[CostRecord] = []
        self._max_history = max_history
        self._lock = threading.Lock()

        # Aggregated totals
        self._total_cost = 0.0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._by_model: Dict[str, Dict[str, float]] = {}
        self._by_provider: Dict[str, Dict[str, float]] = {}

    def set_pricing(self, model: str, pricing: ModelPricing):
        """Set or update pricing for a model."""
        self._pricing[model] = pricing

    def get_pricing(self, model: str) -> Optional[ModelPricing]:
        """Get pricing for a model."""
        # Try exact match
        if model in self._pricing:
            return self._pricing[model]

        # Try prefix match
        model_lower = model.lower()
        for key, pricing in self._pricing.items():
            if model_lower.startswith(key.lower()):
                return pricing

        return None

    def calculate(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Calculate cost for a model call.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost in USD
        """
        pricing = self.get_pricing(model)
        if pricing:
            return pricing.calculate(input_tokens, output_tokens)

        # Fallback: use GPT-4 pricing
        logger.warning(f"No pricing found for model {model}, using GPT-4 pricing")
        return (input_tokens / 1000) * 0.03 + (output_tokens / 1000) * 0.06

    def calculate_detailed(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> Dict[str, Any]:
        """Calculate cost with detailed breakdown.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Dictionary with cost breakdown
        """
        pricing = self.get_pricing(model)

        if pricing:
            input_cost = (input_tokens / 1000) * pricing.input_cost_per_1k
            output_cost = (output_tokens / 1000) * pricing.output_cost_per_1k
            provider = pricing.provider
        else:
            input_cost = (input_tokens / 1000) * 0.03
            output_cost = (output_tokens / 1000) * 0.06
            provider = "unknown"

        return {
            "model": model,
            "provider": provider,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_cost_usd": input_cost,
            "output_cost_usd": output_cost,
            "total_cost_usd": input_cost + output_cost,
        }

    def track(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CostRecord:
        """Track a cost event.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            metadata: Additional metadata (user_id, request_id, etc.)

        Returns:
            CostRecord
        """
        pricing = self.get_pricing(model)
        provider = pricing.provider if pricing else "unknown"
        cost = self.calculate(model, input_tokens, output_tokens)

        record = CostRecord(
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            metadata=metadata or {},
        )

        with self._lock:
            # Add to history
            self._history.append(record)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history // 2 :]

            # Update aggregates
            self._total_cost += cost
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens

            # By model
            if model not in self._by_model:
                self._by_model[model] = {"cost": 0.0, "input": 0, "output": 0, "calls": 0}
            self._by_model[model]["cost"] += cost
            self._by_model[model]["input"] += input_tokens
            self._by_model[model]["output"] += output_tokens
            self._by_model[model]["calls"] += 1

            # By provider
            if provider not in self._by_provider:
                self._by_provider[provider] = {"cost": 0.0, "input": 0, "output": 0, "calls": 0}
            self._by_provider[provider]["cost"] += cost
            self._by_provider[provider]["input"] += input_tokens
            self._by_provider[provider]["output"] += output_tokens
            self._by_provider[provider]["calls"] += 1

        return record

    def get_summary(self) -> Dict[str, Any]:
        """Get cost summary.

        Returns:
            Dictionary with cost summary
        """
        with self._lock:
            calls = sum(m["calls"] for m in self._by_model.values())
            return {
                "total_cost_usd": self._total_cost,
                "total_input_tokens": self._total_input_tokens,
                "total_output_tokens": self._total_output_tokens,
                "total_tokens": self._total_input_tokens + self._total_output_tokens,
                "total_calls": calls,
                "avg_cost_per_call": self._total_cost / calls if calls > 0 else 0,
            }

    def cost_by_model(self) -> Dict[str, Dict[str, float]]:
        """Get costs grouped by model."""
        with self._lock:
            return self._by_model.copy()

    def cost_by_provider(self) -> Dict[str, Dict[str, float]]:
        """Get costs grouped by provider."""
        with self._lock:
            return self._by_provider.copy()

    def cost_by_time(
        self,
        window: timedelta = timedelta(hours=24),
        bucket_size: timedelta = timedelta(hours=1),
    ) -> List[Dict[str, Any]]:
        """Get costs grouped by time buckets.

        Args:
            window: Time window to analyze
            bucket_size: Size of each time bucket

        Returns:
            List of cost buckets
        """
        now = datetime.now()
        window_start = now - window

        buckets: Dict[int, Dict[str, float]] = {}
        bucket_seconds = bucket_size.total_seconds()

        with self._lock:
            for record in self._history:
                if record.timestamp >= window_start:
                    bucket_idx = int(
                        (record.timestamp - window_start).total_seconds() // bucket_seconds
                    )
                    if bucket_idx not in buckets:
                        buckets[bucket_idx] = {"cost": 0.0, "calls": 0}
                    buckets[bucket_idx]["cost"] += record.cost_usd
                    buckets[bucket_idx]["calls"] += 1

        result = []
        for idx in sorted(buckets.keys()):
            bucket_start = window_start + timedelta(seconds=idx * bucket_seconds)
            result.append({
                "timestamp": bucket_start.isoformat(),
                "cost_usd": buckets[idx]["cost"],
                "calls": buckets[idx]["calls"],
            })

        return result

    def estimate_monthly(
        self,
        model: str,
        daily_input_tokens: int,
        daily_output_tokens: int,
    ) -> Dict[str, float]:
        """Estimate monthly cost based on daily usage.

        Args:
            model: Model name
            daily_input_tokens: Expected daily input tokens
            daily_output_tokens: Expected daily output tokens

        Returns:
            Dictionary with daily/weekly/monthly estimates
        """
        daily_cost = self.calculate(model, daily_input_tokens, daily_output_tokens)

        return {
            "daily_cost_usd": daily_cost,
            "weekly_cost_usd": daily_cost * 7,
            "monthly_cost_usd": daily_cost * 30,
            "yearly_cost_usd": daily_cost * 365,
        }

    def compare_models(
        self,
        input_tokens: int,
        output_tokens: int,
    ) -> List[Tuple[str, float]]:
        """Compare costs across all models.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Sorted list of (model, cost) tuples
        """
        costs = []
        for model, pricing in self._pricing.items():
            cost = pricing.calculate(input_tokens, output_tokens)
            costs.append((model, cost))

        costs.sort(key=lambda x: x[1])
        return costs

    def recent_costs(self, count: int = 100) -> List[CostRecord]:
        """Get recent cost records.

        Args:
            count: Number of records to return

        Returns:
            List of recent CostRecord objects
        """
        with self._lock:
            return list(reversed(self._history[-count:]))

    def reset(self):
        """Reset all tracking data."""
        with self._lock:
            self._history.clear()
            self._total_cost = 0.0
            self._total_input_tokens = 0
            self._total_output_tokens = 0
            self._by_model.clear()
            self._by_provider.clear()
