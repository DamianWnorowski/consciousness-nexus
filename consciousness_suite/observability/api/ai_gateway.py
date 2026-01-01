"""AI Gateway Monitor

AI/LLM API gateway metrics and cost tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import threading
import logging

from prometheus_client import Gauge, Counter, Histogram

logger = logging.getLogger(__name__)


class AIProvider(str, Enum):
    """AI/LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE_OPENAI = "azure_openai"
    AWS_BEDROCK = "aws_bedrock"
    COHERE = "cohere"
    MISTRAL = "mistral"
    CUSTOM = "custom"


class ModelTier(str, Enum):
    """Model capability tiers."""
    FLAGSHIP = "flagship"  # GPT-4, Claude 3 Opus
    BALANCED = "balanced"  # GPT-4 Turbo, Claude 3 Sonnet
    FAST = "fast"  # GPT-3.5, Claude 3 Haiku
    EMBEDDING = "embedding"


@dataclass
class ModelUsage:
    """Usage for a specific model."""
    provider: AIProvider
    model: str
    tier: ModelTier
    request_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0
    avg_latency_ms: float = 0
    error_count: int = 0
    first_used: datetime = field(default_factory=datetime.now)
    last_used: datetime = field(default_factory=datetime.now)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def avg_tokens_per_request(self) -> float:
        if self.request_count == 0:
            return 0
        return self.total_tokens / self.request_count

    @property
    def cost_per_request(self) -> float:
        if self.request_count == 0:
            return 0
        return self.total_cost_usd / self.request_count

    @property
    def error_rate(self) -> float:
        if self.request_count == 0:
            return 0
        return self.error_count / self.request_count


@dataclass
class AIProviderMetrics:
    """Metrics for an AI provider."""
    provider: AIProvider
    request_count: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0
    avg_latency_ms: float = 0
    error_count: int = 0
    rate_limit_hits: int = 0
    models_used: List[str] = field(default_factory=list)
    quota_remaining: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def error_rate(self) -> float:
        if self.request_count == 0:
            return 0
        return self.error_count / self.request_count


@dataclass
class CostTracking:
    """Cost tracking for AI usage."""
    period_start: datetime
    period_end: datetime
    total_cost_usd: float = 0
    by_provider: Dict[str, float] = field(default_factory=dict)
    by_model: Dict[str, float] = field(default_factory=dict)
    by_user: Dict[str, float] = field(default_factory=dict)
    by_application: Dict[str, float] = field(default_factory=dict)
    projected_monthly_usd: float = 0
    budget_usd: Optional[float] = None
    budget_remaining_usd: Optional[float] = None
    overage_alerts: List[str] = field(default_factory=list)


@dataclass
class TokenQuota:
    """Token quota configuration and tracking."""
    quota_id: str
    scope: str  # user, api_key, application
    scope_id: str
    daily_limit: int
    monthly_limit: int
    current_daily: int = 0
    current_monthly: int = 0
    daily_reset: datetime = field(default_factory=datetime.now)
    monthly_reset: datetime = field(default_factory=datetime.now)

    @property
    def daily_remaining(self) -> int:
        return max(0, self.daily_limit - self.current_daily)

    @property
    def monthly_remaining(self) -> int:
        return max(0, self.monthly_limit - self.current_monthly)

    @property
    def daily_usage_percent(self) -> float:
        return self.current_daily / self.daily_limit * 100 if self.daily_limit > 0 else 0

    @property
    def monthly_usage_percent(self) -> float:
        return self.current_monthly / self.monthly_limit * 100 if self.monthly_limit > 0 else 0


# Default pricing per 1M tokens (as of early 2024)
DEFAULT_PRICING = {
    "gpt-4": {"input": 30.0, "output": 60.0},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-4o": {"input": 5.0, "output": 15.0},
    "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
    "claude-3-opus": {"input": 15.0, "output": 75.0},
    "claude-3-sonnet": {"input": 3.0, "output": 15.0},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    "claude-3.5-sonnet": {"input": 3.0, "output": 15.0},
    "gemini-pro": {"input": 0.5, "output": 1.5},
    "gemini-1.5-pro": {"input": 3.5, "output": 10.5},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
}


class AIGatewayMonitor:
    """Monitors AI/LLM gateway usage and costs.

    Usage:
        monitor = AIGatewayMonitor()

        # Set pricing
        monitor.set_model_pricing("gpt-4", input=30.0, output=60.0)

        # Record request
        monitor.record_request(
            provider=AIProvider.OPENAI,
            model="gpt-4",
            input_tokens=500,
            output_tokens=200,
            latency_ms=1500,
            user_id="user_123",
        )

        # Check quota
        result = monitor.check_quota("user", "user_123", tokens=1000)

        # Get costs
        costs = monitor.get_cost_tracking()
    """

    def __init__(
        self,
        namespace: str = "consciousness",
        monthly_budget_usd: Optional[float] = None,
    ):
        self.namespace = namespace
        self.monthly_budget_usd = monthly_budget_usd

        self._model_pricing = DEFAULT_PRICING.copy()
        self._model_usage: Dict[str, ModelUsage] = {}
        self._provider_metrics: Dict[str, AIProviderMetrics] = {}
        self._quotas: Dict[str, TokenQuota] = {}
        self._request_history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

        self._cost_alert_callbacks: List[Callable[[str, float], None]] = []
        self._quota_alert_callbacks: List[Callable[[TokenQuota], None]] = []

        # Prometheus metrics
        self.ai_requests = Counter(
            f"{namespace}_ai_gateway_requests_total",
            "Total AI requests",
            ["provider", "model", "tier"],
        )

        self.ai_tokens = Counter(
            f"{namespace}_ai_gateway_tokens_total",
            "Total tokens processed",
            ["provider", "model", "direction"],
        )

        self.ai_cost = Counter(
            f"{namespace}_ai_gateway_cost_usd_total",
            "Total cost in USD",
            ["provider", "model"],
        )

        self.ai_latency = Histogram(
            f"{namespace}_ai_gateway_latency_seconds",
            "Request latency",
            ["provider", "model"],
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
        )

        self.ai_errors = Counter(
            f"{namespace}_ai_gateway_errors_total",
            "Total errors",
            ["provider", "model", "error_type"],
        )

        self.ai_rate_limits = Counter(
            f"{namespace}_ai_gateway_rate_limits_total",
            "Rate limit hits",
            ["provider"],
        )

        self.ai_quota_usage = Gauge(
            f"{namespace}_ai_gateway_quota_usage_percent",
            "Quota usage percentage",
            ["scope", "scope_id", "period"],
        )

        self.ai_cost_daily = Gauge(
            f"{namespace}_ai_gateway_cost_daily_usd",
            "Daily cost in USD",
            ["provider"],
        )

        self.ai_budget_remaining = Gauge(
            f"{namespace}_ai_gateway_budget_remaining_usd",
            "Remaining budget in USD",
        )

    def set_model_pricing(
        self,
        model: str,
        input_per_million: float,
        output_per_million: float,
    ):
        """Set pricing for a model.

        Args:
            model: Model name
            input_per_million: Cost per 1M input tokens
            output_per_million: Cost per 1M output tokens
        """
        self._model_pricing[model] = {
            "input": input_per_million,
            "output": output_per_million,
        }

    def _calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Calculate cost for a request."""
        pricing = self._model_pricing.get(model)
        if not pricing:
            # Try partial match
            for model_key, prices in self._model_pricing.items():
                if model_key in model or model in model_key:
                    pricing = prices
                    break

        if not pricing:
            logger.warning(f"No pricing for model: {model}")
            return 0

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost

    def _get_model_tier(self, model: str) -> ModelTier:
        """Determine model tier."""
        model_lower = model.lower()

        if any(x in model_lower for x in ["opus", "gpt-4", "gemini-1.5-pro"]):
            return ModelTier.FLAGSHIP
        elif any(x in model_lower for x in ["sonnet", "turbo", "4o"]):
            return ModelTier.BALANCED
        elif any(x in model_lower for x in ["haiku", "3.5", "flash"]):
            return ModelTier.FAST
        elif "embed" in model_lower:
            return ModelTier.EMBEDDING

        return ModelTier.BALANCED

    def record_request(
        self,
        provider: AIProvider,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        user_id: Optional[str] = None,
        api_key_id: Optional[str] = None,
        application: Optional[str] = None,
        error: Optional[str] = None,
        rate_limited: bool = False,
    ):
        """Record an AI gateway request.

        Args:
            provider: AI provider
            model: Model name
            input_tokens: Input token count
            output_tokens: Output token count
            latency_ms: Request latency
            user_id: User identifier
            api_key_id: API key identifier
            application: Application name
            error: Error message if failed
            rate_limited: Whether request was rate limited
        """
        cost = self._calculate_cost(model, input_tokens, output_tokens)
        tier = self._get_model_tier(model)
        total_tokens = input_tokens + output_tokens

        with self._lock:
            # Update model usage
            model_key = f"{provider.value}:{model}"
            if model_key not in self._model_usage:
                self._model_usage[model_key] = ModelUsage(
                    provider=provider,
                    model=model,
                    tier=tier,
                )

            usage = self._model_usage[model_key]
            usage.request_count += 1
            usage.total_input_tokens += input_tokens
            usage.total_output_tokens += output_tokens
            usage.total_cost_usd += cost
            usage.avg_latency_ms = (
                (usage.avg_latency_ms * (usage.request_count - 1) + latency_ms)
                / usage.request_count
            )
            if error:
                usage.error_count += 1
            usage.last_used = datetime.now()

            # Update provider metrics
            if provider.value not in self._provider_metrics:
                self._provider_metrics[provider.value] = AIProviderMetrics(
                    provider=provider,
                )

            pm = self._provider_metrics[provider.value]
            pm.request_count += 1
            pm.total_tokens += total_tokens
            pm.total_cost_usd += cost
            pm.avg_latency_ms = (
                (pm.avg_latency_ms * (pm.request_count - 1) + latency_ms)
                / pm.request_count
            )
            if error:
                pm.error_count += 1
            if rate_limited:
                pm.rate_limit_hits += 1
            if model not in pm.models_used:
                pm.models_used.append(model)
            pm.timestamp = datetime.now()

            # Store request
            self._request_history.append({
                "timestamp": datetime.now(),
                "provider": provider.value,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost,
                "latency_ms": latency_ms,
                "user_id": user_id,
                "api_key_id": api_key_id,
                "application": application,
                "error": error,
            })

            # Trim history
            if len(self._request_history) > 100000:
                self._request_history = self._request_history[-50000:]

            # Update quotas
            if user_id:
                self._update_quota("user", user_id, total_tokens)
            if api_key_id:
                self._update_quota("api_key", api_key_id, total_tokens)
            if application:
                self._update_quota("application", application, total_tokens)

        # Update Prometheus metrics
        self.ai_requests.labels(
            provider=provider.value,
            model=model,
            tier=tier.value,
        ).inc()

        self.ai_tokens.labels(
            provider=provider.value,
            model=model,
            direction="input",
        ).inc(input_tokens)

        self.ai_tokens.labels(
            provider=provider.value,
            model=model,
            direction="output",
        ).inc(output_tokens)

        self.ai_cost.labels(
            provider=provider.value,
            model=model,
        ).inc(cost)

        self.ai_latency.labels(
            provider=provider.value,
            model=model,
        ).observe(latency_ms / 1000)

        if error:
            error_type = "rate_limit" if rate_limited else "error"
            self.ai_errors.labels(
                provider=provider.value,
                model=model,
                error_type=error_type,
            ).inc()

        if rate_limited:
            self.ai_rate_limits.labels(provider=provider.value).inc()

        # Check cost alerts
        self._check_cost_alerts(provider.value, cost)

    def _update_quota(self, scope: str, scope_id: str, tokens: int):
        """Update quota usage."""
        quota_key = f"{scope}:{scope_id}"

        if quota_key not in self._quotas:
            return

        quota = self._quotas[quota_key]
        now = datetime.now()

        # Check for daily reset
        if now.date() > quota.daily_reset.date():
            quota.current_daily = 0
            quota.daily_reset = now

        # Check for monthly reset
        if now.month != quota.monthly_reset.month or now.year != quota.monthly_reset.year:
            quota.current_monthly = 0
            quota.monthly_reset = now

        quota.current_daily += tokens
        quota.current_monthly += tokens

        # Update metrics
        self.ai_quota_usage.labels(
            scope=scope,
            scope_id=scope_id,
            period="daily",
        ).set(quota.daily_usage_percent)

        self.ai_quota_usage.labels(
            scope=scope,
            scope_id=scope_id,
            period="monthly",
        ).set(quota.monthly_usage_percent)

        # Check quota alerts
        if quota.daily_usage_percent > 80 or quota.monthly_usage_percent > 80:
            for callback in self._quota_alert_callbacks:
                try:
                    callback(quota)
                except Exception as e:
                    logger.error(f"Quota callback error: {e}")

    def set_quota(
        self,
        scope: str,
        scope_id: str,
        daily_limit: int,
        monthly_limit: int,
    ):
        """Set token quota.

        Args:
            scope: Quota scope (user, api_key, application)
            scope_id: Scope identifier
            daily_limit: Daily token limit
            monthly_limit: Monthly token limit
        """
        quota_key = f"{scope}:{scope_id}"

        with self._lock:
            self._quotas[quota_key] = TokenQuota(
                quota_id=quota_key,
                scope=scope,
                scope_id=scope_id,
                daily_limit=daily_limit,
                monthly_limit=monthly_limit,
            )

    def check_quota(
        self,
        scope: str,
        scope_id: str,
        tokens: int,
    ) -> tuple[bool, Optional[TokenQuota]]:
        """Check if request is within quota.

        Args:
            scope: Quota scope
            scope_id: Scope identifier
            tokens: Required tokens

        Returns:
            (allowed, quota) tuple
        """
        quota_key = f"{scope}:{scope_id}"

        with self._lock:
            quota = self._quotas.get(quota_key)

        if not quota:
            return True, None

        if quota.daily_remaining < tokens:
            return False, quota

        if quota.monthly_remaining < tokens:
            return False, quota

        return True, quota

    def _check_cost_alerts(self, provider: str, cost: float):
        """Check and trigger cost alerts."""
        if self.monthly_budget_usd is None:
            return

        with self._lock:
            total_cost = sum(pm.total_cost_usd for pm in self._provider_metrics.values())

        remaining = self.monthly_budget_usd - total_cost
        self.ai_budget_remaining.set(remaining)

        # Alert at 80% and 100%
        usage_percent = total_cost / self.monthly_budget_usd * 100

        if usage_percent >= 100:
            alert_msg = f"Budget exceeded! Used ${total_cost:.2f} of ${self.monthly_budget_usd:.2f}"
        elif usage_percent >= 80:
            alert_msg = f"Budget warning: 80% used (${total_cost:.2f} of ${self.monthly_budget_usd:.2f})"
        else:
            return

        for callback in self._cost_alert_callbacks:
            try:
                callback(alert_msg, total_cost)
            except Exception as e:
                logger.error(f"Cost alert callback error: {e}")

    def on_cost_alert(self, callback: Callable[[str, float], None]):
        """Register cost alert callback.

        Args:
            callback: Function(message, total_cost)
        """
        self._cost_alert_callbacks.append(callback)

    def on_quota_alert(self, callback: Callable[[TokenQuota], None]):
        """Register quota alert callback.

        Args:
            callback: Function(quota)
        """
        self._quota_alert_callbacks.append(callback)

    def get_model_usage(self, provider: str, model: str) -> Optional[ModelUsage]:
        """Get usage for a specific model.

        Args:
            provider: Provider name
            model: Model name

        Returns:
            ModelUsage or None
        """
        with self._lock:
            return self._model_usage.get(f"{provider}:{model}")

    def get_provider_metrics(self, provider: str) -> Optional[AIProviderMetrics]:
        """Get metrics for a provider.

        Args:
            provider: Provider name

        Returns:
            AIProviderMetrics or None
        """
        with self._lock:
            return self._provider_metrics.get(provider)

    def get_cost_tracking(
        self,
        period_days: int = 30,
    ) -> CostTracking:
        """Get cost tracking summary.

        Args:
            period_days: Period to analyze

        Returns:
            CostTracking
        """
        now = datetime.now()
        period_start = now - timedelta(days=period_days)

        with self._lock:
            requests = [
                r for r in self._request_history
                if r["timestamp"] >= period_start
            ]

        by_provider: Dict[str, float] = {}
        by_model: Dict[str, float] = {}
        by_user: Dict[str, float] = {}
        by_application: Dict[str, float] = {}

        total_cost = 0.0

        for r in requests:
            cost = r["cost_usd"]
            total_cost += cost

            by_provider[r["provider"]] = by_provider.get(r["provider"], 0) + cost
            by_model[r["model"]] = by_model.get(r["model"], 0) + cost

            if r.get("user_id"):
                by_user[r["user_id"]] = by_user.get(r["user_id"], 0) + cost
            if r.get("application"):
                by_application[r["application"]] = by_application.get(r["application"], 0) + cost

        # Project monthly cost
        days_in_period = (now - period_start).days or 1
        daily_avg = total_cost / days_in_period
        projected_monthly = daily_avg * 30

        # Budget tracking
        budget_remaining = None
        overage_alerts = []

        if self.monthly_budget_usd:
            budget_remaining = self.monthly_budget_usd - total_cost
            if budget_remaining < 0:
                overage_alerts.append(f"Over budget by ${-budget_remaining:.2f}")
            elif projected_monthly > self.monthly_budget_usd:
                overage_alerts.append(
                    f"Projected to exceed budget by ${projected_monthly - self.monthly_budget_usd:.2f}"
                )

        return CostTracking(
            period_start=period_start,
            period_end=now,
            total_cost_usd=total_cost,
            by_provider=by_provider,
            by_model=by_model,
            by_user=by_user,
            by_application=by_application,
            projected_monthly_usd=projected_monthly,
            budget_usd=self.monthly_budget_usd,
            budget_remaining_usd=budget_remaining,
            overage_alerts=overage_alerts,
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get AI gateway summary.

        Returns:
            Summary dictionary
        """
        with self._lock:
            providers = dict(self._provider_metrics)
            models = dict(self._model_usage)

        total_requests = sum(p.request_count for p in providers.values())
        total_cost = sum(p.total_cost_usd for p in providers.values())
        total_tokens = sum(p.total_tokens for p in providers.values())

        # Top models by cost
        top_models = sorted(
            models.values(),
            key=lambda m: m.total_cost_usd,
            reverse=True
        )[:5]

        return {
            "total_requests": total_requests,
            "total_cost_usd": total_cost,
            "total_tokens": total_tokens,
            "providers_used": len(providers),
            "models_used": len(models),
            "avg_cost_per_request": total_cost / total_requests if total_requests > 0 else 0,
            "top_models_by_cost": [
                {
                    "model": m.model,
                    "provider": m.provider.value,
                    "cost_usd": m.total_cost_usd,
                    "requests": m.request_count,
                }
                for m in top_models
            ],
            "provider_breakdown": {
                p: {
                    "requests": pm.request_count,
                    "cost_usd": pm.total_cost_usd,
                    "error_rate": pm.error_rate,
                }
                for p, pm in providers.items()
            },
            "budget_status": {
                "monthly_budget": self.monthly_budget_usd,
                "remaining": self.monthly_budget_usd - total_cost if self.monthly_budget_usd else None,
            } if self.monthly_budget_usd else None,
        }
